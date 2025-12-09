import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for linear layers.
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling

class LoRAWrappedLinear(nn.Module):
    """
    Wraps a linear layer with LoRA.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank,
            alpha
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

class LoRAConfig:
    """
    Configuration for LoRA fine-tuning.
    """
    def __init__(
        self,
        r: int = 8,
        lora_alpha: int = 32,
        target_modules: Optional[List[str]] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",  # or "SEQ_CLS" for classification
        inference_mode: bool = False
    ):
        self.r = r  # rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules or ["q_proj", "v_proj"]  # default for Llama models
        self.lora_dropout = lora_dropout
        self.bias = bias
        self.task_type = task_type
        self.inference_mode = inference_mode
        
    def to_peft_config(self) -> LoraConfig:
        """Convert to PEFT library configuration."""
        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS
        }
        
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=task_type_map.get(self.task_type, TaskType.CAUSAL_LM),
            inference_mode=self.inference_mode
        )

class LoRATransformer(nn.Module):
    """
    Main LoRA-enhanced transformer model.
    Supports both causal LM and sequence classification.
    """
    def __init__(
        self,
        model_name_or_path: str,
        lora_config: LoRAConfig,
        num_labels: Optional[int] = None,
        use_peft: bool = True
    ):
        super().__init__()
        
        # Load base model
        if lora_config.task_type == "CAUSAL_LM":
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        elif lora_config.task_type == "SEQ_CLS":
            if num_labels is None:
                raise ValueError("num_labels must be provided for sequence classification")
            self.base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
        else:
            raise ValueError(f"Unsupported task type: {lora_config.task_type}")
        
        self.task_type = lora_config.task_type
        self.num_labels = num_labels
        
        if use_peft:
            # Use PEFT library for LoRA
            self.model = get_peft_model(
                self.base_model,
                lora_config.to_peft_config()
            )
            self.use_custom_lora = False
        else:
            # Use custom LoRA implementation
            self.model = self.base_model
            self._apply_custom_lora(lora_config)
            self.use_custom_lora = True
            
        # Freeze base model parameters (only train LoRA)
        self._freeze_base_model()
        
    def _apply_custom_lora(self, config: LoRAConfig):
        """Apply custom LoRA layers to the model."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check if this module should have LoRA
                should_apply = any(target in name for target in config.target_modules)
                if should_apply:
                    # Replace with LoRA-wrapped version
                    parent_name = ".".join(name.split(".")[:-1])
                    module_name = name.split(".")[-1]
                    parent_module = self.model.get_submodule(parent_name)
                    setattr(
                        parent_module,
                        module_name,
                        LoRAWrappedLinear(module, config.r, config.lora_alpha)
                    )
    
    def _freeze_base_model(self):
        """Freeze all parameters except LoRA ones."""
        for name, param in self.model.named_parameters():
            if 'lora' not in name.lower() and not self.use_custom_lora:
                param.requires_grad = False
            elif self.use_custom_lora and ('lora_A' in name or 'lora_B' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs)
    
    def generate(self, **kwargs):
        """Generate text (for causal LM)."""
        if self.task_type != "CAUSAL_LM":
            raise ValueError("Generate method only available for causal LM models")
        return self.model.generate(**kwargs)
    
    def print_trainable_parameters(self):
        """Print trainable parameters."""
        trainable_params = 0
        all_params = 0
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.4f}%"
        )

class LoRALoss(nn.Module):
    """
    Combined loss function for LoRA fine-tuning.
    Supports various tasks and optional regularization.
    """
    def __init__(
        self,
        task_type: str = "CAUSAL_LM",
        alpha_ce: float = 1.0,
        alpha_lm: float = 0.1,
        beta_lora: float = 0.01,
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        super().__init__()
        self.task_type = task_type
        self.alpha_ce = alpha_ce  # Cross-entropy weight
        self.alpha_lm = alpha_lm  # Language modeling weight (for mixed training)
        self.beta_lora = beta_lora  # LoRA regularization weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        # Base losses
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index
        )
        self.mse_loss = nn.MSELoss()
        
    def compute_lora_regularization(self, model: nn.Module) -> torch.Tensor:
        """Compute regularization loss for LoRA parameters."""
        reg_loss = 0.0
        for name, param in model.named_parameters():
            if 'lora' in name.lower() and param.requires_grad:
                reg_loss += torch.norm(param, p=2)  # L2 regularization
        return reg_loss * self.beta_lora
    
    def forward(
        self,
        model: nn.Module,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        lm_logits: Optional[torch.Tensor] = None,
        lm_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute total loss.
        
        Args:
            model: The LoRA model
            logits: Model output logits
            labels: Ground truth labels
            attention_mask: Attention mask
            lm_logits: Language modeling logits (for mixed training)
            lm_labels: Language modeling labels
        """
        total_loss = 0.0
        
        # Task-specific loss
        if self.task_type == "SEQ_CLS":
            if labels is None:
                raise ValueError("Labels required for sequence classification")
            
            if self.num_labels == 1:  # Regression
                loss = self.mse_loss(logits.squeeze(), labels.float())
            else:  # Classification
                loss = self.ce_loss(logits, labels)
                
            total_loss += self.alpha_ce * loss
            
            # Optional language modeling auxiliary loss
            if lm_logits is not None and lm_labels is not None and self.alpha_lm > 0:
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = lm_labels[..., 1:].contiguous()
                
                if attention_mask is not None:
                    shift_attention_mask = attention_mask[..., 1:].contiguous()
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    lm_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    lm_loss = (lm_loss * shift_attention_mask.view(-1)).sum() / shift_attention_mask.sum()
                else:
                    lm_loss = self.ce_loss(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                
                total_loss += self.alpha_lm * lm_loss
                
        elif self.task_type == "CAUSAL_LM":
            if labels is None:
                raise ValueError("Labels required for causal LM")
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                lm_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss = (lm_loss * shift_attention_mask.view(-1)).sum() / shift_attention_mask.sum()
            else:
                loss = self.ce_loss(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
            total_loss += self.alpha_ce * loss
            
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        # LoRA regularization
        if self.beta_lora > 0:
            lora_reg = self.compute_lora_regularization(model)
            total_loss += lora_reg
            
        return total_loss

class LoRATrainer(Trainer):
    """
    Custom trainer for LoRA fine-tuning with mixed precision support.
    """
    def __init__(self, loss_fn: Optional[LoRALoss] = None, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = loss_fn
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation that supports LoRA regularization.
        """
        # Forward pass
        outputs = model(**inputs)
        
        # Use custom loss if provided, otherwise use default
        if self.loss_fn is not None:
            logits = outputs.logits
            
            # Extract labels if present
            labels = inputs.get("labels")
            
            # Compute loss with custom loss function
            loss = self.loss_fn(
                model=model,
                logits=logits,
                labels=labels,
                attention_mask=inputs.get("attention_mask"),
                lm_logits=logits if self.loss_fn.task_type == "SEQ_CLS" else None,
                lm_labels=labels if self.loss_fn.task_type == "SEQ_CLS" else None
            )
        else:
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs):
        """
        Custom training step with gradient accumulation support.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
            
        if self.args.n_gpu > 1:
            loss = loss.mean()
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        return loss.detach()

def create_lora_model_for_glue(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    task_name: str = "sst-2",
    use_peft: bool = True,
    **lora_kwargs
) -> LoRATransformer:
    """
    Factory function to create LoRA model for GLUE tasks.
    """
    # Task configuration
    glue_config = {
        "cola": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "sst-2": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "mrpc": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "sts-b": {"num_labels": 1, "task_type": "SEQ_CLS"},  # Regression
        "qqp": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "mnli": {"num_labels": 3, "task_type": "SEQ_CLS"},
        "qnli": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "rte": {"num_labels": 2, "task_type": "SEQ_CLS"},
        "wnli": {"num_labels": 2, "task_type": "SEQ_CLS"},
    }
    
    if task_name not in glue_config:
        raise ValueError(f"Unknown task: {task_name}")
    
    config = glue_config[task_name]
    
    # Default LoRA config for GLUE
    default_lora_config = LoRAConfig(
        r=lora_kwargs.get("r", 8),
        lora_alpha=lora_kwargs.get("lora_alpha", 32),
        target_modules=lora_kwargs.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=lora_kwargs.get("lora_dropout", 0.1),
        task_type=config["task_type"]
    )
    
    # Create model
    model = LoRATransformer(
        model_name_or_path=model_name,
        lora_config=default_lora_config,
        num_labels=config["num_labels"],
        use_peft=use_peft
    )
    
    return model

# Example usage:
if __name__ == "__main__":
    # Create LoRA model for sentiment analysis
    model = create_lora_model_for_glue(
        model_name="meta-llama/Llama-2-7b-hf",
        task_name="sst-2",
        use_peft=True,
        r=16,
        lora_alpha=32
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create loss function
    loss_fn = LoRALoss(
        task_type="SEQ_CLS",
        alpha_ce=1.0,
        alpha_lm=0.1,  # Optional auxiliary LM loss
        beta_lora=0.01,  # LoRA regularization
        label_smoothing=0.1
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")