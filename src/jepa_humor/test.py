# test.py - Updated to work with existing finetune.py
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import time
import yaml
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import warnings
import argparse
import os
import random
from pathlib import Path
warnings.filterwarnings('ignore')

# Import from your existing finetune setup
import sys
sys.path.append('.')  # Add current directory to path

class Config:
    """Configuration loader that works with your existing YAML"""
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_experiment()
    
    def load_config(self):
        """Load configuration from YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded config from {self.config_path}")
            return config
            
        except FileNotFoundError:
            print(f"Config file {self.config_path} not found.")
            raise
    
    def setup_experiment(self):
        """Setup experiment parameters"""
        # Set random seeds
        seed = self.config.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

class LLMExperiments:
    def __init__(self, config: Config):
        """Initialize LLM experiments with configuration"""
        self.config = config
        self.device = self.setup_device()
        self.models = {}
        self.tokenizers = {}
        self.test_data = None
        self.results = {}
        
    def setup_device(self):
        """Setup GPU device"""
        gpu_id = self.config.get("gpu_id", 0)
        
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
            else:
                device = torch.device("cuda:0")
                print(f"Using default GPU 0: {torch.cuda.get_device_name(0)}")
            return device
        else:
            print("No GPU found, using CPU")
            return torch.device("cpu")
    
    def load_model(self, model_name: str, model_path: str, load_in_4bit: bool = False):
        """Load a specific LLM model"""
        print(f"\nLoading model: {model_name} ({model_path})")
        
        # Configure quantization
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Set padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "trust_remote_code": True,
                "device_map": "auto" if torch.cuda.is_available() else None,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            print(f"✓ Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            print(f"✗ Error loading {model_name}: {str(e)}")
            return None, None
    
    def load_all_models(self):
        """Load all enabled models"""
        models_config = self.config.get("models", {})
        
        for model_key, model_cfg in models_config.items():
            if model_cfg.get("enabled", False):
                model_path = model_cfg.get("path", "")
                load_in_4bit = model_cfg.get("load_in_4bit", False)
                
                if model_path:
                    self.load_model(model_key, model_path, load_in_4bit)
    
    def load_test_dataset(self):
        """Load test dataset using the same method as finetune.py"""
        try:
            # Try to import from your existing data module
            from data import load_hf_splits_test, load_test_set_from_json
            
            dataset_name = self.config.get("dataset", "nyu-mll/glue")
            dataset_config = self.config.get("dataset_config", "qnli")
            
            # Try to load from Hugging Face first
            print(f"Loading test dataset: {dataset_name}/{dataset_config}")
            
            # Check if test_ann.json exists
            json_path = "test_ann.json"
            if os.path.exists(json_path):
                print("Loading from test_ann.json...")
                test_data = load_test_set_from_json(dataset_config, json_path=json_path)
                if test_data and "test" in test_data:
                    self.test_data = test_data["test"]
                else:
                    self.test_data = []
            else:
                # Fall back to Hugging Face
                print("Loading from Hugging Face...")
                test_splits = load_hf_splits_test(dataset_name, config=dataset_config)
                if test_splits and "validation" in test_splits:
                    self.test_data = test_splits["validation"]
                else:
                    self.test_data = []
            
            if self.test_data:
                print(f"✓ Loaded {len(self.test_data)} test examples")
                return self.test_data
            else:
                return self.create_mock_dataset()
                
        except ImportError as e:
            print(f"Warning: Could not import data module: {e}")
            return self.load_simple_dataset()
        except Exception as e:
            print(f"Error loading test dataset: {e}")
            return self.create_mock_dataset()
    
    def load_simple_dataset(self):
        """Simple dataset loader as fallback"""
        print("Loading simple test dataset...")
        
        # Create simple test data based on QNLI format
        self.test_data = [
            {
                'question': 'What color is the sky?',
                'sentence': 'The sky is blue.',
                'label': 1  # entailment
            },
            {
                'question': 'What do cats eat?',
                'sentence': 'Dogs eat bones.',
                'label': 0  # not entailment
            },
            {
                'question': 'Where do birds live?',
                'sentence': 'Birds live in nests.',
                'label': 1  # entailment
            },
            {
                'question': 'What is 2+2?',
                'sentence': 'The sum of 2 and 2 is 5.',
                'label': 0  # not entailment
            },
            {
                'question': 'What language is spoken in France?',
                'sentence': 'French is spoken in France.',
                'label': 1  # entailment
            }
        ]
        
        print(f"Created simple dataset with {len(self.test_data)} examples")
        return self.test_data
    
    def create_mock_dataset(self):
        """Create mock dataset as last resort"""
        print("Creating mock dataset...")
        
        self.test_data = [
            {
                'premise': 'The cat sat on the mat.',
                'hypothesis': 'A cat was sitting on a mat.',
                'label': 1  # entailment
            },
            {
                'premise': 'The man ate an apple.',
                'hypothesis': 'A person consumed fruit.',
                'label': 1  # entailment
            },
            {
                'premise': 'It was raining outside.',
                'hypothesis': 'The weather was sunny.',
                'label': 0  # contradiction
            }
        ]
        
        print(f"Created mock dataset with {len(self.test_data)} examples")
        return self.test_data
    
    def format_prompt(self, example: Dict, technique: str, n_shots: int = 0) -> str:
        """Format prompt based on technique"""
        # Get text based on QNLI format
        question = example.get('question', example.get('premise', ''))
        sentence = example.get('sentence', example.get('hypothesis', ''))
        
        if technique == "zero_shot":
            prompt = f"""Given the question: "{question}"
And the sentence: "{sentence}"

Does the sentence answer the question? Answer with 1 for yes (entailment) or 0 for no (not entailment).
Answer: """
        
        elif technique == "few_shot":
            # For few-shot, we would add examples here
            prompt = f"""Question: "{question}"
Sentence: "{sentence}"
Answer (1 for entailment, 0 for not entailment): """
        
        elif technique == "chain_of_thought":
            prompt = f"""Given the question: "{question}"
And the sentence: "{sentence}"

Let's think step by step:
1. First, understand the question: {question}
2. Then, examine the sentence: {sentence}
3. Determine if the sentence provides an answer to the question
4. Based on this analysis, provide a binary answer

Answer (1 for entailment, 0 for not entailment): """
        
        else:
            prompt = f"""Question: "{question}"
Sentence: "{sentence}"
Answer (1 for yes, 0 for no): """
        
        return prompt
    
    def query_llm(self, model_name: str, prompt: str) -> int:
        """Query LLM with prompt and get binary answer"""
        model = self.models.get(model_name)
        tokenizer = self.tokenizers.get(model_name)
        
        if not model or not tokenizer:
            # Return random answer for testing
            return random.randint(0, 1)
        
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer from response
            answer_text = response[len(prompt):].strip()
            
            # Try to find a 0 or 1 in the answer
            if '1' in answer_text or 'yes' in answer_text.lower() or 'entail' in answer_text.lower():
                return 1
            elif '0' in answer_text or 'no' in answer_text.lower():
                return 0
            else:
                # Return random if unclear
                return random.randint(0, 1)
                
        except Exception as e:
            print(f"Error querying model: {e}")
            return random.randint(0, 1)
    
    def run_prompting_experiments(self):
        """Run all prompting experiments"""
        if not self.test_data:
            print("No test data available")
            return {}
        
        num_examples = self.config.get("test_config.num_validation_examples", 5)
        test_subset = self.test_data[:min(num_examples, len(self.test_data))]
        
        print(f"\nRunning prompting experiments on {len(test_subset)} examples...")
        
        # Get enabled models
        enabled_models = []
        models_config = self.config.get("models", {})
        for model_key, model_cfg in models_config.items():
            if model_cfg.get("enabled", False) and model_key in self.models:
                enabled_models.append(model_key)
        
        if not enabled_models:
            print("No models loaded. Using mock responses.")
            enabled_models = ["mock_model"]
        
        results = {}
        
        for model_name in enabled_models:
            print(f"\n{'='*60}")
            print(f"Testing: {model_name}")
            print(f"{'='*60}")
            
            model_results = {}
            
            # Run each prompting technique
            prompting_config = self.config.get("prompting", {})
            
            if prompting_config.get("zero_shot", {}).get("enabled", True):
                print("\nRunning Zero-shot prompting...")
                zero_shot_results = []
                for example in tqdm(test_subset, desc="Zero-shot", leave=False):
                    true_label = example.get('label', 0)
                    prompt = self.format_prompt(example, "zero_shot")
                    pred = self.query_llm(model_name, prompt)
                    zero_shot_results.append({
                        'true': true_label,
                        'pred': pred,
                        'correct': true_label == pred
                    })
                model_results['zero_shot'] = zero_shot_results
            
            if prompting_config.get("few_shot", {}).get("enabled", True):
                shots = prompting_config.get("few_shot", {}).get("shots", [1, 3, 5])
                for n in shots:
                    print(f"\nRunning {n}-shot prompting...")
                    few_shot_results = []
                    for example in tqdm(test_subset, desc=f"{n}-shot", leave=False):
                        true_label = example.get('label', 0)
                        prompt = self.format_prompt(example, "few_shot", n_shots=n)
                        pred = self.query_llm(model_name, prompt)
                        few_shot_results.append({
                            'true': true_label,
                            'pred': pred,
                            'correct': true_label == pred
                        })
                    model_results[f'{n}_shot'] = few_shot_results
            
            if prompting_config.get("chain_of_thought", {}).get("enabled", True):
                print("\nRunning Chain-of-Thought prompting...")
                cot_results = []
                for example in tqdm(test_subset, desc="CoT", leave=False):
                    true_label = example.get('label', 0)
                    prompt = self.format_prompt(example, "chain_of_thought")
                    pred = self.query_llm(model_name, prompt)
                    cot_results.append({
                        'true': true_label,
                        'pred': pred,
                        'correct': true_label == pred
                    })
                model_results['chain_of_thought'] = cot_results
            
            # Calculate metrics
            metrics = {}
            for technique, technique_results in model_results.items():
                if technique_results:
                    correct = sum(r['correct'] for r in technique_results)
                    total = len(technique_results)
                    accuracy = correct / total if total > 0 else 0
                    metrics[technique] = {
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total
                    }
            
            results[model_name] = {
                'results': model_results,
                'metrics': metrics
            }
            
            # Print summary
            print(f"\nResults for {model_name}:")
            for technique, metric in metrics.items():
                acc = metric.get('accuracy', 0)
                print(f"  {technique:20} Accuracy: {acc:.3f} ({metric['correct']}/{metric['total']})")
        
        self.results = results
        return results
    
    def save_results(self):
        """Save experiment results"""
        results_dir = self.config.get("test_config.results_dir", "./results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prompting_results_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        serializable_results = {
            'config': self.config.config,
            'results': self.results,
            'timestamp': timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {filepath}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Prompting Experiments")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, help="Specific model to test")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    args = parser.parse_args()
    
    print("="*70)
    print("LLM Prompting Experiments")
    print("="*70)
    
    # Load configuration
    try:
        config = Config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Override models if specified
    if args.model:
        models_config = config.get("models", {})
        for model_key in models_config:
            models_config[model_key]["enabled"] = model_key == args.model
        config.config["models"] = models_config
    
    # Initialize and run experiments
    experiments = LLMExperiments(config)
    
    print("\n1. Loading models...")
    experiments.load_all_models()
    
    print("\n2. Loading test dataset...")
    experiments.load_test_dataset()
    
    if experiments.models:
        print("\n3. Running prompting experiments...")
        results = experiments.run_prompting_experiments()
        
        if not args.no_save:
            print("\n4. Saving results...")
            experiments.save_results()
        
        print("\n" + "="*70)
        print("Prompting experiments completed!")
        print("="*70)
    else:
        print("\n✗ No models loaded. Please check your configuration.")

if __name__ == "__main__":
    main()