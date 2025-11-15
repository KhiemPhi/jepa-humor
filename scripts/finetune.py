import argparse, yaml, os
from jepa_humor.finetune import finetune

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune.yml")
    p.add_argument("--encoder_ckpt", type=str, default="checkpoints/pretrain/best.pt")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only (requires --model_ckpt)")
    p.add_argument("--model_ckpt", type=str, default=None, help="Path to model checkpoint for evaluation (required if --eval_only)")
    args = p.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    if args.eval_only and args.model_ckpt is None:
        # Try to use default checkpoint path from config
        ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
        default_ckpt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(default_ckpt):
            print(f"Using default checkpoint: {default_ckpt}")
            args.model_ckpt = default_ckpt
        else:
            raise ValueError("--model_ckpt must be provided when --eval_only is set")
    
    model = finetune(cfg, encoder_ckpt=args.encoder_ckpt, eval_only=args.eval_only, model_ckpt=args.model_ckpt)
    if not args.eval_only:
        print("✅ Finetune done.")
