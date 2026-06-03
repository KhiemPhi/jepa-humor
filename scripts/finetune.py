import argparse, yaml, os
from jepa_humor.finetune import finetune
from jepa_humor.finetune_wnli import finetune_wnli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune.yml")
    p.add_argument("--encoder_ckpt", type=str, default="fresh")
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only (requires --model_ckpt)")
    p.add_argument("--model_ckpt", type=str, default=None, help="Path to model checkpoint for evaluation (required if --eval_only)")
    p.add_argument("--gpu_id", type=int, default=None)
    args, overrides = p.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    if args.encoder_ckpt == "fresh" and "experiment_name" in cfg:
        cfg["experiment_name"] = cfg["experiment_name"] + "_fresh"

    if args.eval_only and args.model_ckpt is None:
        ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
        default_ckpt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(default_ckpt):
            print(f"Using default checkpoint: {default_ckpt}")
            args.model_ckpt = default_ckpt
        else:
            raise ValueError("--model_ckpt must be provided when --eval_only is set")

    if cfg.get("dataset_config") == "wnli":
        model = finetune_wnli(cfg, encoder_ckpt=args.encoder_ckpt)
    else:
        model = finetune(cfg, encoder_ckpt=args.encoder_ckpt, eval_only=args.eval_only, model_ckpt=args.model_ckpt)
    if not args.eval_only:
        print("✅ Finetune done.")
