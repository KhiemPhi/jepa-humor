import argparse, yaml, os
from jepa_humor.finetune_xlm import finetune_xlm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune_squad_xquad.yml")
    p.add_argument("--jepa_adapter", type=str, default="fresh",
                   help="Path to a JEPA LoRA adapter dir to merge into the encoder, or 'fresh'")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--model_ckpt", type=str, default=None,
                   help="save_pretrained dir of a finetuned QA-LoRA adapter (for --eval_only)")
    p.add_argument("--gpu_id", type=int, default=None)
    args, overrides = p.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    finetune_xlm(cfg, jepa_adapter=args.jepa_adapter,
                 eval_only=args.eval_only, model_ckpt=args.model_ckpt)
    print("✅ finetune_xlm done.")
