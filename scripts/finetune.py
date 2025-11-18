import argparse, yaml
from jepa_humor.finetune import finetune

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune.yml")
    p.add_argument("--encoder_ckpt", type=str, default="checkpoints/pretrain/best.pt")
    args = p.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    model = finetune(cfg, encoder_ckpt=args.encoder_ckpt)
    print("✅ Finetune done.")