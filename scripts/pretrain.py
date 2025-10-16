import argparse, yaml
from jepa_humor.train_ssl import pretrain

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pretrain.yml")
    args = p.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    best_ckpt = pretrain(cfg)
    print(f"✅ Best encoder checkpoint: {best_ckpt}")
