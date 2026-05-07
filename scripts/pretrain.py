import argparse, yaml
from jepa_humor.train_ssl import pretrain

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/pretrain.yml")
    p.add_argument("--gpu_id", type=int, default=None)
    args, overrides = p.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    best_ckpt = pretrain(cfg)
    print(f"✅ Best encoder checkpoint: {best_ckpt}")
