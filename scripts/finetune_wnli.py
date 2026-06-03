import argparse, yaml
from jepa_humor.finetune_wnli import finetune_wnli

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune_wnli.yml")
    p.add_argument("--encoder_ckpt", type=str, default="fresh")
    p.add_argument("--gpu_id", type=int, default=None)
    args, overrides = p.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)

    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    if args.encoder_ckpt == "fresh" and "experiment_name" in cfg:
        cfg["experiment_name"] = cfg["experiment_name"] + "_fresh"

    finetune_wnli(cfg, encoder_ckpt=args.encoder_ckpt)
    print("✅ WNLI finetune done.")
