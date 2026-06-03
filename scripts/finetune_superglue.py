"""
Entry point for SuperGLUE fine-tuning.
  BoolQ, CB, WiC  → standard finetune() (CLS classification)
  COPA            → finetune_copa()
  MultiRC         → finetune_multirc()
  WSC             → finetune_wnli() (super_glue/wsc, MLM scoring)
  RTE             → finetune_superglue_rte() (super_glue/rte, binary NLI)
  ReCoRD          → finetune_record() (super_glue/record, entity scoring)
"""
import argparse, yaml, os
from jepa_humor.finetune import finetune
from jepa_humor.finetune_wnli import finetune_wnli
from jepa_humor.finetune_superglue import (
    finetune_copa, finetune_multirc, finetune_superglue_rte,
    finetune_record, finetune_superglue_wic,
)

_STANDARD = {"boolq", "cb"}   # WiC moved to _CUSTOM for word-in-context support
_CUSTOM   = {
    "copa":    finetune_copa,
    "multirc": finetune_multirc,
    "wsc":     finetune_wnli,
    "rte":     finetune_superglue_rte,
    "record":  finetune_record,
    "wic":     finetune_superglue_wic,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config",       type=str, required=True)
    p.add_argument("--encoder_ckpt", type=str, default="fresh")
    p.add_argument("--gpu_id",       type=int, default=None)
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

    task = cfg.get("dataset_config", "")
    if task in _STANDARD:
        finetune(cfg, encoder_ckpt=args.encoder_ckpt)
    elif task in _CUSTOM:
        _CUSTOM[task](cfg, encoder_ckpt=args.encoder_ckpt)
    else:
        raise ValueError(f"Unknown SuperGLUE task: '{task}'. Expected one of {sorted(_STANDARD | set(_CUSTOM.keys()))}")

    print("✅ SuperGLUE finetune done.")
