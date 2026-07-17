#!/usr/bin/env python
"""
Export the ENCODER BODY from a use_hf_classifier finetune checkpoint into an
encoder_state_dict .pt that finetune.py's --encoder_ckpt can load (JEPA branch).

Use case: MNLI-transfer (AdvGLUE inits rte/mrpc/stsb from deberta-large-mnli).
We finetune our JEPA encoder on MNLI, then export its deberta body here, and pass
that as --encoder_ckpt to the rte/mrpc/stsb finetunes.

Usage:
    python -m scripts.export_finetuned_encoder <in_ckpt.pt> <out_encoder.pt> [base_model_prefix]

base_model_prefix defaults to 'deberta' (DebertaForSequenceClassification.base_model_prefix).
"""
import sys, torch

def main():
    in_ckpt, out_ckpt = sys.argv[1], sys.argv[2]
    prefix = (sys.argv[3] if len(sys.argv) > 3 else "deberta").rstrip(".") + "."
    ckpt = torch.load(in_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    # keys may be wrapped, e.g. 'hf_model.deberta.embeddings...' -> keep everything after 'deberta.'
    def after(sd, marker):
        out = {}
        for k, v in sd.items():
            i = k.find(marker)
            if i != -1:
                out[k[i + len(marker):]] = v
        return out
    body = after(sd, prefix)                       # 'deberta.'
    if not body:
        body = after(sd, "encoder.")               # custom classifier fallback
    if not body:
        raise SystemExit(f"FATAL: no '{prefix}' or 'encoder.' keys found in {in_ckpt}; "
                         f"got sample keys: {list(sd.keys())[:5]}")
    torch.save({"encoder_state_dict": body}, out_ckpt)
    print(f"[export] {len(body)} body tensors ({prefix[:-1]}) -> {out_ckpt}")

if __name__ == "__main__":
    main()
