#!/usr/bin/env python
"""Build final/beir_arms/ and final/tapt/ in the same shape as final/beir/ (user 2026-10-05).

beir_arms: NDCG@10 per BEIR dataset for the three DPR pretraining objectives, each at DPR ckpt2
and ckpt39 -- the QA-corpus arms (experiment/dpr_qapre): every arm pretrains on the DPR multiset
passages themselves ([CLS] title [SEP] passage [SEP], one epoch = 48,886 steps, lr 5e-6, batch 128),
then the same DPR bs256 x 40 epochs. Only the objective differs: jepa (knn_weight 0.1), knn0
(knn_weight 0.0), mlm (BERT MLM, pretrained head). (Not the MS MARCO 5k-step arms.)

tapt: per Don't-Stop-Pretraining task and arm, the best finished 5-seed run (test F1 at best dev,
mean over seeds) with every candidate run and every per-seed value + source file.

Re-runnable: reads result files only, writes final/beir_arms/ and final/tapt/.
"""
import csv, glob, json, os, re
from pathlib import Path

ROOT = Path("/home/tudao/jepa-humor"); FINAL = ROOT / "final"
rel = lambda p: os.path.relpath(p, ROOT)

# ------------------------------------------------------------------ BEIR arms
DPR = ROOT / "experiment/dpr_qapre"
ARMS = {a: {ck: DPR / f"results_beir_qapre_{a}_{ck}" for ck in ("ckpt2", "ckpt39")} for a in ("jepa", "knn0", "mlm")}
DATASETS = ["msmarco", "trec-covid", "nfcorpus", "nq", "hotpotqa", "fiqa", "arguana", "webis-touche2020",
            "cqadupstack", "quora", "dbpedia-entity", "scidocs", "fever", "climate-fever", "scifact"]


def beir_arms(rows):
    out = FINAL / "beir_arms"; out.mkdir(exist_ok=True)
    per = {}  # (arm, ckpt) -> {dataset: value}
    for ds in DATASETS:
        old = json.load(open(FINAL / "beir" / f"{ds}.json"))   # label + public DPR from the report
        cell = {"dataset": ds, "label": old["label"], "ndcg@10": {}, "public_dpr": old["public_dpr"],
                "source_files": {}}
        for arm, cks in ARMS.items():
            for ck, d in cks.items():
                f = d / f"{ds}.json"; key = f"{arm}_{ck}"
                if not f.exists():
                    cell["ndcg@10"][key] = None; cell["source_files"][key] = None; continue
                v = json.load(open(f))["ndcg"]["NDCG@10"]
                cell["ndcg@10"][key] = v; cell["source_files"][key] = rel(f)
                per.setdefault(key, {})[ds] = v
                rows.append(["beir_arms", key, ds, "ndcg@10", "ours", "", f"{v:.6f}", f"{v:.4f}", "", rel(f)])
        have = {k: v for k, v in cell["ndcg@10"].items() if v is not None}
        cell["best"] = max(have, key=have.get) if have else None
        json.dump(cell, open(out / f"{ds}.json", "w"), indent=2)
        rows.append(["beir_arms", "public_dpr", ds, "ndcg@10", "published", "", f"{old['public_dpr']:.6f}",
                     f"{old['public_dpr']:.3f}", "BEIR Table 2 (arXiv 2104.08663)", ""])
    pub = {ds: json.load(open(FINAL / "beir" / f"{ds}.json"))["public_dpr"] for ds in DATASETS}
    zs = [d for d in DATASETS if d != "msmarco"]
    mean = lambda m, ds: sum(m[d] for d in ds) / len(ds) if all(d in m for d in ds) else None
    summ = {"mean_15": {k: mean(m, DATASETS) for k, m in per.items()},
            "mean_14_zero_shot": {k: mean(m, zs) for k, m in per.items()},
            "beats_public_dpr": {k: f"{sum(m[d] > pub[d] for d in m)}/{len(m)}" for k, m in per.items()},
            "wins_by_dataset": {},
            "note": ("MS MARCO is in-domain for DPR-style training, hence the separate 14-set mean. "
                     "None of the QA-pretrain arms was evaluated on msmarco, so mean_15 is null; "
                     "mean_14_zero_shot is the comparable number.")}
    summ["mean_15"]["public_dpr"] = mean(pub, DATASETS); summ["mean_14_zero_shot"]["public_dpr"] = mean(pub, zs)
    for ck in ("ckpt2", "ckpt39"):   # which objective wins each dataset, checkpoint-matched
        w = {a: 0 for a in ARMS}   # zero-shot sets only
        for d in zs:
            w[max(ARMS, key=lambda a: per[f"{a}_{ck}"][d])] += 1
        summ["wins_by_dataset"][ck] = w
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    for k, v in summ["mean_14_zero_shot"].items():
        rows.append(["beir_arms", k, "MEAN(14 zero-shot)", "ndcg@10", "mean", "", f"{v:.6f}", f"{v:.4f}", "", ""])
    return summ


# ------------------------------------------------------------------ TAPT
DSP = ROOT / "experiment/dsp/results"
TASKS = [  # our name, paper name, domain, paper RoBERTa, paper TAPT  (Gururangan et al. 2020, Table 5)
    ("chemprot", "CHEMPROT", "BioMed", 81.9, 82.6), ("rct-20k", "RCT", "BioMed", 87.2, 87.7),
    ("citation_intent", "ACL-ARC", "CS", 63.0, 67.4), ("sciie", "SCIERC", "CS", 77.3, 79.3),
    ("hyperpartisan_news", "HYPERPARTISAN", "News", 86.6, 90.4), ("ag", "AGNEWS", "News", 93.9, 94.5),
    ("amazon", "HELPFULNESS", "Reviews", 65.1, 68.5), ("imdb", "IMDB", "Reviews", 95.0, 95.5)]
MICRO = {"chemprot", "rct-20k"}


def describe(name):
    """Human-readable run description from a result file name."""
    if name.startswith("tapt100_"):
        ck = name.rsplit("_", 1)[1]
        return f"100-epoch TAPT (H100, batch 2048, lr 5e-6, fresh remask) {ck}"
    tapt = ("100-epoch TAPT" if "ep100" in name or "_fresh" not in name else "5-epoch TAPT") + \
           (" (fresh remask)" if "_fresh" in name else " (static mask)")
    ft = ("bs32 lr1e-5 wd0.1" if name.endswith("_wd01_bs32_lr1e5") else "bs32 lr2e-5 wd0.1" if name.endswith("_wd01_bs32")
          else "bs16 lr2e-5 wd0.1" if name.endswith("_wd01") else "bs16 lr2e-5 wd0.01")
    return f"{tapt}, finetune {ft}"


def tapt(rows):
    out = FINAL / "tapt"; out.mkdir(exist_ok=True)
    files = [f for f in glob.glob(str(DSP / "*.json")) + glob.glob(str(DSP / "tapt100/*.json"))
             if not re.search(r"\.s4\d\.json$|_seed4\d\.json$|3seeds|baseline", f)]
    summ = {"metric": "test F1 at the best-dev epoch, mean over 5 seeds (micro-F1 CHEMPROT/RCT, macro-F1 else)",
            "selection": "best run per (task, arm) by that mean, over every finished 5-seed run -- optimistic",
            "tasks": {}, "mean_8": {}}
    best = {}
    for t, pname, dom, p_rob, p_tapt in TASKS:
        cell = {"task": pname, "our_name": t, "domain": dom, "metric": "micro-F1" if t in MICRO else "macro-F1",
                "paper": {"roberta": p_rob, "tapt": p_tapt, "source": "Gururangan et al. 2020, Table 5"}, "ours": {}}
        for arm in ("jepa", "knn0"):
            cands = []
            for f in files:
                n = Path(f).stem
                if not re.match(rf"(tapt100_)?{re.escape(t)}_{arm}_", n): continue
                d = json.load(open(f))
                if len(d.get("seeds", [])) < 5: continue
                cands.append((d["test_mean"], n, f, d))
            m, n, f, d = max(cands)
            per_seed = {}
            for r in d["seeds"]:
                sf = Path(f).with_suffix(f".s{r['seed']}.json")
                per_seed[str(r["seed"])] = {"test": r["test"], "dev": r["dev"], "best_test": r.get("best_test"),
                                            "source_file": rel(sf) if sf.exists() else rel(f)}
                rows.append(["tapt", arm, pname, "test@best_dev", "ours_seed", r["seed"], f"{r['test']:.4f}",
                             f"{r['test']:.2f}", n, per_seed[str(r['seed'])]["source_file"]])
            cell["ours"][arm] = {"value": m, "printed": f"{m:.2f}", "std": d["test_std"], "run": n,
                                 "description": describe(n), "n_seeds": len(d["seeds"]),
                                 "best_test_mean": d.get("best_test_mean"), "source_file": rel(f),
                                 "per_seed": per_seed,
                                 "candidates": {c[1]: round(c[0], 4) for c in sorted(cands, reverse=True)}}
            rows.append(["tapt", arm, pname, "test@best_dev", "ours_reported", "", f"{m:.4f}", f"{m:.2f}", n, rel(f)])
            best.setdefault(arm, []).append(m)
        for k, v in (("paper_roberta", p_rob), ("paper_tapt", p_tapt)):
            rows.append(["tapt", k, pname, "test", "published", "", f"{v:.1f}", f"{v:.1f}", "Gururangan 2020 Table 5", ""])
        json.dump(cell, open(out / f"{t}.json", "w"), indent=2)
        summ["tasks"][pname] = {"jepa": cell["ours"]["jepa"]["value"], "knn0": cell["ours"]["knn0"]["value"],
                                "paper_roberta": p_rob, "paper_tapt": p_tapt}
    summ["mean_8"] = {"jepa": sum(best["jepa"]) / 8, "knn0": sum(best["knn0"]) / 8,
                      "paper_roberta": sum(x[3] for x in TASKS) / 8, "paper_tapt": sum(x[4] for x in TASKS) / 8}
    json.dump(summ, open(out / "summary.json", "w"), indent=2)
    return summ


if __name__ == "__main__":
    rows = []
    b = beir_arms(rows); s = tapt(rows)
    with open(FINAL / "beir_arms_tapt_numbers.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["section", "model", "task", "stat", "scope", "seed", "value", "printed", "run", "source_file"])
        w.writerows(rows)
    print(json.dumps({"beir_mean_14": b["mean_14_zero_shot"], "beir_wins": b["wins_by_dataset"],
                      "tapt_mean_8": s["mean_8"]}, indent=1))
    for k, v in s["tasks"].items():
        print(f"{k:14s} jepa {v['jepa']:.2f}  knn0 {v['knn0']:.2f}  paper {v['paper_roberta']}/{v['paper_tapt']}")
