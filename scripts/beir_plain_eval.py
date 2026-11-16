#!/usr/bin/env python
"""Plain zero-shot BEIR eval of a raw encoder: no DPR/ColBERT finetuning, just embed + dot product.

    python scripts/beir_plain_eval.py --ckpt <encoder.pt | HF dir | none> --tag <name> [--pooling mean|cls]

Queries and documents go through the SAME encoder (title + " " + text for docs), pooled to one
768-d vector, then scored by exact search (BEIR's DenseRetrievalExactSearch) with dot product
(or --score cos_sim). Metric: NDCG@10 (+ recall@100), msmarco on dev, the rest on test;
cqadupstack = mean over its 12 subforums.

--ckpt accepts: a JEPA/knn0/mlm pretrain checkpoint (*.pt with encoder_state_dict), an HF
save_pretrained dir, or 'none' (= plain --model_name, e.g. bert-base-uncased as the reference).
Results: <out_dir>/<tag>.json, written after every dataset; re-running skips datasets already done.
"""
import argparse, json, logging, os, time
import numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

BEIR15 = ["scifact", "nfcorpus", "arguana", "scidocs", "fiqa", "trec-covid", "webis-touche2020", "quora",
          "cqadupstack", "nq", "dbpedia-entity", "hotpotqa", "fever", "climate-fever", "msmarco"]
CQA = ["android", "english", "gaming", "gis", "mathematica", "physics", "programmers", "stats",
       "tex", "unix", "webmasters", "wordpress"]
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True, help="encoder .pt, HF dir, or 'none'")
ap.add_argument("--tag", required=True, help="name of the results file")
ap.add_argument("--model_name", default="bert-base-uncased", help="architecture/tokenizer the ckpt was trained from")
ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
ap.add_argument("--score", default="dot", choices=["dot", "cos_sim"])
ap.add_argument("--datasets", nargs="+", default=BEIR15)
ap.add_argument("--data_dir", default="/tmp/tudao/beir_datasets")
ap.add_argument("--out_dir", default="/home/tudao/jepa-humor/results_beir_plain")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--q_len", type=int, default=64)
ap.add_argument("--d_len", type=int, default=512)
a = ap.parse_args()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
dev = "cuda"


class Encoder:
    """The minimal model interface DRES expects: encode_queries / encode_corpus -> [N, 768] tensor."""

    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(a.model_name)
        if a.ckpt == "none":
            self.model = AutoModel.from_pretrained(a.model_name, add_pooling_layer=False)
        elif os.path.isdir(a.ckpt):
            self.model = AutoModel.from_pretrained(a.ckpt, add_pooling_layer=False)
        else:
            self.model = AutoModel.from_pretrained(a.model_name, add_pooling_layer=False)
            state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
            sd = state.get("encoder_state_dict", state)
            res = self.model.load_state_dict(sd, strict=False)
            missing = [k for k in res.missing_keys if not k.startswith("pooler.")]
            unexpected = [k for k in res.unexpected_keys if not k.startswith("pooler.")]
            # a silent strict=False mismatch would evaluate plain bert-base -- refuse instead
            if missing or unexpected:
                raise SystemExit(f"checkpoint keys do not match {a.model_name}: "
                                 f"missing {missing[:5]} unexpected {unexpected[:5]}")
            print(f"loaded {len(sd)} encoder tensors from {a.ckpt}", flush=True)
        self.model.to(dev).eval()

    @torch.no_grad()
    def _encode(self, texts, max_len, batch_size):
        out = []
        for i in range(0, len(texts), batch_size):
            b = self.tok(texts[i:i + batch_size], padding=True, truncation=True,
                         max_length=max_len, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = self.model(**b).last_hidden_state
            h = h.float()
            if a.pooling == "cls":
                e = h[:, 0]
            else:
                m = b["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(e)
        return torch.cat(out)

    def encode_queries(self, queries, batch_size=256, **kw):
        return self._encode(queries, a.q_len, batch_size)

    def encode_corpus(self, corpus, batch_size=256, **kw):
        if isinstance(corpus, dict):  # beir may pass {"title": [...], "text": [...]}
            corpus = [{"title": t, "text": x} for t, x in zip(corpus["title"], corpus["text"])]
        texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in corpus]
        return self._encode(texts, a.d_len, batch_size)


def load(name, sub=None):
    path = os.path.join(a.data_dir, name)
    if not os.path.isdir(path):
        util.download_and_unzip(URL.format(name), a.data_dir)
    if sub:
        path = os.path.join(path, sub)
    return GenericDataLoader(data_folder=path).load(split="dev" if name == "msmarco" else "test")


def run_one(retriever, name, sub=None):
    corpus, queries, qrels = load(name, sub)
    t = time.time()
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, _p = EvaluateRetrieval.evaluate(qrels, results, [10, 100])
    return {"ndcg@10": ndcg["NDCG@10"], "recall@100": recall["Recall@100"],
            "n_docs": len(corpus), "n_queries": len(queries), "sec": round(time.time() - t)}


os.makedirs(a.out_dir, exist_ok=True)
out = os.path.join(a.out_dir, f"{a.tag}.json")
res = json.load(open(out)) if os.path.exists(out) else {}
res["_config"] = vars(a)
retriever = EvaluateRetrieval(DRES(Encoder(), batch_size=a.batch, corpus_chunk_size=50000),
                              score_function=a.score, k_values=[10, 100])
for name in a.datasets:
    if name in res:
        print(f"[skip] {name} already done: {res[name]['ndcg@10']:.4f}", flush=True)
        continue
    if name == "cqadupstack":
        per = {s: run_one(retriever, name, s) for s in CQA}
        res[name] = {"ndcg@10": float(np.mean([v["ndcg@10"] for v in per.values()])),
                     "recall@100": float(np.mean([v["recall@100"] for v in per.values()])), "subsets": per}
    else:
        res[name] = run_one(retriever, name)
    print(f"[{a.tag}] {name}: NDCG@10 {res[name]['ndcg@10']:.4f}  R@100 {res[name]['recall@100']:.4f}", flush=True)
    json.dump(res, open(out, "w"), indent=1)

done = [d for d in a.datasets if d in res]
res["_mean_ndcg@10"] = float(np.mean([res[d]["ndcg@10"] for d in done]))
json.dump(res, open(out, "w"), indent=1)
print(f"[{a.tag}] MEAN NDCG@10 over {len(done)} datasets: {res['_mean_ndcg@10']:.4f}", flush=True)
