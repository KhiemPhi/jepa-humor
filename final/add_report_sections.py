#!/usr/bin/env python
"""Append sections 12 (QA-corpus objective arms, BEIR) and 13 (TAPT) to REPORT_glue_beir_rankify.{md,docx}
(user 2026-10-05). Numbers come from final/beir_arms/ and final/tapt/ (make_beir_arms_tapt.py).
Idempotent: strips any previous copy of these sections before appending."""
import json
from pathlib import Path
import docx

ROOT = Path("/home/tudao/jepa-humor"); F = ROOT / "final"
MD, DOCX = ROOT / "REPORT_glue_beir_rankify.md", ROOT / "REPORT_glue_beir_rankify.docx"
MARK = "## 12. "
DS = [("trec-covid", "TREC-COVID"), ("nfcorpus", "NFCorpus"), ("nq", "NQ"), ("hotpotqa", "HotpotQA"),
      ("fiqa", "FiQA-2018"), ("arguana", "ArguAna"), ("webis-touche2020", "Touche-2020"),
      ("cqadupstack", "CQADupStack"), ("quora", "Quora"), ("dbpedia-entity", "DBPedia"), ("scidocs", "SCIDOCS"),
      ("fever", "FEVER"), ("climate-fever", "Climate-FEVER"), ("scifact", "SciFact")]
ARMS = ["jepa", "knn0", "mlm"]

beir = {d: json.load(open(F / f"beir_arms/{d}.json")) for d, _ in DS}
bsum = json.load(open(F / "beir_arms/summary.json"))
tapt_order = ["chemprot", "rct-20k", "citation_intent", "sciie", "hyperpartisan_news", "ag", "amazon", "imdb"]
tapt = {t: json.load(open(F / f"tapt/{t}.json")) for t in tapt_order}
tsum = json.load(open(F / "tapt/summary.json"))

# blocks: ("h2"|"h3"|"p"|"li", text) or ("table", header, rows, bold_cells) -- md and docx rendered from the same list
B = []
B.append(("h2", "12. Objective arms on the DPR corpus itself — BEIR (NDCG@10, 14 zero-shot sets)"))
B.append(("p", "Sections 8–11 pretrain on MS MARCO for 5 000 steps. These three arms instead pretrain on the "
               "**DPR multiset passages themselves** (the same 6.26M passages DPR is then trained on, fed as the "
               "`[CLS] title [SEP] passage [SEP]` pair DPR's context encoder sees), for one full epoch "
               "(48 886 steps, lr 5e-6, batch 128, bert-base-uncased). DPR training and BEIR evaluation are "
               "byte-identical to sections 8–11. Only the pretraining objective differs:"))
B.append(("table", ["arm", "pretraining objective"],
          [["jepa", "JEPA, `knn_weight=0.1` (default)"], ["knn0", "JEPA, `knn_weight=0.0`"],
           ["mlm", "BERT MLM (80/10/10, mask 0.15), `cls.predictions` restored from BERT"]], set()))
B.append(("p", "MS MARCO was not evaluated for these arms. `public DPR` is the BEIR Table 2 paper number, as in "
               "section 9. Steps differ from sections 8–11 (48 886 vs 5 000), so these arms are comparable to each "
               "other, not to the MS MARCO arms."))
for ck, title in (("ckpt39", "12.1 ckpt39 (final epoch)"), ("ckpt2", "12.2 ckpt2")):
    B.append(("h3", title))
    rows, bold = [], set()
    for i, (d, lab) in enumerate(DS):
        vals = [beir[d]["ndcg@10"][f"{a}_{ck}"] for a in ARMS]
        rows.append([lab] + [f"{v:.4f}" for v in vals] + [f"{beir[d]['public_dpr']:.3f}"])
        bold.add((i, 1 + vals.index(max(vals))))
    means = [bsum["mean_14_zero_shot"][f"{a}_{ck}"] for a in ARMS]
    rows.append(["**MEAN (14)**"] + [f"{m:.4f}" for m in means] + [f"{bsum['mean_14_zero_shot']['public_dpr']:.4f}"])
    bold.add((len(DS), 1 + means.index(max(means))))
    B.append(("table", ["dataset"] + ARMS + ["public DPR"], rows, bold))
B.append(("p", "Against the published DPR baseline (counts over the 14 sets):"))
B.append(("table", ["", "ckpt39", "ckpt2"],
          [[a, bsum["beats_public_dpr"][f"{a}_ckpt39"], bsum["beats_public_dpr"][f"{a}_ckpt2"]] for a in ARMS], set()))
q = {a: beir["quora"]["ndcg@10"][f"{a}_ckpt39"] for a in ARMS}
m13 = {a: (bsum["mean_14_zero_shot"][f"{a}_ckpt39"] * 14 - q[a]) / 13 for a in ARMS}
gap = bsum["mean_14_zero_shot"]["jepa_ckpt39"] - bsum["mean_14_zero_shot"]["mlm_ckpt39"]
B.append(("p", "Observations:"))
B.append(("li", f"**jepa ckpt39 is the best retriever of any arm in this report** (MEAN-14 "
                f"{bsum['mean_14_zero_shot']['jepa_ckpt39']:.4f}, +{gap*100:.1f} points over mlm ckpt39 and "
                f"+{(bsum['mean_14_zero_shot']['jepa_ckpt39']-bsum['mean_14_zero_shot']['mlm_ckpt2'])*100:.1f} over "
                f"mlm's better checkpoint), and it beats public DPR on 14/14."))
B.append(("li", f"**Quora carries most of that margin.** Both JEPA arms reach {q['jepa']:.3f}/{q['knn0']:.3f} on Quora "
                f"against mlm's {q['mlm']:.3f}. Without Quora the 13-set means are jepa {m13['jepa']:.4f}, "
                f"mlm {m13['mlm']:.4f}, knn0 {m13['knn0']:.4f}, so the jepa–mlm gap shrinks to "
                f"{(m13['jepa']-m13['mlm'])*100:.1f} points. Dataset wins at ckpt39 are split "
                f"(jepa {bsum['wins_by_dataset']['ckpt39']['jepa']}, mlm {bsum['wins_by_dataset']['ckpt39']['mlm']}, "
                f"knn0 {bsum['wins_by_dataset']['ckpt39']['knn0']})."))
B.append(("li", "**The kNN term matters here**: jepa beats knn0 by "
                f"{(bsum['mean_14_zero_shot']['jepa_ckpt39']-bsum['mean_14_zero_shot']['knn0_ckpt39'])*100:.1f} (ckpt39) and "
                f"{(bsum['mean_14_zero_shot']['jepa_ckpt2']-bsum['mean_14_zero_shot']['knn0_ckpt2'])*100:.1f} (ckpt2) points, "
                "mostly on FEVER, TREC-COVID, ArguAna and Climate-FEVER."))
B.append(("li", "**Unlike sections 5 and 9, ckpt39 > ckpt2 for both JEPA arms** (mlm still peaks at ckpt2). "
                "Single runs, no seed variance."))

B.append(("h2", "13. Task-adaptive pretraining (TAPT) — Don't Stop Pretraining tasks"))
B.append(("p", "Gururangan et al. (ACL 2020) setting: continue pretraining roberta-base on each task's own unlabeled "
               "training text, then finetune on the task. Our arms replace their MLM objective with JEPA "
               "(`knn_weight=0.1`) or JEPA without the kNN term (`knn0`). Metric is the paper's: test F1 at the "
               "best-dev epoch (micro-F1 for CHEMPROT and RCT, macro-F1 otherwise), mean over 5 seeds. Paper columns "
               "are their Table 5 (RoBERTa = no continued pretraining, TAPT = their MLM TAPT)."))
B.append(("p", "**Run selection:** each cell is the best finished 5-seed run for that task and arm, over every TAPT "
               "variant (100-epoch TAPT for the four small tasks; 5-epoch and 100-epoch TAPT for the four large ones) "
               "and finetune recipe tried — the same maximise-over-runs procedure as the GLUE sections, so it is "
               "optimistic relative to the paper's single reported result. TAPT used lr 5e-6 (the paper used 1e-4 to "
               "5e-4). `final/tapt/<task>.json` lists the selected run, per-seed values and every candidate."))
rows, bold = [], set()
for i, t in enumerate(tapt_order):
    c = tapt[t]; j, k = c["ours"]["jepa"], c["ours"]["knn0"]
    rows.append([c["domain"], c["task"], f"{c['paper']['roberta']:.1f}", f"{c['paper']['tapt']:.1f}",
                 f"{j['value']:.2f} ±{j['std']:.2f}", f"{k['value']:.2f} ±{k['std']:.2f}"])
    best = max(c["paper"]["tapt"], j["value"], k["value"])
    for col, v in ((3, c["paper"]["tapt"]), (4, j["value"]), (5, k["value"])):
        if abs(v - best) < 1e-9: bold.add((i, col))
m = tsum["mean_8"]
rows.append(["", "**Mean**", f"{m['paper_roberta']:.2f}", f"{m['paper_tapt']:.2f}", f"{m['jepa']:.2f}", f"{m['knn0']:.2f}"])
bold.add((len(tapt_order), 4))
B.append(("table", ["domain", "task", "paper RoBERTa", "paper TAPT", "jepa", "knn0"], rows, bold))
B.append(("p", "Selected runs:"))
B.append(("table", ["task", "jepa", "knn0"],
          [[tapt[t]["task"], tapt[t]["ours"]["jepa"]["description"], tapt[t]["ours"]["knn0"]["description"]]
           for t in tapt_order], set()))
B.append(("p", "Observations:"))
B.append(("li", f"**Both arms beat the paper's TAPT mean** (jepa {m['jepa']:.2f}, knn0 {m['knn0']:.2f} vs "
                f"{m['paper_tapt']:.2f}), winning on ACL-ARC, HYPERPARTISAN, HELPFULNESS and SCIERC and tying IMDB. "
                "They trail it on CHEMPROT, RCT and AGNEWS, where they sit at or just under the paper's no-TAPT RoBERTa."))
B.append(("li", "**The margin sits on the smallest test sets.** HYPERPARTISAN (65 test articles) and ACL-ARC "
                "(139 test examples) carry 2–3 point seed standard deviations and contribute most of the mean gap; "
                "over the other six tasks jepa is level with the paper's TAPT."))
B.append(("li", "**knn0 ≈ jepa**: knn0's pretraining loss barely moves (0.0974 throughout), so knn0 is close to "
                "roberta-base plus our finetuning, yet it matches jepa on most tasks. Much of the gain over the paper "
                "therefore likely comes from the finetuning recipe, not the objective. We ran no local no-TAPT "
                "baseline to confirm this."))
B.append(("li", "**Longer TAPT does not help**: on AGNEWS, RCT and IMDB, the 100-epoch TAPT checkpoints at "
                "epochs 20/40/60/80/100 all finetune within ~0.2 of the 5-epoch TAPT checkpoint."))


def md_inline(s): return s


def render_md():
    out = ["", "---", ""]
    for b in B:
        if out[-1].startswith("- "): out.append("")
        if b[0] == "h2": out += ([] if len(out) <= 4 else ["---", ""]) + [f"## {b[1]}", ""]
        elif b[0] == "h3": out += [f"### {b[1]}", ""]
        elif b[0] == "p": out += [b[1], ""]
        elif b[0] == "li": out += [f"- {b[1]}"] + ([""] if False else [])
        elif b[0] == "table":
            _, hdr, rows, bold = b
            out.append("| " + " | ".join(hdr) + " |"); out.append("|" + "---|" * len(hdr))
            for i, r in enumerate(rows):
                out.append("| " + " | ".join(f"**{c}**" if (i, j) in bold and not c.startswith("**") else c
                                             for j, c in enumerate(r)) + " |")
            out.append("")
        if b[0] != "li": pass
    # blank line after each bullet run
    return "\n".join(out).rstrip("\n") + "\n"


def add_runs(par, text):
    """Add text to a docx paragraph, rendering **bold** and `code` spans."""
    import re
    for tok in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not tok: continue
        if tok.startswith("**"): par.add_run(tok[2:-2]).bold = True
        elif tok.startswith("`"): par.add_run(tok[1:-1]).font.name = "Consolas"
        else: par.add_run(tok)


def render_docx():
    d = docx.Document(DOCX)
    body = d.element.body
    # idempotency: drop everything from an existing "12. " heading on
    start = None
    for el in list(body):
        if el.tag.endswith("}p") and "".join(el.itertext()).startswith("12. Objective arms on the DPR corpus"):
            start = el; break
    if start is not None:
        drop, on = [], False
        for el in list(body):
            if el is start: on = True
            if on and not el.tag.endswith("}sectPr"): drop.append(el)
        for el in drop: body.remove(el)
    style = d.tables[-1].style
    for b in B:
        if b[0] == "h2": d.add_heading(b[1], level=2)
        elif b[0] == "h3": d.add_heading(b[1], level=3)
        elif b[0] == "p": add_runs(d.add_paragraph(), b[1])
        elif b[0] == "li": add_runs(d.add_paragraph(style="List Bullet"), b[1])
        elif b[0] == "table":
            _, hdr, rows, bold = b
            t = d.add_table(rows=1 + len(rows), cols=len(hdr)); t.style = style
            for j, h in enumerate(hdr): t.rows[0].cells[j].text = h
            for i, r in enumerate(rows):
                for j, c in enumerate(r):
                    cell = t.rows[i + 1].cells[j]; cell.text = ""
                    p = cell.paragraphs[0]
                    if (i, j) in bold and not c.startswith("**"): p.add_run(c).bold = True
                    else: add_runs(p, c)
    d.save(DOCX)


md = MD.read_text()
if MARK in md: md = md[:md.index("\n---\n\n" + MARK)] if ("\n---\n\n" + MARK) in md else md[:md.index(MARK)]
MD.write_text(md.rstrip("\n") + "\n" + render_md())
render_docx()
print("appended sections 12-13 to", MD.name, "and", DOCX.name)
