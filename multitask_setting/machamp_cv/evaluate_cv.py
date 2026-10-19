#!/usr/bin/env python3
"""
evaluate_cv.py
==============
Span-level evaluation (exact match of start, end and label; nested spans
from "|" cells included) of a K-fold run, reported two ways:

  mean +- std   metric computed on each fold's test set, then averaged over
                folds (sample std, ddof=1). For a label, folds whose test set
                has no gold span of it are left out of its mean (recall is
                undefined there); the "n" column says how many folds count.
  pooled (OOF)  the K test sets concatenated -- every sentence predicted by
                the model that did not see it in training -- and P/R/F1
                computed once on the whole gold. More reliable for rare labels.

Labels with <= --rare-threshold sentences in the whole gold are flagged: with
that little data no fold scheme gives a stable number, read them as anecdotes.

Usage
-----
    python3 machamp_cv/evaluate_cv.py --folds-dir cv5/

Needs cv5/fold_k/test.bio.tsv and pred.bio.tsv for every fold (run_cv.py
writes them). Output: cv5/cv_summary.tsv, cv5/cv_per_fold.tsv and the
out-of-fold files cv5/oof.gold.bio.tsv / cv5/oof.pred.bio.tsv (all
sentences, original order) for error analysis.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biocv import frame_spans, label_name, load_meta, read_bio_tsv, write_bio_tsv  # noqa: E402

MICRO = "<micro>"
ALL = "<ALL>"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def fold_counts(header, gold_sents, pred_sents):
    """{(frame, label): [tp, fp, fn]} plus gold sentence counts per label."""
    cnt = defaultdict(lambda: [0, 0, 0])
    sent_support = Counter()
    for g, p in zip(gold_sents, pred_sents):
        for c, fr in enumerate(header[1:], start=1):
            gs = Counter(frame_spans(g, c))
            ps = Counter(frame_spans(p, c))
            for lab in {s[2] for s in gs}:
                sent_support[(fr, lab)] += 1
            for sp in set(gs) | set(ps):
                tp = min(gs[sp], ps[sp])
                k = cnt[(fr, sp[2])]
                k[0] += tp
                k[1] += ps[sp] - tp
                k[2] += gs[sp] - tp
    return cnt, sent_support


def add_micro(cnt):
    out = dict(cnt)
    for (fr, lab), (tp, fp, fn) in cnt.items():
        for key in ((fr, MICRO), (ALL, MICRO)):
            v = out.setdefault(key, [0, 0, 0])
            out[key] = [v[0] + tp, v[1] + fp, v[2] + fn]
    return out


def mean_std(xs):
    if not xs:
        return math.nan, math.nan
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else math.nan)


def fmt(x, pct=True):
    return "  -  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:5.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path)
    ap.add_argument("--pred-name", default="pred.bio.tsv", help="prediction file inside each fold dir")
    ap.add_argument("--rare-threshold", type=int, default=5,
                    help="flag labels with at most this many gold sentences (default 5)")
    args = ap.parse_args()

    meta = load_meta(args.folds_dir)
    header, k = meta["header"], meta["k"]

    per_fold = []           # [{key: [tp, fp, fn]}]
    gold_sents_total = Counter()
    oof_gold = [None] * meta["n_sentences"]
    oof_pred = [None] * meta["n_sentences"]
    for f in range(1, k + 1):
        d = args.folds_dir / f"fold_{f}"
        gh, gold = read_bio_tsv(d / "test.bio.tsv")
        ph, pred = read_bio_tsv(d / args.pred_name)
        if gh != header or ph != header:
            raise SystemExit(f"{d}: header differs from folds.json")
        if len(gold) != len(pred):
            raise SystemExit(f"{d}: {len(gold)} gold vs {len(pred)} predicted sentences")
        for i, (g, p) in enumerate(zip(gold, pred)):
            if [r[0] for r in g] != [r[0] for r in p]:
                raise SystemExit(f"{d}: tokens of test sentence {i + 1} differ between gold and prediction")
        cnt, sup = fold_counts(header, gold, pred)
        per_fold.append(add_micro(cnt))
        gold_sents_total.update(sup)
        for idx, g, p in zip(meta["splits"][f - 1]["test"], gold, pred):
            oof_gold[idx], oof_pred[idx] = g, p

    labels = sorted({key for pf in per_fold for key in pf if key[1] != MICRO})
    by_label = Counter(l for _, l in labels)
    ambiguous = {l for l, c in by_label.items() if c > 1}
    frames = [fr for fr in header[1:] if any(fr == x for x, _ in labels)]
    keys = []
    for fr in frames:
        keys += [key for key in labels if key[0] == fr] + [(fr, MICRO)]
    keys.append((ALL, MICRO))

    rows = []
    long_rows = ["fold\tframe\tlabel\ttp\tfp\tfn\tgold_spans\tprecision\trecall\tf1"]
    for key in keys:
        fr, lab = key
        tot = [sum(pf.get(key, [0, 0, 0])[i] for pf in per_fold) for i in range(3)]
        gold_spans = tot[0] + tot[2]
        ps, rs, fs = [], [], []
        for f, pf in enumerate(per_fold, start=1):
            tp, fp, fn = pf.get(key, [0, 0, 0])
            p, r, f1 = prf(tp, fp, fn)
            long_rows.append(f"{f}\t{fr}\t{lab}\t{tp}\t{fp}\t{fn}\t{tp + fn}\t{p:.4f}\t{r:.4f}\t{f1:.4f}")
            if tp + fn > 0:
                ps.append(p), rs.append(r), fs.append(f1)
        pp, pr, pf1 = prf(*tot)
        n_sent = gold_sents_total[key] if lab != MICRO else None
        note = ""
        if lab != MICRO and n_sent <= args.rare_threshold:
            note = f"<= {args.rare_threshold} sentences: unstable"
        elif len(fs) < k:
            note = f"absent from {k - len(fs)} test fold(s)"
        rows.append(dict(frame=fr, label=lab, gold_spans=gold_spans, gold_sents=n_sent, n=len(fs),
                         P=mean_std(ps), R=mean_std(rs), F=mean_std(fs),
                         pooled=(pp, pr, pf1), tp=tot[0], fp=tot[1], fn=tot[2], note=note))

    # macro-F1 over labels (labels with gold support in that fold / in the pool)
    macro_folds = []
    for pf in per_fold:
        f1s = [prf(*pf[key])[2] for key in labels if key in pf and pf[key][0] + pf[key][2] > 0]
        if f1s:
            macro_folds.append(statistics.fmean(f1s))
    pooled_f1s = [r["pooled"][2] for r in rows if r["label"] != MICRO and r["gold_spans"] > 0]
    macro = (mean_std(macro_folds), statistics.fmean(pooled_f1s) if pooled_f1s else math.nan)

    # ------------------------------------------------------------------ write
    tsv = ["frame\tlabel\tgold_spans\tgold_sentences\tn_folds\t"
           "P_mean\tP_std\tR_mean\tR_std\tF1_mean\tF1_std\t"
           "P_pooled\tR_pooled\tF1_pooled\ttp\tfp\tfn\tnote"]
    for r in rows:
        tsv.append("\t".join(str(x) for x in [
            r["frame"], r["label"], r["gold_spans"], "" if r["gold_sents"] is None else r["gold_sents"], r["n"],
            *(f"{v:.4f}" for v in (*r["P"], *r["R"], *r["F"], *r["pooled"])),
            r["tp"], r["fp"], r["fn"], r["note"]]))
    tsv.append("\t".join([ALL, "<macro>", "", "", str(len(macro_folds)), "", "", "", "",
                          f"{macro[0][0]:.4f}", f"{macro[0][1]:.4f}", "", "", f"{macro[1]:.4f}",
                          "", "", "", "macro-F1 over labels"]))
    (args.folds_dir / "cv_summary.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    (args.folds_dir / "cv_per_fold.tsv").write_text("\n".join(long_rows) + "\n", encoding="utf-8")
    if all(s is not None for s in oof_gold):
        write_bio_tsv(args.folds_dir / "oof.gold.bio.tsv", header, oof_gold)
        write_bio_tsv(args.folds_dir / "oof.pred.bio.tsv", header, oof_pred)

    # ------------------------------------------------------------------ print
    def name(r):
        if r["label"] == MICRO:
            return "micro (all labels)" if r["frame"] == ALL else f"{r['frame']} micro"
        return "  " + label_name(r["frame"], r["label"], ambiguous)

    w = max(len(name(r)) for r in rows) + 2
    print(f"Span-level exact match, {k}-fold CV. mean±std over folds (n = folds with gold spans) "
          f"| pooled out-of-fold. Values in %.\n")
    print(f"{'label':<{w}}{'spans':>6}{'sents':>6}   {'F1 mean ± std':>15}{'n':>3}   "
          f"{'P':>5} {'R':>5} {'F1':>5}  (pooled)   note")
    for r in rows:
        if r["label"] == MICRO and r["frame"] == ALL:
            print("-" * (w + 60))
        (fm, fs) = r["F"]
        pp, pr, pf1 = r["pooled"]
        sents = "" if r["gold_sents"] is None else r["gold_sents"]
        print(f"{name(r):<{w}}{r['gold_spans']:>6}{sents:>6}   {fmt(fm)} ± {fmt(fs)}{r['n']:>5}   "
              f"{fmt(pp)} {fmt(pr)} {fmt(pf1)}             {r['note']}")
    (mm, ms), mp = macro
    print(f"{'macro-F1 (labels)':<{w}}{'':>12}   {fmt(mm)} ± {fmt(ms)}{len(macro_folds):>5}   "
          f"{'':>11} {fmt(mp)}")
    rare = [r for r in rows if r["note"].startswith("<=")]
    if rare:
        print(f"\n{len(rare)} label(s) with <= {args.rare_threshold} gold sentences: "
              + ", ".join(name(r).strip() for r in rare)
              + ".\nTheir scores rest on a handful of spans; no fold scheme makes them stable.")
    print(f"\nWritten: {args.folds_dir}/cv_summary.tsv, cv_per_fold.tsv, oof.gold.bio.tsv, oof.pred.bio.tsv")


if __name__ == "__main__":
    main()
