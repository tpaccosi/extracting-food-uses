#!/usr/bin/env python3
"""
make_folds_from.py
==================
Builds a new cv dir for training on a DIFFERENT annotation of the same
sentences (e.g. merged_test_final_bio.tsv) and testing on the OLD test folds.

For every fold K of an existing cv dir made by make_folds.py:
    train = sentences of the new file whose gold sentence was in train of fold K
    dev   = sentences of the new file whose gold sentence was in dev   of fold K
    test  = the OLD fold_K/test.bio.tsv, unchanged (all its sentences)
so no test sentence ever ends up in train/dev, and the scores are directly
comparable with the old cv run.

Sentences are matched by their token sequence, walking the gold in its original
order (rebuilt from folds.json + fold_*/test.bio.tsv), so a sentence that occurs
twice goes to the right fold. Gold sentences missing from the new file just make
that fold's train/dev smaller; sentences of the new file not found in the gold
are left out. Both are reported.

The layered columns (FRAME__L1, ...) are recomputed on the NEW file, since it
may nest deeper than the old gold, and the old test is re-written as .conll
with those same columns, so the model and the test files agree.

Usage
-----
    python3 machamp_cv/make_folds_from.py --cvdir cv5/ \\
        --input merged_test_final_bio.tsv --outdir cv5_merged/

    --min-layer-spans N   same meaning as in make_folds.py
    --dataset-name NAME   default: the one in cv5/folds.json

Output (same layout as make_folds.py, so run_cv.py works on it)
------
    cv5_merged/folds.json
    cv5_merged/fold_1/ ... fold_K/
        train.bio.tsv dev.bio.tsv    from the new file
        test.bio.tsv                 old test, copied as is
        train.conll dev.conll test.conll
        machamp_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import biocv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biocv import (decompose_sentence, frame_depths, layered_columns,  # noqa: E402
                   read_bio_tsv, sentence_layer_counts, write_bio_tsv, write_conll,
                   write_machamp_config)


def key(s):
    return tuple(r[0] for r in s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cvdir", required=True, type=Path, help="existing cv dir (folds.json, fold_*/test.bio.tsv)")
    ap.add_argument("--input", required=True, type=Path, help="new BIO TSV to train on (with header)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--dataset-name", help="default: the one in <cvdir>/folds.json")
    ap.add_argument("--min-layer-spans", type=int, default=0,
                    help="layer columns with fewer spans than this in the new file are not trained (default 0)")
    ap.add_argument("--require-dev-spans", action="store_true",
                    help="drop layer columns that have no span in the dev set of some fold "
                         "(otherwise MAChamp crashes in callback.plot with a KeyError)")
    ap.add_argument("--n-frames", type=int, help="as in make_folds.py")
    ap.add_argument("--lenient", action="store_true", help="as in make_folds.py")
    ap.add_argument("--nested-frames", nargs="*",
                help="frame che mantengono i layer annidati (es. CURE); per tutti gli altri "
                     "si usa solo la prima annotazione di ogni cella. Default: tutti nested")
    args = ap.parse_args()
    if args.outdir.resolve() == args.cvdir.resolve():
        raise SystemExit("--outdir must differ from --cvdir (the old folds would be overwritten)")

    meta = json.loads((args.cvdir / "folds.json").read_text(encoding="utf-8"))
    k, n = meta["k"], meta["n_sentences"]
    dataset = args.dataset_name or meta["dataset_name"]

    # ------------------------------------------- old gold in original order
    gold, old_tests = [None] * n, []
    for f in range(k):
        hdr, sents = read_bio_tsv(args.cvdir / f"fold_{f + 1}" / "test.bio.tsv",
                                  strict=not args.lenient, n_frames=args.n_frames)
        idx = meta["splits"][f]["test"]
        if len(sents) != len(idx):
            raise SystemExit(f"fold_{f + 1}/test.bio.tsv: {len(sents)} sentences, folds.json says {len(idx)}")
        for i, s in zip(idx, sents):
            gold[i] = s
        old_tests.append((hdr, sents))
    if any(g is None for g in gold):
        raise SystemExit("folds.json test splits do not cover every gold sentence")

    header, new = read_bio_tsv(args.input, strict=not args.lenient, n_frames=args.n_frames)
    if header != meta["header"]:
        raise SystemExit(f"header differs from the old gold:\n  old: {meta['header']}\n  new: {header}")
    if args.nested_frames is not None:
        unknown = set(args.nested_frames) - set(header[1:])
        if unknown:
            raise SystemExit(f"--nested-frames non presenti nell'header: {sorted(unknown)}\n"
                             f"  frame disponibili: {header[1:]}")
        biocv.NESTED_FRAMES = set(args.nested_frames)
        print(f"layer annidati solo per: {sorted(biocv.NESTED_FRAMES)}; "
              f"gli altri frame: prima annotazione")

    # ------------------------------------------------------------ alignment
    by_key = {}
    for i, s in enumerate(gold):
        by_key.setdefault(key(s), []).append(i)
    used, match, j = set(), [], 0
    for s in new:
        kk = key(s)
        while j < n and key(gold[j]) != kk:
            j += 1
        if j < n and j not in used:
            gi, j = j, j + 1
        else:
            gi = next((i for i in by_key.get(kk, []) if i not in used), None)
        if gi is not None:
            used.add(gi)
        match.append(gi)
    new_of_gold = {g: p for p, g in enumerate(match) if g is not None}
    unmatched = [p for p, g in enumerate(match) if g is None]
    missing = sorted(set(range(n)) - used)

    # ---------------------------------------------- layers from the new file
    frames = header[1:]
    depths = frame_depths(header, new)
    old_depths = Counter(fr for fr, _, _ in meta["columns"])
    full_columns = layered_columns(frames, depths)
    layer_total = Counter()
    for s in new:
        layer_total.update(sentence_layer_counts(header, s))
    layer_items = [sentence_layer_counts(header, s) for s in new]
    parts_of = []
    for f in range(k):
        sp = meta["splits"][f]
        parts_of.append({
            "train": [new_of_gold[i] for i in sp["train"] if i in new_of_gold],
            "dev": [new_of_gold[i] for i in sp["dev"] if i in new_of_gold],
        })

    def dev_spans(name, f):
        return sum(layer_items[p][name] for p in parts_of[f]["dev"])

    columns, skipped = [], []
    for fr, li, name in full_columns:
        if li > 0:
            why = None
            if any(x[0] == fr for x in skipped):
                why = "deeper layer of a dropped one"
            elif layer_total[name] < args.min_layer_spans:
                why = f"< --min-layer-spans {args.min_layer_spans}"
            elif args.require_dev_spans and any(dev_spans(name, f) == 0 for f in range(k)):
                why = "no span in the dev of fold(s) " + str([f + 1 for f in range(k) if dev_spans(name, f) == 0])
            if why:
                skipped.append((fr, name, layer_total[name], why))
                continue
        columns.append((fr, li, name))

    print(f"{len(new)} sentences in {args.input.name}; matched to the gold: {len(new) - len(unmatched)}; "
          f"gold sentences not in it: {len(missing)}")
    print(f"MAChamp tasks: {len(columns)} (old run: {len(meta['columns'])})")
    for fr, li, name in columns:
        if li:
            print(f"  {name:<28}{layer_total[name]:>6} spans")
    for _, nm, c, why in skipped:
        print(f"  not trained: {nm} ({c} spans) -- {why}")
    if skipped:
        print("  their spans are left out of training (in the test they count as missed, as in the old run)")
    empty_dev = [(name, [f + 1 for f in range(k) if dev_spans(name, f) == 0])
                 for fr, li, name in columns if li > 0]
    empty_dev = [(nm, fl) for nm, fl in empty_dev if fl]
    if empty_dev:
        print("WARNING  layer task(s) with no gold span in the DEV of some fold -> MAChamp crashes in "
              "callback.plot (KeyError):")
        for nm, fl in empty_dev:
            print(f"  {nm}: fold(s) {fl}")
        print("  re-run with --require-dev-spans (drops them) or patch MAChamp's callback.py")
    deeper = [fr for fr in frames if sum(1 for c in columns if c[0] == fr) != old_depths[fr]]
    if deeper:
        print(f"  NOTE layers differ from the old run for: {deeper} "
              f"(use --min-layer-spans to drop sparse layers)")

    # ------------------------------------------------------------- write
    splits_new = []
    for f in range(k):
        d = args.outdir / f"fold_{f + 1}"
        sp = meta["splits"][f]
        parts = parts_of[f]
        leak = set(parts["train"] + parts["dev"]) & {new_of_gold.get(i) for i in sp["test"]}
        if leak - {None}:
            raise SystemExit(f"fold_{f + 1}: test sentence(s) in train/dev, this should not happen")
        dropped = {}
        for part, idx in parts.items():
            sents = [new[p] for p in idx]
            write_bio_tsv(d / f"{part}.bio.tsv", header, sents)
            layered = []
            dropped[part] = 0
            for s in sents:
                rows, dr = decompose_sentence(header, s, columns)
                dropped[part] += dr
                layered.append(rows)
            write_conll(d / f"{part}.conll", layered)
        thdr, tsents = old_tests[f]
        write_bio_tsv(d / "test.bio.tsv", thdr, tsents)
        write_conll(d / "test.conll", [decompose_sentence(thdr, s, columns)[0] for s in tsents])
        write_machamp_config(d / "machamp_config.json", d, dataset, columns)
        splits_new.append({"train": parts["train"], "dev": parts["dev"], "test": sp["test"]})
        print(f"  fold_{f + 1}: train {len(parts['train'])}/{len(sp['train'])}  "
              f"dev {len(parts['dev'])}/{len(sp['dev'])}  test {len(tsents)} (old, unchanged)"
              + (f"  nested spans not trained: {dropped['train']}" if dropped["train"] else ""))

    out = dict(meta)
    out.update({
        "input": str(args.input.resolve()),
        "dataset_name": dataset,
        "columns": [list(c) for c in columns],
        "train_source": str(args.input.resolve()),
        "test_source": str((args.cvdir / "folds.json").resolve()),
        "note": "splits[*].train/dev index sentences of train_source; splits[*].test and "
                "fold_of_sentence index the old gold (test_source)",
        "splits": splits_new,
        "new_to_gold_index": match,
        "gold_missing_from_new": missing,
        "nested_frames": sorted(biocv.NESTED_FRAMES) if biocv.NESTED_FRAMES is not None else None,
    })
    (args.outdir / "folds.json").write_text(json.dumps(out), encoding="utf-8")

    if unmatched:
        print(f"WARNING  {len(unmatched)} sentence(s) of {args.input.name} not in the gold, left out: #"
              + ", #".join(str(p + 1) for p in unmatched[:10]))
    if missing:
        print(f"NOTE  {len(missing)} gold sentence(s) not in {args.input.name}; per fold "
              f"missing from train+dev: "
              + str([sum(1 for i in missing if i in set(meta['splits'][f]['train'] + meta['splits'][f]['dev']))
                     for f in range(k)]))
    print(f"\nWritten to {args.outdir}/  -- next: run_cv.py on it (or train each fold_*/machamp_config.json)")


if __name__ == "__main__":
    main()