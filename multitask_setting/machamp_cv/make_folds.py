#!/usr/bin/env python3
"""
make_folds.py
=============
Splits ONE gold BIO TSV (token<TAB>FRAME1<TAB>..., blank line per sentence)
into K stratified folds for cross-validation and prepares everything MAChamp
needs for each fold.

Stratification
    Sentences are assigned with multi-label iterative stratification on the
    (frame, label) spans they contain, rarest label first, so rare labels are
    spread over the folds instead of piling up in one. The script prints the
    label x fold table (span counts in each test fold) and warns when a label
    is missing from some fold's test or train set.

Nested cells ("|")
    A cell such as "I-CURE_Affliction|B-CURE_LU" holds more than one span.
    seq_bio would treat the whole string as a brand-new label and its BIO
    constraints would break, so each frame column is split into layers:
    outer spans stay in FRAME, spans nested inside them go to FRAME__L1,
    FRAME__L2, ... (one plain BIO column each). recompose_predictions.py
    merges them back into "|" cells. The number of layers per frame is taken
    from the whole gold so all folds share the same columns. The script
    checks the round trip on every nested cell and reports it.

Inside each fold, test = that fold; dev = a stratified --dev-frac of the
other K-1 folds; train = the rest.

Usage
-----
    python3 machamp_cv/make_folds.py --input teresa_gold.bio.tsv \\
        --k 5 --dev-frac 0.1 --seed 42 --outdir cv5/ --dataset-name FOOD_FRAMES

    --n-frames 6   keep token + the first 6 frame columns, dropping any extra
                   column on the right (rows with a trailing extra column)

Output
------
    cv5/folds.json               assignment + layered column layout (used by the other scripts)
    cv5/label_fold_table.tsv     label x fold span counts
    cv5/fold_1/ ... fold_K/
        train.bio.tsv dev.bio.tsv test.bio.tsv   original format, sentences unchanged
        train.conll   dev.conll   test.conll     MAChamp format, "|" cells split into layers
        machamp_config.json                      one seq_bio task per layered column
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biocv import (ParseStats, compose_sentence, decompose_sentence, frame_depths,  # noqa: E402
                   iterative_stratification, label_name, layered_columns, read_bio_tsv,
                   sentence_label_counts, sentence_layer_counts, write_bio_tsv, write_conll, write_machamp_config,
                   frame_spans)


def roundtrip_report(header, sentences, columns):
    """Every sentence goes original -> layers -> original; compares cells.
    Returns (nested cells, identical, only tag order differs, spans differ,
    sentences with identical spans, examples of real differences)."""
    frames = header[1:]
    nested = identical = reordered = differ = same_spans = 0
    examples = []
    for si, sent in enumerate(sentences):
        layered, _ = decompose_sentence(header, sent, columns)
        back = compose_sentence(frames, layered, columns)
        spans_ok = all(frame_spans(sent, c) == frame_spans(back, c) for c in range(1, len(header)))
        same_spans += spans_ok
        for t, (orig_row, back_row) in enumerate(zip(sent, back)):
            for c in range(1, len(header)):
                if "|" not in orig_row[c]:
                    continue
                nested += 1
                if orig_row[c] == back_row[c]:
                    identical += 1
                elif spans_ok and sorted(orig_row[c].split("|")) == sorted(back_row[c].split("|")):
                    reordered += 1
                else:
                    differ += 1
                    if len(examples) < 5:
                        examples.append((si, t, header[c], orig_row[c], back_row[c]))
    return nested, identical, reordered, differ, same_spans, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="gold BIO TSV (with header)")
    ap.add_argument("--k", type=int, default=5, help="number of folds (default 5)")
    ap.add_argument("--dev-frac", type=float, default=0.1,
                    help="fraction of each fold's training pool held out as dev (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--dataset-name", default="FOOD_FRAMES")
    ap.add_argument("--min-layer-spans", type=int, default=0,
                    help="layer columns (FRAME__L1, ...) with fewer spans than this in the whole gold "
                         "are not trained as MAChamp tasks (default 0 = train every layer)")
    ap.add_argument("--n-frames", type=int,
                    help="keep only token + the first N frame columns (drops extra columns on the right)")
    ap.add_argument("--lenient", action="store_true",
                    help="pad rows that have too few columns with 'O' (reported) instead of stopping")
    args = ap.parse_args()
    if args.k < 2:
        raise SystemExit("--k must be >= 2")
    if not 0 < args.dev_frac < 1:
        raise SystemExit("--dev-frac must be in (0, 1)")

    header, sentences = read_bio_tsv(args.input, strict=not args.lenient, n_frames=args.n_frames)
    frames = header[1:]
    n = len(sentences)
    if n < args.k:
        raise SystemExit(f"only {n} sentences, cannot make {args.k} folds")

    odd = "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029\r"
    weird = [(si, t, row[0]) for si, s in enumerate(sentences) for t, row in enumerate(s)
             if any(ch in row[0] for ch in odd)]
    if weird:
        print(f"NOTE  {len(weird)} token(s) contain line-separator characters "
              f"(e.g. sentence {weird[0][0] + 1}, token {weird[0][1] + 1}: {weird[0][2]!r}); "
              f"read correctly here, but tools that split lines with str.splitlines() will break them.")

    # ---------------------------------------------------------------- layers
    stats = ParseStats()
    for sent in sentences:
        for c in range(1, len(header)):
            frame_spans(sent, c, stats)
    depths = frame_depths(header, sentences)
    full_columns = layered_columns(frames, depths)
    nested, identical, reordered, differ, same_spans, examples = roundtrip_report(header, sentences, full_columns)

    # spans per layer column (whole gold) + per sentence, for stratification
    layer_items = [sentence_layer_counts(header, s) for s in sentences]
    layer_total = Counter()
    for it in layer_items:
        layer_total.update(it)
    # sparse layers: from the first layer under --min-layer-spans on, a frame gets no more tasks
    columns, skipped = [], []
    for fr, li, name in full_columns:
        if li > 0 and (layer_total[name] < args.min_layer_spans
                       or any(s[0] == fr for s in skipped)):
            skipped.append((fr, name, layer_total[name]))
            continue
        columns.append((fr, li, name))

    print(f"{n} sentences, {len(frames)} frames")
    print(f"MAChamp tasks: {len(columns)}. Spans per layer column (whole gold):")
    for fr, li, name in columns:
        if depths[fr] > 1:
            print(f"  {name:<28}{layer_total[name] if li else '(outer)':>8}")
    if skipped:
        print(f"  not trained (< --min-layer-spans {args.min_layer_spans}): "
              + ", ".join(f"{name} ({c} spans)" for _, name, c in skipped)
              + "\n  their spans are left out of training and count as missed (FN) in the evaluation")
    print(f"Cells with '|': {nested}; after decompose+recompose: {identical} identical, "
          f"{reordered} same tags in another order, {differ} different; "
          f"sentences with identical spans: {same_spans}/{n}")
    if reordered:
        print(f"  ({reordered} cell(s) only change the order of their tags, e.g. two spans over the same "
              f"tokens: the order inside a cell carries no information; spans are unchanged)")
    for si, t, fr, orig, back in examples:
        print(f"  WARNING  sentence {si + 1}, token {t + 1}, {fr}: {orig!r} -> {back!r} (spans differ)")
    if stats.orphan_i:
        print(f"  WARNING  {stats.orphan_i} I- tags with no span to continue were read as B- "
              f"(they come back as B- after recomposition)")
    if stats.bad_tags:
        print(f"  WARNING  unrecognised tags treated as O: {dict(stats.bad_tags.most_common(10))}")

    # --------------------------------------------------------------- folds
    items = [sentence_label_counts(header, s) for s in sentences]
    # layer columns are stratified too, so each fold's dev/test gets some nested spans
    strat = [it + li for it, li in zip(items, layer_items)]
    fold_of = iterative_stratification(strat, [1.0 / args.k] * args.k, args.seed)
    folds = [[i for i in range(n) if fold_of[i] == f] for f in range(args.k)]

    splits = []
    for f in range(args.k):
        pool = [i for i in range(n) if fold_of[i] != f]
        sub = iterative_stratification([strat[i] for i in pool],
                                       [1 - args.dev_frac, args.dev_frac], args.seed + f + 1)
        train = [i for i, g in zip(pool, sub) if g == 0]
        dev = [i for i, g in zip(pool, sub) if g == 1]
        splits.append({"train": train, "dev": dev, "test": folds[f]})

    # ---------------------------------------------------------------- write
    args.outdir.mkdir(parents=True, exist_ok=True)
    dropped_train = []
    for f, sp in enumerate(splits, start=1):
        d = args.outdir / f"fold_{f}"
        for part in ("train", "dev", "test"):
            sents = [sentences[i] for i in sp[part]]
            write_bio_tsv(d / f"{part}.bio.tsv", header, sents)
            layered = []
            dropped_total = 0
            for s in sents:
                rows, dropped = decompose_sentence(header, s, columns)
                dropped_total += dropped
                layered.append(rows)
            if part == "train":
                dropped_train.append(dropped_total)
            write_conll(d / f"{part}.conll", layered)
        write_machamp_config(d / "machamp_config.json", d, args.dataset_name, columns)

    # layer tasks without gold spans in some fold's dev: MAChamp logs "-" as dev score for
    # them and older versions crash while plotting (KeyError in callback.plot)
    layer_cols = [name for fr, li, name in columns if li > 0]
    empty_dev = []
    for name in layer_cols:
        missing = [f + 1 for f, sp in enumerate(splits)
                   if sum(layer_items[i][name] for i in sp["dev"]) == 0]
        if missing:
            empty_dev.append((name, missing))

    meta = {
        "input": str(args.input.resolve()),
        "k": args.k, "seed": args.seed, "dev_frac": args.dev_frac,
        "dataset_name": args.dataset_name,
        "header": header,
        "columns": [list(c) for c in columns],
        "n_sentences": n,
        "fold_of_sentence": fold_of,
        "splits": splits,
    }
    (args.outdir / "folds.json").write_text(json.dumps(meta), encoding="utf-8")

    # ------------------------------------------------------ label x fold table
    labels = sorted({l for it in items for l in it})
    by_label = Counter(l for _, l in labels)
    ambiguous = {l for l, c in by_label.items() if c > 1}
    sent_count = Counter(l for it in items for l in it)

    def count(idx, lab):
        return sum(items[i][lab] for i in idx)

    names = [label_name(fr, l, ambiguous) for fr, l in labels]
    w = max([len(x) for x in names] + [12])
    fold_hdr = "".join(f"{'F' + str(f):>6}" for f in range(1, args.k + 1))
    fw = max(len(fr) for fr in header[1:]) + 2
    lines = [f"{'frame':<{fw}}{'label':<{w}}{fold_hdr}{'spans':>7}{'sents':>7}"]
    tsv = ["frame\tlabel\t" + "\t".join(f"fold_{f}" for f in range(1, args.k + 1)) + "\tspans\tsentences"]
    missing_test, missing_train = [], []
    for (fr, l), name in zip(labels, names):
        per = [count(folds[f], (fr, l)) for f in range(args.k)]
        tot = sum(per)
        lines.append(f"{fr:<{fw}}{name:<{w}}" + "".join(f"{c:>6}" for c in per)
                     + f"{tot:>7}{sent_count[(fr, l)]:>7}")
        tsv.append("\t".join([fr, l] + [str(c) for c in per] + [str(tot), str(sent_count[(fr, l)])]))
        if 0 in per:
            missing_test.append((name, [f + 1 for f, c in enumerate(per) if c == 0], sent_count[(fr, l)]))
        no_train = [f + 1 for f, sp in enumerate(splits) if count(sp["train"], (fr, l)) == 0]
        if no_train:
            missing_train.append((name, no_train))
    sizes = [len(fl) for fl in folds]
    lines.append(f"{'':<{fw}}{'#sentences':<{w}}" + "".join(f"{c:>6}" for c in sizes) + f"{n:>7}")
    (args.outdir / "label_fold_table.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")

    print(f"\nFolds (seed={args.seed}): test sizes {sizes}; "
          f"train/dev/test per fold: "
          + ", ".join(f"{len(s['train'])}/{len(s['dev'])}/{len(s['test'])}" for s in splits))
    print("\nSpans per label in each TEST fold:")
    print("\n".join(lines))
    if missing_test:
        print(f"\nWARNING  {len(missing_test)} label(s) missing from at least one test fold "
              f"(too few sentences to cover all {args.k}):")
        for name, fl, sc in missing_test:
            print(f"  {name}: absent from fold(s) {fl} ({sc} sentence(s) in the gold)")
    if missing_train:
        print(f"\nWARNING  {len(missing_train)} label(s) absent from the TRAIN set of some fold "
              f"(the model cannot learn them there):")
        for name, fl in missing_train:
            print(f"  {name}: fold(s) {fl}")
    if empty_dev:
        print(f"\nWARNING  layer task(s) with no gold span in the DEV set of some fold:")
        for name, fl in empty_dev:
            print(f"  {name} ({layer_total[name]} spans in the gold): fold(s) {fl}")
        print("  MAChamp then has no dev score for that task and may crash in callback.plot "
              "(KeyError). Patch MAChamp (see README) or raise --min-layer-spans.")
    if any(dropped_train):
        print(f"\nNested spans left out of training per fold (untrained layers): {dropped_train}")
    print(f"\nWritten to {args.outdir}/  -- next: run_cv.py (or train each fold_*/machamp_config.json by hand)")


if __name__ == "__main__":
    main()
