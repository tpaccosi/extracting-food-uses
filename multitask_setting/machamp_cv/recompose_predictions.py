#!/usr/bin/env python3
"""
recompose_predictions.py
========================
Turns a MAChamp prediction file (layered columns: FRAME, FRAME__L1, ...)
back into our BIO TSV format, merging the layers of each frame into "|"
cells (outer span first) exactly as in the gold.

run_cv.py calls this automatically; use it by hand if you ran predict.py
yourself:

    python3 machamp_cv/recompose_predictions.py --folds-dir cv5/ \\
        --pred cv5/fold_1/pred.conll --out cv5/fold_1/pred.bio.tsv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from biocv import columns_from_meta, compose_sentence, load_meta, read_conll, write_bio_tsv  # noqa: E402


def recompose(folds_dir: Path, pred: Path, out: Path, gold: Path | None = None) -> int:
    meta = load_meta(folds_dir)
    header = meta["header"]
    columns = columns_from_meta(meta)
    layered = read_conll(pred, 1 + len(columns))
    sents = [compose_sentence(header[1:], s, columns) for s in layered]
    if gold is not None:
        gold_layered = read_conll(gold, 1 + len(columns))
        if len(gold_layered) != len(layered):
            raise SystemExit(f"{pred}: {len(layered)} sentences, {gold} has {len(gold_layered)}")
        for i, (g, p) in enumerate(zip(gold_layered, layered)):
            if [r[0] for r in g] != [r[0] for r in p]:
                raise SystemExit(f"{pred}: tokens of sentence {i + 1} differ from {gold}")
    write_bio_tsv(out, header, sents)
    return len(sents)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path, help="output dir of make_folds.py")
    ap.add_argument("--pred", required=True, type=Path, help="MAChamp prediction file")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gold-conll", type=Path, help="optional: check sentence/token alignment against this file")
    args = ap.parse_args()
    n = recompose(args.folds_dir, args.pred, args.out, args.gold_conll)
    print(f"wrote {args.out} ({n} sentences)")


if __name__ == "__main__":
    main()
