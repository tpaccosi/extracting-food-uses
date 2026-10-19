#!/usr/bin/env python3
"""
lexical_baseline.py -- context-free lexical baseline for the food-frame folds.

For each fold and each frame:
  1. Lexicon. Every annotated span of the frame in train + dev is lowercased and
     counted with its label. For each such form, its unannotated occurrences
     (the same lowercased token sequence appearing in train + dev without being
     a span of that frame) are counted as "O". The form is kept with its
     majority label; forms whose majority is O are discarded (ties are kept).
  2. Tagging. On the test sentences, lexicon forms are matched left to right
     with greedy longest match, regardless of context.

Predictions are written as cv-dir/fold_k/<pred-name> (same format as the gold,
nested cells joined with "|"), so they can be scored with the same evaluator
as the neural models:

    python3 lexical_baseline.py --folds-dir cv5/ --outdir cv5_lexical/
    python3 machamp_cv/evaluate_cv.py --folds-dir cv5_lexical/

The fold layout (folds.json, fold_k/{train,dev,test}.bio.tsv) is the one
produced by machamp_cv/make_folds.py. --outdir receives a copy of folds.json
and of each test.bio.tsv, so the fold directory is never modified.
"""
import argparse
import collections
import json
import shutil
from pathlib import Path


# ---------------------------------------------------------------- BIO I/O
def read_bio(path):
    sents, cur, header = [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if header is None:
                header = line.split("\t")
                continue
            if not line.strip():
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            cur.append(line.split("\t"))
    if cur:
        sents.append(cur)
    return header, sents


def decode(sent, col):
    """Spans (start, end, label) of one column; '|' separates nesting layers."""
    out = set()
    cells = [r[col].split("|") if col < len(r) else ["O"] for r in sent]
    for layer in range(max(len(c) for c in cells)):
        tags = [c[layer] if layer < len(c) else "O" for c in cells] + ["O"]
        start, lab = None, None
        for i, t in enumerate(tags):
            if t in ("O", "_", "") or t.startswith("B-") or (t.startswith("I-") and t[2:] != lab):
                if start is not None:
                    out.add((start, i, lab))
                start, lab = None, None
                if t.startswith(("B-", "I-")):
                    start, lab = i, t[2:]
    return out


def encode(spans, n):
    """BIO cells for n tokens; overlapping spans go to separate layers, outer first."""
    layers = []
    for sp in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]), x[2])):
        for layer in layers:
            if all(sp[1] <= o[0] or o[1] <= sp[0] for o in layer):
                layer.append(sp)
                break
        else:
            layers.append([sp])
    cells = [[] for _ in range(n)]
    for layer in layers:
        tags = ["O"] * n
        for a, b, lab in layer:
            tags[a] = "B-" + lab
            for i in range(a + 1, b):
                tags[i] = "I-" + lab
        for i in range(n):
            cells[i].append(tags[i])
    out = []
    for c in cells:
        while len(c) > 1 and c[-1] == "O":
            c.pop()
        out.append("|".join(c) if c else "O")
    return out


# ---------------------------------------------------------------- lexicon
def build_lexicon(sents, col):
    labels = collections.defaultdict(collections.Counter)   # form -> label counts
    positions = collections.defaultdict(set)                # form -> annotated (sent, start)
    n_max = 1
    lowered = []
    for si, s in enumerate(sents):
        toks = [r[0].lower() for r in s]
        lowered.append(toks)
        for a, b, lab in decode(s, col):
            form = tuple(toks[a:b])
            labels[form][lab] += 1
            positions[form].add((si, a))
            n_max = max(n_max, b - a)

    occurrences = collections.Counter()                     # all occurrences of lexicon forms
    for toks in lowered:
        for i in range(len(toks)):
            for n in range(1, min(n_max, len(toks) - i) + 1):
                form = tuple(toks[i:i + n])
                if form in labels:
                    occurrences[form] += 1

    lexicon = {}
    for form, cnt in labels.items():
        lab, c = cnt.most_common(1)[0]
        unannotated = occurrences[form] - len(positions[form])
        if c >= unannotated:                                 # majority is not O
            lexicon[form] = lab
    return lexicon, n_max


def tag(sent, lexicon, n_max):
    toks = [r[0].lower() for r in sent]
    out, i = set(), 0
    while i < len(toks):
        for n in range(min(n_max, len(toks) - i), 0, -1):
            form = tuple(toks[i:i + n])
            if form in lexicon:
                out.add((i, i + n, lexicon[form]))
                i += n
                break
        else:
            i += 1
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path, help="output of make_folds.py (cv5/)")
    ap.add_argument("--outdir", required=True, type=Path, help="where to write the predictions")
    ap.add_argument("--pred-name", default="pred.bio.tsv")
    args = ap.parse_args()
    if args.folds_dir.resolve() == args.outdir.resolve():
        raise SystemExit("--outdir must differ from --folds-dir")

    folds = sorted(p for p in args.folds_dir.glob("fold_*") if p.is_dir())
    if not folds:
        raise SystemExit(f"no fold_* directories in {args.folds_dir}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    meta = args.folds_dir / "folds.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["system"] = "lexical_baseline"
        (args.outdir / "folds.json").write_text(json.dumps(m), encoding="utf-8")
    else:
        print(f"note: {meta} not found; evaluate_cv.py needs it in {args.outdir}")

    for fd in folds:
        header, train = read_bio(fd / "train.bio.tsv")
        _, dev = read_bio(fd / "dev.bio.tsv")
        _, test = read_bio(fd / "test.bio.tsv")
        frames = header[1:]
        pred = {f: [] for f in frames}
        sizes = {}
        for ci, f in enumerate(frames, 1):
            lexicon, n_max = build_lexicon(train + dev, ci)
            sizes[f] = len(lexicon)
            pred[f] = [encode(tag(s, lexicon, n_max), len(s)) for s in test]
        od = args.outdir / fd.name
        od.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fd / "test.bio.tsv", od / "test.bio.tsv")
        with open(od / args.pred_name, "w", encoding="utf-8") as out:
            out.write("\t".join(header) + "\n")
            for si, s in enumerate(test):
                for ti, r in enumerate(s):
                    out.write("\t".join([r[0]] + [pred[f][si][ti] for f in frames]) + "\n")
                out.write("\n")
        print(f"{fd.name}: lexicon sizes " + ", ".join(f"{f}={n}" for f, n in sizes.items())
              + f"  -> {od / args.pred_name}")


if __name__ == "__main__":
    main()
