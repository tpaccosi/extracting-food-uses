#!/usr/bin/env python3
"""
Inter-annotator agreement (Krippendorff's alpha) for WebAnno TSV 3.3 exports
with a single span layer (custom.Span|label), overall and per frame.

Input: one or more folders, each containing exactly two annotator files
(e.g. 1792419489086_teresa.tsv and 1792419489086_Gauri_B.tsv). The two files
of a folder must share the same tokenisation (checked). Results are given per
folder and pooled over all folders.

Unit of analysis: token x frame. For each token and frame, the value is the set
of labels of that frame covering the token (span ids such as [12] and B/I
position are ignored; nested spans with the same label count once), or "O" if
none. Alpha is nominal, computed from the coincidence matrix (two coders, no
missing values). "Overall" pools all token x frame units of the selected frames.

As a complement, span-level agreement is reported as F1 between the two
annotators (exact boundaries and label; symmetric, so no annotator is gold).

Usage:
  python3 iaa_krippendorff.py folder1 folder2 folder3
  python3 iaa_krippendorff.py root_dir --recursive        # every sub-folder with 2 .tsv
  python3 iaa_krippendorff.py ... --frames new             # only new-scheme frames (default)
  python3 iaa_krippendorff.py ... --frames all             # also INGREDIENT, INGESTION
"""
import argparse
import collections
import glob
import os
import re

# label prefix -> frame (checked in order; FOOD_LU before the other FOOD_ labels)
FRAME_OF = [
    ("FOOD_LU", "FOOD_ENTITY"),
    ("FOOD_", "FOOD_ATTRIBUTES"),
    ("DESCR_", "DESCRIPTOR"),
    ("COOKING_CREATION_", "COOKING_CREATION"),
    ("PR_", "PRESERVING"),
    ("CURE_", "CURE"),
    ("INGR_", "INGREDIENT"),
    ("INGESTION_", "INGESTION"),
]
NEW_FRAMES = ["FOOD_ENTITY", "FOOD_ATTRIBUTES", "DESCRIPTOR",
              "COOKING_CREATION", "PRESERVING", "CURE"]
OLD_ONLY = ["INGREDIENT", "INGESTION"]
SPAN_ID = re.compile(r"^(.*?)(?:\[(\d+)\])?$")


def frame_of(label):
    for prefix, frame in FRAME_OF:
        if label == prefix or label.startswith(prefix):
            return frame
    return "OTHER"


# ---------------------------------------------------------------- reading
def read_webanno(path):
    """Return list of tokens: (token_id, offsets, form, [(label, span_id or None), ...])."""
    toks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            tid, off, form = cols[0], cols[1], cols[2]
            cell = cols[3] if len(cols) > 3 else "_"
            anns = []
            if cell not in ("_", "", "*"):
                for item in cell.split("|"):
                    item = item.replace("\\_", "_")
                    m = SPAN_ID.match(item)
                    anns.append((m.group(1), m.group(2)))
            toks.append((tid, off, form, anns))
    return toks


def token_values(toks, frames):
    """{frame: [value per token]}; value = '+'.join(sorted labels) or 'O'."""
    vals = {f: [] for f in frames}
    for _, _, _, anns in toks:
        per = collections.defaultdict(set)
        for lab, _ in anns:
            per[frame_of(lab)].add(lab)
        for f in frames:
            vals[f].append("+".join(sorted(per[f])) if per[f] else "O")
    return vals


def spans(toks):
    """Set of (frame, label, start_index, end_index) spans, from WebAnno span ids."""
    by_id = collections.defaultdict(list)
    out = set()
    for i, (_, _, _, anns) in enumerate(toks):
        for lab, sid in anns:
            if sid is None:                        # single-token span
                out.add((frame_of(lab), lab, i, i + 1))
            else:
                by_id[(lab, sid)].append(i)
    for (lab, _), idx in by_id.items():
        out.add((frame_of(lab), lab, min(idx), max(idx) + 1))
    return out


# ---------------------------------------------------------------- alpha
def nominal_alpha(pairs):
    """Krippendorff's alpha (nominal) for two coders, no missing values.
    pairs: list of (value_coder1, value_coder2)."""
    if not pairs:
        return float("nan")
    o = collections.Counter()                      # coincidence matrix
    for a, b in pairs:
        o[a, b] += 1
        o[b, a] += 1
    n_c = collections.Counter()
    for (a, _), c in o.items():
        n_c[a] += c
    n = sum(n_c.values())                          # = 2 * units
    d_o = sum(c for (a, b), c in o.items() if a != b) / n
    d_e = (n * n - sum(v * v for v in n_c.values())) / (n * (n - 1))
    if d_e == 0:
        return float("nan")                        # no variation (e.g. frame never annotated)
    return 1 - d_o / d_e


def span_f1(s1, s2):
    tp = len(s1 & s2)
    p = tp / len(s1) if s1 else 0.0
    r = tp / len(s2) if s2 else 0.0
    return (2 * p * r / (p + r) if p + r else (1.0 if not s1 and not s2 else 0.0)), len(s1), len(s2)


# ---------------------------------------------------------------- driver
def load_pair(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tsv")))
    if len(files) != 2:
        raise SystemExit(f"{folder}: expected 2 .tsv files, found {len(files)}")
    t1, t2 = read_webanno(files[0]), read_webanno(files[1])
    if [t[:3] for t in t1] != [t[:3] for t in t2]:
        raise SystemExit(f"{folder}: the two files have different tokenisation")
    return files, t1, t2


def evaluate(pairs_of_tokens, frames):
    """pairs_of_tokens: list of (toks1, toks2). Returns per-frame and overall results."""
    unit_pairs = {f: [] for f in frames}
    s1_all, s2_all = set(), set()
    for k, (t1, t2) in enumerate(pairs_of_tokens):
        v1, v2 = token_values(t1, frames), token_values(t2, frames)
        for f in frames:
            unit_pairs[f].extend(zip(v1[f], v2[f]))
        s1_all |= {(k,) + s for s in spans(t1) if s[0] in frames}
        s2_all |= {(k,) + s for s in spans(t2) if s[0] in frames}
    res = {}
    for f in frames:
        f1, n1, n2 = span_f1({s for s in s1_all if s[1] == f}, {s for s in s2_all if s[1] == f})
        res[f] = (nominal_alpha(unit_pairs[f]), f1, n1, n2)
    overall_pairs = [p for f in frames for p in unit_pairs[f]]
    f1, n1, n2 = span_f1(s1_all, s2_all)
    res["OVERALL"] = (nominal_alpha(overall_pairs), f1, n1, n2)
    return res


def short_names(names):
    """Strip the common prefix (e.g. the document id) from the two file names."""
    pre = os.path.commonprefix(names)
    out = [n[len(pre):].strip("_-. ") or n for n in names]
    return out


def print_table(title, res, frames, names):
    a_n, b_n = short_names(names)
    print(f"\n== {title}")
    print(f"{'Frame':18s} {'alpha':>7s} {'span F1':>8s} {'spans ' + a_n:>16s} {'spans ' + b_n:>16s}")
    for f in frames + ["OVERALL"]:
        a, f1, n1, n2 = res[f]
        if n1 == 0 and n2 == 0:
            print(f"{f:18s} {'n/a':>7s} {'n/a':>8s} {n1:16d} {n2:16d}")
        else:
            a_s = f"{a:7.3f}" if a == a else f"{'n/a':>7s}"
            print(f"{f:18s} {a_s} {100 * f1:8.1f} {n1:16d} {n2:16d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--recursive", action="store_true",
                    help="use every sub-folder (at any depth) containing exactly two .tsv files")
    ap.add_argument("--frames", choices=["new", "all"], default="new")
    args = ap.parse_args()

    folders = []
    for d in args.folders:
        if args.recursive:
            for root, _, files in os.walk(d):
                if len([x for x in files if x.endswith(".tsv")]) == 2:
                    folders.append(root)
        else:
            folders.append(d)
    folders = sorted(set(folders))
    if not folders:
        raise SystemExit("no folder with two .tsv files found")

    frames = NEW_FRAMES + (OLD_ONLY if args.frames == "all" else [])
    loaded = [load_pair(d) for d in folders]

    unknown = collections.Counter(lab for _, t1, t2 in loaded for t in (t1, t2)
                                  for *_, anns in t for lab, _ in anns if frame_of(lab) == "OTHER")
    if unknown:
        print("WARNING: labels not mapped to any frame (ignored):", dict(unknown))

    for files, t1, t2 in loaded:
        names = [os.path.splitext(os.path.basename(x))[0] for x in files]
        res = evaluate([(t1, t2)], frames)
        print_table(f"{os.path.dirname(files[0]) or '.'}  ({len(t1)} tokens; {names[0]} vs {names[1]})",
                    res, frames, names)

    if len(loaded) > 1:
        res = evaluate([(t1, t2) for _, t1, t2 in loaded], frames)
        n_tok = sum(len(t1) for _, t1, _ in loaded)
        print_table(f"POOLED over {len(loaded)} folders ({n_tok} tokens)", res, frames, ["annotator 1", "annotator 2"])


if __name__ == "__main__":
    main()