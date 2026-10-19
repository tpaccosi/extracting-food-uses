"""Baseline veloce: classificatore lineare (LinearSVC) con feature a finestra ±2.
Serve come sanity check della pipeline e come riferimento minimo per i transformer.

Uso:
    python baseline_linear.py --data gold.tsv --folds folds.json --out results/baseline
"""
import argparse
import json
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from common import (clean_docs, evaluate, load_folds, load_gold, repair_bio,
                    write_predictions)


def shape(w):
    s = "".join("X" if c.isupper() else "x" if c.islower() else "d" if c.isdigit() else c
                for c in w)
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def token_feats(toks, i):
    f = {"bias": 1.0}
    for off in (-2, -1, 0, 1, 2):
        j = i + off
        if 0 <= j < len(toks):
            w = toks[j]
            f[f"{off}:w={w.lower()}"] = 1.0
            f[f"{off}:shape={shape(w)}"] = 1.0
            if off == 0:
                f[f"suf3={w[-3:].lower()}"] = 1.0
                f[f"suf2={w[-2:].lower()}"] = 1.0
                f[f"pre3={w[:3].lower()}"] = 1.0
                f["istitle"] = float(w.istitle())
                f["isupper"] = float(w.isupper())
        else:
            f[f"{off}:PAD"] = 1.0
    return f


def featurize(docs):
    return [token_feats(d["tokens"], i) for d in docs for i in range(len(d["tokens"]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="results/baseline")
    ap.add_argument("--layers", default="all", help="'all' oppure lista separata da virgole")
    ap.add_argument("--nested", default="first", choices=["first", "last"])
    ap.add_argument("--C", type=float, default=0.1)
    args = ap.parse_args()

    layers, docs = load_gold(args.data)
    docs = clean_docs(docs, layers, args.nested)
    folds = load_folds(args.folds)
    sel = layers if args.layers == "all" else args.layers.split(",")

    for layer in sel:
        for fd in folds:
            k = fd["fold"]
            # il baseline non usa il dev: si allena su train+dev
            tr = [docs[i] for i in fd["train"] + fd["dev"]]
            te = [docs[i] for i in fd["test"]]
            vec = DictVectorizer()
            Xtr = vec.fit_transform(featurize(tr))
            ytr = [l for d in tr for l in d["labels"][layer]]
            clf = LinearSVC(C=args.C).fit(Xtr, ytr)
            flat = list(clf.predict(vec.transform(featurize(te))))

            preds, pos = [], 0
            for d in te:
                n = len(d["tokens"])
                preds.append(repair_bio(flat[pos:pos + n]))
                pos += n
            gold = [d["labels"][layer] for d in te]
            m = evaluate(gold, preds)

            odir = os.path.join(args.out, layer, f"fold{k}")
            os.makedirs(odir, exist_ok=True)
            with open(os.path.join(odir, "metrics.json"), "w") as f:
                json.dump(m, f, indent=2)
            write_predictions(os.path.join(odir, "predictions.tsv"),
                              [d["tokens"] for d in te], gold, preds)
            print(f"{layer:18s} fold{k}  span-F1 micro={m['span_micro_f1']:.3f} "
                  f"macro={m['span_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
