"""Crea i 5 fold a livello di documento e li salva, così tutti i modelli usano gli stessi split.

Uso:
    python make_folds.py --data gold.tsv --out folds.json --k 5 --seed 42
"""
import argparse
from collections import Counter

from common import clean_docs, load_gold, make_folds, save_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="folds.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    args = ap.parse_args()

    layers, docs = load_gold(args.data)
    docs = clean_docs(docs, layers)
    folds = make_folds(len(docs), args.k, args.seed, args.dev_ratio)
    save_folds(folds, args.out)

    print(f"{len(docs)} documenti, {sum(len(d['tokens']) for d in docs)} token, layer: {layers}")
    for fd in folds:
        ntok = {s: sum(len(docs[i]["tokens"]) for i in fd[s]) for s in ("train", "dev", "test")}
        print(f"fold {fd['fold']}: " + " | ".join(
            f"{s} {len(fd[s])} doc / {ntok[s]} tok" for s in ("train", "dev", "test")))
    print("\nSpan per tipo nel test di ogni fold (controllo di copertura):")
    for lay in layers:
        print(f"  {lay}")
        for fd in folds:
            c = Counter(l[2:] for i in fd["test"] for l in docs[i]["labels"][lay]
                        if l.startswith("B-"))
            print(f"    fold {fd['fold']}: " + ", ".join(f"{t}={n}" for t, n in sorted(c.items())))
    print(f"\nFold salvati in {args.out}")


if __name__ == "__main__":
    main()
