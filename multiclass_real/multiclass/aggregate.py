"""Aggrega le metriche dei 5 fold: media ± deviazione standard per layer e per classe.

Uso:
    python aggregate.py results/baseline results/roberta-base
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

MAIN = ["span_micro_f1", "span_macro_f1", "span_micro_precision",
        "span_micro_recall", "token_macro_f1"]


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.3f}"


def aggregate(run_dir):
    summary, per_class = [], []
    for layer_dir in sorted(glob.glob(os.path.join(run_dir, "*"))):
        files = sorted(glob.glob(os.path.join(layer_dir, "fold*", "metrics.json")))
        if not files:
            continue
        ms_ = [json.load(open(f)) for f in files]
        layer = os.path.basename(layer_dir)
        row = {"layer": layer, "n_folds": len(ms_)}
        for k in MAIN:
            row[k] = ms([m[k] for m in ms_])
        summary.append(row)
        classes = sorted({c for m in ms_ for c in m["per_class"]})
        for c in classes:
            vals = [m["per_class"].get(c, {"f1": 0, "support": 0}) for m in ms_]
            per_class.append({"layer": layer, "class": c,
                              "f1": ms([v["f1"] for v in vals]),
                              "support_tot": sum(v["support"] for v in vals)})
    return pd.DataFrame(summary), pd.DataFrame(per_class)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    for run in sys.argv[1:]:
        s, c = aggregate(run)
        print(f"\n=== {run} ===")
        print(s.to_string(index=False))
        print()
        print(c.to_string(index=False))
        s.to_csv(os.path.join(run, "summary.csv"), index=False)
        c.to_csv(os.path.join(run, "per_class.csv"), index=False)


if __name__ == "__main__":
    main()
