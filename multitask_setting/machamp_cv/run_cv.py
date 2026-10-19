#!/usr/bin/env python3
"""
run_cv.py
=========
Runs the K-fold experiment with MAChamp: for every fold it trains on
train.conll (dev.conll for validation), predicts test.conll, recomposes the
layered prediction into pred.bio.tsv and finally calls evaluate_cv.py.

Usage
-----
    python3 machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir ~/machamp --device 0

    --folds 1 3          only some folds
    --skip-existing      reuse a fold's model / prediction if already there (resume after a crash)
    --parameters-config  passed to MAChamp's train.py (default: MAChamp's own configs/params.json)
    --dry-run            print the commands without running them

Per fold, cv5/fold_k/ gets: pred.conll (raw MAChamp output), pred.bio.tsv
(recomposed, our format) and run.json (model path, commands).
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from biocv import columns_from_meta, load_meta, write_machamp_config  # noqa: E402
from recompose_predictions import recompose  # noqa: E402


def run(cmd, cwd, dry):
    print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def newest_model(machamp_dir: Path, name: str) -> Path | None:
    models = sorted((machamp_dir / "logs" / name).glob("*/model.pt"), key=lambda p: p.stat().st_mtime)
    return models[-1] if models else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path)
    ap.add_argument("--machamp-dir", required=True, type=Path, help="clone of github.com/machamp-nlp/machamp")
    ap.add_argument("--device", default="0", help="GPU id, -1 for CPU")
    ap.add_argument("--parameters-config", type=Path)
    ap.add_argument("--name-prefix", help="MAChamp run name prefix (default: dataset name)")
    ap.add_argument("--folds", type=int, nargs="*", help="fold numbers to run (default: all)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folds_dir = args.folds_dir.resolve()
    machamp_dir = args.machamp_dir.resolve()
    meta = load_meta(folds_dir)
    columns = columns_from_meta(meta)
    dataset = meta["dataset_name"]
    prefix = args.name_prefix or dataset
    folds = args.folds or list(range(1, meta["k"] + 1))
    if not args.dry_run and not (machamp_dir / "train.py").exists():
        raise SystemExit(f"{machamp_dir}/train.py not found -- is --machamp-dir a MAChamp clone?")

    for f in folds:
        d = folds_dir / f"fold_{f}"
        name = f"{prefix}_fold{f}"
        print(f"\n=== fold {f}/{meta['k']}  ({name})", flush=True)
        cfg = d / "machamp_config.json"
        write_machamp_config(cfg, d, dataset, columns)  # refresh absolute paths if the dir moved

        model = newest_model(machamp_dir, name) if args.skip_existing else None
        train_cmd = [args.python, "train.py", "--dataset_configs", cfg, "--device", args.device, "--name", name]
        if args.parameters_config:
            train_cmd += ["--parameters_config", args.parameters_config.resolve()]
        if model is None:
            run(train_cmd, machamp_dir, args.dry_run)
            model = newest_model(machamp_dir, name) if not args.dry_run else \
                machamp_dir / "logs" / name / "<DATE>" / "model.pt"
            if model is None:
                raise SystemExit(f"no model.pt under {machamp_dir}/logs/{name}/ after training")
        else:
            print(f"  reusing {model}")

        pred = d / "pred.conll"
        pred_cmd = [args.python, "predict.py", model, d / "test.conll", pred,
                    "--dataset", dataset, "--device", args.device]
        if not (args.skip_existing and pred.exists()):
            run(pred_cmd, machamp_dir, args.dry_run)
        if not args.dry_run:
            n = recompose(folds_dir, pred, d / "pred.bio.tsv", d / "test.conll")
            print(f"  wrote {d / 'pred.bio.tsv'} ({n} sentences)")
            (d / "run.json").write_text(json.dumps({
                "model": str(model),
                "train_cmd": [str(c) for c in train_cmd],
                "predict_cmd": [str(c) for c in pred_cmd],
            }, indent=2), encoding="utf-8")

    if args.dry_run:
        return
    missing = [f for f in range(1, meta["k"] + 1) if not (folds_dir / f"fold_{f}" / "pred.bio.tsv").exists()]
    if missing:
        print(f"\nFolds without predictions yet: {missing} -- run them, then evaluate_cv.py")
        return
    print("\n=== evaluation")
    subprocess.run([args.python, str(HERE / "evaluate_cv.py"), "--folds-dir", str(folds_dir)], check=True)


if __name__ == "__main__":
    main()
