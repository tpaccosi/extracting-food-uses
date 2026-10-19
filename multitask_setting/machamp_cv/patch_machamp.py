#!/usr/bin/env python3
"""
patch_machamp.py
================
MAChamp crashes at the end of an epoch with

    File ".../machamp/model/callback.py", in plot
        x[-1].append(self.dev_scores[epoch][task_name][main_metric][sum_metric])
    KeyError: 'DESCRIPTOR__L1'

when a task has no dev score in that epoch (a sparse layer task with no gold
span in dev, where the model also predicts none -- MAChamp prints "-" for it).
The crash is only in the loss/score *plot*; this patch makes the plot use 0.0
for a missing score. Training and model selection are not changed.

Usage:
    python3 machamp_cv/patch_machamp.py --machamp-dir machamp/

Idempotent; the original is kept as callback.py.orig.
"""
import argparse
import re
import shutil
from pathlib import Path

PAT = re.compile(r"self\.(dev|train)_scores\[(\w+)\]\[(\w+)\]\[(\w+)\]\[(\w+)\]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machamp-dir", required=True, type=Path)
    args = ap.parse_args()
    path = args.machamp_dir / "machamp" / "model" / "callback.py"
    if not path.exists():
        raise SystemExit(f"{path} not found -- is --machamp-dir the MAChamp clone?")
    lines = path.read_text(encoding="utf-8").split("\n")
    changed = []
    for i, line in enumerate(lines):
        if ".append(" in line and PAT.search(line):   # only reads inside append(...), never assignments
            new = PAT.sub(r"self.\1_scores[\2].get(\3, {}).get(\4, {}).get(\5, 0.0)", line)
            if new != line:
                lines[i] = new
                changed.append((i + 1, line.strip(), new.strip()))
    if not changed:
        print(f"nothing to patch in {path} (already patched, or a MAChamp version without this code)")
        return
    backup = path.with_name(path.name + ".orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text("\n".join(lines), encoding="utf-8")
    for lineno, old, new in changed:
        print(f"{path}:{lineno}\n  - {old}\n  + {new}")
    print(f"backup: {backup}")


if __name__ == "__main__":
    main()
