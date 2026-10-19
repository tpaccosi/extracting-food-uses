# Multi-task food-frame labelling with MaChAmp (5-fold CV)

Span-level sequence labelling of food entities and their uses (`FOOD_ENTITY`, `FOOD_ATTRIBUTES`,
`DESCRIPTOR`, `COOKING_CREATION`, `PRESERVING`, `CURE`), one BIO task per frame with
[MaChAmp](https://github.com/machamp-nlp/machamp).

## Input

Multi-column BIO TSV with a header row and a blank line between sentences.
Nested spans share a cell, separated by `|` (outer span first):

```
token   FOOD_ENTITY  FOOD_ATTRIBUTES  DESCRIPTOR  COOKING_CREATION  PRESERVING  CURE
eyes    O            O                O           O                 O           I-CURE_Affliction|B-CURE_Body_Part
```

Nested spans are trained as extra layer tasks (`FRAME__L1`, …) and merged back after prediction.

## Usage

```bash
# once: patch MaChAmp (avoids a KeyError in callback.plot)
python3 machamp_cv/patch_machamp.py --machamp-dir machamp/

# 1. stratified folds + MaChAmp files
python3 machamp_cv/make_folds.py --input gold.bio.tsv --k 5 --seed 42 --outdir cv5/

# 2. train, predict and evaluate every fold
python3 machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir machamp/ --device 0
```

Options: `--folds 2 4` (some folds only), `--skip-existing` (resume), `--dry-run` (print commands).

To train on another annotation of the same sentences (e.g. uncorrected pre-annotations)
and test on the original gold folds:

```bash
python3 machamp_cv/make_folds_from.py --cvdir cv5/ --input silver.bio.tsv --outdir cv5_silver/
python3 machamp_cv/run_cv.py --folds-dir cv5_silver/ --machamp-dir machamp/ --device 0
```

## Output (`cv5/`)

* `cv_summary.tsv`: P/R/F1 per label and frame, mean ± std over folds and pooled out-of-fold;
* `cv_per_fold.tsv`, `label_fold_table.tsv`;
* `oof.gold.bio.tsv`, `oof.pred.bio.tsv`: out-of-fold predictions for error analysis;
* `fold_k/pred.bio.tsv` and `fold_k/run.json` (path of the trained `model.pt`).

Scores are span-level, exact match, nested spans included.

`folds/` contains the folds used in the paper; `multi_results/` the summaries of all systems
and notebooks to compare them.
