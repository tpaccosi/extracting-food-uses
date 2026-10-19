# Multiclass (HuggingFace) vs. multi-task (MaChAmp): same folds, same evaluation

The multi-task system lives in `machamp_cv/`. This folder adds the multiclass counterpart:
a HuggingFace transformer that predicts **one class per token**, i.e. the combination of the
token's tags across all frames (`biocv.joint_label`, identical to `make_multiclass.py`;
nested `|` cells are kept whole inside the class).

To guarantee identical data reading and scoring, the script imports
`machamp_cv/biocv.py` and `machamp_cv/evaluate_cv.py` directly.

## Usage
```bash
pip install torch "transformers>=4.40" accelerate

# 1. folds (once; shared by both systems)
python3 machamp_cv/make_folds.py --input gold.tsv --k 5 --seed 42 --outdir cv5/

# 2. multi-task (MaChAmp)
python3 machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir machamp/ --device 0

# 3. multiclass (HuggingFace): writes cv5_hf_mc/fold_k/pred.bio.tsv and runs evaluate_cv.py
python3 hf_multiclass/run_hf_multiclass.py --folds-dir cv5/ --outdir cv5_hf_mc/ \
    --model bert-base-multilingual-cased --fp16

# 4. label-by-label comparison (mean F1 over folds and pooled)
python3 machamp_cv/compare_cv.py cv5/ cv5_hf_mc/
```

Useful options: `--folds 1 3` (run only some folds), `--skip-existing` (resume after a crash),
`--keep-checkpoints` (keep the trained models in `fold_k/ckpt/`; by default they are deleted
after prediction), `--oracle` (no training, see below).

## Output (`cv5_hf_mc/`)
Same format as `cv5/`: `folds.json`, `fold_k/{test,pred}.bio.tsv`, `fold_k/run.json`
(configuration, number of classes, best dev F1, test F1), `cv_summary.tsv`,
`cv_per_fold.tsv`, `oof.gold.bio.tsv`, `oof.pred.bio.tsv`.

## Comparability notes
* **Encoder**: `--model` should match `transformer_model` in MaChAmp's `params.json`.
* **Hyperparameters**: `--lr`, `--batch-size` and `--epochs` should match the MaChAmp
  configuration, so that differences between systems do not reflect tuning.
* **Model selection**: the multiclass model selects the best epoch by span micro-F1 over
  all frames on the dev set, whereas MaChAmp combines the dev scores of its tasks.
* **Seeds**: with 5 folds, differences of 1–2 points may be noise. Running several seeds
  per system, with one output folder per seed, gives a more reliable comparison:
  `--seed 1 --outdir cv5_hf_mc_s1/`, `--seed 2 --outdir cv5_hf_mc_s2/`, …
