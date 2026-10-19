# Extracting Culinary, Medicinal and Preservation Uses of Food from Historical Texts

> **Note:** The fine-tuned models will be released on Zenodo after the anonymity period. All data, prompts and code needed to reproduce the experiments are already available in this repository.

Data and code for extracting food entities and their uses from historical English texts
(18th–20th centuries) with a frame-semantic annotation scheme, plus an exploratory
pipeline for historical Dutch.



The scheme covers six annotation layers:

| Frame | Labels |
|---|---|
| `FOOD_ENTITY` | `FOOD_LU` |
| `FOOD_ATTRIBUTES` | `Count`, `Unit`, `Constituent_Part` |
| `DESCRIPTOR` | `Neutral`, `Positive`, `Negative` |
| `COOKING_CREATION` | `LU`, `Food_Material`, `Food_Product` |
| `PRESERVING` | `LU`, `Food_Patient`, `Medium` |
| `CURE` | `LU`, `Affliction`, `Food_Treatment`, `Body_Part`, `Patient`, `Healer` (spans may be nested) |

See `revised_annotation_guidelines.pdf` for the full guidelines.

## Repository structure

```
data/english/
  gold_final.tsv          gold standard (810 sentences): LLM pre-annotation + manual correction
  gold_pilot.tsv          pilot sentences annotated from scratch
  llm results/            outputs of the seven LLMs compared as pre-annotators
data/dutch/               Dutch data
prompts/english/          two-stage prompting pipeline (food entities, then one frame per prompt)
prompts/dutch/            prompts for the Dutch cascade
agreement/                inter-annotator agreement (Krippendorff's alpha, IAA.py)
multitask_setting/        MaChAmp multi-task models, 5-fold CV, lexical baseline
multiclass_real/          HuggingFace multiclass models (one joint label per token)
```

Each folder with code has its own README with usage instructions.

## Data format

Multi-column BIO TSV: a header row, one token per line, a blank line between sentences.
Nested spans share a cell, separated by `|` (outer span first):

```
token   FOOD_ENTITY  FOOD_ATTRIBUTES  DESCRIPTOR  COOKING_CREATION  PRESERVING  CURE
pain    O            O                O           O                 O           B-CURE_Affliction
in      O            O                O           O                 O           I-CURE_Affliction
the     O            O                O           O                 O           I-CURE_Affliction
eyes    O            O                O           O                 O           I-CURE_Affliction|B-CURE_Body_Part
```

## Quick start

```bash
# folds, training and evaluation of the multi-task models
python3 multitask_setting/machamp_cv/make_folds.py --input data/english/gold_final.tsv \
    --k 5 --seed 42 --outdir cv5/
python3 multitask_setting/machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir machamp/ --device 0

# lexical baseline on the same folds
python3 multitask_setting/lexical_baseline.py --folds-dir cv5/ --outdir cv5_lexical/
python3 multitask_setting/machamp_cv/evaluate_cv.py --folds-dir cv5_lexical/

# inter-annotator agreement
python3 agreement/IAA.py agreement/ --recursive
```

All systems are evaluated at span level (exact match, nested spans included), as mean ± std
over the 5 folds and pooled out-of-fold.

## License

See ``` LICENSE.md ```
