# Dutch data

Exploratory data for the historical-Dutch pipeline, from the EU TRIFECTA grant
project (`trifecta-annotation`), sourced from the DBNL historical Dutch corpus.

There are two separate KWIC batches behind this pipeline, not one:

1. **Manual batch** (`food_snippets_kwic_sample.csv`) — the original,
   hand-curated KWIC batch. The Step A/B/C cascade's few-shot prompt
   examples (see `prompts/dutch/prompts.txt`) were selected from this batch.
2. **INCEpTION batch** (`kwic_inputs.jsonl`) — a later, separate
   multi-annotator effort. This is the input to GijsBERT training and to
   the gold-labelling batch sampling, not to the cascade's few-shot prompts.

## Files

- `food_snippets_kwic_sample.csv` — 315-row sample of the manual batch
  (deduped on exact snippet text, up to 5 snippets per source work, 63
  works represented). The full batch is ~31,000 rows / 70MB (plus a 759MB
  exploded-per-keyword version); too large for this repo, sent separately.
- `kwic_inputs.jsonl` — 3,118 normalized KWIC records from the INCEpTION
  WebAnno export; input to GijsBERT training and gold-labelling sampling.
- `gijsbert_train.jsonl` — 2,690 rows, frame-classification training export
  (Step B). 105 rows of synthetic English calibration sentences (13 unique
  sentences, used for a cross-lingual scratch experiment) were filtered out
  of the original 2,795-row export to keep this file Dutch-only.
- `gijsbert_eval.jsonl` — 157 rows, held-out evaluation split for the same
  classifier.
- `inception_nl_v2_iaa_agreed_gold.csv` — 394 rows where two independent
  INCEpTION annotators agreed, frame-classification gold.

## Caveat on the gold set

`inception_nl_v2_iaa_agreed_gold.csv` is a **curated, easier, positive-biased
slice** — rows where annotators converged quickly — not a representative
sample of corpus difficulty. Treat it as a sanity-check set, not as
equivalent in coverage to `data/english/gold_final.tsv`.
