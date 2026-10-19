# 5-fold CV per token classification su gold.tsv

## Passi
```bash
pip install -r requirements.txt

# 1. fold fissi a livello di documento (tutti i modelli usano gli stessi split)
python make_folds.py --data gold.tsv --out folds.json

# 2. baseline lineare (CPU, pochi secondi)
python baseline_linear.py --data gold.tsv --folds folds.json --out results/baseline

# 3. transformer (GPU consigliata); un modello per layer e per fold
python train_transformer.py --data gold.tsv --folds folds.json --model roberta-base --fp16
#    solo alcuni layer/fold:  --layers CURE,PRESERVING  --only_folds 0,1

# 4. tabella media ± dev.std sui fold
python aggregate.py results/baseline results/roberta-base
```

## Scelte di design
- Ogni colonna (FOOD_ENTITY, FOOD_ATTRIBUTES, DESCRIPTOR, COOKING_CREATION, PRESERVING, CURE)
  è un task BIO multiclasse separato.
- Etichette annidate `A|B`: si tiene la prima (span esterno); `--nested last` per l'interno.
- Refusi normalizzati (`B_`→`B-`, `i-`→`I-`, `V-`→`B-`, `o`/`Ot`/`Os`/`BO`/vuoto→`O`),
  poi riparazione BIO (I- orfano → B-).
- Split per documento (niente leakage); il 10% del train fa da dev per early stopping.
- Documenti oltre i 512 sub-token vengono spezzati a fine frase e ricomposti prima della valutazione.
- Metrica principale: F1 span-level a match esatto (micro e macro), più F1 token-level.

## Output
`results/<run>/<layer>/fold<k>/metrics.json` e `predictions.tsv` (token, gold, pred);
`aggregate.py` scrive `summary.csv` e `per_class.csv` in ogni run.
