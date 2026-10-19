# Multiclass (HuggingFace) vs multi-task (MAChamp), stessi fold e stessa valutazione

Il multi-task resta quello in `machamp_cv/`. Questa cartella aggiunge solo il lato multiclass:
un transformer HuggingFace con **una classe per token**, cioè la combinazione dei tag di tutti i frame
(`biocv.joint_label`, identica a `make_multiclass.py`; le celle `|` restano intere).

Per garantire la stessa lettura dei dati e la stessa metrica, lo script importa direttamente
`machamp_cv/biocv.py` ed `machamp_cv/evaluate_cv.py`.

## Struttura attesa
```
progetto/
  machamp_cv/          # il tuo codice (biocv.py, evaluate_cv.py, compare_cv.py, ...)
  hf_multiclass/       # questa cartella
  gold.tsv
```

## Passi
```bash
pip install torch "transformers>=4.40" accelerate

# 1. fold (una volta sola, li usano entrambi i sistemi)
python3 machamp_cv/make_folds.py --input gold.tsv --k 5 --seed 42 --outdir cv5/

# 2. multi-task (MAChamp)
python3 machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir machamp/ --device 0

# 3. multiclass (HuggingFace) -- scrive cv5_hf_mc/fold_k/pred.bio.tsv e lancia evaluate_cv.py
python3 hf_multiclass/run_hf_multiclass.py --folds-dir cv5/ --outdir cv5_hf_mc/ \
    --model bert-base-multilingual-cased --fp16

# 4. confronto etichetta per etichetta (F1 media sui fold e pooled)
python3 machamp_cv/compare_cv.py cv5/ cv5_hf_mc/
```

Opzioni utili: `--folds 1 3` (solo alcuni fold), `--skip-existing` (riprendere dopo un crash),
`--oracle` (nessun training: vedi sotto).

## `--oracle`: il tetto dell'approccio multiclass
Predice la classe gold quando quella combinazione compare nel train del fold, altrimenti `O`.
Misura quanto si perde **solo** perché alcune combinazioni non sono mai viste in training;
nessun modello multiclass può superarlo. Sul gold attuale: micro-F1 97.9, macro-F1 96.9.
Le etichette più penalizzate sono DESCR_Negative (87) e COOKING_CREATION_Food_Product (87).
Serve anche per provare l'intera catena senza GPU.

## Output (`cv5_hf_mc/`)
Stesso formato di `cv5/`: `folds.json`, `fold_k/{test,pred}.bio.tsv`, `fold_k/run.json`
(config, n. classi, F1 dev migliore, F1 test), `cv_summary.tsv`, `cv_per_fold.tsv`,
`oof.gold.bio.tsv`, `oof.pred.bio.tsv`.

## Perché il confronto sia equo
* **Stesso encoder**: `--model` deve essere il `transformer_model` del `params.json` di MAChamp.
* **Iperparametri**: allinea `--lr`, `--batch-size`, `--epochs` a quelli di MAChamp; altrimenti
  la differenza tra i sistemi include anche il tuning.
* **Selezione dell'epoca**: qui si sceglie sul dev con lo span micro-F1 di tutti i frame; MAChamp
  combina gli score di dev dei suoi task. È una differenza da dichiarare.
* **Più seed**: con 5 fold, differenze di 1-2 punti possono essere rumore; meglio 3 seed per sistema,
  una cartella per seed: `--seed 1 --outdir cv5_hf_mc_s1/`, `--seed 2 --outdir cv5_hf_mc_s2/`…
