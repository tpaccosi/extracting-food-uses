# machamp_cv — 5-fold cross-validation con MAChamp

Tre passi, dal gold BIO TSV al riepilogo:

```bash
# 1. fold stratificati + file MAChamp per ogni fold
python3 machamp_cv/make_folds.py --input teresa_gold.bio.tsv --k 5 --seed 42 --outdir cv5/

#    se alcune righe hanno una colonna in più a destra: --n-frames 6 (tiene token + le prime 6)

#    layer annidati troppo rari: --min-layer-spans 20 (non li addestra; i loro span contano come FN)

# 1b. una volta sola: evita il crash di MAChamp (KeyError in callback.plot) sui task senza score di dev
python3 machamp_cv/patch_machamp.py --machamp-dir ~/machamp

# 2. training + predizione su ogni fold, poi valutazione automatica
python3 machamp_cv/run_cv.py --folds-dir cv5/ --machamp-dir ~/machamp --device 0
#    (--dry-run per vedere i comandi, --folds 2 4 per alcuni fold, --skip-existing per riprendere)

# 3. (già chiamato da run_cv.py) valutazione
python3 machamp_cv/evaluate_cv.py --folds-dir cv5/
```

## Cosa fa ogni script

| script | ruolo |
|---|---|
| `make_folds.py` | assegna le frasi ai fold con stratificazione iterativa multi-etichetta (etichetta più rara prima, pesata sul numero di span); per ogni fold test = quel fold, dev = 10% stratificato del resto (`--dev-frac`), train = il resto. Stampa e salva la tabella etichette × fold e avvisa se un'etichetta manca dal test o dal train di qualche fold. |
| `run_cv.py` | per ogni fold: `train.py` di MAChamp, `predict.py` sul test, ricomposizione in `pred.bio.tsv`. |
| `recompose_predictions.py` | da colonne a layer (`FRAME`, `FRAME__L1`, …) di nuovo a celle con `\|`. Utile se lanci MAChamp a mano. |
| `evaluate_cv.py` | valutazione a livello di span (match esatto, span annidati inclusi): media ± dev. std. tra fold **e** pooled out-of-fold. |
| `biocv.py` | funzioni condivise. |

## Celle con `|`

Una cella come `I-CURE_Affliction|B-CURE_LU` contiene due span. Ogni colonna frame viene scomposta in layer senza sovrapposizioni: gli span esterni restano in `FRAME`, quelli annidati vanno in `FRAME__L1`, `FRAME__L2`… (quanti layer servono lo decide il gold intero, così tutti i fold hanno le stesse colonne). Ogni layer è un task `seq_bio` normale. Dopo la predizione i layer vengono riuniti (layer esterno per primo). `make_folds.py` verifica il giro completo su ogni cella annidata e stampa quante tornano identiche.

Limite: due span **dello stesso tipo che si incrociano** senza essere uno dentro l'altro non sono ricostruibili in modo univoco dal formato con `|`; se capita, `make_folds.py` lo segnala come "spans differ".

## Come leggere i risultati

* **media ± std**: variabilità tra fold. Per un'etichetta, i fold senza span gold di quella etichetta sono esclusi dalla media (colonna `n`).
* **pooled**: i 5 test set concatenati, ogni frase predetta da un modello che non l'ha vista. Per le etichette rare è il numero più affidabile.
* Etichette con ≤ 5 frasi nel gold (`--rare-threshold`) sono marcate "unstable": nessuno schema di fold dà un numero stabile.

File prodotti in `cv5/`: `cv_summary.tsv`, `cv_per_fold.tsv`, `label_fold_table.tsv`, `oof.gold.bio.tsv` / `oof.pred.bio.tsv` (tutte le frasi nell'ordine originale, per l'analisi degli errori).
