"""5-fold CV di token classification con un transformer HuggingFace.
Un modello separato per ogni layer di annotazione (ognuno è un task multiclasse BIO).

Uso:
    python train_transformer.py --data gold.tsv --folds folds.json \
        --model roberta-base --layers all --epochs 10 --out results
"""
import argparse
import inspect
import json
import os
import shutil

import numpy as np
import torch
from transformers import (AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

from common import (clean_docs, evaluate, label_list, load_folds, load_gold, repair_bio,
                    write_predictions)

SENT_END = {".", ";", "!", "?", ":"}
PREFIX_SPACE_MODELS = {"roberta", "longformer", "bart", "gpt2", "mpnet", "deberta"}


# ----------------------------------------------------------------------------
# Chunking dei documenti lunghi (alcuni superano i 512 sub-token)
# ----------------------------------------------------------------------------

def chunk_doc(tokens, tokenizer, max_len):
    """Divide un documento in segmenti di parole che stanno in max_len sub-token,
    tagliando preferibilmente dopo un segno di fine frase. Ritorna liste di (start, end)."""
    enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
    n_sub = [0] * len(tokens)
    for w in enc.word_ids():
        if w is not None:
            n_sub[w] += 1
    budget = max_len - tokenizer.num_special_tokens_to_add()
    chunks, start, used, last_break = [], 0, 0, None
    i = 0
    while i < len(tokens):
        if used + n_sub[i] > budget and i > start:
            cut = last_break + 1 if last_break is not None and last_break >= start else i
            chunks.append((start, cut))
            start, used, last_break = cut, 0, None
            i = cut
            continue
        used += n_sub[i]
        if tokens[i] in SENT_END:
            last_break = i
        i += 1
    if start < len(tokens):
        chunks.append((start, len(tokens)))
    return chunks


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, docs, layer, tokenizer, label2id, max_len):
        self.items, self.meta = [], []  # meta: (doc_idx, start, end, word_ids)
        for di, d in enumerate(docs):
            for s, e in chunk_doc(d["tokens"], tokenizer, max_len):
                words = d["tokens"][s:e]
                labs = d["labels"][layer][s:e]
                enc = tokenizer(words, is_split_into_words=True, truncation=True,
                                max_length=max_len)
                wids = enc.word_ids()
                lab_ids, prev = [], None
                for w in wids:
                    if w is None or w == prev:
                        lab_ids.append(-100)  # etichetta solo sul primo sub-token
                    else:
                        lab_ids.append(label2id[labs[w]])
                    prev = w
                item = {k: v for k, v in enc.items()}
                item["labels"] = lab_ids
                self.items.append(item)
                self.meta.append((di, s, e, wids))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def decode(dataset, logits, docs, id2label):
    """Ricompone le predizioni a livello di parola e di documento."""
    preds = [["O"] * len(d["tokens"]) for d in docs]
    pred_ids = logits.argmax(-1)
    for row, (di, s, e, wids) in enumerate(dataset.meta):
        prev = None
        for pos, w in enumerate(wids):
            if w is not None and w != prev:
                preds[di][s + w] = id2label[int(pred_ids[row, pos])]
            prev = w
    return [repair_bio(p) for p in preds]


# ----------------------------------------------------------------------------

def make_training_args(**kw):
    # compatibilità: 'evaluation_strategy' è stato rinominato 'eval_strategy'
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**kw)


def run_fold(args, layer, fd, docs, labels, tokenizer):
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    k = fd["fold"]
    odir = os.path.join(args.out, args.run_name, layer, f"fold{k}")
    os.makedirs(odir, exist_ok=True)
    set_seed(args.seed)

    tr_docs = [docs[i] for i in fd["train"]]
    dv_docs = [docs[i] for i in fd["dev"]]
    te_docs = [docs[i] for i in fd["test"]]
    ds_tr = ChunkDataset(tr_docs, layer, tokenizer, label2id, args.max_len)
    ds_dv = ChunkDataset(dv_docs, layer, tokenizer, label2id, args.max_len)
    ds_te = ChunkDataset(te_docs, layer, tokenizer, label2id, args.max_len)

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def compute_metrics(ev):
        logits, _ = ev
        preds = decode(ds_dv, logits, dv_docs, id2label)
        m = evaluate([d["labels"][layer] for d in dv_docs], preds)
        return {"span_f1": m["span_micro_f1"], "span_macro_f1": m["span_macro_f1"]}

    targs = make_training_args(
        output_dir=os.path.join(odir, "ckpt"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="span_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available() and args.fp16,
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=ds_tr, eval_dataset=ds_dv,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    out = trainer.predict(ds_te)
    preds = decode(ds_te, out.predictions, te_docs, id2label)
    gold = [d["labels"][layer] for d in te_docs]
    m = evaluate(gold, preds)
    m["best_dev_span_f1"] = trainer.state.best_metric
    m["config"] = vars(args)
    with open(os.path.join(odir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2)
    write_predictions(os.path.join(odir, "predictions.tsv"),
                      [d["tokens"] for d in te_docs], gold, preds)
    if not args.keep_checkpoints:
        shutil.rmtree(os.path.join(odir, "ckpt"), ignore_errors=True)
    print(f">>> {layer} fold{k}: span-F1 micro={m['span_micro_f1']:.4f} "
          f"macro={m['span_macro_f1']:.4f}")
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--run_name", default=None, help="default: nome del modello")
    ap.add_argument("--out", default="results")
    ap.add_argument("--layers", default="all", help="'all' oppure lista separata da virgole")
    ap.add_argument("--only_folds", default=None, help="es. '0,1' per lanciare solo alcuni fold")
    ap.add_argument("--nested", default="first", choices=["first", "last"])
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--keep_checkpoints", action="store_true")
    args = ap.parse_args()
    args.run_name = args.run_name or args.model.replace("/", "_")

    layers, docs = load_gold(args.data)
    docs = clean_docs(docs, layers, args.nested)
    folds = load_folds(args.folds)
    if args.only_folds:
        keep = {int(x) for x in args.only_folds.split(",")}
        folds = [f for f in folds if f["fold"] in keep]
    sel = layers if args.layers == "all" else args.layers.split(",")

    mtype = AutoConfig.from_pretrained(args.model).model_type
    tok_kw = {"add_prefix_space": True} if mtype in PREFIX_SPACE_MODELS else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **tok_kw)
    args.max_len = min(args.max_len, tokenizer.model_max_length)

    for layer in sel:
        labels = label_list(docs, layer)  # stesso label set per tutti i fold
        print(f"\n===== {layer}: {len(labels)} etichette =====")
        for fd in folds:
            run_fold(args, layer, fd, docs, labels, tokenizer)


if __name__ == "__main__":
    main()
