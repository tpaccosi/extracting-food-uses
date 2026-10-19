#!/usr/bin/env python3
"""
run_hf_multiclass.py
====================
Multiclass token classification (ONE label per token) with a HuggingFace
transformer, on the SAME folds as the MAChamp multi-task run and scored by
the SAME evaluator, so the two are directly comparable.

Class = combination of the token's tags in all frames (same definition as
make_multiclass.py), e.g.

    eyes  O O O O O I-CURE_Affliction|B-CURE_Body_Part
      -> CURE=I-CURE_Affliction|B-CURE_Body_Part

Nested "|" cells stay whole inside the class, so no layers are needed.
After prediction the class is split back into one cell per frame and written
as pred.bio.tsv; then machamp_cv/evaluate_cv.py scores it exactly like the
multi-task run (span-level exact match, nested spans included, mean±std and
pooled).

Usage
-----
    python3 run_hf_multiclass.py --folds-dir cv5/ --outdir cv5_hf_mc/ \\
        --machamp-cv machamp_cv/ --model bert-base-multilingual-cased
    python3 machamp_cv/compare_cv.py cv5/ cv5_hf_mc/

Options
    --folds 1 3        only some folds (evaluation runs when all K have pred.bio.tsv)
    --skip-existing    keep folds whose pred.bio.tsv already exists
    --oracle           no training: predicts the gold class whenever it was seen
                       in the fold's train set, else O. Gives the CEILING of the
                       joint-label approach and tests the pipeline without a GPU.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

SENT_END = {".", ";", "!", "?", ":"}
PREFIX_SPACE_MODELS = {"roberta", "longformer", "bart", "gpt2", "mpnet", "deberta"}

# identical to joint_label / split_joint_label in the newer machamp_cv/biocv.py
JOINT_SEP, JOINT_EQ = ";", "="


def _has_tags(cell):
    for part in cell.split("|"):
        part = part.strip()
        if len(part) > 2 and part[1] == "-" and part[0] in "BI":
            return True
    return False


def joint_label_fallback(frames, row):
    parts = [f"{fr}{JOINT_EQ}{cell}" for fr, cell in zip(frames, row[1:]) if _has_tags(cell)]
    return JOINT_SEP.join(parts) if parts else "O"


def split_joint_label_fallback(frames, label):
    cells = {fr: "O" for fr in frames}
    if label not in ("O", ""):
        for part in label.split(JOINT_SEP):
            fr, _, cell = part.partition(JOINT_EQ)
            if fr in cells:
                cells[fr] = cell
    return [cells[fr] for fr in frames]


# ----------------------------------------------------------------------------- data

def load_helpers(machamp_cv: Path):
    """Imports the user's biocv / evaluate_cv so parsing and scoring are the
    very same code used for the multi-task run."""
    p = machamp_cv.resolve()
    if not (p / "biocv.py").exists() or not (p / "evaluate_cv.py").exists():
        raise SystemExit(f"{p}: biocv.py / evaluate_cv.py not found (--machamp-cv)")
    sys.path.insert(0, str(p))
    import biocv  # noqa: E402
    import evaluate_cv  # noqa: E402

    need_b = ["load_meta", "read_bio_tsv", "write_bio_tsv", "frame_spans"]
    need_e = ["fold_counts", "add_micro", "prf", "ALL", "MICRO"]
    miss = [f"biocv.{n}" for n in need_b if not hasattr(biocv, n)] + \
           [f"evaluate_cv.{n}" for n in need_e if not hasattr(evaluate_cv, n)]
    if miss:
        raise SystemExit(f"{p}: this version of machamp_cv lacks {miss}. "
                         f"Use the machamp_cv/ folder that produced your multi-task results.")

    if not all(hasattr(biocv, n) for n in ("joint_label", "split_joint_label", "JOINT_SEP", "JOINT_EQ")):
        print(f"note: {p}/biocv.py has no joint-label helpers (older version); "
              f"using the built-in copy (same definition as make_multiclass.py)")
        biocv.JOINT_SEP, biocv.JOINT_EQ = JOINT_SEP, JOINT_EQ
        biocv.joint_label = joint_label_fallback
        biocv.split_joint_label = split_joint_label_fallback
    return biocv, evaluate_cv


def prepare(args, biocv):
    """Reads cv5/fold_k/{train,dev,test}.bio.tsv, writes outdir/folds.json and
    outdir/fold_k/test.bio.tsv (what evaluate_cv.py needs)."""
    meta = biocv.load_meta(args.folds_dir)
    header, k = meta["header"], meta["k"]
    frames = header[1:]
    if args.folds_dir.resolve() == args.outdir.resolve():
        raise SystemExit("--outdir must differ from --folds-dir")
    args.outdir.mkdir(parents=True, exist_ok=True)

    data, mism = {}, 0
    for f in range(1, k + 1):
        src = args.folds_dir / f"fold_{f}"
        data[f] = {}
        for part in ("train", "dev", "test"):
            h, sents = biocv.read_bio_tsv(src / f"{part}.bio.tsv")
            if h != header:
                raise SystemExit(f"{src}/{part}.bio.tsv: header differs from folds.json")
            for s in sents:
                for row in s:
                    for cell in row[1:]:
                        if biocv.JOINT_SEP in cell or biocv.JOINT_EQ in cell:
                            raise SystemExit(f"cell {cell!r} contains a joint-label separator")
                    if part == "test":   # the K test sets cover every sentence exactly once
                        back = biocv.split_joint_label(frames, biocv.joint_label(frames, row))
                        if back != [c if c.strip() not in ("", "_") else "O" for c in row[1:]]:
                            mism += 1
            data[f][part] = sents
        dst = args.outdir / f"fold_{f}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / "test.bio.tsv", dst / "test.bio.tsv")

    new_meta = dict(meta)
    new_meta.update({"mode": "joint", "task_type": "seq", "columns": [["JOINT", 0, "JOINT"]],
                     "source_folds": str(args.folds_dir.resolve()),
                     "system": "hf_multiclass", "model": args.model})
    (args.outdir / "folds.json").write_text(json.dumps(new_meta), encoding="utf-8")
    print(f"{meta['n_sentences']} sentences, {k} folds from {args.folds_dir}")
    print("round trip token -> class -> cells: " + ("OK" if mism == 0 else
          f"{mism} token(s) with unrecognised tags (e.g. 'B_CURE_LU', 'o') become O -- "
          f"evaluate_cv.py treats them as O in the gold too, so scores are unaffected"))
    return meta, header, data


def joint_seq(biocv, frames, sent):
    return [biocv.joint_label(frames, row) for row in sent]


def to_rows(biocv, frames, sent, joint):
    """Joint labels of one sentence -> rows in the original format."""
    return [[row[0]] + biocv.split_joint_label(frames, lab) for row, lab in zip(sent, joint)]


def span_f1(evaluate_cv, header, gold_sents, pred_rows):
    cnt, _ = evaluate_cv.fold_counts(header, gold_sents, pred_rows)
    tp, fp, fn = evaluate_cv.add_micro(cnt).get((evaluate_cv.ALL, evaluate_cv.MICRO), [0, 0, 0])
    return evaluate_cv.prf(tp, fp, fn)[2]


# ----------------------------------------------------------------------------- model

def chunk_sentence(tokens, tokenizer, max_len):
    """Word spans (start, end) that fit in max_len sub-tokens, cut after
    sentence punctuation when possible (some 'sentences' are whole recipes)."""
    enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False)
    n_sub = [0] * len(tokens)
    for w in enc.word_ids():
        if w is not None:
            n_sub[w] += 1
    budget = max_len - tokenizer.num_special_tokens_to_add()
    chunks, start, used, last_break, i = [], 0, 0, None, 0
    while i < len(tokens):
        if used + n_sub[i] > budget and i > start:
            cut = last_break + 1 if last_break is not None and last_break >= start else i
            chunks.append((start, cut))
            start, used, last_break, i = cut, 0, None, cut
            continue
        used += n_sub[i]
        if tokens[i] in SENT_END:
            last_break = i
        i += 1
    if start < len(tokens):
        chunks.append((start, len(tokens)))
    return chunks


def build_dataset(torch, sents, joints, tokenizer, label2id, max_len):
    items, meta = [], []
    for si, (sent, joint) in enumerate(zip(sents, joints)):
        tokens = [r[0] for r in sent]
        for s, e in chunk_sentence(tokens, tokenizer, max_len):
            enc = tokenizer(tokens[s:e], is_split_into_words=True, truncation=True, max_length=max_len)
            wids = enc.word_ids()
            labs, prev = [], None
            for w in wids:
                if w is None or w == prev:
                    labs.append(-100)                      # label on first sub-token only
                else:
                    labs.append(label2id.get(joint[s + w], -100))  # class unseen in train: ignored
                prev = w
            item = dict(enc)
            item["labels"] = labs
            items.append(item)
            meta.append((si, s, wids))

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(items)

        def __getitem__(self, i):
            return items[i]

    ds = DS()
    ds.meta = meta
    return ds


def decode(ds, logits, sents, id2label):
    out = [["O"] * len(s) for s in sents]
    ids = logits.argmax(-1)
    for row, (si, s, wids) in enumerate(ds.meta):
        prev = None
        for pos, w in enumerate(wids):
            if w is not None and w != prev:
                out[si][s + w] = id2label[int(ids[row, pos])]
            prev = w
    return out


def train_and_predict(args, header, fold, labels, biocv, evaluate_cv, odir):
    import torch
    from transformers import (AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, EarlyStoppingCallback,
                              Trainer, TrainingArguments, set_seed)
    frames = header[1:]
    set_seed(args.seed)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    mtype = AutoConfig.from_pretrained(args.model).model_type
    kw = {"add_prefix_space": True} if mtype in PREFIX_SPACE_MODELS else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, **kw)
    max_len = min(args.max_len, tokenizer.model_max_length)

    js = {p: [joint_seq(biocv, frames, s) for s in fold[p]] for p in fold}
    ds = {p: build_dataset(torch, fold[p], js[p], tokenizer, label2id, max_len) for p in fold}

    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id)

    def compute_metrics(ev):
        pred = decode(ds["dev"], ev.predictions, fold["dev"], id2label)
        rows = [to_rows(biocv, frames, s, p) for s, p in zip(fold["dev"], pred)]
        return {"span_f1": span_f1(evaluate_cv, header, fold["dev"], rows)}

    # warmup = 10% of all optimisation steps, as an integer: works in every
    # transformers version (v5 removed warmup_ratio)
    steps_per_epoch = math.ceil(len(ds["train"]) / args.batch_size)
    warmup_steps = math.ceil(0.1 * steps_per_epoch * args.epochs)
    targs = dict(
        output_dir=str(odir / "ckpt"), learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs, weight_decay=0.01, warmup_steps=warmup_steps,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="span_f1", greater_is_better=True,
        logging_steps=50, fp16=torch.cuda.is_available() and args.fp16,
        report_to="none", seed=args.seed)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params:          # transformers < 4.41
        targs["evaluation_strategy"] = targs.pop("eval_strategy")
    essential = {"output_dir", "learning_rate", "num_train_epochs", "evaluation_strategy",
                 "eval_strategy", "save_strategy", "load_best_model_at_end", "metric_for_best_model"}
    for key in list(targs):
        if key not in params:
            if key in essential:
                raise SystemExit(f"TrainingArguments of this transformers version has no '{key}'")
            print(f"note: TrainingArguments has no '{key}' in this transformers version, skipped")
            targs.pop(key)
    trainer = Trainer(model=model, args=TrainingArguments(**targs),
                      train_dataset=ds["train"], eval_dataset=ds["dev"],
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])
    trainer.train()
    logits = trainer.predict(ds["test"]).predictions
    pred = decode(ds["test"], logits, fold["test"], id2label)
    best = trainer.state.best_metric
    if not args.keep_checkpoints:
        shutil.rmtree(odir / "ckpt", ignore_errors=True)
    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, best


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds-dir", required=True, type=Path, help="output of machamp_cv/make_folds.py (cv5/)")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--machamp-cv", type=Path, default=Path("machamp_cv"),
                    help="folder with biocv.py and evaluate_cv.py (default: ./machamp_cv)")
    ap.add_argument("--model", default="bert-base-multilingual-cased",
                    help="use the same encoder as MAChamp's params.json for a fair comparison")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="ceiling of the joint-label approach, no training")
    args = ap.parse_args()

    biocv, evaluate_cv = load_helpers(args.machamp_cv)
    meta, header, data = prepare(args, biocv)
    frames = header[1:]
    folds = args.folds or list(range(1, meta["k"] + 1))

    print("\nper fold: classes in train | test tokens whose class never occurs in train")
    for f in folds:
        fold = data[f]
        train_cls = Counter(l for s in fold["train"] for l in joint_seq(biocv, frames, s))
        test_j = [l for s in fold["test"] for l in joint_seq(biocv, frames, s)]
        unseen = sum(1 for l in test_j if l not in train_cls)
        labels = ["O"] + sorted(l for l in train_cls if l != "O")
        odir = args.outdir / f"fold_{f}"
        pred_path = odir / "pred.bio.tsv"
        print(f"=== fold {f}: {len(labels)} classes | unseen {unseen}/{len(test_j)} "
              f"({100 * unseen / max(len(test_j), 1):.1f}%)", flush=True)
        if args.skip_existing and pred_path.exists():
            print("  pred.bio.tsv exists, skipped")
            continue

        if args.oracle:
            seen = set(labels)
            pred = [[l if l in seen else "O" for l in joint_seq(biocv, frames, s)] for s in fold["test"]]
            best = None
        else:
            pred, best = train_and_predict(args, header, fold, labels, biocv, evaluate_cv, odir)

        rows = [to_rows(biocv, frames, s, p) for s, p in zip(fold["test"], pred)]
        biocv.write_bio_tsv(pred_path, header, rows)
        f1 = span_f1(evaluate_cv, header, fold["test"], rows)
        (odir / "run.json").write_text(json.dumps({
            "fold": f, "n_classes": len(labels), "test_tokens_unseen_class": unseen,
            "best_dev_span_f1": best, "test_span_micro_f1": f1,
            "config": {k: str(v) for k, v in vars(args).items()}}, indent=2), encoding="utf-8")
        print(f"  test span micro-F1 (all frames): {100 * f1:.1f}   -> {pred_path}", flush=True)

    missing = [f for f in range(1, meta["k"] + 1) if not (args.outdir / f"fold_{f}" / "pred.bio.tsv").exists()]
    if missing:
        print(f"\nfolds without predictions yet: {missing} -- run them, then evaluate_cv.py")
        return
    print("\n=== evaluation (machamp_cv/evaluate_cv.py)")
    subprocess.run([sys.executable, str(args.machamp_cv.resolve() / "evaluate_cv.py"),
                    "--folds-dir", str(args.outdir)], check=True)
    print(f"\nnext: python3 {args.machamp_cv}/compare_cv.py {args.folds_dir} {args.outdir}")


if __name__ == "__main__":
    main()