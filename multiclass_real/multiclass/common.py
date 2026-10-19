"""Utility condivise: caricamento del TSV, pulizia delle etichette, fold, metriche."""
import json
import random
import re
from collections import Counter, defaultdict

import numpy as np

# ----------------------------------------------------------------------------
# Caricamento
# ----------------------------------------------------------------------------

def load_gold(path):
    """Legge il TSV in stile CoNLL. Documenti separati da righe vuote.
    Ritorna (layers, docs) dove docs = [{"tokens": [...], "labels": {layer: [...]}}]."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    header = lines[0].rstrip("\r").split("\t")
    layers = header[1:]
    docs, cur = [], []
    for line in lines[1:]:
        line = line.rstrip("\r")
        if not line.strip():
            if cur:
                docs.append(cur)
                cur = []
            continue
        parts = line.split("\t")
        parts += [""] * (len(header) - len(parts))
        cur.append(parts)
    if cur:
        docs.append(cur)

    out = []
    for d in docs:
        out.append({
            "tokens": [r[0] for r in d],
            "labels": {lay: [r[i + 1] for r in d] for i, lay in enumerate(layers)},
        })
    return layers, out


# ----------------------------------------------------------------------------
# Pulizia etichette
# ----------------------------------------------------------------------------

_NOISE_O = {"", "o", "os", "ot", "bo", "0"}


def clean_label(raw, nested="first", log=None):
    """Normalizza una singola etichetta.
    nested: 'first' = tiene la prima etichetta di 'A|B' (span esterno),
            'last'  = tiene l'ultima (span più interno)."""
    lab = raw.strip()
    orig = lab
    if "|" in lab:
        parts = [p for p in lab.split("|") if p.strip()]
        lab = parts[0] if nested == "first" else parts[-1]
    lab = lab.strip()
    if lab.lower() in _NOISE_O or lab.upper().startswith("O-"):
        lab = "O"
    else:
        m = re.match(r"^([BIVbiv])[-_](.+)$", lab)
        if m:
            prefix = m.group(1).upper()
            prefix = "B" if prefix == "V" else prefix  # 'V' è un refuso per 'B' (tasto adiacente)
            lab = f"{prefix}-{m.group(2)}"
        elif lab != "O":
            lab = "O"  # formato irriconoscibile
    if log is not None and lab != orig:
        log[(orig, lab)] += 1
    return lab


def repair_bio(labels):
    """I-X senza un B-X/I-X precedente diventa B-X."""
    out, prev = [], "O"
    for lab in labels:
        if lab.startswith("I-"):
            typ = lab[2:]
            if prev == "O" or prev[2:] != typ:
                lab = "B-" + typ
        out.append(lab)
        prev = lab
    return out


def clean_docs(docs, layers, nested="first", verbose=True):
    log = Counter()
    for d in docs:
        for lay in layers:
            labs = [clean_label(x, nested, log) for x in d["labels"][lay]]
            fixed = repair_bio(labs)
            log[("BIO-repair", "")] += sum(a != b for a, b in zip(labs, fixed))
            d["labels"][lay] = fixed
    if verbose:
        n_nested = sum(v for (o, _), v in log.items() if "|" in o)
        n_other = sum(v for (o, n), v in log.items() if "|" not in o and o != "BIO-repair")
        print(f"[pulizia] etichette annidate risolte ({nested}): {n_nested} | "
              f"refusi corretti: {n_other} | riparazioni BIO: {log[('BIO-repair', '')]}")
    return docs


def label_list(docs, layer):
    types = sorted({l[2:] for d in docs for l in d["labels"][layer] if l != "O"})
    return ["O"] + [f"{p}-{t}" for t in types for p in ("B", "I")]


# ----------------------------------------------------------------------------
# Fold (a livello di documento, per evitare leakage)
# ----------------------------------------------------------------------------

def make_folds(n_docs, k=5, seed=42, dev_ratio=0.1):
    """Ritorna una lista di k dict {train, dev, test} con indici di documenti.
    Il dev è estratto dal train e serve solo per early stopping / scelta epoca."""
    idx = list(range(n_docs))
    random.Random(seed).shuffle(idx)
    chunks = [idx[i::k] for i in range(k)]
    folds = []
    for i in range(k):
        test = sorted(chunks[i])
        rest = [j for c in range(k) if c != i for j in chunks[c]]
        random.Random(seed + i).shuffle(rest)
        n_dev = max(1, int(len(rest) * dev_ratio))
        folds.append({"fold": i, "train": sorted(rest[n_dev:]),
                      "dev": sorted(rest[:n_dev]), "test": test})
    return folds


def save_folds(folds, path):
    with open(path, "w") as f:
        json.dump(folds, f)


def load_folds(path):
    with open(path) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Metriche (span-level stile CoNLL, match esatto) + token-level
# ----------------------------------------------------------------------------

def get_spans(labels):
    spans, start, typ = set(), None, None
    for i, lab in enumerate(list(labels) + ["O"]):
        if lab == "O" or lab.startswith("B-") or (lab.startswith("I-") and lab[2:] != typ):
            if typ is not None:
                spans.add((start, i, typ))
            start, typ = (i, lab[2:]) if lab != "O" else (None, None)
    return spans


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(gold_seqs, pred_seqs):
    """gold_seqs, pred_seqs: liste di liste di etichette BIO (una per documento)."""
    cnt = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn per tipo
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = get_spans(g), get_spans(p)
        for s in gs & ps:
            cnt[s[2]][0] += 1
        for s in ps - gs:
            cnt[s[2]][1] += 1
        for s in gs - ps:
            cnt[s[2]][2] += 1

    per_class = {}
    for t, (tp, fp, fn) in sorted(cnt.items()):
        p, r, f = _prf(tp, fp, fn)
        per_class[t] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
    TP = sum(v[0] for v in cnt.values())
    FP = sum(v[1] for v in cnt.values())
    FN = sum(v[2] for v in cnt.values())
    mp, mr, mf = _prf(TP, FP, FN)
    supported = [v for v in per_class.values() if v["support"] > 0]
    macro_f1 = float(np.mean([v["f1"] for v in supported])) if supported else 0.0

    # token-level: macro-F1 sulle classi non-O (tag completi B-/I-)
    flat_g = [x for s in gold_seqs for x in s]
    flat_p = [x for s in pred_seqs for x in s]
    tags = sorted(set(flat_g) - {"O"})
    tok_f1s = []
    for t in tags:
        tp = sum(1 for a, b in zip(flat_g, flat_p) if a == t and b == t)
        fp = sum(1 for a, b in zip(flat_g, flat_p) if a != t and b == t)
        fn = sum(1 for a, b in zip(flat_g, flat_p) if a == t and b != t)
        tok_f1s.append(_prf(tp, fp, fn)[2])
    acc = float(np.mean([a == b for a, b in zip(flat_g, flat_p)])) if flat_g else 0.0

    return {
        "span_micro_precision": mp, "span_micro_recall": mr, "span_micro_f1": mf,
        "span_macro_f1": macro_f1,
        "token_macro_f1": float(np.mean(tok_f1s)) if tok_f1s else 0.0,
        "token_accuracy": acc,
        "per_class": per_class,
    }


def write_predictions(path, docs_tokens, gold, pred):
    with open(path, "w", encoding="utf-8") as f:
        f.write("token\tgold\tpred\n")
        for toks, g, p in zip(docs_tokens, gold, pred):
            for t, a, b in zip(toks, g, p):
                f.write(f"{t}\t{a}\t{b}\n")
            f.write("\n")
