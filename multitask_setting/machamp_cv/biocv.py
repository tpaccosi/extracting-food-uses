"""
biocv.py -- shared helpers for the 5-fold cross-validation pipeline.

* reading / writing our BIO TSV files (header "token<TAB>FRAME1<TAB>...",
  blank line between sentences)
* turning a frame column into spans, including cells that stack several
  tags with "|" (nested / overlapping spans, e.g. "I-CURE_Affliction|B-CURE_LU")
* splitting those spans into non-overlapping "layers" (one plain BIO column
  each, so MAChamp's seq_bio CRF sees only valid B-/I-/O labels) and
  composing the layers back into "|" cells after prediction
* iterative (multi-label) stratification of sentences into folds
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

LAYER_SEP = "__L"          # FRAME, FRAME__L1, FRAME__L2, ...
Span = Tuple[int, int, str]  # (start, end inclusive, label type)
Row = List[str]
Sentence = List[Row]


NESTED_FRAMES: set | None = None   # None = tutti nested (comportamento vecchio)


def is_nested(frame: str) -> bool:
    return NESTED_FRAMES is None or frame in NESTED_FRAMES


def first_tag(cell: str) -> str:
    """'I-A|B-B' -> 'I-A'; 'O' / '' / '_' -> 'O'. Prima annotazione della cella."""
    for part in cell.split("|"):
        part = part.strip()
        if part not in ("", "O", "_"):
            return part
    return "O"


# --------------------------------------------------------------------------- IO

def _lines(path: Path) -> List[str]:
    """Splits on '\\n' only. str.splitlines() would also break lines on
    characters that can occur inside a token (\\x0b, \\x0c, \\x1c-\\x1e,
    \\x85, \\u2028, \\u2029), cutting one row into two shorter ones."""
    return [l[:-1] if l.endswith("\r") else l
            for l in Path(path).read_text(encoding="utf-8").split("\n")]


def read_bio_tsv(path: Path, strict: bool = True,
                 n_frames: int | None = None) -> Tuple[List[str], List[Sentence]]:
    """Returns (header, sentences); each sentence is a list of rows (cells).

    n_frames: keep only the token column + the first n_frames frame columns,
      in the header and in every row; anything to the right is dropped
      (non-'O' cells that get dropped are reported, since that loses labels).
    strict=False: rows with too few cells are padded with 'O' at the end and
      reported (as warnings on stderr) instead of stopping."""
    lines = _lines(path)
    if not lines or lines[0].split("\t")[0] != "token":
        raise SystemExit(f"{path}: expected a header row starting with 'token'")
    header = lines[0].split("\t")
    msgs = []
    if n_frames is not None:
        n_cols = n_frames + 1
        if len(header) < n_cols:
            raise SystemExit(f"{path}: header has only {len(header) - 1} frame columns, "
                             f"--n-frames asks for {n_frames}")
        if len(header) > n_cols:
            msgs.append(f"{path}: header columns dropped: {header[n_cols:]}")
        header = header[:n_cols]
    n_cols = len(header)

    sentences: List[Sentence] = []
    cur: Sentence = []
    bad = []                     # rows with too few columns (or too many, without n_frames)
    trimmed = 0                  # rows cut down to n_cols
    lost = []                    # (lineno, dropped non-O cells)
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            if cur:
                sentences.append(cur)
                cur = []
            continue
        cells = line.split("\t")
        if len(cells) > n_cols and n_frames is not None:
            extra = [c for c in cells[n_cols:] if c.strip() not in ("", "O", "_")]
            if extra:
                lost.append((lineno, extra))
            cells = cells[:n_cols]
            trimmed += 1
        if len(cells) != n_cols:
            bad.append((lineno, len(cells), line))
            if len(cells) < n_cols:
                cells = cells + ["O"] * (n_cols - len(cells))
        cur.append(cells)
    if cur:
        sentences.append(cur)

    if trimmed:
        msgs.append(f"{path}: {trimmed} row(s) had more than {n_cols} columns; "
                    f"kept token + first {n_cols - 1} frames.")
    if lost:
        msgs.append(f"  WARNING  {len(lost)} of them had non-O labels in the dropped column(s):")
        for lineno, extra in lost[:10]:
            msgs.append(f"    line {lineno}: {extra}")
        if len(lost) > 10:
            msgs.append(f"    ... and {len(lost) - 10} more")
    if bad:
        too_many = [b for b in bad if b[1] > n_cols]
        msg = [f"{path}: {len(bad)} row(s) do not have {n_cols} tab-separated columns "
               f"(token + {n_cols - 1} frames):"]
        for lineno, n, line in bad[:15]:
            msg.append(f"  line {lineno}: {n} columns: {line!r}")
        if len(bad) > 15:
            msg.append(f"  ... and {len(bad) - 15} more")
        if strict or too_many:
            if too_many:
                msg.append("Rows with extra columns: use --n-frames N to keep token + the first N frames.")
            msg.append("Rows with missing columns: fix them in the file, or rerun with --lenient "
                       "to pad them with 'O'.")
            raise SystemExit("\n".join(msgs + msg))
        msg.append("  -> short rows padded with 'O' in the missing (last) column(s). CHECK they are right.")
        msgs += msg
    if msgs:
        print("\n".join(msgs), file=sys.stderr)
    return header, sentences


def write_bio_tsv(path: Path, header: Sequence[str], sentences: Iterable[Sentence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for sent in sentences:
            for row in sent:
                f.write("\t".join(row) + "\n")
            f.write("\n")


def write_conll(path: Path, sentences: Iterable[Sentence]) -> None:
    """MAChamp column format: no header, blank line between sentences."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sent in sentences:
            for row in sent:
                f.write("\t".join(row) + "\n")
            f.write("\n")


def read_conll(path: Path, n_cols: int) -> List[Sentence]:
    """Reads a MAChamp input/output file. Lines that don't have n_cols
    columns and start with '#' are treated as comments and skipped."""
    sentences: List[Sentence] = []
    cur: Sentence = []
    for lineno, line in enumerate(_lines(path), start=1):
        if not line.strip():
            if cur:
                sentences.append(cur)
                cur = []
            continue
        cells = line.split("\t")
        if len(cells) != n_cols:
            if line.startswith("#"):
                continue
            raise SystemExit(f"{path}:{lineno}: {len(cells)} columns, expected {n_cols}")
        cur.append(cells)
    if cur:
        sentences.append(cur)
    return sentences


# ------------------------------------------------------------------ cell/spans

class ParseStats:
    def __init__(self):
        self.orphan_i = 0      # I-X with nothing to continue -> treated as B-X
        self.bad_tags = Counter()  # tags that are neither O nor B-/I-

    def merge(self, other: "ParseStats") -> None:
        self.orphan_i += other.orphan_i
        self.bad_tags.update(other.bad_tags)


def parse_cell(cell: str, stats: ParseStats | None = None) -> List[Tuple[str, str]]:
    """'I-A|B-B' -> [('I','A'), ('B','B')]; 'O' / '' / '_' -> []."""
    out = []
    for part in cell.split("|"):
        part = part.strip()
        if part in ("", "O", "_"):
            continue
        if len(part) > 2 and part[1] == "-" and part[0] in "BI":
            out.append((part[0], part[2:]))
        elif stats is not None:
            stats.bad_tags[part] += 1
    return out


def spans_from_cells(cells: Sequence[str], stats: ParseStats | None = None) -> List[Span]:
    """Reads one frame column of one sentence (cells may contain '|') and
    returns its spans. An I-X continues an X span that ended on the previous
    token; if several qualify, the one that sat at the same position inside
    the previous cell wins (then the innermost / most recently started)."""
    spans: List[dict] = []
    for t, cell in enumerate(cells):
        used = set()
        for pos, (prefix, typ) in enumerate(parse_cell(cell, stats)):
            if prefix == "I":
                cands = [i for i, s in enumerate(spans)
                         if s["type"] == typ and s["end"] == t - 1 and i not in used]
                if cands:
                    i = min(cands, key=lambda i: (abs(spans[i]["pos"] - pos), -spans[i]["start"]))
                    spans[i]["end"] = t
                    spans[i]["pos"] = pos
                    used.add(i)
                    continue
                if stats is not None:
                    stats.orphan_i += 1
            spans.append({"start": t, "end": t, "type": typ, "pos": pos})
            used.add(len(spans) - 1)
    return sorted((s["start"], s["end"], s["type"]) for s in spans)


def assign_layers(spans: Sequence[Span]) -> List[List[Span]]:
    """Greedy interval colouring: outer (earlier, longer) spans go to layer 0,
    anything overlapping them to layer 1, and so on. Uses the minimum number
    of layers (= maximum nesting/overlap depth)."""
    layers: List[List[Span]] = []
    layer_end: List[int] = []
    for sp in sorted(spans, key=lambda s: (s[0], -s[1], s[2])):
        for li, end in enumerate(layer_end):
            if end < sp[0]:
                layers[li].append(sp)
                layer_end[li] = sp[1]
                break
        else:
            layers.append([sp])
            layer_end.append(sp[1])
    return layers


def spans_to_bio(spans: Sequence[Span], n: int) -> List[str]:
    tags = ["O"] * n
    for start, end, typ in spans:
        tags[start] = f"B-{typ}"
        for t in range(start + 1, end + 1):
            tags[t] = f"I-{typ}"
    return tags


def spans_to_cells(spans: Sequence[Span], n: int) -> List[str]:
    """Inverse of spans_from_cells: outer layer first, joined with '|'."""
    per_token: List[List[str]] = [[] for _ in range(n)]
    for layer in assign_layers(spans):
        for t, tag in enumerate(spans_to_bio(layer, n)):
            if tag != "O":
                per_token[t].append(tag)
    return ["|".join(tags) if tags else "O" for tags in per_token]


# def frame_spans(sentence: Sentence, col: int, stats: ParseStats | None = None) -> List[Span]:
#     return spans_from_cells([row[col] for row in sentence], stats)

def frame_spans(sentence: Sentence, col: int, stats: ParseStats | None = None,
                nested: bool = True) -> List[Span]:
    cells = [row[col] for row in sentence]
    if not nested:
        cells = [first_tag(c) for c in cells]
    return spans_from_cells(cells, stats)

# ------------------------------------------------------------ layered columns

def layered_columns(frames: Sequence[str], depths: Dict[str, int]) -> List[Tuple[str, int, str]]:
    """[(frame, layer, column_name)] in MAChamp column order (token is col 0)."""
    cols = []
    for fr in frames:
        for li in range(max(1, depths.get(fr, 1))):
            cols.append((fr, li, fr if li == 0 else f"{fr}{LAYER_SEP}{li}"))
    return cols


def frame_depths(header: Sequence[str], sentences: Sequence[Sentence]) -> Dict[str, int]:
    depths = {fr: 1 for fr in header[1:]}
    for sent in sentences:
        for c, fr in enumerate(header[1:], start=1):
            # depths[fr] = max(depths[fr], len(assign_layers(frame_spans(sent, c))))
            depths[fr] = max(depths[fr], len(assign_layers(frame_spans(sent, c, nested=is_nested(fr)))))

    return depths


def decompose_sentence(header: Sequence[str], sentence: Sentence,
                       columns: Sequence[Tuple[str, int, str]],
                       stats: ParseStats | None = None) -> Tuple[Sentence, int]:
    """Original rows -> layered rows (token + one plain BIO tag per column).
    Returns (rows, number of spans dropped because the sentence needs more
    layers than the configured depth -- 0 when depths come from this data)."""
    n = len(sentence)
    frame_idx = {fr: i for i, fr in enumerate(header)}
    layer_tags: Dict[Tuple[str, int], List[str]] = {}
    dropped = 0
    depth = Counter(fr for fr, _, _ in columns)
    for fr in depth:
        # layers = assign_layers(frame_spans(sentence, frame_idx[fr], stats))
        layers = assign_layers(frame_spans(sentence, frame_idx[fr], stats, nested=is_nested(fr)))
        for li in range(depth[fr]):
            layer_tags[(fr, li)] = spans_to_bio(layers[li] if li < len(layers) else [], n)
        dropped += sum(len(l) for l in layers[depth[fr]:])
    rows = [[sentence[t][0]] + [layer_tags[(fr, li)][t] for fr, li, _ in columns] for t in range(n)]
    return rows, dropped


def compose_sentence(frames: Sequence[str], layered: Sentence,
                     columns: Sequence[Tuple[str, int, str]]) -> Sentence:
    """Layered rows (e.g. MAChamp predictions) -> original-format rows."""
    n = len(layered)
    by_frame: Dict[str, List[Span]] = {fr: [] for fr in frames}
    for c, (fr, _, _) in enumerate(columns, start=1):
        by_frame[fr].extend(spans_from_cells([row[c] for row in layered]))
    out_cols = {fr: spans_to_cells(sorted(set(sp)), n) for fr, sp in by_frame.items()}
    return [[layered[t][0]] + [out_cols[fr][t] for fr in frames] for t in range(n)]


# -------------------------------------------------------------- stratification

def sentence_label_counts(header: Sequence[str], sentence: Sentence) -> Counter:
    """Counter of (frame, label) -> number of spans (all layers)."""
    c = Counter()
    for col, fr in enumerate(header[1:], start=1):
        # for _, _, typ in frame_spans(sentence, col):
        for _, _, typ in frame_spans(sentence, col, nested=is_nested(fr)):
            c[(fr, typ)] += 1
    return c


def sentence_layer_counts(header: Sequence[str], sentence: Sentence) -> Counter:
    """Counter of layer column name (FRAME__L1, ...) -> spans on that layer."""
    c = Counter()
    for col, fr in enumerate(header[1:], start=1):
        # for li, layer in enumerate(assign_layers(frame_spans(sentence, col))):
        for li, layer in enumerate(assign_layers(frame_spans(sentence, col, nested=is_nested(fr)))):
            if li > 0:
                c[f"{fr}{LAYER_SEP}{li}"] += len(layer)
    return c


def iterative_stratification(items: Sequence[Counter], proportions: Sequence[float],
                             seed: int) -> List[int]:
    """Multi-label iterative stratification (Sechidis et al., 2011), weighted
    by span counts. The rarest label (fewest remaining spans) is placed first,
    each of its sentences going to the group that still 'wants' that label
    the most; ties -> group that wants the most sentences -> random."""
    rng = random.Random(seed)
    n, k = len(items), len(proportions)
    total = Counter()
    for it in items:
        total.update(it)
    want_label = [{l: total[l] * p for l in total} for p in proportions]
    want_size = [n * p for p in proportions]
    assign = [-1] * n
    remaining = set(range(n))
    left = Counter(total)

    def place(i: int, label=None) -> None:
        key = lambda g: ((want_label[g][label] if label is not None else 0.0),
                         want_size[g], rng.random())
        g = max(range(k), key=key)
        assign[i] = g
        remaining.discard(i)
        want_size[g] -= 1
        for l, c in items[i].items():
            want_label[g][l] -= c
            left[l] -= c

    while True:
        live = [l for l in left if left[l] > 0]
        if not live:
            break
        label = min(live, key=lambda l: (left[l], str(l)))
        todo = sorted(i for i in remaining if items[i][label] > 0)
        rng.shuffle(todo)
        for i in todo:
            place(i, label)
    rest = sorted(remaining)
    rng.shuffle(rest)
    for i in rest:
        place(i)
    return assign


# ---------------------------------------------------------------- misc

def load_meta(folds_dir: Path) -> dict:
    return json.loads((Path(folds_dir) / "folds.json").read_text(encoding="utf-8"))


def columns_from_meta(meta: dict) -> List[Tuple[str, int, str]]:
    return [tuple(c) for c in meta["columns"]]


def write_machamp_config(path: Path, fold_dir: Path, dataset_name: str,
                         columns: Sequence[Tuple[str, int, str]]) -> None:
    fold_dir = Path(fold_dir).resolve()
    tasks = {name: {"task_type": "seq_bio", "column_idx": i}
             for i, (_, _, name) in enumerate(columns, start=1)}
    cfg = {dataset_name: {
        "train_data_path": str(fold_dir / "train.conll"),
        "dev_data_path": str(fold_dir / "dev.conll"),
        "word_idx": 0,
        "tasks": tasks,
    }}
    Path(path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def label_name(frame: str, label: str, ambiguous: set) -> str:
    return f"{frame}:{label}" if label in ambiguous else label
