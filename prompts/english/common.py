"""
common.py
=========
Shared code between call_llms.py (stage 1: query APIs, save raw JSON) and
json_to_bio.py (stage 2: convert raw JSON to aligned BIO TSV).

Schema model
------------
This is a DOUBLE ANNOTATION scheme: each frame in config.yaml (FOOD,
PRESERVING, ...) gets its own output column and its own independent set of
BIO tags. A token can carry a tag in every column at once -- e.g. a food
entity keeps FOOD_LU in the FOOD column AND gets PR_Food_Patient in the
PRESERVING column if it's also a preserving-frame argument. Columns are
never overwritten/relabeled; they are filled in parallel.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

TOKEN_RE = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")
_NAIVE_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'\u201c])')


def get_api_key(env_var: str) -> str:
    """Reads an API key from the environment and strips surrounding
    whitespace/newlines before use. Guards against a common shell mistake
    (a stray newline pasted into `export KEY="..."`) that otherwise
    surfaces as a confusing low-level HTTP error ("Illegal header value")
    deep in the networking stack instead of a clear message here.
    """
    value = os.environ.get(env_var)
    if value is None:
        raise RuntimeError(f"{env_var} is not set. Run: export {env_var}=\"...\"")
    stripped = value.strip()
    if not stripped:
        raise RuntimeError(f"{env_var} is set but empty after stripping whitespace.")
    if stripped != value:
        import sys
        print(f"NOTE: {env_var} had leading/trailing whitespace or a newline -- stripped it.",
              file=sys.stderr)
    return stripped


# Models known (as of the Anthropic SDK v1.0 / Aug 2026 migration) to hard-reject
# ANY non-default temperature/top_p/top_k at the API level, regardless of how it's
# sent (typed kwarg or extra_body) -- for these, skip straight to a bare default
# call instead of wasting a network round-trip (and its billed input tokens) on a
# request we already know will fail. Match by substring since model IDs carry
# dated suffixes (e.g. "claude-haiku-4-5-20251001") that would break exact match.
ANTHROPIC_NO_SAMPLING_MODEL_SUBSTRINGS = (
    "claude-sonnet-5", "claude-opus-5", "claude-fable-5", "claude-mythos-5",
    "claude-mythos-preview", "claude-opus-4-8", "claude-opus-4-7",
)


def anthropic_supports_custom_temperature(model_name: str) -> bool:
    name = model_name.lower()
    return not any(s in name for s in ANTHROPIC_NO_SAMPLING_MODEL_SUBSTRINGS)


# Same idea for OpenAI: "reasoning" models (o1/o3/o4-style) and some current
# gpt-5.x models hard-reject any non-default temperature with an
# "unsupported_value" 400 error. Unlike the Anthropic list (sourced from
# official docs), this one is inferred from observed errors -- treat it as
# best-effort and expect to extend it as new rejecting models turn up.
OPENAI_NO_SAMPLING_MODEL_SUBSTRINGS = ("gpt-5.5", "gpt-5.6")
OPENAI_NO_SAMPLING_MODEL_PREFIXES = ("o1", "o3", "o4")  # reasoning-model family


def openai_supports_custom_temperature(model_name: str) -> bool:
    name = model_name.lower()
    if name.startswith(OPENAI_NO_SAMPLING_MODEL_PREFIXES):
        return False
    return not any(s in name for s in OPENAI_NO_SAMPLING_MODEL_SUBSTRINGS)


def build_cached_system(system_prompt: str, ttl: str = "1h") -> list:
    """Wraps a system prompt string into Anthropic's prompt-caching content-block
    format. The system prompt is identical across every call within one
    frame/stage, so caching it turns a $10/MTok base-rate cost into a
    $0.25/MTok cache-hit cost (Fable 5.1/Mythos 5.1 rate; 0.1x base for other
    models) on every call after the first. ttl="1h" survives the full
    duration of a batch run; "5m" is cheaper to write but expires between
    calls in a slow synchronous loop. Note: caching only activates once the
    cached block is at least ~1024 tokens for some models -- below that it's
    silently a no-op (no error, just no discount), so it's always safe to
    leave this on even for a smaller system prompt.
    """
    return [{"type": "text", "text": system_prompt,
             "cache_control": {"type": "ephemeral", "ttl": ttl}}]


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def split_sentences(text: str) -> List[str]:
    """Try nltk punkt if available, else fall back to a naive regex splitter.
    For historical text with irregular punctuation, this is the weakest
    link -- swap in a domain-tuned splitter if needed.
    """
    text = text.strip()
    if not text:
        return []
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = _NAIVE_SENT_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]


@dataclass
class FrameDef:
    name: str                    # e.g. "FOOD", "PRESERVING"
    labels: List[str]            # all base label names for this frame's column
    lu_label: str | None         # the label marking this frame's triggering LU, or
                                  # None for attribute-only frames with no separate trigger
    definitions: Dict[str, str]  # label -> definition text
    description: str = ""        # short prose description of the frame
    note: str = ""                # optional extra guidance (special cases, aggregated frames, etc.)

    @property
    def json_key(self) -> str:
        return f"{self.name.lower()}_tags"

    def bio_vocab(self) -> set:
        vocab = {"O"}
        for lab in self.labels:
            vocab.add(f"B-{lab}")
            vocab.add(f"I-{lab}")
        return vocab


@dataclass
class Schema:
    frames: Dict[str, FrameDef] = field(default_factory=dict)
    few_shot: List[Dict[str, Any]] = field(default_factory=list)
    general_rules: str = ""

    @classmethod
    def from_config(cls, cfg: dict) -> "Schema":
        frames: Dict[str, FrameDef] = {}
        for frame_name, frame_cfg in cfg["frames"].items():
            entries = frame_cfg["labels"]
            labels = [e["name"] for e in entries]
            definitions = {e["name"]: e.get("definition", "").strip() for e in entries}
            has_lu = frame_cfg.get("has_lu", True)
            if has_lu:
                lu_candidates = [e["name"] for e in entries if e.get("is_lu")]
                lu_label = lu_candidates[0] if lu_candidates else labels[0]
            else:
                lu_label = None
            description = (frame_cfg.get("description") or "").strip()
            note = (frame_cfg.get("note") or "").strip()
            frames[frame_name] = FrameDef(
                name=frame_name, labels=labels, lu_label=lu_label,
                definitions=definitions, description=description, note=note,
            )
        return cls(
            frames=frames,
            few_shot=cfg.get("few_shot_examples", []),
            general_rules=(cfg.get("general_rules") or "").strip(),
        )

    def frame_names(self) -> List[str]:
        return list(self.frames.keys())

    def json_keys(self) -> List[str]:
        return [f.json_key for f in self.frames.values()]

    def subset(self, frame_names: List[str]) -> "Schema":
        """Returns a new Schema restricted to the given frames (same
        FrameDef objects, same few_shot/general_rules) -- used to build a
        smaller system prompt for one call when frames are split into
        groups instead of annotated all at once.
        """
        missing = [f for f in frame_names if f not in self.frames]
        if missing:
            raise KeyError(f"subset() requested unknown frame(s): {missing}")
        return Schema(
            frames={f: self.frames[f] for f in frame_names},
            few_shot=self.few_shot,
            general_rules=self.general_rules,
        )


def _label_line(name: str, definitions: Dict[str, str]) -> str:
    d = definitions.get(name, "")
    return f"  - {name}: {d}" if d else f"  - {name}"


def build_system_prompt(schema: Schema) -> str:
    frame_blocks = []
    for frame in schema.frames.values():
        desc_line = f"{frame.description}\n" if frame.description else ""
        note_line = f"NOTE: {frame.note}\n" if frame.note else ""
        if frame.lu_label is not None:
            fe_labels = [l for l in frame.labels if l != frame.lu_label]
            lu_def = _label_line(frame.lu_label, frame.definitions).strip()
            lu_block = (
                f"Lexical unit label (the word/expression that triggers this frame):\n"
                f"  {lu_def}\n"
            )
        else:
            fe_labels = list(frame.labels)
            lu_block = (
                "This frame has NO separate lexical-unit trigger word: annotate its\n"
                "labels directly on the qualifying span wherever it occurs.\n"
            )
        fe_def_lines = "\n".join(_label_line(l, frame.definitions) for l in fe_labels)
        frame_blocks.append(
            f'FRAME "{frame.name}" (output key: "{frame.json_key}")\n'
            f"{desc_line}"
            f"{note_line}"
            f"{lu_block}"
            f"Frame element (FE) labels (arguments of the frame):\n"
            f"{fe_def_lines}"
        )
    frames_block = "\n\n".join(frame_blocks)

    food_group_note = ""
    if "FOOD" not in schema.frames:
        food_group_note = (
            "\nNOTE: the FOOD frame is not part of this call. Where a frame element "
            "definition below says to \"re-annotate what is already FOOD_LU in the FOOD "
            "column\", that column isn't present here -- just identify the relevant food "
            "entity directly for the frame(s) below, using the same span rule (no "
            "determiner) as FOOD_LU normally would.\n"
        )

    json_keys = ", ".join(f'"{k}"' for k in schema.json_keys())
    all_vocab_lines = []
    for frame in schema.frames.values():
        all_vocab_lines.append(f'{frame.json_key} base labels: {", ".join(frame.labels)}')
    vocab_block = "\n".join(all_vocab_lines)

    examples_txt = []
    for ex in schema.few_shot:
        output_obj = {frame.json_key: ex[frame.json_key] for frame in schema.frames.values()}
        examples_txt.append(
            "Input tokens: " + json.dumps(ex["tokens"]) + "\n"
            "Output: " + json.dumps(output_obj)
        )
    examples_block = "\n\n".join(examples_txt)

    general_rules_block = schema.general_rules if schema.general_rules else ""

    food_frame = schema.frames.get("FOOD")
    attribute_only_frames = [f for f in schema.frames.values()
                              if f.lu_label is None and f.name != "FOOD"]
    lu_frames = [f for f in schema.frames.values()
                 if f.lu_label is not None and f.name != "FOOD"]

    step_lines = []
    if food_frame is not None:
        step_lines.append(
            "STEP A -- FOOD ENTITY GATE\n"
            "For every candidate entity in the sentence, decide whether it is a food "
            "entity in this context (not a metaphor, not out of scope). For each one "
            "that is, tag it FOOD_LU in the FOOD sequence. Do this pass first, before "
            "anything else -- every later step re-uses the food entities you settle on "
            "here."
        )
        other_food_fe = [l for l in food_frame.labels if l != food_frame.lu_label]
        attr_names = ", ".join(f.name for f in attribute_only_frames) if attribute_only_frames else ""
        attr_clause = (
            f" Also handle the attribute-only frame(s) here, since they describe these "
            f"same food entities and have no trigger word of their own: {attr_names}."
        ) if attribute_only_frames else ""
        step_lines.append(
            "STEP B -- OTHER FOOD-ENTITY ATTRIBUTES\n"
            f"For the food entities you just tagged in Step A, annotate any of their "
            f"other FOOD frame elements that are present ({', '.join(other_food_fe)})."
            f"{attr_clause} Nothing here changes a FOOD_LU tag -- you are only adding "
            f"attributes alongside it."
        )
    else:
        step_lines.append(
            "STEP A/B -- NOT APPLICABLE THIS CALL\n"
            "The FOOD frame isn't part of this call, so there's no separate food-entity "
            "gate to run first. Identify the relevant food entities directly within "
            "Steps C/D below, using the same span rule (no determiner) FOOD_LU would "
            "normally use."
        )

    if lu_frames:
        lu_frame_names = ", ".join(f.name for f in lu_frames)
        step_lines.append(
            f"STEP C -- OTHER FRAMES' LEXICAL UNITS\n"
            f"For every frame besides FOOD that has its own trigger word ({lu_frame_names}): "
            f"scan the sentence for that frame's lexical unit (the word/expression that "
            f"evokes it) and tag it with that frame's LU label in that frame's own "
            f"sequence. Do this for all such frames before moving to Step D -- first "
            f"settle which frame instances actually exist in the sentence, then fill "
            f"their arguments."
        )
        step_lines.append(
            "STEP D -- REMAINING FRAME ELEMENTS\n"
            "For each frame instance you found an LU for in Step C, identify its other "
            "frame elements (e.g. who is the Agent, what is the Patient, what is the "
            "Medium) and tag them in THAT frame's own sequence -- independently of "
            "whatever those same entities are already tagged as in the FOOD sequence "
            "or any other frame's sequence. This is where the double-annotation "
            "described above actually happens: a food entity tagged FOOD_LU in Step A "
            "gets its ADDITIONAL tag here, in the relevant other frame's sequence, "
            "without losing the FOOD_LU tag."
        )

    step_lines.append(
        "An entity not involved in a given frame is \"O\" in that frame's sequence, "
        "regardless of what it is tagged as in other sequences."
    )
    steps_block = "\n\n".join(step_lines)

    return f"""You are an expert linguistic annotator performing frame-semantic
annotation of food-related mentions in (possibly historical) English text.

TASK
You will receive a sentence already split into tokens, given as a numbered
JSON list. For EACH frame listed below you must produce one BIO-tagged
sequence, with EXACTLY one tag per input token, in the same order.

THIS IS A DOUBLE-ANNOTATION SCHEME, NOT A RELABELING SCHEME.
Each frame's output sequence is independent. If a token is the FOOD_LU of
the FOOD frame, it KEEPS that tag in the FOOD sequence. If that same food
entity is ALSO an argument of another frame (e.g. it is being preserved),
it gets an ADDITIONAL tag of its own in that frame's sequence (e.g.
PR_Food_Patient in the PRESERVING sequence) -- you never remove or replace
the FOOD_LU tag to make room for the other frame's tag. A single token can
therefore be non-"O" in more than one output sequence at once.

STEP-BY-STEP LOGIC -- work through these IN ORDER, A then B then C then D.
Do not fill in a frame element before you've committed to the LU that
frame instance belongs to; do not skip straight to guessing frame elements
without first doing the entity-gate and LU-identification passes.

{steps_block}

MULTI-WORD / MODIFIED LEXICAL UNITS:
A lexical unit can include its own adverbial modifier as a single span,
e.g. "[season lightly]" as one COOKING_CREATION_LU, "[chop finely]" as
one COOKING_CREATION_LU.

NESTED SPANS WITHIN THE SAME FRAME:
Each frame has exactly ONE output column, but a token MAY carry more than
one of that frame's own labels at once if two of its labels genuinely
apply to overlapping/nested spans (e.g. a Material entity nested inside
a larger Product compound, or an Affliction nested inside a larger
Patient phrase). In that case, join the applicable tags for that token
with a pipe "|", each one following normal BIO form, e.g.
"B-INGR_Food_Product|B-INGR_Material_LU". Most tokens will still have
exactly one tag (or "O") -- only use "|" when a genuine nested overlap
occurs within that SAME frame, as described in each frame's notes below.

{general_rules_block}

FRAMES
{food_group_note}{frames_block}

ALLOWED LABELS PER SEQUENCE
{vocab_block}
Every tag is either "O" or "B-<label>"/"I-<label>" using only the labels
listed for that sequence above.

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences, with these
keys: {json_keys}. Every array must have exactly the same length as the
input token list.

EXAMPLES
{examples_block}
"""


def build_user_prompt(tokens: List[str]) -> str:
    numbered = [f"{i}: {t}" for i, t in enumerate(tokens)]
    return (
        "Tokens (index: token):\n" + "\n".join(numbered) +
        f"\n\nInput tokens as JSON list: {json.dumps(tokens)}\n"
        "Return the JSON object now."
    )
