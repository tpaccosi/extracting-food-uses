#!/usr/bin/env python3
"""
stage1_food_entity.py
======================
STAGE 1 of the two-stage FOOD_ENTITY -> PRESERVING pipeline. Ignores every
other frame. A clean, dedicated (non-generic) prompt: identifies FOOD_LU
spans only, nothing else.

Usage
-----
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...
    python3 stage1_food_entity.py --input input.txt --config config.yaml --outdir stage1_out/

Output (per model)
-------------------
    stage1_out/<input_stem>.<model>.raw.jsonl   -- raw model responses
    stage1_out/<input_stem>.<model>.food_entity.bio.tsv  -- final BIO TSV,
        one column "FOOD_ENTITY", ready for eval_bio.py against a gold
        file with a matching column (see derive_gold_subset.py).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List

import yaml

from common import tokenize, split_sentences, get_api_key, anthropic_supports_custom_temperature, build_cached_system, openai_supports_custom_temperature

SYSTEM_PROMPT = """You are an expert linguistic annotator identifying food entities in
(possibly historical) English text.

TASK
You will receive a sentence already split into tokens, given as a
numbered JSON list. Tag every FOOD_LU span with a BIO-tagged sequence,
"food_entity_tags", with EXACTLY one tag per input token, in the same
order as the input.

WHAT COUNTS AS FOOD_LU
Word(s) referring to a food item -- any exemplar of food, per an extended
definition: not restricted to consumption, includes non-ingestive uses
(e.g. a food substance applied externally, such as on the skin). Annotate
the food entity without its determiner, unless the determiner is
indispensable to define the entity.

Do NOT tag:
  - A word used metaphorically or figuratively (e.g. "he's a real peach"
    -- describing a person, not the fruit; "that's the cream of the
    crop" -- meaning "the best", not dairy).
  - A word that is not a food reference in this context at all, even if
    it can refer to food in other contexts.

CRITICAL -- FOOD_LU IS THE BARE ENTITY ONLY, NEVER ITS DESCRIPTORS:
Tag ONLY the head noun (or fixed noun compound) naming the food item
itself. Do NOT include any descriptive adjective, quality, intensity, or
type-modifier in the FOOD_LU span, no matter how closely it sits next to
the food word -- those belong to a separate frame (DESCRIPTOR), entirely
out of scope for this task. This applies to EVERY adjective describing
the food, not just obviously evaluative ones: "fresh apples" -> tag only
"apples"; "kippered herring" -> tag only "herring"; "really delicious
fish" -> tag only "fish". The adjective is simply left as "O" here (it is
not your job to tag it as anything in this task).
The one exception is the material-to-product compound rule below, which
is NOT a descriptor relationship (it's naming what something is made of/
from, not describing a quality of it).

For a material-to-product compound like "vegetable oil", tag BOTH
"vegetable" and "vegetable oil" as FOOD_LU (this frame doesn't
distinguish material from product -- that distinction is out of scope
here).

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"food_entity_tags": [...]}
The array must have exactly the same length as the input token list.
Every tag is "O", "B-FOOD_LU", or "I-FOOD_LU".

EXAMPLES

Input tokens: ["I", "pickle", "my", "cucumbers", "in", "vinegar", "."]
Output: {"food_entity_tags": ["O", "O", "O", "B-FOOD_LU", "O", "B-FOOD_LU", "O"]}

Input tokens: ["My", "mother", "smokes", "two", "pounds", "of", "salmon", "in", "the", "old", "smokehouse", "."]
Output: {"food_entity_tags": ["O", "O", "O", "O", "O", "O", "B-FOOD_LU", "O", "O", "O", "O", "O"]}

Input tokens: ["We", "enjoyed", "planked", "fish", "."]
Output: {"food_entity_tags": ["O", "O", "O", "B-FOOD_LU", "O"]}

Input tokens: ["Add", "the", "boiling", "water", "to", "the", "beaten", "egg", "."]
Output: {"food_entity_tags": ["O", "O", "O", "B-FOOD_LU", "O", "O", "O", "B-FOOD_LU", "O"]}

Input tokens: ["To", "make", "orange", "cake", "take", "two", "oranges", ",", "butter", "and", "sugar", "."]
Output: {"food_entity_tags": ["O", "O", "B-FOOD_LU", "I-FOOD_LU", "O", "O", "B-FOOD_LU", "O", "B-FOOD_LU", "O", "B-FOOD_LU", "O"]}

Input tokens: ["All", "sufferers", "from", "the", "gout", "should", "eat", "apples", "."]
Output: {"food_entity_tags": ["O", "O", "O", "O", "O", "O", "O", "B-FOOD_LU", "O"]}

Input tokens: ["We", "bought", "fresh", "apples", "and", "kippered", "herring", "."]
Output: {"food_entity_tags": ["O", "O", "O", "B-FOOD_LU", "O", "O", "B-FOOD_LU", "O"]}
NOTE: "fresh" and "kippered" are descriptors, NOT part of the FOOD_LU span
-- they are left "O" here (DESCRIPTOR is a separate frame, out of scope
for this task). Only the bare food nouns "apples" and "herring" are tagged.
"""

VALID_TAGS = {"O", "B-FOOD_LU", "I-FOOD_LU"}


def build_user_prompt(tokens: List[str]) -> str:
    numbered = [f"{i}: {t}" for i, t in enumerate(tokens)]
    return (
        "Tokens (index: token):\n" + "\n".join(numbered) +
        f"\n\nInput tokens as JSON list: {json.dumps(tokens)}\n"
        "Return the JSON object now."
    )


_warned_no_temperature_support: set = set()


def call_anthropic(system_prompt: str, user_prompt: str, model_cfg: dict) -> str:
    """See call_llms.py's call_anthropic for the full explanation: Anthropic SDK
    v1.0 dropped the typed `temperature` kwarg, and several current models
    (Opus 5, Sonnet 5, Fable 5, Mythos 5/Preview, Opus 4.7/4.8) hard-reject any
    non-default sampling value at the API level regardless of transport. Tries
    the old kwarg, then extra_body, then falls back to the model's default,
    warning once per model if it has to go that far.
    """
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))
    model_name = model_cfg["name"]
    temperature = model_cfg.get("temperature", 0)
    base_kwargs = dict(
        model=model_name, max_tokens=1000,
        system=build_cached_system(system_prompt), messages=[{"role": "user", "content": user_prompt}],
    )

    if not anthropic_supports_custom_temperature(model_name):
        if model_name not in _warned_no_temperature_support:
            _warned_no_temperature_support.add(model_name)
            print(f"NOTE: {model_name} is known to hard-reject custom temperature -- "
                  f"skipping straight to its default (instead of {temperature}) without "
                  f"wasting a request on it.", file=sys.stderr)
        resp = client.messages.create(**base_kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")

    try:
        resp = client.messages.create(temperature=temperature, **base_kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")
    except TypeError as e:
        if "temperature" not in str(e):
            raise

    try:
        resp = client.messages.create(extra_body={"temperature": temperature}, **base_kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        pass

    if model_name not in _warned_no_temperature_support:
        _warned_no_temperature_support.add(model_name)
        print(f"NOTE: {model_name} does not accept a custom temperature (rejected both "
              f"the legacy kwarg and extra_body) -- using this model's default temperature "
              f"for ALL calls to it instead of {temperature}. This affects determinism; "
              f"worth noting in your methodology.", file=sys.stderr)
    resp = client.messages.create(**base_kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")


_warned_no_temperature_support_openai: set = set()


def call_openai(system_prompt: str, user_prompt: str, model_cfg: dict) -> str:
    """Some current OpenAI models (o1/o3/o4-style reasoning models, gpt-5.5,
    gpt-5.6) reject any non-default temperature with a 400 'unsupported_value'
    error. Skips the wasted request entirely for models already known to
    reject it; for others, tries the configured temperature first and falls
    back to the model's default at runtime if rejected (covers models not
    yet added to the known list).
    """
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    model_name = model_cfg["name"]
    temperature = model_cfg.get("temperature", 0)
    base_kwargs = dict(
        model=model_name,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    if not openai_supports_custom_temperature(model_name):
        if model_name not in _warned_no_temperature_support_openai:
            _warned_no_temperature_support_openai.add(model_name)
            print(f"NOTE: {model_name} is known to reject custom temperature -- "
                  f"using its default (instead of {temperature}) without wasting a request.",
                  file=sys.stderr)
        resp = client.chat.completions.create(**base_kwargs)
        return resp.choices[0].message.content

    try:
        resp = client.chat.completions.create(temperature=temperature, **base_kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        msg = str(e)
        if "temperature" not in msg or "unsupported_value" not in msg:
            raise

    if model_name not in _warned_no_temperature_support_openai:
        _warned_no_temperature_support_openai.add(model_name)
        print(f"NOTE: {model_name} does not accept a custom temperature (rejected with "
              f"'unsupported_value') -- using this model's default temperature for ALL "
              f"calls to it instead of {temperature}. This affects determinism; worth "
              f"noting in your methodology.", file=sys.stderr)
    resp = client.chat.completions.create(**base_kwargs)
    return resp.choices[0].message.content


PROVIDERS = {"anthropic": call_anthropic, "openai": call_openai}


def extract_json(raw: str) -> dict:
    raw = re.sub(r"^```(json)?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def repair_bio(tags: List[str]) -> List[str]:
    fixed, prev = [], "O"
    for tag in tags:
        if tag not in VALID_TAGS:
            tag = "O"
        if tag == "I-FOOD_LU" and prev not in ("B-FOOD_LU", "I-FOOD_LU"):
            tag = "B-FOOD_LU"
        fixed.append(tag)
        prev = tag
    return fixed


def align(tags: List[str], n: int) -> List[str]:
    if len(tags) < n:
        return tags + ["O"] * (n - len(tags))
    return tags[:n]


def run(input_path: Path, config_path: Path, outdir: Path) -> None:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    text = input_path.read_text(encoding="utf-8")
    sentences_raw = split_sentences(text)
    sentences_tok = [tokenize(s) for s in sentences_raw]
    outdir.mkdir(parents=True, exist_ok=True)

    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        provider_fn = PROVIDERS.get(model_cfg["provider"])
        if provider_fn is None:
            print(f"[{model_name}] unknown provider, skipping", file=sys.stderr)
            continue

        raw_path = outdir / f"{input_path.stem}.{model_name}.raw.jsonl"
        bio_path = outdir / f"{input_path.stem}.{model_name}.food_entity.bio.tsv"
        err_path = outdir / f"{input_path.stem}.{model_name}.food_entity.errors.jsonl"

        print(f"[{model_name}] stage 1: {len(sentences_tok)} sentences -> {bio_path}", file=sys.stderr)

        results = []
        with open(raw_path, "w", encoding="utf-8") as rawf, open(err_path, "w", encoding="utf-8") as errf:
            for i, tokens in enumerate(sentences_tok):
                if not tokens:
                    continue
                record = {"sentence_index": i, "sentence_text": sentences_raw[i], "tokens": tokens, "model": model_name}
                problems = []
                tags = ["O"] * len(tokens)
                try:
                    raw = provider_fn(SYSTEM_PROMPT, build_user_prompt(tokens), model_cfg)
                    record["raw_response"] = raw
                    data = extract_json(raw)
                    raw_tags = data.get("food_entity_tags", [])
                    if len(raw_tags) != len(tokens):
                        problems.append(f"length {len(raw_tags)} != {len(tokens)} tokens -- padded/truncated")
                    tags = repair_bio(align(list(raw_tags), len(tokens)))
                except Exception as e:
                    record["raw_response"] = None
                    record["error"] = str(e)
                    problems.append(f"FAILED: {e}")
                rawf.write(json.dumps(record, ensure_ascii=False) + "\n")
                if problems:
                    errf.write(json.dumps({"sentence_index": i, "sentence_text": sentences_raw[i], "problems": problems}) + "\n")
                results.append({"tokens": tokens, "tags": tags})

        with open(bio_path, "w", encoding="utf-8") as f:
            f.write("token\tFOOD_ENTITY\n")
            for r in results:
                for tok, tag in zip(r["tokens"], r["tags"]):
                    f.write(f"{tok}\t{tag}\n")
                f.write("\n")

        print(f"[{model_name}] done.", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    run(args.input, args.config, args.outdir)


if __name__ == "__main__":
    main()
