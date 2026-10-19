#!/usr/bin/env python3
"""
stage2_preserving.py
=====================
STAGE 2, PRESERVING frame only. Given the FOOD_LU spans found by
stage1_food_entity.py (marked inline as ground truth), first identifies
PR_LU, then fills PR_Agent / PR_Medium / PR_Food_Patient.

Usage
-----
    python3 stage2_preserving.py --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_preserving_out/
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stage2_common import run_stage2

SYSTEM_PROMPT = """You are an expert linguistic annotator applying frame semantics to
(possibly historical) English text about food.

CONTEXT ALREADY GIVEN TO YOU
Food entities (FOOD_LU) have ALREADY been identified in the sentence and
are marked inline in the token list below as <FOOD>word</FOOD>. Treat
these spans as ground truth -- do not second-guess or re-derive them.

TASK
Produce ONE BIO-tagged sequence, "preserving_tags", one tag per input
token, following this order of operations:

STEP 1 -- Identify PR_LU: every word expressing the act of preserving
something from decaying (e.g. can, cure, dry, pickle, preserve, salt,
smoke, tinned). This is NOT restricted to verb forms -- nominalized or
gerund forms of the same act count equally, e.g. "preservation",
"salting", "curing", "drying", "canning", "pickling", "smoking" used as
nouns are ALL valid PR_LU triggers, just as much as their verb-form
counterparts. Scan for both. If no such word is present, every token is
"O" and you're done.

STEP 2 -- If a PR_LU IS present, for that frame instance identify:
  - PR_Agent: the person performing the preservation (e.g. "my mom").
  - PR_Medium: the substance the food is preserved in/with, or any means/
    instrument/METHOD used to preserve (e.g. "the smoke", "household
    methods").
  - PR_Food_Patient: the food entity being preserved. This is almost
    always one of the ALREADY-KNOWN food entities marked <FOOD>...</FOOD>
    above -- tag it here EVEN THOUGH it's already marked as a food
    entity; this is intentional double-annotation, not a mistake.
    Likewise PR_Medium can itself be a food entity.

SPAN RULES: include adjectives/quantifiers/modifiers of the head noun in
the same span, but ALWAYS exclude articles (a, an, the) and personal/
possessive pronouns (I, you, he, my, your, etc.) from the span, even when
adjacent to an included modifier.

A PR_LU can include its own adverbial modifier as one span (e.g.
"[salted heavily]" as one PR_LU).

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"preserving_tags": [...]}
The array must have exactly the same length as the input token list
(count <FOOD>/</FOOD> markers as zero-width).
Allowed labels: PR_LU, PR_Agent, PR_Medium, PR_Food_Patient.
Every tag is "O", "B-<label>", or "I-<label>".

EXAMPLES

Input tokens (FOOD_LU pre-marked): ["I", "pickle", "my", "<FOOD>cucumbers</FOOD>", "in", "<FOOD>vinegar</FOOD>", "."]
Output: {"preserving_tags": ["O", "B-PR_LU", "O", "B-PR_Food_Patient", "O", "B-PR_Medium", "O"]}

Input tokens (FOOD_LU pre-marked): ["My", "mother", "smokes", "two", "pounds", "of", "<FOOD>salmon</FOOD>", "in", "the", "old", "smokehouse", "."]
Output: {"preserving_tags": ["B-PR_Agent", "I-PR_Agent", "B-PR_LU", "O", "O", "O", "B-PR_Food_Patient", "O", "B-PR_Medium", "I-PR_Medium", "I-PR_Medium", "O"]}

Input tokens (FOOD_LU pre-marked): ["We", "enjoyed", "planked", "<FOOD>fish</FOOD>", "."]
Output: {"preserving_tags": ["O", "O", "O", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["We", "keep", "<FOOD>vegetables</FOOD>", "fresh", "by", "household", "methods", "of", "preservation", "."]
Output: {"preserving_tags": ["O", "O", "B-PR_Food_Patient", "O", "O", "B-PR_Medium", "I-PR_Medium", "O", "B-PR_LU", "O"]}

Input tokens (FOOD_LU pre-marked): ["I", "love", "cured", "<FOOD>meat</FOOD>", "."]
Output: {"preserving_tags": ["O", "O", "B-PR_LU", "B-PR_Food_Patient", "O"]}

Input tokens (FOOD_LU pre-marked): ["Salting", "is", "nothing", "else", "than", "a", "species", "of", "drying", "."]
Output: {"preserving_tags": ["B-PR_LU", "O", "O", "O", "O", "O", "O", "O", "B-PR_LU", "O"]}
NOTE: this example has TWO separate PR_LU instances, both nominal/gerund
forms ("Salting", "drying"), neither with an explicit Patient or Medium
in this sentence -- that's fine, tag only what's actually present.

Input tokens (FOOD_LU pre-marked): ["they", "are", "restricted", "to", "the", "preservation", "of", "the", "<FOOD>flesh</FOOD>", "."]
Output: {"preserving_tags": ["O", "O", "O", "O", "O", "B-PR_LU", "O", "O", "B-PR_Food_Patient", "O"]}
NOTE: "preservation" (a noun, not a verb) is still PR_LU here.
"""

VALID_TAGS = {
    "preserving_tags": {"O", "B-PR_LU", "I-PR_LU", "B-PR_Agent", "I-PR_Agent",
                         "B-PR_Medium", "I-PR_Medium", "B-PR_Food_Patient", "I-PR_Food_Patient"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--stage1-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    run_stage2(
        frame_tag="preserving", json_keys=["preserving_tags"], column_names=["PRESERVING"],
        valid_tags=VALID_TAGS, system_prompt=SYSTEM_PROMPT,
        input_stem=args.input.stem, stage1_dir=args.stage1_dir,
        config_path=args.config, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
