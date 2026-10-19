#!/usr/bin/env python3
"""
stage2_food_attributes.py
===========================
STAGE 2, FOOD's own remaining attributes only (FOOD_Constituent_Part,
FOOD_Count, FOOD_Unit) -- the frame elements of the FOOD frame besides
FOOD_LU itself, which stage1_food_entity.py already found.

Run this ONCE per input, independently of which other stage2_X.py scripts
you also run (PRESERVING, CURE, etc.) -- it doesn't need to be repeated
per other frame.

Usage
-----
    python3 stage2_food_attributes.py --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_food_attributes_out/
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
these spans as ground truth -- do not second-guess or re-derive them, and
do not tag FOOD_LU yourself (it isn't part of this task's output).

TASK
For each already-known food entity, check whether it has any of these
attributes nearby, and tag them in ONE BIO sequence, "food_attributes_tags":

  - FOOD_Constituent_Part: a part/component of the food entity. Annotate
    the full relevant span including any descriptor of the part in the
    SAME span (e.g. "yellow rind" as one span in "grate the yellow rind
    of lemon" -- the descriptor "yellow" stays inside the Constituent_Part
    span, it is not tagged separately).
  - FOOD_Count: the number of units of the food entity. Annotate numerical
    expressions ("two", "three") as well as vague-quantity expressions
    ("enough", "a little", "a few", "some", and quantifiers implying a
    small non-specific number like "several", "many", "dozens").
  - FOOD_Unit: the standardized quantity the food entity is measured in
    (e.g. "pounds", "slice", "piece", "chop", "morsel", "layers"). For
    expressions like "a spoonful", "a drop", "a pinch", tag the WHOLE
    expression INCLUDING the determiner "a" as one FOOD_Unit span (this
    is the one exception to the general "exclude determiners" rule).

SPAN RULES: for Constituent_Part and Count, include adjectives/quantifiers
of the head noun in the same span, but exclude articles (a, an, the) and
personal/possessive pronouns. FOOD_Unit is the exception noted above --
it keeps "a" when the expression is a measure phrase like "a spoonful".

Most food entities will have NO attributes at all -- only tag what's
actually present in the sentence; don't force an attribute that isn't there.

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"food_attributes_tags": [...]}
The array must have exactly the same length as the input token list
(count <FOOD>/</FOOD> markers as zero-width).
Allowed labels: FOOD_Constituent_Part, FOOD_Count, FOOD_Unit.
Every tag is "O", "B-<label>", or "I-<label>".

EXAMPLES

Input tokens (FOOD_LU pre-marked): ["I", "pickle", "my", "<FOOD>cucumbers</FOOD>", "in", "<FOOD>vinegar</FOOD>", "."]
Output: {"food_attributes_tags": ["O", "O", "O", "O", "O", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["My", "mother", "smokes", "two", "pounds", "of", "<FOOD>salmon</FOOD>", "in", "the", "old", "smokehouse", "."]
Output: {"food_attributes_tags": ["O", "O", "O", "B-FOOD_Count", "B-FOOD_Unit", "O", "O", "O", "O", "O", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["Grate", "the", "yellow", "rind", "of", "the", "<FOOD>lemon</FOOD>", "."]
Output: {"food_attributes_tags": ["O", "O", "B-FOOD_Constituent_Part", "I-FOOD_Constituent_Part", "O", "O", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["Add", "a", "pinch", "of", "<FOOD>salt</FOOD>", "and", "a", "few", "<FOOD>cloves</FOOD>", "."]
Output: {"food_attributes_tags": ["O", "B-FOOD_Unit", "I-FOOD_Unit", "O", "O", "O", "B-FOOD_Count", "I-FOOD_Count", "O", "O"]}
NOTE: "a pinch" is one FOOD_Unit span including the determiner "a"
(exception); "a few" is one FOOD_Count span (also an exception -- see
the general rule's "a few" carve-out).
"""

VALID_TAGS = {
    "food_attributes_tags": {"O", "B-FOOD_Constituent_Part", "I-FOOD_Constituent_Part",
                              "B-FOOD_Count", "I-FOOD_Count", "B-FOOD_Unit", "I-FOOD_Unit"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--stage1-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    run_stage2(
        frame_tag="food_attributes", json_keys=["food_attributes_tags"], column_names=["FOOD_ATTRIBUTES"],
        valid_tags=VALID_TAGS, system_prompt=SYSTEM_PROMPT,
        input_stem=args.input.stem, stage1_dir=args.stage1_dir,
        config_path=args.config, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
