#!/usr/bin/env python3
"""
stage2_descriptor.py
======================
STAGE 2, DESCRIPTOR frame only (DESCR_Positive / DESCR_Negative /
DESCR_Neutral). No lexical unit of its own -- tags a descriptive span
directly, only when it describes an already-known food entity.

Usage
-----
    python3 stage2_descriptor.py --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_descriptor_out/
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
Tag DESCRIPTOR spans in ONE BIO sequence, "descriptor_tags" -- a
characteristic, description, or type of a food entity, polarity-typed:
  - DESCR_Positive: a positive characteristic (intensity, volume/reach,
    duration, state, character, hedonic quality). Include intensifying
    modifiers in the same span (e.g. "really delicious").
  - DESCR_Negative: a negative characteristic, same rules as Positive.
  - DESCR_Neutral: a neutral characteristic (this covers most
    process-descriptive adjectives like "boiling" or "planked"), same
    rules as Positive.

CRITICAL RULE: only tag a DESCRIPTOR when there is an already-known food
entity (marked <FOOD>...</FOOD>) NEARBY that it describes. A stand-alone
adjective with no associated food entity in the expression is NOT tagged.
Usually this is an adjective or a type specification functioning like a
compound modifier, e.g. "Hamburg steak", "Spring Chicken", "cane sugar",
"loaf sugar".

SPECIAL CASE -- participial/attributive process words: when the
descriptive word is ITSELF a participial/attributive verb-form describing
a process (e.g. "planked fish", "boiling water", "beaten egg"), it is
STILL tagged as DESCRIPTOR here (typically DESCR_Neutral) even though it
is simultaneously a trigger word for a different frame (COOKING_CREATION)
that isn't part of this call's output -- tag it as DESCRIPTOR regardless.

SPAN RULES: include intensifying modifiers in the same span; exclude
articles and personal/possessive pronouns.

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"descriptor_tags": [...]}
The array must have exactly the same length as the input token list
(count <FOOD>/</FOOD> markers as zero-width).
Allowed labels: DESCR_Positive, DESCR_Negative, DESCR_Neutral.
Every tag is "O", "B-<label>", or "I-<label>".

EXAMPLES

Input tokens (FOOD_LU pre-marked): ["We", "bought", "fresh", "<FOOD>apples</FOOD>", "and", "kippered", "<FOOD>herring</FOOD>", "."]
Output: {"descriptor_tags": ["O", "O", "B-DESCR_Positive", "O", "O", "B-DESCR_Neutral", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["We", "enjoyed", "planked", "<FOOD>fish</FOOD>", "."]
Output: {"descriptor_tags": ["O", "O", "B-DESCR_Neutral", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["Add", "the", "boiling", "<FOOD>water</FOOD>", "to", "the", "beaten", "<FOOD>egg</FOOD>", "."]
Output: {"descriptor_tags": ["O", "O", "B-DESCR_Neutral", "O", "O", "O", "B-DESCR_Neutral", "O", "O"]}

Input tokens (FOOD_LU pre-marked): ["This", "<FOOD>fish</FOOD>", "is", "really", "delicious", "."]
Output: {"descriptor_tags": ["O", "O", "O", "B-DESCR_Positive", "I-DESCR_Positive", "O"]}
NOTE: the intensifier "really" is included in the same DESCR_Positive
span as "delicious".

Input tokens (FOOD_LU pre-marked): ["The", "weather", "was", "cold", "."]
Output: {"descriptor_tags": ["O", "O", "O", "O", "O"]}
NOTE: "cold" describes the weather, not a food entity -- not tagged.
"""

VALID_TAGS = {
    "descriptor_tags": {"O", "B-DESCR_Positive", "I-DESCR_Positive",
                         "B-DESCR_Negative", "I-DESCR_Negative",
                         "B-DESCR_Neutral", "I-DESCR_Neutral"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--stage1-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    run_stage2(
        frame_tag="descriptor", json_keys=["descriptor_tags"], column_names=["DESCRIPTOR"],
        valid_tags=VALID_TAGS, system_prompt=SYSTEM_PROMPT,
        input_stem=args.input.stem, stage1_dir=args.stage1_dir,
        config_path=args.config, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
