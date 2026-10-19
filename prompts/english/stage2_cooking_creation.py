#!/usr/bin/env python3
"""
stage2_cooking_creation.py
============================
STAGE 2, COOKING_CREATION frame only (COOKING_CREATION_LU, Cook,
Food_Product, Food_Material). Aggregates FrameNet's Cooking_creation and
Apply_heat frames.

Usage
-----
    python3 stage2_cooking_creation.py --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_cooking_creation_out/
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
Tag the COOKING_CREATION frame in ONE BIO sequence, "cooking_creation_tags":
food/meal preparation and manipulation by an agent, either creating or
modifying it.

STEP 1 -- Identify COOKING_CREATION_LU: every word which evokes the
cooking/culinary manipulation process, whether it emphasizes the process
(e.g. boiling potatoes) or the resulting edible entity (e.g. baking a
pie). LUs include: bake, baking, barbecue, blanch, boil, boiling, braise,
broil, brown, char, coddle, concoct, cook, cooking, deep fry, fix, fry,
frying, grill, grilling, make, melt, melting, microwave, parboil, plank,
poach, prepare, preparation, put together, roast, roasting, saute, scald,
scorch, sear, simmer, simmering, singe, steam, steaming, steep, stew,
stewing, toast, toasting, whip up.

EXCLUDED: verbs describing an action that does NOT change/transform the
food are NOT annotated as COOKING_CREATION_LU -- e.g. mix, stir,
sprinkle, put, pour, add, shake. Do not tag these even if food is nearby.

ALSO EXCLUDED -- PRESERVATION VERBS: words expressing food preservation
belong to a separate frame (PRESERVING, out of scope here), even though
they also involve manipulating food and could otherwise look like
COOKING_CREATION_LU candidates. Do NOT tag any of these, in any form
(verb, gerund, or noun): can, cure, dry, pickle, preservation, preserve, 
salt, tinned, smoke, buccan, bucan, pot, kipper, collar, souse 
-- nor their derived forms (canning, curing, drying, pickling,
preservation, preserving, salting, smoking etc.). E.g. in "we smoke the
salmon", "smoke" is PRESERVING's job, not COOKING_CREATION's -- leave it
untagged here.

A COOKING_CREATION_LU can include its own adverbial modifier as one span,
e.g. "[season lightly]", "[chop finely]" as a single LU span.

PARTICIPIAL/ATTRIBUTIVE LUs: if the LU word is itself a participial/
attributive adjective describing the food (e.g. "planked fish", "boiling
water", "beaten egg"), tag it as COOKING_CREATION_LU here (a separate,
independent DESCRIPTOR frame -- out of scope for this call -- would ALSO
tag it, but that's not your concern here).

STEP 2 -- For each COOKING_CREATION_LU instance you found, identify:
  - COOKING_CREATION_Cook: the person who prepares the Produced_Food.
    Include determiners/specifications (e.g. "the chef", "my dad").
  - COOKING_CREATION_Food_Product: the resulting food produced. This is
    usually one of the ALREADY-KNOWN food entities marked <FOOD>...</FOOD>
    -- tag it here EVEN THOUGH it's already marked as a food entity
    (intentional double-annotation).
  - COOKING_CREATION_Food_Material: the raw food DIRECTLY undergoing a
    single-step transformation named by the LU (e.g. "boil potatoes" ->
    potatoes is Food_Material). Do NOT use this for ingredients being
    COMBINED into a final product (a recipe's ingredient list) -- that
    belongs to a different frame (INGREDIENT, out of scope here): if a
    verb like "make"/"bake" produces a final product out of SEVERAL
    combined ingredients, tag ONLY the final product as
    COOKING_CREATION_Food_Product, and do NOT tag the individual combined
    ingredients as Food_Material here at all.

SPAN RULES: exclude articles and personal/possessive pronouns from spans.

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"cooking_creation_tags": [...]}
The array must have exactly the same length as the input token list
(count <FOOD>/</FOOD> markers as zero-width).
Allowed labels: COOKING_CREATION_LU, COOKING_CREATION_Cook,
COOKING_CREATION_Food_Product, COOKING_CREATION_Food_Material.
Every tag is "O", "B-<label>", or "I-<label>".

EXAMPLES

Input tokens (FOOD_LU pre-marked): ["The", "chef", "baked", "an", "<FOOD>apple</FOOD>", "<FOOD>pie</FOOD>", "from", "<FOOD>apples</FOOD>", "."]
Output: {"cooking_creation_tags": ["B-COOKING_CREATION_Cook", "I-COOKING_CREATION_Cook", "B-COOKING_CREATION_LU", "O", "B-COOKING_CREATION_Food_Product", "I-COOKING_CREATION_Food_Product", "O", "B-COOKING_CREATION_Food_Material", "O"]}

Input tokens (FOOD_LU pre-marked): ["To", "make", "<FOOD>orange</FOOD>", "<FOOD>cake</FOOD>", "take", "two", "<FOOD>oranges</FOOD>", ",", "<FOOD>butter</FOOD>", "and", "<FOOD>sugar</FOOD>", "."]
Output: {"cooking_creation_tags": ["O", "B-COOKING_CREATION_LU", "B-COOKING_CREATION_Food_Product", "I-COOKING_CREATION_Food_Product", "O", "O", "O", "O", "O", "O", "O", "O"]}
NOTE: oranges/butter/sugar are combined ingredients (INGREDIENT frame's
job, out of scope here) -- NOT tagged as Food_Material.

Input tokens (FOOD_LU pre-marked): ["We", "enjoyed", "planked", "<FOOD>fish</FOOD>", "."]
Output: {"cooking_creation_tags": ["O", "O", "B-COOKING_CREATION_LU", "B-COOKING_CREATION_Food_Material", "O"]}

Input tokens (FOOD_LU pre-marked): ["Add", "the", "boiling", "<FOOD>water</FOOD>", "to", "the", "beaten", "<FOOD>egg</FOOD>", "."]
Output: {"cooking_creation_tags": ["O", "O", "B-COOKING_CREATION_LU", "B-COOKING_CREATION_Food_Material", "O", "O", "B-COOKING_CREATION_LU", "B-COOKING_CREATION_Food_Material", "O"]}
NOTE: "Add" is excluded (does not transform the food) -- not tagged.
"""

VALID_TAGS = {
    "cooking_creation_tags": {
        "O", "B-COOKING_CREATION_LU", "I-COOKING_CREATION_LU",
        "B-COOKING_CREATION_Cook", "I-COOKING_CREATION_Cook",
        "B-COOKING_CREATION_Food_Product", "I-COOKING_CREATION_Food_Product",
        "B-COOKING_CREATION_Food_Material", "I-COOKING_CREATION_Food_Material",
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--stage1-dir", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    args = ap.parse_args()
    run_stage2(
        frame_tag="cooking_creation", json_keys=["cooking_creation_tags"], column_names=["COOKING_CREATION"],
        valid_tags=VALID_TAGS, system_prompt=SYSTEM_PROMPT,
        input_stem=args.input.stem, stage1_dir=args.stage1_dir,
        config_path=args.config, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
