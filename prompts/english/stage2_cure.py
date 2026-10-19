#!/usr/bin/env python3
"""
stage2_cure.py
================
STAGE 2, CURE frame only (CURE_LU, Affliction, Body_Part, Healer,
Food_Treatment, Patient). Includes the hybrid/implicit-LU behavior and
the metonymic "used for/as" construction.

Usage
-----
    python3 stage2_cure.py --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_cure_out/
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
Tag the CURE frame in ONE BIO sequence, "cure_tags": a Healer treats/cures
an Affliction affecting a Patient, using a food item as Treatment.

STEP 1 -- Identify CURE_LU. LUs include: alleviate, cure, ease, heal,
improve, nurse, palliate, palliative, rehabilitate, remedy, resuscitate,
soothe, subdue, therapy, treat, treatment.

METONYMIC / IMPLICIT LU: when there is no explicit CURE_LU verb, the
construction "used for/as", "is good for", "excellent for", or simply a
pronoun + "for" can itself function as CURE_LU. When NONE of these are
present either, but the sentence clearly describes a food used as a
medicine, the Affliction or Patient noun ITSELF functions as the LU: tag
it with its FE label pipe-joined with CURE_LU, e.g.
"B-CURE_Affliction|B-CURE_LU". If BOTH an Affliction and a Patient are
present, only the AFFLICTION gets this extra CURE_LU pipe-tag (never the
Patient), even though both FE spans are still tagged normally.

STEP 2 -- For that frame instance, identify:
  - CURE_Affliction: the disease/injury/condition. Include determiners/
    specifications (e.g. "the plague", "the worst fever I have ever had").
  - CURE_Body_Part: the specific body area treated. Include determiners
    (e.g. "on the body", "his arms").
  - CURE_Healer: who treats/cures the Patient. Include determiners (e.g.
    "Persian doctors", "my mother").
  - CURE_Food_Treatment: the food substance used to cure. This is almost
    always one of the ALREADY-KNOWN food entities marked <FOOD>...</FOOD>
    -- tag it here EVEN THOUGH already marked as food (double-annotation).
  - CURE_Patient: the sufferer.

NESTED PATIENT/AFFLICTION: a Patient phrase often syntactically CONTAINS
its Affliction (e.g. "sufferers from the gout"). Tag the whole Patient
phrase with CURE_Patient as normal, and on the token(s) where the
Affliction sits, pipe-join it with the ongoing Patient tag, e.g.
"I-CURE_Patient|B-CURE_Affliction|B-CURE_LU" (the LU part only if this is
also the implicit-LU case above). Only tag CURE_Patient alone (no pipe)
when it is a genuinely separate span, not wrapped around an Affliction.

SPAN RULES: exclude articles and personal/possessive pronouns from spans
(except where they are inside a larger included span, as in the nested
case above).

OUTPUT FORMAT
Return ONLY a single JSON object, no prose, no markdown fences:
{"cure_tags": [...]}
The array must have exactly the same length as the input token list
(count <FOOD>/</FOOD> markers as zero-width).
Allowed labels: CURE_LU, CURE_Affliction, CURE_Body_Part, CURE_Healer,
CURE_Food_Treatment, CURE_Patient.
Every tag is "O", "B-<label>", "I-<label>", or a pipe-joined combination.

EXAMPLES

Input tokens (FOOD_LU pre-marked): ["<FOOD>Honey</FOOD>", "soothes", "my", "sore", "throat", "."]
Output: {"cure_tags": ["B-CURE_Food_Treatment", "B-CURE_LU", "O", "B-CURE_Affliction", "I-CURE_Affliction", "O"]}

Input tokens (FOOD_LU pre-marked): ["All", "sufferers", "from", "the", "gout", "are", "cured", "by", "<FOOD>apples</FOOD>", "."]
Output: {"cure_tags": ["O", "B-CURE_Patient", "I-CURE_Patient", "I-CURE_Patient", "I-CURE_Patient|B-CURE_Affliction", "O", "B-CURE_LU", "O", "B-CURE_Food_Treatment", "O"]}
NOTE: "gout" is nested inside the Patient span -- two tags pipe-joined on that one
token.

Input tokens (FOOD_LU pre-marked): ["My", "dad", "takes", "<FOOD>honey</FOOD>", "for", "his", "cough", "."]
Output: {"cure_tags": ["B-CURE_Patient", "I-CURE_Patient", "B-CURE_LU", "B-CURE_Food_Treatment", "O", "O", "B-CURE_Affliction", "O"]}
NOTE: here Patient ("My dad") and Affliction ("cough") are genuinely
separate spans, not nested -- no pipe needed, and "takes" functions as the
CURE_LU.
"""

VALID_TAGS = {
    "cure_tags": {
        "O", "B-CURE_LU", "I-CURE_LU",
        "B-CURE_Affliction", "I-CURE_Affliction",
        "B-CURE_Body_Part", "I-CURE_Body_Part",
        "B-CURE_Healer", "I-CURE_Healer",
        "B-CURE_Food_Treatment", "I-CURE_Food_Treatment",
        "B-CURE_Patient", "I-CURE_Patient",
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
        frame_tag="cure", json_keys=["cure_tags"], column_names=["CURE"],
        valid_tags=VALID_TAGS, system_prompt=SYSTEM_PROMPT,
        input_stem=args.input.stem, stage1_dir=args.stage1_dir,
        config_path=args.config, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
