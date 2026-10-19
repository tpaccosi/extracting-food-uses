#!/usr/bin/env python3
"""
stage2_preserving_batch.py
=======================
Batch-API version of stage2_preserving.py -- reuses its exact SYSTEM_PROMPT and
VALID_TAGS (written and tested once, there), only the transport differs.

Usage
-----
    python3 stage2_preserving_batch.py submit --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_preserving_out/
    python3 stage2_preserving_batch.py fetch --outdir stage2_preserving_out/ --wait
"""
from __future__ import annotations

from stage2_preserving import SYSTEM_PROMPT, VALID_TAGS
from stage2_common_batch import build_arg_parser, cmd_submit, cmd_fetch

FRAME_TAG = "preserving"
JSON_KEYS = ['preserving_tags']
COLUMN_NAMES = ['PRESERVING']


def main():
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()
    if args.cmd == "submit":
        cmd_submit(FRAME_TAG, SYSTEM_PROMPT, args)
    else:
        cmd_fetch(FRAME_TAG, JSON_KEYS, COLUMN_NAMES, VALID_TAGS, args)


if __name__ == "__main__":
    main()
