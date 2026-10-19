#!/usr/bin/env python3
"""
stage2_descriptor_batch.py
=======================
Batch-API version of stage2_descriptor.py -- reuses its exact SYSTEM_PROMPT and
VALID_TAGS (written and tested once, there), only the transport differs.

Usage
-----
    python3 stage2_descriptor_batch.py submit --input input.txt --stage1-dir stage1_out/ \
        --config config.yaml --outdir stage2_descriptor_out/
    python3 stage2_descriptor_batch.py fetch --outdir stage2_descriptor_out/ --wait
"""
from __future__ import annotations

from stage2_descriptor import SYSTEM_PROMPT, VALID_TAGS
from stage2_common_batch import build_arg_parser, cmd_submit, cmd_fetch

FRAME_TAG = "descriptor"
JSON_KEYS = ['descriptor_tags']
COLUMN_NAMES = ['DESCRIPTOR']


def main():
    ap = build_arg_parser(__doc__)
    args = ap.parse_args()
    if args.cmd == "submit":
        cmd_submit(FRAME_TAG, SYSTEM_PROMPT, args)
    else:
        cmd_fetch(FRAME_TAG, JSON_KEYS, COLUMN_NAMES, VALID_TAGS, args)


if __name__ == "__main__":
    main()
