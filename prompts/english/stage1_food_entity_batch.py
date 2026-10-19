#!/usr/bin/env python3
"""
stage1_food_entity_batch.py
=============================
Batch-API version of stage1_food_entity.py (50% cheaper, async instead of
synchronous). Reuses the exact same SYSTEM_PROMPT/parsing/repair logic
from stage1_food_entity.py -- only the transport differs.

Usage
-----
    export ANTHROPIC_API_KEY=... OPENAI_API_KEY=...

    python3 stage1_food_entity_batch.py submit --input input.txt --config config.yaml --outdir stage1_out/
    python3 stage1_food_entity_batch.py fetch  --outdir stage1_out/ --wait

Output (per model), once fetched -- identical shape to the sync version:
    stage1_out/<input_stem>.<model>.raw.jsonl
    stage1_out/<input_stem>.<model>.food_entity.bio.tsv
    stage1_out/<input_stem>.<model>.food_entity.errors.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml

from common import tokenize, split_sentences, get_api_key, anthropic_supports_custom_temperature, build_cached_system, openai_supports_custom_temperature
from stage1_food_entity import SYSTEM_PROMPT, build_user_prompt, extract_json, repair_bio, align

MANIFEST_SUFFIX = ".stage1.batch_manifest.json"


def _custom_id(model_name: str, sentence_index: int) -> str:
    return f"{model_name}__{sentence_index}"


def _tokenize_input(input_path: Path):
    text = input_path.read_text(encoding="utf-8")
    sentences_raw = split_sentences(text)
    sentences_tok = [tokenize(s) for s in sentences_raw]
    return [(i, sentences_raw[i], toks) for i, toks in enumerate(sentences_tok) if toks]


# --------------------------------------------------------------------------
# SUBMIT
# --------------------------------------------------------------------------

def submit_anthropic(items, model_cfgs: List[dict]) -> Dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))

    requests, manifest_items = [], {}
    for model_cfg in model_cfgs:
        model_name = model_cfg["name"]
        temperature = model_cfg.get("temperature", 0)
        for i, sentence_text, tokens in items:
            cid = _custom_id(model_name, i)
            params = {
                "model": model_name, "max_tokens": 1000,
                "system": build_cached_system(SYSTEM_PROMPT),
                "messages": [{"role": "user", "content": build_user_prompt(tokens)}],
            }
            if anthropic_supports_custom_temperature(model_name):
                params["temperature"] = temperature  # else omit -- known to hard-reject it
            requests.append({"custom_id": cid, "params": params})
            manifest_items[cid] = {"sentence_index": i, "sentence_text": sentence_text,
                                    "tokens": tokens, "model": model_name}

    batch = client.beta.messages.batches.create(requests=requests)
    print(f"[anthropic] submitted batch {batch.id} with {len(requests)} requests", file=sys.stderr)
    return {"batch_id": batch.id, "items": manifest_items}


def submit_openai(items, model_cfgs: List[dict], workdir: Path) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    jsonl_path = workdir / "_openai_stage1_batch_input.jsonl"
    manifest_items = {}
    warned_models = set()
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for model_cfg in model_cfgs:
            model_name = model_cfg["name"]
            temperature = model_cfg.get("temperature", 0)
            if not openai_supports_custom_temperature(model_name) and model_name not in warned_models:
                warned_models.add(model_name)
                print(f"NOTE: {model_name} is known to reject custom temperature -- "
                      f"omitting it from all its batch requests (uses the model's default).",
                      file=sys.stderr)
            for i, sentence_text, tokens in items:
                cid = _custom_id(model_name, i)
                body = {
                    "model": model_name,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(tokens)},
                    ],
                }
                if openai_supports_custom_temperature(model_name):
                    body["temperature"] = temperature
                line = {"custom_id": cid, "method": "POST", "url": "/v1/chat/completions", "body": body}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                manifest_items[cid] = {"sentence_index": i, "sentence_text": sentence_text,
                                        "tokens": tokens, "model": model_name}

    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                   completion_window="24h")
    print(f"[openai] submitted batch {batch.id} with {len(manifest_items)} requests", file=sys.stderr)
    return {"batch_id": batch.id, "items": manifest_items}


def cmd_submit(args):
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    items = _tokenize_input(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)

    anthropic_models = [m for m in cfg["models"] if m["provider"] == "anthropic"]
    openai_models = [m for m in cfg["models"] if m["provider"] == "openai"]

    manifest = {"input_stem": args.input.stem, "providers": {}}
    if anthropic_models:
        manifest["providers"]["anthropic"] = submit_anthropic(items, anthropic_models)
    if openai_models:
        manifest["providers"]["openai"] = submit_openai(items, openai_models, args.outdir)

    manifest_path = args.outdir / f"{args.input.stem}{MANIFEST_SUFFIX}"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {manifest_path}. Run `fetch` once the batch(es) complete.", file=sys.stderr)


# --------------------------------------------------------------------------
# FETCH
# --------------------------------------------------------------------------

def fetch_anthropic(batch_id: str) -> Dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))
    batch = client.beta.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        return {"status": batch.processing_status, "results": {}}

    results = {}
    for entry in client.beta.messages.batches.results(batch_id):
        cid = entry.custom_id
        if entry.result.type == "succeeded":
            msg = entry.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            results[cid] = {"raw_response": text, "error": None}
        else:
            results[cid] = {"raw_response": None, "error": f"{entry.result.type}"}
    return {"status": "ended", "results": results}


def fetch_openai(batch_id: str) -> Dict[str, Any]:
    """Handles both output_file_id and error_file_id -- if every request in the
    batch fails validation (e.g. bad model name), output_file_id can be None
    with only error_file_id populated."""
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        return {"status": batch.status, "results": {}}

    results: Dict[str, Any] = {}
    if batch.output_file_id:
        content = client.files.content(batch.output_file_id).text
        for line in content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            if obj.get("error"):
                results[cid] = {"raw_response": None, "error": str(obj["error"])}
            else:
                text = obj["response"]["body"]["choices"][0]["message"]["content"]
                results[cid] = {"raw_response": text, "error": None}
    if batch.error_file_id:
        err_content = client.files.content(batch.error_file_id).text
        for line in err_content.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            if cid:
                err_detail = obj.get("error") or obj.get("response", {}).get("body") or obj
                results[cid] = {"raw_response": None, "error": str(err_detail)}
    if not batch.output_file_id and not batch.error_file_id:
        print(f"[openai] batch {batch_id} completed but has neither output nor error file", file=sys.stderr)
    return {"status": "completed", "results": results}


FETCHERS = {"anthropic": fetch_anthropic, "openai": fetch_openai}


def _write_outputs(outdir: Path, input_stem: str, manifest_items: Dict[str, Any],
                    results: Dict[str, Any]) -> None:
    by_model: Dict[str, List[dict]] = {}
    for cid, meta in manifest_items.items():
        res = results.get(cid)
        if res is None:
            continue
        record = {
            "sentence_index": meta["sentence_index"], "sentence_text": meta["sentence_text"],
            "tokens": meta["tokens"], "model": meta["model"], "raw_response": res["raw_response"],
        }
        if res["error"]:
            record["error"] = res["error"]
        by_model.setdefault(meta["model"], []).append(record)

    for model_name, records in by_model.items():
        records.sort(key=lambda r: r["sentence_index"])

        raw_path = outdir / f"{input_stem}.{model_name}.raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        bio_path = outdir / f"{input_stem}.{model_name}.food_entity.bio.tsv"
        err_path = outdir / f"{input_stem}.{model_name}.food_entity.errors.jsonl"
        n_flagged = 0
        with open(bio_path, "w", encoding="utf-8") as biof, open(err_path, "w", encoding="utf-8") as errf:
            biof.write("token\tFOOD_ENTITY\n")
            for r in records:
                tokens = r["tokens"]
                problems = []
                tags = ["O"] * len(tokens)
                if r.get("error"):
                    problems.append(f"FAILED: {r['error']}")
                else:
                    try:
                        data = extract_json(r["raw_response"])
                        raw_tags = data.get("food_entity_tags", [])
                        if len(raw_tags) != len(tokens):
                            problems.append(f"length {len(raw_tags)} != {len(tokens)} tokens -- padded/truncated")
                        tags = repair_bio(align(list(raw_tags), len(tokens)))
                    except Exception as e:
                        problems.append(f"FAILED to parse: {e}")
                for tok, tag in zip(tokens, tags):
                    biof.write(f"{tok}\t{tag}\n")
                biof.write("\n")
                if problems:
                    n_flagged += 1
                    errf.write(json.dumps({"sentence_index": r["sentence_index"],
                                           "sentence_text": r["sentence_text"], "problems": problems}) + "\n")
        print(f"[{model_name}] wrote {bio_path} ({len(records)} sentences, {n_flagged} flagged)", file=sys.stderr)


def cmd_fetch(args):
    manifest_paths = sorted(args.outdir.glob(f"*{MANIFEST_SUFFIX}"))
    if not manifest_paths:
        print(f"No {MANIFEST_SUFFIX} manifest found in {args.outdir}. Run `submit` first.", file=sys.stderr)
        return

    deadline = time.time() + args.max_wait if args.wait else None
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        input_stem = manifest["input_stem"]

        while True:
            all_done = True
            for provider, info in manifest["providers"].items():
                status_result = FETCHERS[provider](info["batch_id"])
                status = status_result["status"]
                is_done = status in ("ended", "completed")
                print(f"[{provider}] batch {info['batch_id']}: {status}", file=sys.stderr)
                if is_done:
                    _write_outputs(args.outdir, input_stem, info["items"], status_result["results"])
                else:
                    all_done = False
            if all_done or not args.wait:
                break
            if deadline and time.time() > deadline:
                print("max-wait exceeded, exiting -- re-run fetch later", file=sys.stderr)
                break
            time.sleep(args.poll_interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit")
    sp.add_argument("--input", required=True, type=Path)
    sp.add_argument("--config", required=True, type=Path)
    sp.add_argument("--outdir", required=True, type=Path)
    sp.set_defaults(func=cmd_submit)

    fp = sub.add_parser("fetch")
    fp.add_argument("--outdir", required=True, type=Path)
    fp.add_argument("--wait", action="store_true")
    fp.add_argument("--max-wait", type=int, default=3600)
    fp.add_argument("--poll-interval", type=int, default=30)
    fp.set_defaults(func=cmd_fetch)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
