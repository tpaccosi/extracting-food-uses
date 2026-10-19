"""
stage2_common_batch.py
========================
Shared submit/fetch engine for every stage2_X_batch.py script. Each thin
wrapper script imports its own SYSTEM_PROMPT/VALID_TAGS from the existing
SYNCHRONOUS stage2_X.py module (so the prompt text is written and tested
exactly once) and calls submit_stage2_batch(...) / fetch_stage2_batch(...)
here for the actual batch-API mechanics.

Depends on stage1 already being fetched/complete (sync or batch, doesn't
matter) -- reads its BIO TSV + raw JSONL via stage2_common.load_stage1(),
exactly like the synchronous run_stage2() does.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import yaml

from common import get_api_key, anthropic_supports_custom_temperature, build_cached_system, openai_supports_custom_temperature
from stage2_common import load_stage1, build_user_prompt, extract_json, repair_bio, align


def _custom_id(model_name: str, sentence_index: int) -> str:
    return f"{model_name}__{sentence_index}"


# --------------------------------------------------------------------------
# SUBMIT
# --------------------------------------------------------------------------

def submit_anthropic(system_prompt: str, stage1_by_model: Dict[str, dict], model_cfgs: List[dict]) -> Dict[str, Any]:
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("ANTHROPIC_API_KEY"))

    requests, manifest_items = [], {}
    for model_cfg in model_cfgs:
        model_name = model_cfg["name"]
        temperature = model_cfg.get("temperature", 0)
        stage1_sents = stage1_by_model.get(model_name)
        if not stage1_sents:
            continue
        for i in sorted(stage1_sents.keys()):
            tokens, food_entity_tags, sentence_text = stage1_sents[i]
            cid = _custom_id(model_name, i)
            params = {
                "model": model_name, "max_tokens": 1200,
                "system": build_cached_system(system_prompt),
                "messages": [{"role": "user", "content": build_user_prompt(tokens, food_entity_tags)}],
            }
            if anthropic_supports_custom_temperature(model_name):
                params["temperature"] = temperature
            requests.append({"custom_id": cid, "params": params})
            manifest_items[cid] = {"sentence_index": i, "sentence_text": sentence_text,
                                    "tokens": tokens, "food_entity_tags": food_entity_tags, "model": model_name}

    if not requests:
        return {"batch_id": None, "items": {}}
    batch = client.beta.messages.batches.create(requests=requests)
    print(f"[anthropic] submitted batch {batch.id} with {len(requests)} requests", file=sys.stderr)
    return {"batch_id": batch.id, "items": manifest_items}


def submit_openai(system_prompt: str, stage1_by_model: Dict[str, dict], model_cfgs: List[dict],
                   workdir: Path, frame_tag: str) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    jsonl_path = workdir / f"_openai_{frame_tag}_batch_input.jsonl"
    manifest_items = {}
    n_written = 0
    warned_models = set()
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for model_cfg in model_cfgs:
            model_name = model_cfg["name"]
            temperature = model_cfg.get("temperature", 0)
            stage1_sents = stage1_by_model.get(model_name)
            if not stage1_sents:
                continue
            if not openai_supports_custom_temperature(model_name) and model_name not in warned_models:
                warned_models.add(model_name)
                print(f"NOTE: {model_name} is known to reject custom temperature -- "
                      f"omitting it from all its batch requests (uses the model's default).",
                      file=sys.stderr)
            for i in sorted(stage1_sents.keys()):
                tokens, food_entity_tags, sentence_text = stage1_sents[i]
                cid = _custom_id(model_name, i)
                body = {
                    "model": model_name,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": build_user_prompt(tokens, food_entity_tags)},
                    ],
                }
                if openai_supports_custom_temperature(model_name):
                    body["temperature"] = temperature
                f.write(json.dumps({"custom_id": cid, "method": "POST",
                                     "url": "/v1/chat/completions", "body": body}, ensure_ascii=False) + "\n")
                manifest_items[cid] = {"sentence_index": i, "sentence_text": sentence_text,
                                        "tokens": tokens, "food_entity_tags": food_entity_tags, "model": model_name}
                n_written += 1

    if n_written == 0:
        return {"batch_id": None, "items": {}}
    uploaded = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/chat/completions",
                                   completion_window="24h")
    print(f"[openai] submitted batch {batch.id} with {n_written} requests", file=sys.stderr)
    return {"batch_id": batch.id, "items": manifest_items}


def cmd_submit(frame_tag: str, system_prompt: str, args) -> None:
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.outdir.mkdir(parents=True, exist_ok=True)
    input_stem = args.input.stem

    anthropic_models = [m for m in cfg["models"] if m["provider"] == "anthropic"]
    openai_models = [m for m in cfg["models"] if m["provider"] == "openai"]

    stage1_by_model: Dict[str, dict] = {}
    for model_cfg in cfg["models"]:
        model_name = model_cfg["name"]
        try:
            stage1_by_model[model_name] = load_stage1(args.stage1_dir, input_stem, model_name)
        except FileNotFoundError as e:
            print(f"[{model_name}] {e}", file=sys.stderr)

    manifest = {"input_stem": input_stem, "frame_tag": frame_tag, "providers": {}}
    if anthropic_models:
        result = submit_anthropic(system_prompt, stage1_by_model, anthropic_models)
        if result["batch_id"]:
            manifest["providers"]["anthropic"] = result
    if openai_models:
        result = submit_openai(system_prompt, stage1_by_model, openai_models, args.outdir, frame_tag)
        if result["batch_id"]:
            manifest["providers"]["openai"] = result

    manifest_path = args.outdir / f"{input_stem}.{frame_tag}.batch_manifest.json"
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


def _write_outputs(outdir: Path, input_stem: str, frame_tag: str, json_keys: List[str],
                    column_names: List[str], valid_tags: Dict[str, set],
                    manifest_items: Dict[str, Any], results: Dict[str, Any]) -> None:
    by_model: Dict[str, List[dict]] = {}
    for cid, meta in manifest_items.items():
        res = results.get(cid)
        if res is None:
            continue
        record = dict(meta)
        record["raw_response"] = res["raw_response"]
        if res["error"]:
            record["error"] = res["error"]
        by_model.setdefault(meta["model"], []).append(record)

    for model_name, records in by_model.items():
        records.sort(key=lambda r: r["sentence_index"])

        raw_path = outdir / f"{input_stem}.{model_name}.raw.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps({k: v for k, v in r.items()}, ensure_ascii=False) + "\n")

        bio_path = outdir / f"{input_stem}.{model_name}.{frame_tag}.bio.tsv"
        err_path = outdir / f"{input_stem}.{model_name}.{frame_tag}.errors.jsonl"
        n_flagged = 0
        with open(bio_path, "w", encoding="utf-8") as biof, open(err_path, "w", encoding="utf-8") as errf:
            biof.write("token\tFOOD_ENTITY\t" + "\t".join(column_names) + "\n")
            for r in records:
                tokens = r["tokens"]
                food_entity_tags = r["food_entity_tags"]
                problems = []
                out_tags = {k: ["O"] * len(tokens) for k in json_keys}
                if r.get("error"):
                    problems.append(f"FAILED: {r['error']}")
                else:
                    try:
                        data = extract_json(r["raw_response"])
                        for k in json_keys:
                            raw_k_tags = data.get(k, [])
                            if len(raw_k_tags) != len(tokens):
                                problems.append(f"{k} length {len(raw_k_tags)} != {len(tokens)} -- padded/truncated")
                            out_tags[k] = repair_bio(align(list(raw_k_tags), len(tokens)), valid_tags[k])
                    except Exception as e:
                        problems.append(f"FAILED to parse: {e}")
                for idx in range(len(tokens)):
                    row = [tokens[idx], food_entity_tags[idx]] + [out_tags[jk][idx] for jk in json_keys]
                    biof.write("\t".join(row) + "\n")
                biof.write("\n")
                if problems:
                    n_flagged += 1
                    errf.write(json.dumps({"sentence_index": r["sentence_index"],
                                           "sentence_text": r.get("sentence_text"), "problems": problems}) + "\n")
        print(f"[{model_name}] wrote {bio_path} ({len(records)} sentences, {n_flagged} flagged)", file=sys.stderr)


def cmd_fetch(frame_tag: str, json_keys: List[str], column_names: List[str],
              valid_tags: Dict[str, set], args) -> None:
    manifest_paths = sorted(args.outdir.glob(f"*.{frame_tag}.batch_manifest.json"))
    if not manifest_paths:
        print(f"No *.{frame_tag}.batch_manifest.json found in {args.outdir}. Run `submit` first.", file=sys.stderr)
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
                    _write_outputs(args.outdir, input_stem, frame_tag, json_keys, column_names,
                                    valid_tags, info["items"], status_result["results"])
                else:
                    all_done = False
            if all_done or not args.wait:
                break
            if deadline and time.time() > deadline:
                print("max-wait exceeded, exiting -- re-run fetch later", file=sys.stderr)
                break
            time.sleep(args.poll_interval)


def build_arg_parser(description: str):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit")
    sp.add_argument("--input", required=True, type=Path)
    sp.add_argument("--stage1-dir", required=True, type=Path)
    sp.add_argument("--config", required=True, type=Path)
    sp.add_argument("--outdir", required=True, type=Path)

    fp = sub.add_parser("fetch")
    fp.add_argument("--outdir", required=True, type=Path)
    fp.add_argument("--wait", action="store_true")
    fp.add_argument("--max-wait", type=int, default=3600)
    fp.add_argument("--poll-interval", type=int, default=30)

    return ap
