"""Debug script: run preprocessing + chat template on multiple rows to
identify whether <tool_call> blocks are being correctly attached & rendered.

Run from project root with: .venv/bin/python tmp/debug_tool_call.py
"""
import json
import sys
import uuid

sys.path.insert(0, "/Users/vasim/Programming/ai-engineering/fine-tuning-qwen")


def to_qwen_tool_format(example):
    convos = example["conversations"]
    new_convo = []
    pending_call_ids = []
    pending_call_args = []
    for msg in convos:
        role = msg["from"]
        text = msg["value"]
        if role == "human":
            new_convo.append({"role": "user", "content": text})
        elif role == "gpt":
            entry = {"role": "assistant", "content": text}
            if pending_call_ids:
                entry["tool_calls"] = pending_call_args
                pending_call_ids = []
                pending_call_args = []
            new_convo.append(entry)
        elif role == "function_call":
            try:
                fc = json.loads(text)
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                arg_str = json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                pending_call_args = [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fc.get("name", ""), "arguments": arg_str},
                }]
                pending_call_ids.append(call_id)
            except Exception:
                pass
        elif role == "observation":
            if pending_call_ids:
                call_id = pending_call_ids.pop(0)
                pending_call_args.pop(0)
                new_convo.append({"role": "tool", "tool_call_id": call_id, "content": text})

    raw_tools = example.get("tools", []) or []
    tools = []
    for t in raw_tools:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:
                continue
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            if "type" not in t:
                t["type"] = "function"
            tools.append(t)
        elif "name" in t:
            tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
    return {"conversations": new_convo, "tools": tools}


from datasets import load_dataset

ds = load_dataset(
    "hiyouga/glaive-function-calling-v2-sharegpt",
    split="train",
    cache_dir="../data/",
)

print("=" * 80)
print("Scanning rows for tool_calls attachment + rendering")
print("=" * 80)

from transformers import AutoTokenizer
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit")
from unsloth.chat_templates import get_chat_template
tok = get_chat_template(tok, chat_template="qwen3-instruct")
print("Tokenizer loaded.\n")

found_tool_call = 0
for row_idx in [0, 1, 2, 3, 4, 5, 6, 7, 100, 202, 500, 1000, 5000]:
    if row_idx >= len(ds):
        continue
    raw = ds[row_idx]
    processed = to_qwen_tool_format(raw)
    convos = processed["conversations"]
    tools = processed["tools"]

    has_tool_calls = any("tool_calls" in m for m in convos if m.get("role") == "assistant")
    if not has_tool_calls:
        continue

    found_tool_call += 1
    print(f"\n--- ROW {row_idx} ---")
    print(f"  # tools: {len(tools)}")
    print(f"  # convos: {len(convos)}")
    for i, m in enumerate(convos):
        if "tool_calls" in m:
            print(f"  [turn {i}] assistant tool_calls: {m['tool_calls']}")
            print(f"           content: {m.get('content','')[:80]}")

    try:
        text = tok.apply_chat_template(
            convos, tools=tools if tools else None,
            tokenize=False, add_generation_prompt=False,
        )
        has_tool_call_token = "<tool_call>" in text
        print(f"  Rendered contains <tool_call>: {has_tool_call_token}")
        if not has_tool_call_token:
            print(f"  Rendered (first 800 chars):\n{text[:800]}")
    except Exception as e:
        print(f"  RENDER ERROR: {type(e).__name__}: {e}")

    if found_tool_call >= 4:
        break

print(f"\nScanned {found_tool_call} rows with tool_calls attached.")
