"""End-to-end test: dataset.map (preprocess) -> dataset.map (format)."""
import json
import sys

sys.path.insert(0, ".")

import uuid


def to_qwen_tool_format(example):
    convos = example["conversations"]
    new_convo = []
    pending_tool_call_msgs = []
    for msg in convos:
        role = msg["from"]
        text = msg["value"]
        if role == "human":
            new_convo.append({"role": "user", "content": text})
        elif role == "gpt":
            if pending_tool_call_msgs:
                tool_calls = [m["args"] for m in pending_tool_call_msgs if m["role"] == "tool_call"]
                tool_responses = [
                    (m["call_id"], m["content"])
                    for m in pending_tool_call_msgs
                    if m["role"] == "tool_response"
                ]
                if tool_calls:
                    new_convo.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                for call_id, content in tool_responses:
                    new_convo.append({"role": "tool", "tool_call_id": call_id, "content": content})
                pending_tool_call_msgs = []
            new_convo.append({"role": "assistant", "content": text})
        elif role == "function_call":
            try:
                fc = json.loads(text)
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                arg_str = json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                pending_tool_call_msgs.append({
                    "role": "tool_call",
                    "call_id": call_id,
                    "args": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": fc.get("name", ""), "arguments": arg_str},
                    },
                })
            except Exception:
                pass
        elif role == "observation":
            if pending_tool_call_msgs:
                latest = None
                for item in reversed(pending_tool_call_msgs):
                    if item["role"] == "tool_call":
                        latest = item
                        break
                if latest:
                    pending_tool_call_msgs.append({"role": "tool_response", "call_id": latest["call_id"], "content": text})
    if pending_tool_call_msgs:
        tool_calls = [m["args"] for m in pending_tool_call_msgs if m["role"] == "tool_call"]
        tool_responses = [(m["call_id"], m["content"]) for m in pending_tool_call_msgs if m["role"] == "tool_response"]
        if tool_calls:
            new_convo.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for call_id, content in tool_responses:
            new_convo.append({"role": "tool", "tool_call_id": call_id, "content": content})
    raw_tools_str = example.get("tools", "") or ""
    return {"conversations": new_convo, "tools": raw_tools_str}


def _parse_tools(tools_str):
    if not tools_str or not isinstance(tools_str, str):
        return []
    s = tools_str.strip()
    if not s or s == "[]":
        return []
    try:
        parsed = json.loads(s)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out = []
    for t in parsed:
        if not isinstance(t, dict):
            continue
        if "function" in t and isinstance(t["function"], dict):
            if "type" not in t:
                t["type"] = "function"
            out.append(t)
        elif "name" in t:
            out.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
            })
    return out


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    tools_batch = [_parse_tools(t) for t in examples.get("tools", [""] * len(convos))]
    texts = [
        tok.apply_chat_template(
            convo, tools=(tools if tools else None),
            tokenize=False, add_generation_prompt=False,
        )
        for convo, tools in zip(convos, tools_batch)
    ]
    return {"text": texts}


from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template

ds = load_dataset(
    "hiyouga/glaive-function-calling-v2-sharegpt",
    split="train",
    cache_dir="../data/",
)

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit")
tok = get_chat_template(tok, chat_template="qwen3-instruct")
print("Tokenizer loaded.")

print(f"\nOriginal dataset: {len(ds)} rows, columns: {ds.column_names}")
print("Running preprocess map...")
ds = ds.map(
    to_qwen_tool_format,
    remove_columns=[c for c in ds.column_names if c != "tools"],
)
print(f"After preprocess: {len(ds)} rows, columns: {ds.column_names}")
print(f"Sample row 0 tools type: {type(ds[0]['tools']).__name__}, len: {len(ds[0]['tools'])}")

print("\nRunning format map...")
ds = ds.map(formatting_prompts_func, batched=True)
print(f"After format: {len(ds)} rows, columns: {ds.column_names}")
print(f"Sample row 1 text length: {len(ds[1]['text'])}")
print(f"Sample row 1 text first 500 chars:\n{ds[1]['text'][:500]}")

# Count tool_call tokens
n_tc = sum(t.count("<tool_call>") for t in ds[i]["text"] for i in range(min(10, len(ds))))
print(f"\nTotal <tool_call> in first 10 rows: {n_tc}")
print("\nSUCCESS!")
