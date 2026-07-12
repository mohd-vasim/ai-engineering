"""Test fixed preprocessing: attach tool_calls to assistant, keep tool message
in observation, but DON'T pop pending_call_args on observation."""
import json
import uuid


def to_qwen_tool_format(example):
    convos = example["conversations"]
    new_convo = []
    pending_call_ids = []   # list of tool_call_ids awaiting attachment
    pending_call_args = []  # parallel list of tool_call dicts
    for msg in convos:
        role = msg["from"]
        text = msg["value"]
        if role == "human":
            new_convo.append({"role": "user", "content": text})
        elif role == "gpt":
            entry = {"role": "assistant", "content": text}
            if pending_call_ids:
                # Attach ALL pending tool_calls to this assistant message
                entry["tool_calls"] = pending_call_args
                pending_call_ids = []
                pending_call_args = []
            new_convo.append(entry)
        elif role == "function_call":
            try:
                fc = json.loads(text)
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                arg_str = json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                pending_call_args.append({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": fc.get("name", ""), "arguments": arg_str},
                })
                pending_call_ids.append(call_id)
            except Exception:
                pass
        elif role == "observation":
            # Pair this observation with the EARLIEST pending call (FIFO)
            if pending_call_ids:
                call_id = pending_call_ids[0]  # peek, don't pop
                new_convo.append({"role": "tool", "tool_call_id": call_id, "content": text})
                # Do NOT pop - the tool_calls are still attached to the next gpt message
            # else: orphan observation -> skip

    # Tools: parse the JSON string
    raw_tools_str = example.get("tools", "") or ""
    tools = []
    if isinstance(raw_tools_str, str) and raw_tools_str.strip():
        try:
            parsed = json.loads(raw_tools_str)
            if isinstance(parsed, list):
                for t in parsed:
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
        except Exception:
            pass

    return {"conversations": new_convo, "tools": tools}


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
print("Tokenizer loaded.\n")

# Test on row 1
print("=" * 80)
print("ROW 1 — full output")
print("=" * 80)
processed = to_qwen_tool_format(ds[1])
convos = processed["conversations"]
tools = processed["tools"]

print("\n=== CONVERSATIONS ===")
for i, m in enumerate(convos):
    has_tc = " [HAS tool_calls]" if "tool_calls" in m else ""
    print(f"[{i}] {m['role']}{has_tc}: {m.get('content','')[:80]!r}")
    if "tool_calls" in m:
        for tc in m["tool_calls"]:
            print(f"    tool_call: {tc['function']['name']}({tc['function']['arguments']})")

print("\n=== RENDERED TEXT (first 3000 chars) ===")
text = tok.apply_chat_template(
    convos, tools=tools if tools else None,
    tokenize=False, add_generation_prompt=False,
)
print(text[:3000])
print(f"\n... [truncated; total length: {len(text)} chars]")

# Count tool_call tokens
import re
n_tool_call_open = text.count("<tool_call>")
n_tool_call_close = text.count("</tool_call>")
n_tool_response_open = text.count("<tool_response>")
print(f"\nCount: <tool_call>={n_tool_call_open}, </tool_call>={n_tool_call_close}, <tool_response>={n_tool_response_open}")
