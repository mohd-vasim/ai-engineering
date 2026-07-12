"""Trace through preprocessing for row 1 step by step."""
import json
import uuid

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
            except Exception as e:
                print(f"  ! function_call parse error: {e}")
        elif role == "observation":
            if pending_call_ids:
                call_id = pending_call_ids.pop(0)
                pending_call_args.pop(0)
                new_convo.append({"role": "tool", "tool_call_id": call_id, "content": text})

    raw_tools = example.get("tools", []) or []
    print(f"\n  tools field type: {type(raw_tools).__name__}, value: {str(raw_tools)[:150]}")

    tools = []
    for t in raw_tools:
        print(f"  iterating tool: type={type(t).__name__}, value={str(t)[:100]}")
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception as e:
                print(f"    ! json.loads error: {e}")
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
    print(f"  -> final tools: {tools}")
    return {"conversations": new_convo, "tools": tools}


from datasets import load_dataset

ds = load_dataset(
    "hiyouga/glaive-function-calling-v2-sharegpt",
    split="train",
    cache_dir="../data/",
)

print("=== Row 1 PREPROCESSED ===")
result = to_qwen_tool_format(ds[1])
print()
print("=== Conversations (truncated) ===")
for i, m in enumerate(result["conversations"]):
    print(f"[{i}] role={m['role']}, content={m.get('content','')[:60]!r}")
    if "tool_calls" in m:
        print(f"    tool_calls={m['tool_calls']}")
