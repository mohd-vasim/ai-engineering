import json, sys, uuid
sys.path.insert(0, ".")

def to_qwen_tool_format(example):
    convos = example["conversations"]
    new_convo = []
    pending_tool_call_msgs = []
    for msg in convos:
        role = msg["from"]; text = msg["value"]
        if role == "human":
            new_convo.append({"role": "user", "content": text})
        elif role == "gpt":
            if pending_tool_call_msgs:
                tool_calls = [m["args"] for m in pending_tool_call_msgs if m["role"] == "tool_call"]
                tool_responses = [(m["call_id"], m["content"]) for m in pending_tool_call_msgs if m["role"] == "tool_response"]
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
                pending_tool_call_msgs.append({"role": "tool_call", "call_id": call_id, "args": {"id": call_id, "type": "function", "function": {"name": fc.get("name", ""), "arguments": arg_str}}})
            except Exception: pass
        elif role == "observation":
            if pending_tool_call_msgs:
                latest = None
                for item in reversed(pending_tool_call_msgs):
                    if item["role"] == "tool_call": latest = item; break
                if latest:
                    pending_tool_call_msgs.append({"role": "tool_response", "call_id": latest["call_id"], "content": text})
    if pending_tool_call_msgs:
        tool_calls = [m["args"] for m in pending_tool_call_msgs if m["role"] == "tool_call"]
        tool_responses = [(m["call_id"], m["content"]) for m in pending_tool_call_msgs if m["role"] == "tool_response"]
        if tool_calls: new_convo.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for call_id, content in tool_responses:
            new_convo.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return {"conversations": new_convo, "tools": example.get("tools", "") or ""}


def _parse_tools(tools_str):
    if not tools_str or not isinstance(tools_str, str): return []
    s = tools_str.strip()
    if not s or s == "[]": return []
    try: parsed = json.loads(s)
    except Exception: return []
    if not isinstance(parsed, list): return []
    out = []
    for t in parsed:
        if not isinstance(t, dict): continue
        if "function" in t and isinstance(t["function"], dict):
            if "type" not in t: t["type"] = "function"
            out.append(t)
        elif "name" in t:
            out.append({"type": "function", "function": {"name": t.get("name", ""), "description": t.get("description", ""), "parameters": t.get("parameters", {"type": "object", "properties": {}})}})
    return out


def formatting_prompts_func(examples):
    convos = examples["conversations"]
    tools_batch = [_parse_tools(t) for t in examples.get("tools", [""] * len(convos))]
    texts = [tok.apply_chat_template(convo, tools=(tools if tools else None), tokenize=False, add_generation_prompt=False) for convo, tools in zip(convos, tools_batch)]
    return {"text": texts}


from datasets import load_dataset
from transformers import AutoTokenizer
from unsloth.chat_templates import get_chat_template

ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train", cache_dir="../data/")
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-1.5B-unsloth-bnb-4bit")
tok = get_chat_template(tok, chat_template="qwen3-instruct")
print(f"Original: {len(ds)} rows")
ds = ds.map(to_qwen_tool_format, remove_columns=[c for c in ds.column_names if c != "tools"])
print(f"After preprocess: {len(ds)} rows, cols: {ds.column_names}")
ds = ds.map(formatting_prompts_func, batched=True)
print(f"After format: {len(ds)} rows, cols: {ds.column_names}")

# Stats
n_with_tc = 0
n_with_tr = 0
total_tc = 0
for i in range(min(1000, len(ds))):
    t = ds[i]["text"]
    if "<tool_call>" in t: n_with_tc += 1
    if "<tool_response>" in t: n_with_tr += 1
    total_tc += t.count("<tool_call>")
print(f"\nIn first 1000 rows: {n_with_tc} have <tool_call>, {n_with_tr} have <tool_response>")
print(f"Total <tool_call> count: {total_tc}")

# Print row 1
print("\n=== ROW 1 FULL TEXT ===")
print(ds[1]["text"][:2500])
