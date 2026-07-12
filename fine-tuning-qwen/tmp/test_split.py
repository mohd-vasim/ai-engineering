"""Final preprocessing: split the post-observation gpt message so the
tool_calls come BEFORE the tool response, not after.

The source data has the order:
  human -> function_call -> observation -> gpt(answer)

For SFT, we need:
  human -> assistant(tool_calls=...) -> tool(...) -> assistant(answer)

So when we see a function_call -> observation sequence, we should:
  1. Emit an assistant turn with the tool_calls (no content)
  2. Emit a tool turn with the observation
  3. Emit the assistant turn with the answer (no tool_calls)
"""
import json
import uuid


def to_qwen_tool_format(example):
    convos = example["conversations"]
    new_convo = []
    # Buffer: messages accumulated before seeing a 'gpt' that closes a tool call
    pending_tool_call_msgs = []  # list of (role, content) for tool/tool_call, in order

    for msg in convos:
        role = msg["from"]
        text = msg["value"]
        if role == "human":
            new_convo.append({"role": "user", "content": text})
        elif role == "gpt":
            # Before emitting this gpt message, check if we have any pending
            # tool_call/tool messages buffered. If so, emit them as
            # assistant(tool_calls) -> tool(...) sequence BEFORE the answer.
            if pending_tool_call_msgs:
                # All pending items are tool_calls followed by tool responses
                # Group: collect all function_calls in order, then interleave with observations
                tool_calls = []  # ordered list of call dicts
                tool_responses = []  # ordered list of (call_id, content)
                call_id_to_args = {}
                for tc_msg in pending_tool_call_msgs:
                    if tc_msg["role"] == "tool_call":
                        call_id_to_args[tc_msg["call_id"]] = tc_msg["args"]
                        tool_calls.append(tc_msg["args"])
                    elif tc_msg["role"] == "tool_response":
                        tool_responses.append((tc_msg["call_id"], tc_msg["content"]))
                # Emit an assistant turn with all the tool_calls
                if tool_calls:
                    new_convo.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    })
                # Emit each tool response as a tool turn
                for call_id, content in tool_responses:
                    new_convo.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content,
                    })
                pending_tool_call_msgs = []
            # Now emit the gpt as the final assistant turn (the answer)
            new_convo.append({"role": "assistant", "content": text})
        elif role == "function_call":
            try:
                fc = json.loads(text)
                call_id = f"call_{uuid.uuid4().hex[:24]}"
                arg_str = json.dumps(fc.get("arguments", {}), ensure_ascii=False)
                tool_call_dict = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": arg_str,
                    },
                }
                pending_tool_call_msgs.append({
                    "role": "tool_call",
                    "call_id": call_id,
                    "args": tool_call_dict,
                })
            except Exception:
                pass
        elif role == "observation":
            if pending_tool_call_msgs:
                # Find the most recent tool_call to pair with this observation
                latest = None
                for item in reversed(pending_tool_call_msgs):
                    if item["role"] == "tool_call":
                        latest = item
                        break
                if latest:
                    pending_tool_call_msgs.append({
                        "role": "tool_response",
                        "call_id": latest["call_id"],
                        "content": text,
                    })
            # else: orphan observation -> skip

    # If there are still pending tool_call_msgs at the end (no following gpt),
    # emit them as best we can (probably incomplete conversation)
    if pending_tool_call_msgs:
        tool_calls = [m["args"] for m in pending_tool_call_msgs if m["role"] == "tool_call"]
        tool_responses = [(m["call_id"], m["content"]) for m in pending_tool_call_msgs if m["role"] == "tool_response"]
        if tool_calls:
            new_convo.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
        for call_id, content in tool_responses:
            new_convo.append({"role": "tool", "tool_call_id": call_id, "content": content})

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


def test_row(row_idx):
    print(f"\n{'=' * 80}\nROW {row_idx}\n{'=' * 80}")
    processed = to_qwen_tool_format(ds[row_idx])
    convos = processed["conversations"]
    tools = processed["tools"]
    print(f"  # tools: {len(tools)}")
    print(f"  # convos: {len(convos)}")
    print("\n  CONVERSATIONS:")
    for i, m in enumerate(convos):
        marker = ""
        if "tool_calls" in m:
            marker = f" [tool_calls={[tc['function']['name'] for tc in m['tool_calls']]}]"
        content = m.get("content", "")[:60]
        print(f"  [{i}] {m['role']}{marker}: {content!r}")
    text = tok.apply_chat_template(
        convos, tools=tools if tools else None,
        tokenize=False, add_generation_prompt=False,
    )
    n_tc = text.count("<tool_call>")
    n_tr = text.count("<tool_response>")
    print(f"\n  Rendered: <tool_call>={n_tc}, <tool_response>={n_tr}, total_chars={len(text)}")
    # Find the first <tool_call> block and its surrounding context
    idx = text.find("<tool_call>")
    if idx > 0:
        # Find the preceding <|im_start|>assistant
        prev_assistant = text.rfind("<|im_start|>assistant", 0, idx)
        print(f"  First <tool_call> at position {idx}, preceding <|im_start|>assistant at {prev_assistant}")
        # Show the assistant turn containing it
        next_end = text.find("<|im_end|>", idx)
        snippet = text[prev_assistant:next_end + len("<|im_end|>")]
        print(f"  Assistant turn (with 1st <tool_call>):\n    {snippet!r}")
    return text


for idx in [0, 1, 2, 3, 4, 5, 6, 7, 100, 202]:
    if idx < len(ds):
        test_row(idx)
