"""Print full rendered text for row 1 to verify tool_call format."""
import json
import sys
sys.path.insert(0, ".")
from tmp.test_fixed import to_qwen_tool_format, ds, tok

processed = to_qwen_tool_format(ds[1])
convos = processed["conversations"]
tools = processed["tools"]

print("=== TOOLS ===")
print(json.dumps(tools, indent=2)[:1000])
print()
print("=== CONVERSATIONS (parsed) ===")
print(json.dumps(convos, indent=2)[:3000])
print()
print("=== RENDERED TEXT ===")
text = tok.apply_chat_template(
    convos, tools=tools if tools else None,
    tokenize=False, add_generation_prompt=False,
)
print(text)
