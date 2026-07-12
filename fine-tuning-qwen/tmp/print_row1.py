"""Print full rendered text for row 1 to verify final structure."""
import json
import sys
sys.path.insert(0, ".")
from tmp.test_split import to_qwen_tool_format, ds, tok

processed = to_qwen_tool_format(ds[1])
convos = processed["conversations"]
tools = processed["tools"]

text = tok.apply_chat_template(
    convos, tools=tools if tools else None,
    tokenize=False, add_generation_prompt=False,
)
print(text)
