"""Trace through preprocessing for row 1 to see why tool_calls isn't attached."""
import json
import uuid

from datasets import load_dataset

ds = load_dataset(
    "hiyouga/glaive-function-calling-v2-sharegpt",
    split="train",
    cache_dir="../data/",
)

# Show raw structure of row 1
print("=== ROW 1 RAW ===")
for i, msg in enumerate(ds[1]["conversations"]):
    print(f"[{i}] from={msg['from']!r}")
    print(f"    value (first 200 chars): {msg['value'][:200]}")
    print()

# Check what tools field looks like
print("=== ROW 1 TOOLS ===")
print(f"Type: {type(ds[1].get('tools'))}")
print(f"Value: {ds[1].get('tools')}")
