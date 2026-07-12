"""Check the type of the 'tools' field across multiple rows."""
from datasets import load_dataset

ds = load_dataset(
    "hiyouga/glaive-function-calling-v2-sharegpt",
    split="train",
    cache_dir="../data/",
)

print(f"Total rows: {len(ds)}")
print(f"Column types:")
for col in ds.column_names:
    sample = ds[0][col]
    print(f"  {col}: {type(sample).__name__}")
    if isinstance(sample, str):
        print(f"    first 100 chars: {sample[:100]}")

# Check 10 random rows
print("\n=== Tools field across 10 rows ===")
for i in [0, 1, 2, 5, 10, 50, 100, 500, 1000, 50000]:
    if i < len(ds):
        t = ds[i]["tools"]
        print(f"row {i}: type={type(t).__name__}, len={len(t) if hasattr(t, '__len__') else '?'}")
        if isinstance(t, str):
            print(f"   first 80: {t[:80]!r}")
        elif isinstance(t, list):
            print(f"   list with {len(t)} items, first item type: {type(t[0]).__name__ if t else 'empty'}")
