import json
import re

from langchain_ollama import ChatOllama


def get_llm(
    model: str = "qwen3.5:4b-mlx",
    temperature: float = 0.0,
    **kwargs,
) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        format="json",
        reasoning=False,
        **kwargs,
    )


def clean_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not extract JSON from: {raw[:200]!r}")
