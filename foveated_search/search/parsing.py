"""Strict JSON parsing of model answers (Step 2). Step 4 schemas/parsers live in schemas.py."""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


# vLLM structured output (OpenAI response_format) for the point answer. Chosen in Step 2:
# 100% parse rate vs 75% for JSON-by-prompt, same TTFT (NOTES.md "Output format").
POINT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "point_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "point": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                         "minItems": 2, "maxItems": 2},
                        {"type": "null"},
                    ]
                }
            },
            "required": ["point"],
            "additionalProperties": False,
        },
    },
}


class ParseError(ValueError):
    pass


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    keys = [k for k, _ in pairs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        # Seen in Step 2: the model echoes both template options, {"point": [x, y], "point": null}.
        # json.loads would silently keep the last one; treat it as ambiguous instead.
        raise ParseError(f"duplicate keys {sorted(dup)} (ambiguous answer)")
    return dict(pairs)


def _json_candidates(text: str) -> list[str]:
    t = text.strip()
    cands = [m.group(1).strip() for m in _FENCE.finditer(t)]
    cands.append(t)
    # first {...} block anywhere in the text
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        cands.append(t[i : j + 1])
    return cands


def loads_strict(text: str):
    """Parse the first JSON value found in the answer (fenced, bare, or embedded); duplicate keys
    raise ParseError (ambiguous). Raises ParseError if nothing parses."""
    for cand in _json_candidates(text):
        try:
            return json.loads(cand, object_pairs_hook=_no_duplicate_keys)
        except json.JSONDecodeError:
            continue
    raise ParseError(f"no JSON in answer: {text[:120]!r}")


def parse_point(text: str) -> tuple[float, float] | None:
    """Parse {"point": [x, y]} / {"point": null}. Also accepts Qwen's native "point_2d" key
    and a one-element list of such objects. Returns (x, y) in model coords, None = not visible.
    Raises ParseError if the answer is not in an accepted format.
    """
    for cand in _json_candidates(text):
        try:
            obj = json.loads(cand, object_pairs_hook=_no_duplicate_keys)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        if not isinstance(obj, dict):
            continue
        key = "point" if "point" in obj else "point_2d" if "point_2d" in obj else None
        if key is None:
            continue
        p = obj[key]
        if p is None:
            return None
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p):
            return float(p[0]), float(p[1])
        raise ParseError(f"bad point value: {p!r}")
    raise ParseError(f"no point JSON in answer: {text[:120]!r}")
