"""Smallest JSON schemas that keep grounding quality, and strict parsers for them.

Output tokens dominate per-call latency (Step 2: ~14.6 ms each at BF16), so answers are short and
"no"/"absent" branches are cheap (anyOf). One measured exception to "shortest keys": points use
Qwen3-VL's trained key `point_2d`. A one-letter key costs ~3 fewer tokens but full-frame hit rate
drops from 92.5% to 42.5% with points biased up-left (Step 4, results/step4_point_format.json).
  point   : {"point_2d":[x,y]} | {"absent":true}             x, y in 0..1000 on the image sent
  confirm : {"y":false} | {"y":true[,"point_2d":[x,y]][,"l":"label"]}
"""

from __future__ import annotations

from dataclasses import dataclass

from foveated_search.search.parsing import ParseError, loads_strict

_POINT = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000}, "minItems": 2, "maxItems": 2}


POINT_KEY = "point_2d"   # Qwen3-VL's native grounding key (see module docstring)


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def point_schema() -> dict:
    return {"anyOf": [_obj({POINT_KEY: _POINT}), _obj({"absent": {"const": True}})]}


def confirm_schema(refine_point: bool, label: bool, label_max_chars: int) -> dict:
    yes_props: dict = {"y": {"const": True}}
    if refine_point:
        yes_props[POINT_KEY] = _POINT
    if label:
        yes_props["l"] = {"type": "string", "minLength": 1, "maxLength": int(label_max_chars)}
    return {"anyOf": [_obj({"y": {"const": False}}), _obj(yes_props)]}


def structured_body(schema: dict) -> dict:
    """extra_body for the OpenAI client: vLLM per-request structured outputs. Whitespace inside the
    JSON is controlled server-side (server.structured_disable_any_whitespace): the per-request
    `disable_any_whitespace` field is ignored by vLLM 0.31 (Step 4)."""
    return {"structured_outputs": {"json": schema}}


@dataclass
class ConfirmAnswer:
    yes: bool
    point: tuple[float, float] | None = None
    label: str | None = None


def _coords(v) -> tuple[float, float]:
    if isinstance(v, list) and len(v) == 2 and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v):
        return float(v[0]), float(v[1])
    raise ParseError(f"bad point {v!r}")


def parse_p(text: str) -> tuple[float, float] | None:
    """Point answer {"point_2d":[x,y]} -> (x, y); {"absent":true} -> None. Raises ParseError otherwise."""
    obj = loads_strict(text)
    if isinstance(obj, dict) and set(obj) == {POINT_KEY}:
        return _coords(obj[POINT_KEY])
    if isinstance(obj, dict) and obj == {"absent": True}:
        return None
    raise ParseError(f"expected {{'{POINT_KEY}': [x, y]}} or {{'absent': true}}: {text[:80]!r}")


def parse_confirm(text: str) -> ConfirmAnswer:
    obj = loads_strict(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("y"), bool):
        raise ParseError(f"expected boolean 'y': {text[:80]!r}")
    if not obj["y"]:
        return ConfirmAnswer(False)
    point = _coords(obj[POINT_KEY]) if POINT_KEY in obj else None
    label = obj.get("l")
    if label is not None and not isinstance(label, str):
        raise ParseError(f"bad label {label!r}")
    return ConfirmAnswer(True, point, label)
