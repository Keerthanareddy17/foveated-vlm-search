"""Metrics over traces (one dict per query, as written by the runner).

Definitions (CLAUDE.md): a positive is FOUND when the strategy confirmed it AND its final point lies
inside any ground-truth box. A negative is correct when the search ends without a confirmed point.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from foveated_search.search.geometry import in_any_xywh


def is_found(trace: dict) -> bool:
    """Positive found: decision found AND final point inside a gt box."""
    m = trace["meta"]
    return bool(trace["found"] and trace["final_point"] and in_any_xywh(*trace["final_point"], m["gt_boxes"]))


def is_correct(trace: dict) -> bool:
    return (not trace["found"]) if trace["meta"]["is_negative"] else is_found(trace)


def bootstrap_ci(values: list[float], resamples: int, level: float, seed: int) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed."""
    if not values:
        return (float("nan"), float("nan"))
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(resamples, len(a)))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def _mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def summarize(traces: list[dict], resamples: int = 2000, level: float = 0.95, seed: int = 42) -> dict:
    pos = [t for t in traces if not t["meta"]["is_negative"]]
    neg = [t for t in traces if t["meta"]["is_negative"]]
    found = [float(is_found(t)) for t in pos]
    fp = [float(t["found"]) for t in neg]
    correct = [float(is_correct(t)) for t in traces]
    out = {
        "n": len(traces), "n_pos": len(pos), "n_neg": len(neg),
        "found": _mean(found), "found_ci": bootstrap_ci(found, resamples, level, seed),
        "fp_rate": _mean(fp), "fp_ci": bootstrap_ci(fp, resamples, level, seed),
        "accuracy": _mean(correct), "accuracy_ci": bootstrap_ci(correct, resamples, level, seed),
        # positives the strategy said "found" but with the point outside every gt box
        "wrong_location": _mean(float(t["found"] and not is_found(t)) for t in pos),
        "calls": _mean(t["n_calls"] for t in traces),
        "fixations": _mean(len(t["fixations"]) for t in traces),
        "pixels_frac": _mean(t["pixels_sampled_frac"] for t in traces),
        "pixels_unique_frac": _mean(t["pixels_unique_frac"] for t in traces),
        "pixels": _mean(t["pixels_sent"] for t in traces),
        "vision_tokens": _mean(t["vision_tokens"] for t in traces),
        "completion_tokens": _mean(t["completion_tokens"] for t in traces),
        "png_bytes": _mean(t["png_bytes"] for t in traces),
        "parse_errors": int(sum(t["parse_errors"] for t in traces)),
    }
    return out


def latency(traces: list[dict]) -> dict:
    """p50 / p95 of time-to-answer (sum of call times) and wall clock, in seconds."""
    tta = [t["time_to_answer_s"] for t in traces]
    wall = [t["wall_s"] for t in traces]
    if not tta:
        return {"n": 0}
    return {"n": len(tta), "tta_p50": float(np.percentile(tta, 50)), "tta_p95": float(np.percentile(tta, 95)),
            "tta_mean": float(np.mean(tta)), "wall_p50": float(np.percentile(wall, 50)),
            "wall_p95": float(np.percentile(wall, 95))}


def breakdown(traces: list[dict], key, resamples: int = 2000, level: float = 0.95, seed: int = 42) -> dict:
    """summarize() per group; key(trace) -> group name."""
    groups: dict = defaultdict(list)
    for t in traces:
        groups[key(t)].append(t)
    return {g: summarize(ts, resamples, level, seed) for g, ts in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def by_size(t: dict) -> str:
    return "negative" if t["meta"]["is_negative"] else t["meta"]["size_bucket"]


def by_query_type(t: dict) -> str:
    return "food" if t["meta"]["query_type"] == "supercategory" else "single category"


def by_polarity(t: dict) -> str:
    return "negative" if t["meta"]["is_negative"] else "positive"



def paired_compare(a: list[dict], b: list[dict], metric: str, resamples: int = 2000, level: float = 0.95,
                   seed: int = 42) -> dict:
    """b minus a on the SAME queries (matched by query_id). metric: "found" (positives only) or "accuracy".
    Paired percentile bootstrap over queries; flip counts wrong->right and right->wrong."""
    am = {t["meta"]["query_id"]: t for t in a}
    bm = {t["meta"]["query_id"]: t for t in b}
    ids = sorted(set(am) & set(bm))
    if metric == "found":
        ids = [i for i in ids if not am[i]["meta"]["is_negative"]]
        fa, fb = [float(is_found(am[i])) for i in ids], [float(is_found(bm[i])) for i in ids]
    elif metric == "accuracy":
        fa, fb = [float(is_correct(am[i])) for i in ids], [float(is_correct(bm[i])) for i in ids]
    else:
        raise ValueError(metric)
    xa, xb = np.asarray(fa), np.asarray(fb)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(resamples, len(ids)))
    diffs = xb[idx].mean(axis=1) - xa[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return {"n": len(ids), "a": float(xa.mean()), "b": float(xb.mean()), "diff": float(xb.mean() - xa.mean()),
            "diff_ci": (float(lo), float(hi)), "wrong_to_right": int(((xa == 0) & (xb == 1)).sum()),
            "right_to_wrong": int(((xa == 1) & (xb == 0)).sum())}


def second_fixation(traces: list[dict]) -> dict:
    """Searches with a rejected crop; how many then took another fixation; how many of those found the target."""
    rej = [t for t in traces if any(c["role"] == "confirm" and c.get("parsed") and not c["parsed"]["y"] for c in t["calls"])]
    second = [t for t in rej if len(t["fixations"]) >= 2]
    pos_second = [t for t in second if not t["meta"]["is_negative"]]
    return {"searches": len(traces), "with_rejected_crop": len(rej),
            "rejected_on_positives": sum(not t["meta"]["is_negative"] for t in rej),
            "took_second_fixation": len(second),
            "second_fixation_on_positives": len(pos_second),
            "found_after_second_fixation": sum(is_found(t) for t in pos_second),
            "false_positive_after_second_fixation": sum(t["found"] for t in second if t["meta"]["is_negative"]),
            "third_fixation": sum(len(t["fixations"]) >= 3 for t in traces), "repeats": sum(t["repeats"] for t in traces)}
