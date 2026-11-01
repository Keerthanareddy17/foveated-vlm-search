"""Eval metrics, splits and runner helpers (GPU-free)."""

import pytest

from foveated_search.eval.metrics import (bootstrap_ci, breakdown, by_query_type, by_size, is_correct, is_found,
                                          latency, summarize)
from foveated_search.eval.runner import apply_overrides, eval_config
from foveated_search.eval.splits import load_split


def tr(neg=False, found=False, point=None, gt=((10, 10, 20, 20),), size="medium", qt="category", calls=2, tta=0.5):
    return {"meta": {"is_negative": neg, "gt_boxes": [] if neg else [list(b) for b in gt], "size_bucket": None if neg else size,
                     "query_type": qt}, "found": found, "final_point": point, "n_calls": calls, "fixations": [[0, 0]],
            "pixels_sampled_frac": 0.2, "pixels_unique_frac": 0.1, "pixels_sent": 100, "vision_tokens": 55,
            "completion_tokens": 30, "png_bytes": 1000, "parse_errors": 0, "time_to_answer_s": tta, "wall_s": tta + 0.01}


def test_found_requires_confirmation_and_point_in_box():
    assert is_found(tr(found=True, point=[15, 15]))
    assert not is_found(tr(found=True, point=[40, 15]))          # confirmed, wrong place
    assert not is_found(tr(found=False, point=[15, 15]))         # point but not confirmed
    assert is_correct(tr(neg=True, found=False)) and not is_correct(tr(neg=True, found=True, point=[1, 1]))


def test_summarize_counts_and_rates():
    ts = [tr(found=True, point=[15, 15]), tr(found=True, point=[99, 99]), tr(), tr(neg=True), tr(neg=True, found=True, point=[0, 0])]
    s = summarize(ts, resamples=500)
    assert (s["n"], s["n_pos"], s["n_neg"]) == (5, 3, 2)
    assert s["found"] == pytest.approx(1 / 3) and s["fp_rate"] == pytest.approx(0.5)
    assert s["accuracy"] == pytest.approx(2 / 5) and s["wrong_location"] == pytest.approx(1 / 3)
    assert s["calls"] == 2 and s["vision_tokens"] == 55 and s["pixels_frac"] == pytest.approx(0.2)
    lo, hi = s["found_ci"]
    assert 0 <= lo <= s["found"] <= hi <= 1


def test_bootstrap_ci_deterministic_and_degenerate():
    v = [1.0] * 30 + [0.0] * 70
    assert bootstrap_ci(v, 2000, 0.95, 42) == bootstrap_ci(v, 2000, 0.95, 42)
    lo, hi = bootstrap_ci(v, 2000, 0.95, 42)
    assert 0.2 < lo < 0.3 < hi < 0.4
    assert bootstrap_ci([1.0] * 10, 100, 0.95, 0) == (1.0, 1.0)


def test_latency_percentiles():
    lat = latency([tr(tta=t / 100) for t in range(1, 101)])
    assert lat["n"] == 100 and lat["tta_p50"] == pytest.approx(0.505) and lat["tta_p95"] == pytest.approx(0.9505)


def test_breakdowns():
    ts = [tr(found=True, point=[15, 15], size="small"), tr(size="large", qt="supercategory"), tr(neg=True)]
    b = breakdown(ts, by_size, resamples=100)
    assert set(b) == {"small", "large", "negative"} and b["small"]["found"] == 1.0
    q = breakdown(ts, by_query_type, resamples=100)
    assert q["food"]["n"] == 1 and q["single category"]["n"] == 2


def test_splits_and_latency_subset():
    cfg = eval_config()
    dev, test, all_ = (load_split(cfg, s) for s in ["dev", "test", "all"])
    assert (len(dev), len(test), len(all_)) == (50, 450, 500)
    assert not {r["query_id"] for r in dev} & {r["query_id"] for r in test}
    lat = load_split(cfg, "latency")
    assert len(lat) == 100 and {r["query_id"] for r in lat} <= {r["query_id"] for r in test}
    assert [r["query_id"] for r in lat] == [r["query_id"] for r in load_split(cfg, "latency")]   # seeded


def test_overrides_apply_and_reject_unknown_keys():
    cfg = eval_config({"search.fixation_size": 160, "search.confirm.refine_point": False})
    assert cfg["search"]["fixation_size"] == 160 and cfg["search"]["confirm"]["refine_point"] is False
    with pytest.raises(KeyError):
        apply_overrides(cfg, {"search.fixation_sz": 1})
