"""Step 8 helpers (GPU-free): paired comparison, second-fixation stats, cached-token parsing, R7 server flags,
off-loop image encoding."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from foveated_search.config import load_config
from foveated_search.eval.metrics import paired_compare, second_fixation
from foveated_search.serving.client import parse_cached
from foveated_search.serving.launch import vllm_args


def tr(qid, neg=False, found=False, point=None, fixations=1, rejected=False, repeats=0):
    calls = [{"role": "point", "parsed": {"point": [1, 1]}}]
    if rejected:
        calls.append({"role": "confirm", "parsed": {"y": False}})
    return {"meta": {"query_id": qid, "is_negative": neg, "gt_boxes": [] if neg else [[0, 0, 10, 10]]},
            "found": found, "final_point": point, "calls": calls, "fixations": [[0, 0]] * fixations, "repeats": repeats}


def test_paired_compare_matches_queries_and_counts_flips():
    a = [tr("q1", found=True, point=[5, 5]), tr("q2"), tr("q3"), tr("q4", neg=True), tr("q9")]
    b = [tr("q3", found=True, point=[5, 5]), tr("q2", found=True, point=[5, 5]), tr("q1"), tr("q4", neg=True, found=True)]
    f = paired_compare(a, b, "found", resamples=500)
    assert f["n"] == 3 and f["wrong_to_right"] == 2 and f["right_to_wrong"] == 1   # q9 (only in a) ignored
    assert f["diff"] == pytest.approx(1 / 3) and f["diff_ci"][0] <= f["diff"] <= f["diff_ci"][1]
    acc = paired_compare(a, b, "accuracy", resamples=500)
    assert acc["n"] == 4 and acc["right_to_wrong"] == 2 and acc["wrong_to_right"] == 2   # q4 negative turned FP
    same = paired_compare(a, a, "accuracy", resamples=200)
    assert same["diff"] == 0 and same["diff_ci"] == (0.0, 0.0)


def test_second_fixation_stats():
    ts = [tr("a", rejected=True, fixations=2, found=True, point=[5, 5]), tr("b", rejected=True, fixations=1),
          tr("c", neg=True, rejected=True, fixations=2), tr("d", fixations=3, repeats=2), tr("e")]
    x = second_fixation(ts)
    assert (x["with_rejected_crop"], x["rejected_on_positives"], x["took_second_fixation"]) == (3, 2, 2)
    assert (x["second_fixation_on_positives"], x["found_after_second_fixation"]) == (1, 1)
    assert x["false_positive_after_second_fixation"] == 0 and x["third_fixation"] == 1 and x["repeats"] == 2


def test_parse_cached_tokens():
    assert parse_cached(SimpleNamespace(prompt_tokens_details=SimpleNamespace(cached_tokens=48))) == 48
    assert parse_cached({"prompt_tokens_details": {"cached_tokens": 0}}) == 0
    assert parse_cached(SimpleNamespace(prompt_tokens_details=None)) is None and parse_cached(None) is None


def test_r7_overlay_flags_only_on_r7():
    base = vllm_args(load_config())
    r6 = vllm_args(load_config(overlays=["configs/serve_finetuned_fp8.yaml"]))
    r7 = vllm_args(load_config(overlays=["configs/serve_finetuned_fp8.yaml", "configs/serve_finetuned_fp8_prefix.yaml"]))
    eager = vllm_args(load_config(overlays=["configs/serve_finetuned_fp8.yaml", "configs/serve_finetuned_fp8_eager.yaml"]))
    assert "--no-enable-prefix-caching" in base and "--no-enable-prefix-caching" in r6
    assert "--enable-prefix-caching" in r7 and "--enable-prompt-tokens-details" in r7
    assert r7[r7.index("--mm-processor-cache-gb") + 1] == "4" and r6[r6.index("--mm-processor-cache-gb") + 1] == "0"
    assert "--enable-prompt-tokens-details" not in r6 and "--enforce-eager" in eager and "--enforce-eager" not in r6
    assert r6[1].endswith("qwen3-vl-2b-foveated-search-fp8")
    # everything except the model path is identical between the BF16 and FP8 servers (same server config)
    assert [x for x in base if "Qwen" not in x] == [x for x in r6 if "foveated-search-fp8" not in x]


def test_aencode_matches_encode():
    from foveated_search.serving.client import VLMClient

    c = VLMClient(load_config(), counter=False)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8), "RGB")
    assert asyncio.run(c.aencode(img)).png == c.encode(img).png
