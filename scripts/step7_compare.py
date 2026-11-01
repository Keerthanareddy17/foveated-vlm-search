#!/usr/bin/env python
"""Step 7 checks: base vs fine-tuned, side by side. Writes results/step7_checks.md and .json.

(b/c) val samples: results/step6_base_on_val.json vs results/step7_finetuned_on_val.json, per type and polarity,
      with an explicit flag when accuracy on negatives drops by more than NEG_DROP_FLAG.
(d)   dev eye search: dev_eye_plain (base, same settings as R4) vs dev_eye_finetuned.
(e)   dev full frame (forgetting): R1_full_frame filtered to dev vs dev_full_frame_finetuned.
"""

from __future__ import annotations

import json

from foveated_search.config import REPO_ROOT
from foveated_search.eval.metrics import summarize
from foveated_search.eval.runner import eval_config
from foveated_search.eval.splits import split_ids
from foveated_search.eval.table import load_run

NEG_DROP_FLAG = 0.05


def rate(rows, cond) -> tuple[float, int]:
    rs = [r for r in rows if cond(r)]
    return (sum(r["ok"] for r in rs) / len(rs) if rs else float("nan")), len(rs)


def eye_extra(traces: list[dict]) -> dict:
    rej = [t for t in traces if any(c["role"] == "confirm" and c["parsed"] and not c["parsed"]["y"] for c in t["calls"])]
    second = [t for t in rej if len(t["fixations"]) >= 2]
    return {"searches_with_rejected_crop": len(rej), "took_second_fixation_after_rejection": len(second),
            "second_fixation_rate": (len(second) / len(rej)) if rej else float("nan"),
            "repeats": sum(t["repeats"] for t in traces),
            "max_fixations_used": max((len(t["fixations"]) for t in traces), default=0)}


def pct(v: float) -> str:
    return "n/a" if v != v else f"{100 * v:.1f}"


def main() -> None:
    cfg = eval_config()
    base = json.loads((REPO_ROOT / "results/step6_base_on_val.json").read_text())
    ft = json.loads((REPO_ROOT / "results/step7_finetuned_on_val.json").read_text())
    assert [r["id"] for r in base["rows"]] == [r["id"] for r in ft["rows"]], "different val samples"
    L = ["# Step 7: base vs fine-tuned checks", "",
         f"Base: {base['model']} ({base['precision']}). Fine-tuned: {ft['model']} ({ft['precision']}). "
         "Same val samples, prompts, schemas and parsers (Step 6 scorer).", "",
         "## Val samples (300 per type; success %)", "",
         "| skill | subset | n | base | fine-tuned | change |", "|---|---|---|---|---|---|"]
    out = {"val": {}, "flags": []}
    subsets = [("where (first glance)", "where", [("all", None), ("positives", "pos"), ("negatives (absent)", "neg")]),
               ("look again", "look_again", [("all", None), ("positives", "pos"), ("negatives (absent)", "neg")]),
               ("confirm", "confirm", [("all", None), ("yes crops", "yes"), ("no crops", "no")])]
    for label, t, parts in subsets:
        for name, pol in parts:
            cond = (lambda r, t=t, pol=pol: r["type"] == t and (pol is None or r["polarity"] == pol))
            b, n = rate(base["rows"], cond)
            f, _ = rate(ft["rows"], cond)
            out["val"][f"{t}/{name}"] = {"n": n, "base": b, "finetuned": f}
            L.append(f"| {label} | {name} | {n} | {pct(b)} | {pct(f)} | {100 * (f - b):+.1f} pp |")
            if pol in ("neg", "no") and f < b - NEG_DROP_FLAG:
                out["flags"].append(f"{t} {name}: {pct(b)}% -> {pct(f)}% (drop > {100 * NEG_DROP_FLAG:.0f} pp)")
    for t in ["confirm"]:
        for k in ["near_miss", "distractor", "background"]:
            cond = lambda r, k=k: r["type"] == "confirm" and r["no_kind"] == k  # noqa: E731
            b, n = rate(base["rows"], cond)
            f, _ = rate(ft["rows"], cond)
            L.append(f"| confirm | no: {k} | {n} | {pct(b)} | {pct(f)} | {100 * (f - b):+.1f} pp |")
    L += ["", "**Negatives check:** " + ("; ".join(out["flags"]) if out["flags"] else
                                          f"no negative subset dropped by more than {100 * NEG_DROP_FLAG:.0f} pp."), ""]

    dev = split_ids(cfg, "dev")
    def dev_traces(run):
        r = load_run(run)
        return [t for t in r[0] if t["meta"]["query_id"] in dev] if r else None

    pairs = {"eye search (dev, 50)": ("dev_eye_plain", "dev_eye_finetuned"),
             "full frame (dev, 50)": ("R1_full_frame", "dev_full_frame_finetuned")}
    L += ["## Dev split sanity runs (accuracy runs, concurrency 8; R5 on test is Step 8)", "",
          "| strategy | model | found % | FP % (neg) | accuracy % | wrong loc. % | calls / query | fixations / query | "
          "searches with a rejected crop | took a 2nd fixation after it |", "|---|---|---|---|---|---|---|---|---|---|"]
    out["dev"] = {}
    for label, (b_run, f_run) in pairs.items():
        for model, run in [("base", b_run), ("fine-tuned", f_run)]:
            tr = dev_traces(run)
            if not tr:
                L.append(f"| {label} | {model} | missing run {run} | | | | | | | |")
                continue
            s = summarize(tr)
            x = eye_extra(tr) if "eye" in label else {}
            out["dev"][f"{label}/{model}"] = {**{k: s[k] for k in ["n", "found", "fp_rate", "accuracy", "wrong_location",
                                                                   "calls", "fixations"]}, **x}
            sec = f"{x['took_second_fixation_after_rejection']} / {x['searches_with_rejected_crop']}" if x else "-"
            L.append(f"| {label} | {model} | {pct(s['found'])} | {pct(s['fp_rate'])} | {pct(s['accuracy'])} | "
                     f"{pct(s['wrong_location'])} | {s['calls']:.2f} | {s['fixations']:.2f} | "
                     f"{x.get('searches_with_rejected_crop', '-')} | {sec} |")
    (REPO_ROOT / "results/step7_checks.md").write_text("\n".join(L) + "\n")
    (REPO_ROOT / "results/step7_checks.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
