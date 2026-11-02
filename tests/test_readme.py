"""Step 10 (GPU-free): the README generator renders only known facts/tables and flags hand-typed digits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_readme_tables as B  # noqa: E402


def facts():
    F = B.Facts()
    F.add("a.x", "94.2", "results/x")
    return F, {"t": "| h |\n|---|\n| 1 |"}


def test_render_substitutes_facts_and_tables():
    F, T = facts()
    out = B.render("acc {{a.x}}%\n{{table:t}}", F, T)
    assert out == "acc 94.2%\n| h |\n|---|\n| 1 |"


def test_unknown_placeholder_fails():
    F, T = facts()
    with pytest.raises(KeyError):
        B.render("{{a.missing}}", F, T)
    with pytest.raises(KeyError):
        B.render("{{table:missing}}", F, T)


def test_fact_cannot_change_value():
    F, _ = facts()
    F.add("a.x", "94.2", "results/x")          # same value again is fine
    with pytest.raises(ValueError):
        F.add("a.x", "94.3", "results/x")


def test_untraced_ignores_generated_values_links_and_bibtex():
    F, T = facts()
    marked = B.render("acc {{a.x}}% on 7 runs [l](docs/a_2.png)\n```bibtex\nyear = {2025}\n```", F, T, mark=True)
    hits = B.untraced(marked)
    assert len(hits) == 1 and hits[0].strip().startswith("7")


def test_signed_ci_format():
    assert B.sci(0.031, 0.1) == "[+3.1, +10.0]" and B.ci(0.92, 0.964) == "[92.0, 96.4]"
