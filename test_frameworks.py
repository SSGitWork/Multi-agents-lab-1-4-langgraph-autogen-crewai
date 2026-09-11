"""
tests/test_frameworks.py  –  Behavioural tests for Lab 1.4.

These tests verify that each framework implementation:
  1. Produces a word_freq.py file with the required function.
  2. Produces valid Python (no SyntaxErrors).
  3. Returns a tracker with at least one recorded step.
  4. Reports fewer than 15 steps (loop guard working).

Tests are skipped automatically if a framework module raises
NotImplementedError, so you can run the suite incrementally.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FRAMEWORK_MODULES = {
    "langgraph": "agent_langgraph",
    "autogen": "agent_autogen",
    "crewai": "agent_crewai",
}


def _load_module(name: str) -> Any:
    """Import a framework module, returning None if not yet implemented."""
    if name in sys.modules:
        del sys.modules[name]
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        pytest.skip(f"Could not import {name}: {exc}")


def _try_run(module: Any) -> None:
    """Call module.run(), skipping the test if NotImplementedError is raised."""
    try:
        module.run()
    except NotImplementedError:
        pytest.skip(f"{module.__name__} is not yet implemented.")


# ---------------------------------------------------------------------------
# Parametrised fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=list(FRAMEWORK_MODULES.values()))
def framework_module(request):
    mod = _load_module(request.param)
    _try_run(mod)
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOutputFile:
    """Each framework must produce a word_freq.py file."""

    def test_word_freq_file_exists(self, framework_module):
        from tools.code_tools import WORKSPACE

        target = WORKSPACE / "word_freq.py"
        assert target.exists(), (
            f"{framework_module.__name__}: expected word_freq.py in workspace "
            f"({WORKSPACE}), but it was not found."
        )

    def test_word_freq_is_valid_python(self, framework_module):
        from tools.code_tools import WORKSPACE

        source = (WORKSPACE / "word_freq.py").read_text()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(
                f"{framework_module.__name__}: word_freq.py has a syntax error: {exc}"
            )

    def test_word_freq_contains_function(self, framework_module):
        from tools.code_tools import WORKSPACE

        source = (WORKSPACE / "word_freq.py").read_text()
        tree = ast.parse(source)
        func_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "word_frequency" in func_names, (
            f"{framework_module.__name__}: word_freq.py must contain a "
            "function named 'word_frequency'."
        )

    def test_word_freq_function_has_type_annotations(self, framework_module):
        from tools.code_tools import WORKSPACE

        source = (WORKSPACE / "word_freq.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "word_frequency":
                # Check return annotation exists
                assert node.returns is not None, (
                    f"{framework_module.__name__}: word_frequency() must have "
                    "a return type annotation."
                )
                # Check at least one argument has an annotation
                annotated_args = [
                    a for a in node.args.args if a.annotation is not None
                ]
                assert annotated_args, (
                    f"{framework_module.__name__}: word_frequency() must have "
                    "type-annotated parameters."
                )


class TestTrackerUsage:
    """Each framework must record at least one LLM step."""

    def test_tracker_has_steps(self, framework_module):
        assert framework_module.tracker.step_count >= 1, (
            f"{framework_module.__name__}: tracker recorded 0 steps. "
            "Did you call tracker.record_step() after each LLM call?"
        )

    def test_step_count_within_limit(self, framework_module):
        assert framework_module.tracker.step_count < 15, (
            f"{framework_module.__name__}: agent took "
            f"{framework_module.tracker.step_count} steps (limit is 15). "
            "Check your termination condition."
        )

    def test_total_tokens_recorded(self, framework_module):
        assert framework_module.tracker.total_tokens > 0, (
            f"{framework_module.__name__}: total_tokens is 0. "
            "Token usage is not being recorded."
        )


class TestSampleFile:
    """The agent should also create sample.txt as instructed."""

    def test_sample_txt_exists(self, framework_module):
        from tools.code_tools import WORKSPACE

        target = WORKSPACE / "sample.txt"
        assert target.exists(), (
            f"{framework_module.__name__}: expected sample.txt in workspace. "
            "The agent should have created it as part of the task."
        )

    def test_sample_txt_has_enough_words(self, framework_module):
        from tools.code_tools import WORKSPACE

        text = (WORKSPACE / "sample.txt").read_text()
        word_count = len(text.split())
        assert word_count >= 20, (
            f"{framework_module.__name__}: sample.txt has only {word_count} words; "
            "the task requires at least 20."
        )


# ---------------------------------------------------------------------------
# Cross-framework comparison test (runs once, not parametrised)
# ---------------------------------------------------------------------------

class TestCrossFrameworkComparison:
    """After all three frameworks run, verify comparison data is present."""

    def test_results_json_exists(self):
        results_path = Path("comparison/results.json")
        if not results_path.exists():
            pytest.skip("comparison/results.json not found; run `python compare.py` first.")
        import json

        data = json.loads(results_path.read_text())
        assert "results" in data, "comparison/results.json is missing the 'results' key."

    def test_tradeoffs_md_has_content(self):
        tradeoffs_path = Path("comparison/tradeoffs.md")
        assert tradeoffs_path.exists(), "comparison/tradeoffs.md not found."
        content = tradeoffs_path.read_text()
        # Check that at least some answers have been filled in
        placeholder_count = content.count("*Your answer:*")
        total_sections = 6
        filled = total_sections - placeholder_count
        assert filled >= 2, (
            f"comparison/tradeoffs.md: only {filled}/{total_sections} sections "
            "appear to be filled in. Complete at least 2 before submitting."
        )
