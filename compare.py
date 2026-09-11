"""
compare.py  –  Run all three framework agents and print a comparison table.

Usage:
    python compare.py                   # runs all three
    python compare.py --only langgraph  # runs one framework

The script prints a summary table and writes comparison/results.json
so you can fill in the qualitative observations in comparison/tradeoffs.md.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime
from pathlib import Path


def run_framework(name: str) -> dict:
    """Import and run one framework, returning its summary dict."""
    module_map = {
        "langgraph": "agent_langgraph",
        "autogen": "agent_autogen",
        "crewai": "agent_crewai",
    }
    module_name = module_map[name]
    try:
        mod = __import__(module_name)
        mod.run()
        summary = mod.tracker.summary()
        summary["status"] = "ok"
    except NotImplementedError as exc:
        summary = {"framework": name, "status": "not_implemented", "error": str(exc)}
    except Exception as exc:
        summary = {
            "framework": name,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    return summary


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 75)
    print(f"  {'Framework':<12} {'Status':<14} {'Steps':>6} {'Prompt':>10} {'Completion':>12} {'Total':>10}")
    print("-" * 75)
    for r in results:
        framework = r.get("framework", "?")
        status = r.get("status", "?")
        steps = r.get("step_count", "-")
        prompt = r.get("total_prompt_tokens", "-")
        completion = r.get("total_completion_tokens", "-")
        total = r.get("total_tokens", "-")
        print(f"  {framework:<12} {status:<14} {str(steps):>6} {str(prompt):>10} {str(completion):>12} {str(total):>10}")
    print("=" * 75 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=["langgraph", "autogen", "crewai"],
        help="Run only one framework.",
    )
    args = parser.parse_args()

    frameworks = [args.only] if args.only else ["langgraph", "autogen", "crewai"]
    results = [run_framework(f) for f in frameworks]

    print_table(results)

    out_path = Path("comparison/results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps({"run_at": datetime.utcnow().isoformat(), "results": results}, indent=2)
    )
    print(f"Results saved to {out_path}")
    print("Now fill in comparison/tradeoffs.md with your qualitative observations.\n")


if __name__ == "__main__":
    main()
