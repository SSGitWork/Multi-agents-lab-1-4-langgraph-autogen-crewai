"""
tools/token_counter.py  –  Per-run token and step tracking.

Import UsageTracker in each framework skeleton and call record_step()
after every LLM round-trip so you end up with a comparable table.

Students do NOT modify this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class StepRecord:
    step: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float


@dataclass
class UsageTracker:
    framework: str
    steps: list[StepRecord] = field(default_factory=list)
    _start: float = field(default_factory=time.time, repr=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_step(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        latency_s: float | None = None,
    ) -> None:
        """Append one LLM round-trip to the log."""
        self.steps.append(
            StepRecord(
                step=len(self.steps) + 1,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_s=latency_s if latency_s is not None else 0.0,
            )
        )

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.prompt_tokens for s in self.steps)

    @property
    def total_completion_tokens(self) -> int:
        return sum(s.completion_tokens for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._start

    def summary(self) -> dict:
        return {
            "framework": self.framework,
            "step_count": self.step_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(f"\n{'=' * 55}")
        print(f"  Framework : {s['framework']}")
        print(f"  Steps     : {s['step_count']}")
        print(f"  Prompt    : {s['total_prompt_tokens']:,} tokens")
        print(f"  Completion: {s['total_completion_tokens']:,} tokens")
        print(f"  Total     : {s['total_tokens']:,} tokens")
        print(f"  Elapsed   : {s['elapsed_s']} s")
        print(f"{'=' * 55}\n")
