"""Stretch goal: Haiku vs Sonnet generation comparison (latency/cost/correctness).

Runs the full 55-question gold set through the eval harness twice - once with
Claude Sonnet 5 (the production default) as the generation model, once with Claude
Haiku 4.5 - holding everything else fixed (same Retriever, same Judge, always
Claude Sonnet 5 so correctness grading doesn't itself become a second variable, and
the same GroundingChecker). This isolates the one variable the comparison is about:
does the generation model matter, and by how much, on latency/cost/correctness?

Real API cost: ~55 generate() calls per arm (110 total) plus a judge call on every
question both sides agree is answerable (up to ~50 per arm). Confirmed via a prior
live smoke test that Claude Haiku 4.5 accepts the same forced tool_choice schema
Generator already uses for Claude Sonnet 5 - not assumed from documentation alone.

request_config is a declarative record of what Generator.generate() actually sends
(read from generation/generator.py's source, not scraped per-response, since none
of these parameters vary call-to-call) so the comparison's fairness is auditable
from the output JSON itself rather than only asserted in prose. In particular: the
two models are NOT equally capable of deterministic sampling (Sonnet 5 runs
adaptive thinking by default and 400s on temperature/top_p/top_k; Haiku 4.5 runs no
thinking by default and does accept sampling params) - neither is pinned, to keep
the two requests configured identically from this codebase's point of view, and
that asymmetry is recorded explicitly rather than glossed over.

Usage:
    python scripts/compare_generation_models.py
    python scripts/compare_generation_models.py --config config/config.yaml
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from tqdm import tqdm

from rag_receipts.config import load_config
from rag_receipts.eval.dataset import load_eval_questions
from rag_receipts.eval.judge import Judge
from rag_receipts.eval.runner import run_eval, summarize
from rag_receipts.generation.config import GenerationConfig
from rag_receipts.generation.generator import ANSWER_TOOL, ANSWER_TOOL_NAME, Generator, SYSTEM_PROMPT
from rag_receipts.grounding.checker import GroundingChecker
from rag_receipts.retrieval.pipeline import Retriever
from rag_receipts.telemetry.timing import summarize_durations

HAIKU_MODEL = "claude-haiku-4-5"

# Anthropic first-party API pricing, per 1M tokens, captured 2026-10-02. Not tracked
# anywhere else in this repo - re-verify before reusing this script far in the future.
PRICING_PER_MILLION_TOKENS = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    HAIKU_MODEL: {"input": 1.00, "output": 5.00},
}


@dataclass
class TimingGenerator(Generator):
    """Wraps Generator.generate() with a wall-clock Stopwatch per call, matching
    measure_latency.py's/run_vertex_eval.py's inline-timing pattern, without
    touching eval/runner.py or eval/models.py (neither captures latency today)."""

    durations_s: list[float] = field(default_factory=list)

    def generate(self, query, chunks):
        t0 = time.perf_counter()
        result = super().generate(query, chunks)
        self.durations_s.append(time.perf_counter() - t0)
        return result


def build_request_config(model: str) -> dict:
    """Declarative record of what Generator.generate() sends for `model`, read
    from generation/generator.py:133-140 - not scraped per-response, since none of
    these parameters vary call-to-call. If generate() ever starts varying one of
    these per call, this function needs revisiting."""
    return {
        "model": model,
        "max_tokens": 4096,
        "system_prompt": SYSTEM_PROMPT,
        "tools": [ANSWER_TOOL],
        "tool_choice": {"type": "tool", "name": ANSWER_TOOL_NAME},
        "thinking": "not set (parameter omitted)",
        "thinking_behavior_note": (
            "Same omitted parameter on both arms does not mean same inference "
            "behavior: claude-sonnet-5 runs adaptive thinking by default when "
            "`thinking` is omitted, while claude-haiku-4-5 runs no thinking at all "
            "when omitted (Haiku only thinks if explicitly given "
            "{'type':'enabled','budget_tokens':N}, which this comparison does not do)."
        ),
        "temperature": "not set",
        "top_p": "not set",
        "top_k": "not set",
        "sampling_params_note": (
            "Neither arm pins sampling params. claude-sonnet-5 would reject them "
            "with a 400 while adaptive thinking is active; claude-haiku-4-5 would "
            "accept them. The two models are not equally capable of being made "
            "deterministic - leaving both unset keeps the request shape identical "
            "rather than pinning one model and not the other."
        ),
        "output_config_effort": "not set (effort errors on both claude-sonnet-5 and claude-haiku-4-5)",
        "prompt_caching": "not used (no cache_control sent)",
    }


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = PRICING_PER_MILLION_TOKENS[model]
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def run_arm(
    model: str,
    *,
    questions,
    retriever,
    judge,
    grounding_checker,
    base_generation_config: GenerationConfig,
) -> dict:
    print(f"\n--- {model} ---")
    generation_config = replace(base_generation_config, model=model)
    generator = TimingGenerator.from_config(generation_config)

    results = []
    for question in tqdm(questions, desc=model):
        results.extend(
            run_eval(
                [question],
                retriever=retriever,
                generator=generator,
                judge=judge,
                grounding_checker=grounding_checker,
            )
        )

    summary = summarize(results)

    gen_input_tokens = sum(r.generated.input_tokens for r in results if r.generated is not None)
    gen_output_tokens = sum(r.generated.output_tokens for r in results if r.generated is not None)
    judge_input_tokens = sum(r.judge.input_tokens for r in results if r.judge is not None)
    judge_output_tokens = sum(r.judge.output_tokens for r in results if r.judge is not None)

    generation_cost = cost_usd(model, gen_input_tokens, gen_output_tokens)
    # Judge is always claude-sonnet-5 regardless of which model generated the answer.
    judge_cost = cost_usd("claude-sonnet-5", judge_input_tokens, judge_output_tokens)

    return {
        "generation_model": model,
        "request_config": build_request_config(model),
        "eval_summary": asdict(summary),
        "latency_s": summarize_durations(generator.durations_s),
        "tokens": {
            "generation_input": gen_input_tokens,
            "generation_output": gen_output_tokens,
            "judge_input": judge_input_tokens,
            "judge_output": judge_output_tokens,
        },
        "cost_usd": {
            "generation": generation_cost,
            "judge": judge_cost,
            "total": generation_cost + judge_cost,
        },
        "results": [asdict(r) for r in results],
    }


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]
    cfg = load_config(config_path)

    questions = load_eval_questions(cfg.eval.dataset_path)
    print(f"Loaded {len(questions)} questions from {cfg.eval.dataset_path}")

    retriever = Retriever.from_config(cfg)
    judge = Judge.from_config(cfg.eval)
    grounding_checker = GroundingChecker.from_config(cfg.grounding)

    models = [cfg.generation.model, HAIKU_MODEL]
    arms = {
        model: run_arm(
            model,
            questions=questions,
            retriever=retriever,
            judge=judge,
            grounding_checker=grounding_checker,
            base_generation_config=cfg.generation,
        )
        for model in models
    }

    report = {
        "run_metadata": {
            "anthropic_sdk_version": version("anthropic"),
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "models": arms,
        "pricing_per_million_tokens": PRICING_PER_MILLION_TOKENS,
        "pricing_captured": "2026-10-02",
        "methodology_notes": {
            "judge_fixed_at_sonnet_5": (
                "The judge model is claude-sonnet-5 in both arms, even when grading "
                "Haiku-generated answers - isolates generation as the only variable "
                "and matches this project's locked decision that Sonnet 5 is the "
                "correctness judge regardless of which model is under test."
            ),
            "no_temperature_pinned": (
                "Neither arm sets temperature/top_p/top_k, despite claude-haiku-4-5 "
                "supporting them and claude-sonnet-5 rejecting them - see each arm's "
                "request_config.sampling_params_note."
            ),
            "thinking_default_asymmetry": (
                "Neither arm sets `thinking`, but the two models behave differently "
                "when it's omitted - see each arm's request_config.thinking_behavior_note."
            ),
            "latency_scope": (
                "latency_s measures only the wall-clock time of each Generator.generate() "
                "call (network + inference for the real Claude API round-trip). Retrieval "
                "latency is identical across arms (same Retriever, reranker-independent of "
                "generation model) and already measured in Step 7's latency_report.json - "
                "not re-measured here."
            ),
            "non_determinism_caveat": (
                "Neither model's generation or judging calls pin temperature, so "
                "correctness/grounding numbers here are not bit-reproducible across "
                "re-runs, consistent with the same caveat already documented for "
                "Steps 5-7's eval runs."
            ),
        },
    }

    output_path = Path("results/generation_model_comparison.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote comparison to {output_path}\n")
    print(f"{'model':20s} {'accuracy':>9s} {'grounded':>9s} {'lat_p50':>9s} {'lat_p95':>9s} {'cost':>9s}")
    for model, arm in arms.items():
        s = arm["eval_summary"]
        lat = arm["latency_s"]
        cost = arm["cost_usd"]["total"]
        print(
            f"{model:20s} "
            f"{s['correctness_accuracy']:9.3f} "
            f"{s['grounding']['grounded_rate']:9.3f} "
            f"{lat['p50'] * 1000:7.0f}ms "
            f"{lat['p95'] * 1000:7.0f}ms "
            f"${cost:8.4f}"
        )


if __name__ == "__main__":
    main()
