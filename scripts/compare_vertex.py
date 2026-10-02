"""Builds the combined Vertex AI Search comparison report: baseline (hand-built
pipeline) vs config A (Vertex Search -> our Generator/Judge/GroundingChecker)
vs config B (Vertex's own Answer API end-to-end).

Reads the already-committed baseline reports (results/eval_report.json,
results/latency_report.json) plus the two Vertex reports produced by
scripts/run_vertex_eval.py (results/vertex_search_eval.json,
results/vertex_answer_eval.json), and writes results/vertex_comparison.json.

A hardcoded methodology_notes field carries this comparison's two structural
asymmetries verbatim, so the numbers can't be copied out of the JSON without
them: (1) Vertex ingested our already-chunked corpus, so chunking itself is
not part of what's being compared; (2) our pipeline's latency is staged
(embed/dense/rerank/generate), Vertex's is end-to-end-only per call.

Usage:
    python scripts/compare_vertex.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METHODOLOGY_NOTES = {
    "chunking_asymmetry": (
        "Vertex AI Search ingested this project's already-chunked 955 objects "
        "(one GCS object per existing chunk_id), not raw pages - so Vertex's "
        "results reflect its retrieval/ranking quality only, not its own "
        "chunking/segmentation. This was a deliberate choice to keep retrieval "
        "scoring an exact chunk_id match against the existing gold set; it "
        "means this comparison does not measure 'point Vertex at raw documents "
        "and let it chunk,' which is how Vertex AI Search is normally used."
    ),
    "latency_split_asymmetry": (
        "The baseline's latency is staged (embed_query / dense_search / rerank "
        "/ generate, each measured separately - see results/latency_report.json). "
        "Vertex's Search API (config A) and Answer API (config B) each expose "
        "only one end-to-end wall-clock number per call - there is no way to "
        "recover Vertex's internal retrieval/ranking/generation split from the "
        "outside. Vertex's numbers below are end-to-end only; do not compare "
        "them against a single baseline stage in isolation."
    ),
    "config_b_retrieval_asymmetry": (
        "Config B's retrieval precision/recall/MRR/hit-rate are scored against "
        "the chunk_ids Vertex's Answer API actually cited, not a separately "
        "exposed ranked search-results list (the Answer API doesn't expose one "
        "the way the Search API does). This is a narrower signal than the "
        "baseline's or config A's full top-k - treat config B's retrieval "
        "numbers as 'what Vertex chose to cite,' not 'everything it retrieved.'"
    ),
    "config_b_abstention_asymmetry": (
        "Confirmed live against a true out-of-corpus question: Vertex's Answer "
        "API has no structured abstention signal - a query it can't answer "
        "still gets a normal non-empty answer_text (prose explaining no "
        "information was found), and answer_skipped_reasons stays empty. Our "
        "own pipeline's Generator returns an explicit answerable=false via a "
        "forced tool schema; Vertex has no equivalent. Config B's "
        "answerable flag is therefore a bool(answer_text) heuristic, not a "
        "verified equivalent signal - expect config B's correctness breakdown "
        "to show more 'incorrectly_answered' on the deliberately unanswerable "
        "gold questions than the baseline does, as an artifact of this gap "
        "rather than a true hallucination-rate difference."
    ),
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _retrieval_row(summary: dict) -> dict:
    r = summary["retrieval"]
    return {
        "hit_rate": r["hit_rate"],
        "mean_recall": r["mean_recall"],
        "mean_mrr": r["mean_mrr"],
        "mean_precision": r["mean_precision"],
    }


def _grounding_row(summary: dict) -> dict:
    g = summary["grounding"]
    return {
        "grounded_rate": g["grounded_rate"],
        "contradicted_rate": g["contradicted_rate"],
        "ungrounded_rate": g["ungrounded_rate"],
        "fully_grounded_answer_rate": g["fully_grounded_answer_rate"],
    }


def main() -> None:
    config_path = "config/config.yaml"
    if "--config" in sys.argv:
        config_path = sys.argv[sys.argv.index("--config") + 1]

    from rag_receipts.config import load_config

    cfg = load_config(config_path)

    baseline = _load_json(Path(cfg.eval.output_path))
    latency_baseline = _load_json(Path("results/latency_report.json"))
    vertex_search = _load_json(Path(cfg.vertex.search_results_path))
    vertex_answer = _load_json(Path(cfg.vertex.answer_results_path))

    missing = [
        name
        for name, report in [
            ("baseline eval report", baseline),
            ("baseline latency report", latency_baseline),
            ("vertex search eval report", vertex_search),
            ("vertex answer eval report", vertex_answer),
        ]
        if report is None
    ]
    if missing:
        print("Missing report(s), comparison will have null entries for: " + ", ".join(missing))

    comparison = {
        "methodology_notes": METHODOLOGY_NOTES,
        "retrieval": {
            "baseline": _retrieval_row(baseline["summary"]) if baseline else None,
            "config_a_vertex_search": _retrieval_row(vertex_search["summary"]) if vertex_search else None,
            "config_b_vertex_answer": _retrieval_row(vertex_answer["summary"]) if vertex_answer else None,
        },
        "correctness_accuracy": {
            "baseline": baseline["summary"]["correctness_accuracy"] if baseline else None,
            "config_a_vertex_search": vertex_search["summary"]["correctness_accuracy"] if vertex_search else None,
            "config_b_vertex_answer": vertex_answer["summary"]["correctness_accuracy"] if vertex_answer else None,
        },
        "grounding": {
            "baseline": _grounding_row(baseline["summary"]) if baseline else None,
            "config_a_vertex_search": _grounding_row(vertex_search["summary"]) if vertex_search else None,
            "config_b_vertex_answer": _grounding_row(vertex_answer["summary"]) if vertex_answer else None,
            "config_b_vertex_self_reported_grounding_score": (
                vertex_answer.get("mean_vertex_grounding_score") if vertex_answer else None
            ),
        },
        "latency": {
            "baseline_staged": latency_baseline["stages"] if latency_baseline else None,
            "config_a_vertex_search_end_to_end": vertex_search.get("latency_end_to_end_s") if vertex_search else None,
            "config_b_vertex_answer_end_to_end": vertex_answer.get("latency_end_to_end_s") if vertex_answer else None,
        },
    }

    output_path = Path("results/vertex_comparison.json")
    output_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")
    if missing:
        print("Re-run after the missing report(s) exist to fill in the null entries.")


if __name__ == "__main__":
    main()
