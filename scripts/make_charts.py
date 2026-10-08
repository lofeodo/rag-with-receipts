"""Render the README's result charts as standalone SVGs from results/*.json.

Usage: python scripts/make_charts.py
Writes docs/img/chart_{correctness,reranker,latency}.svg plus a *_dark.svg
variant of each, selected in the README with a <picture> element.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

THEMES = {
    "": dict(card="#f6f8fa", stroke="#d0d7de", ink="#1f2328", sub="#57606a",
             track="#e6e9ee", muted="#c2c8d0", accent="#6d5bd0"),
    "_dark": dict(card="#161b22", stroke="#30363d", ink="#e6edf3", sub="#8b949e",
                  track="#21262d", muted="#3d444d", accent="#8b7df0"),
}
T = THEMES[""]  # active theme; set per render pass in __main__
SUFFIX = ""
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

W = 720
LABEL_W = 190
PAD = 28
BAR_H = 22
ROW_H = 40


def bar_panel(x, y, width, title, rows, max_value, fmt):
    """One horizontal bar panel. rows: (label, value, highlighted). Returns (svg, height)."""
    track_w = width - LABEL_W - 74
    parts = [
        f'<text x="{x}" y="{y}" font-size="13" font-weight="600" fill="{T["sub"]}" '
        f'letter-spacing="0.4">{escape(title.upper())}</text>'
    ]
    cy = y + 22
    for label, value, hi in rows:
        bw = max(3, track_w * value / max_value)
        color = T["accent"] if hi else T["muted"]
        weight = "700" if hi else "500"
        parts.append(
            f'<text x="{x}" y="{cy + 16}" font-size="14" font-weight="{weight}" '
            f'fill="{T["ink"]}">{escape(label)}</text>'
            f'<rect x="{x + LABEL_W}" y="{cy}" width="{track_w}" height="{BAR_H}" rx="6" fill="{T["track"]}"/>'
            f'<rect x="{x + LABEL_W}" y="{cy}" width="{bw:.1f}" height="{BAR_H}" rx="6" fill="{color}"/>'
            f'<text x="{x + LABEL_W + track_w + 10}" y="{cy + 16}" font-size="14" '
            f'font-weight="700" fill="{T["ink"]}">{escape(fmt(value))}</text>'
        )
        cy += ROW_H
    return "".join(parts), cy - y


def figure(height, body, heading, caption):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" '
        f'width="{W}" height="{height}" font-family="{FONT}" role="img" '
        f'aria-label="{escape(heading)}">'
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{height - 1}" rx="14" fill="{T["card"]}" stroke="{T["stroke"]}"/>'
        f'<text x="{PAD}" y="{PAD + 12}" font-size="18" font-weight="700" fill="{T["ink"]}">{escape(heading)}</text>'
        f'<text x="{PAD}" y="{PAD + 32}" font-size="13" fill="{T["sub"]}">{escape(caption)}</text>'
        f"{body}</svg>"
    )


def write(name, svg):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name.replace(".svg", f"{SUFFIX}.svg")).write_text(svg, encoding="utf-8")
    print("wrote", name.replace(".svg", f"{SUFFIX}.svg"))


def correctness(summary):
    by = summary["correctness_by_type"]
    rows = [
        ("Single-hop (n=35)", by["single_hop"]["accuracy"] * 100, True),
        ("Multi-hop (n=15)", by["multi_hop"]["accuracy"] * 100, False),
        ("Overall (n=55)", summary["correctness_accuracy"] * 100, False),
    ]
    body, h = bar_panel(PAD, 96, W - 2 * PAD, "Answer correctness (LLM judge)", rows, 100, lambda v: f"{v:.0f}%")
    write(
        "chart_correctness.svg",
        figure(96 + h + 22, body, "Single-hop is solved; multi-hop is the open problem",
               "Share of answers judged correct, by question type"),
    )


def reranker(sweep):
    names = {
        "BAAI/bge-reranker-base": "bge-base (old default)",
        "BAAI/bge-reranker-v2-m3": "bge-v2-m3",
        "cross-encoder/ms-marco-MiniLM-L-6-v2": "MiniLM (adopted)",
    }
    adopted = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hit = [(names[r["model_name"]], r["retrieval"]["hit_rate"] * 100, r["model_name"] == adopted) for r in sweep]
    lat = [(names[r["model_name"]], r["rerank_latency"]["p50"] * 1000, r["model_name"] == adopted) for r in sweep]
    half = (W - 2 * PAD - 24) // 2
    # Panels stacked rather than side by side so labels keep room to breathe.
    p1, h1 = bar_panel(PAD, 96, W - 2 * PAD, "Retrieval hit rate (higher is better)", hit, 100, lambda v: f"{v:.0f}%")
    y2 = 96 + h1 + 22
    p2, h2 = bar_panel(PAD, y2, W - 2 * PAD, "Rerank latency, p50 (lower is better)", lat, 3000, lambda v: f"{v:,.0f} ms")
    del half
    write(
        "chart_reranker.svg",
        figure(y2 + h2 + 22, p1 + p2, "The adopted reranker beat the old default on accuracy and speed",
               "Three rerankers compared on the same 55-question gold set"),
    )


def latency(report):
    st = report["stages"]
    rows = [
        ("Embed query", st["embed_query_s"]["p50"] * 1000, False),
        ("Vector search", st["dense_search_s"]["p50"] * 1000, False),
        ("Rerank", st["rerank_s"]["p50"] * 1000, False),
        ("LLM generation", st["generate_s"]["p50"] * 1000, True),
    ]

    def fmt(v):
        return f"{v:.1f} ms" if v < 10 else f"{v:,.0f} ms"

    body, h = bar_panel(PAD, 96, W - 2 * PAD, "Median time per stage", rows, 3000, fmt)
    write(
        "chart_latency.svg",
        figure(96 + h + 22, body, "The LLM call is the bottleneck, not retrieval",
               "Per-stage latency, p50, measured on the real pipeline"),
    )


if __name__ == "__main__":
    res = ROOT / "results"
    summary = json.loads((res / "eval_report.json").read_text())["summary"]
    sweep = json.loads((res / "reranker_sweep.json").read_text())
    report = json.loads((res / "latency_report.json").read_text())
    for SUFFIX, T in THEMES.items():
        correctness(summary)
        reranker(sweep)
        latency(report)
