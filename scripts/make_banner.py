"""Render the README banner (docs/img/banner.svg) from results/*.json.

Usage: python scripts/make_banner.py

The banner is self-contained (dark card, system fonts only) so it looks the
same in GitHub light and dark mode. The figures on the receipt come from the
committed results files, so they can't drift from the benchmarks.
"""

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img" / "banner.svg"

W, H = 1200, 360
SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "'SFMono-Regular', Consolas, 'Courier New', monospace"
ACCENT = "#8b7df0"
ACCENT_DEEP = "#6d5bd0"

PILLS_ROW1 = ["Cited answers", "Golden dataset", "Eval harness", "Reranking"]
PILLS_ROW2 = ["Latency tuned", "FAISS + pgvector", "GCP Cloud Run"]


def pill(x, y, label):
    w = 8.1 * len(label) + 30
    svg = (
        f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="32" rx="16" '
        f'fill="#ffffff" fill-opacity="0.07" stroke="#ffffff" stroke-opacity="0.22"/>'
        f'<text x="{x + w / 2:.1f}" y="{y + 21}" text-anchor="middle" font-size="15" '
        f'font-weight="600" fill="#e6e9f2">{escape(label)}</text>'
    )
    return svg, w


def pill_row(x, y, labels):
    parts = []
    for label in labels:
        svg, w = pill(x, y, label)
        parts.append(svg)
        x += w + 10
    return "".join(parts)


def receipt(summary, latency):
    """A tilted paper receipt listing the headline measured numbers."""
    rows = [
        ("Retrieval hit rate", f"{summary['retrieval']['hit_rate'] * 100:.0f}%"),
        ("Answer correctness", f"{summary['correctness_accuracy'] * 100:.0f}%"),
        ("Claims grounded", f"{summary['grounding']['grounded_rate'] * 100:.0f}%"),
        ("p50 latency", f"{latency['stages']['end_to_end_s']['p50']:.1f}s"),
    ]
    w = 300
    body = [
        f'<text x="{w / 2}" y="40" text-anchor="middle" font-family="{MONO}" font-size="16" '
        f'font-weight="700" fill="#2a2a2a" letter-spacing="2">RECEIPT</text>',
        f'<line x1="22" y1="56" x2="{w - 22}" y2="56" stroke="#9a968a" stroke-dasharray="4 4"/>',
    ]
    y = 90
    for label, value in rows:
        body.append(
            f'<text x="22" y="{y}" font-family="{MONO}" font-size="15" fill="#3a3a3a">{escape(label)}</text>'
            f'<text x="{w - 22}" y="{y}" text-anchor="end" font-family="{MONO}" font-size="17" '
            f'font-weight="700" fill="#1a1a1a">{escape(value)}</text>'
        )
        y += 34
    body.append(f'<line x1="22" y1="{y - 12}" x2="{w - 22}" y2="{y - 12}" stroke="#9a968a" stroke-dasharray="4 4"/>')
    y += 16
    body.append(
        f'<text x="22" y="{y}" font-family="{MONO}" font-size="14" font-weight="700" '
        f'fill="#1a1a1a" letter-spacing="1">ALL MEASURED</text>'
        f'<path d="M{w - 52} {y - 8} l8 8 l16 -18" fill="none" stroke="#1f9d6b" '
        f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    height = y + 34
    # Zig-zag bottom edge, like torn receipt paper.
    paper = f"M0 0 H{w} V{height - 10}" + "".join(
        f" l-8 10 l-8 -10" for _ in range(int(w // 16))
    ) + " Z"
    return (
        f'<g transform="translate(838 {(H - height) / 2:.0f}) rotate(3 150 {height / 2:.0f})" filter="url(#shadow)">'
        f'<path d="{paper}" fill="#f6f3ea"/>{"".join(body)}</g>'
    )


def build(summary, latency):
    cta_y = 262
    cta = (
        f'<g filter="url(#glow)">'
        f'<rect x="60" y="{cta_y}" width="330" height="62" rx="31" fill="url(#cta)"/></g>'
        f'<rect x="60" y="{cta_y}" width="330" height="62" rx="31" fill="none" '
        f'stroke="#ffffff" stroke-opacity="0.55" stroke-width="2"/>'
        f'<circle cx="98" cy="{cta_y + 31}" r="15" fill="#ffffff"/>'
        f'<path d="M93 {cta_y + 22} L106 {cta_y + 31} L93 {cta_y + 40} Z" fill="{ACCENT_DEEP}"/>'
        f'<text x="128" y="{cta_y + 40}" font-size="24" font-weight="800" fill="#ffffff">Try the live demo</text>'
        f'<path d="M345 {cta_y + 31} h20 m-8 -8 l8 8 l-8 8" fill="none" stroke="#ffffff" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="{SANS}" role="img" aria-label="RAG With Receipts: try the live demo">'
        f"<defs>"
        f'<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#0d1117"/><stop offset="1" stop-color="#1c1745"/></linearGradient>'
        f'<linearGradient id="cta" x1="0" y1="0" x2="1" y2="0">'
        f'<stop offset="0" stop-color="{ACCENT_DEEP}"/><stop offset="1" stop-color="{ACCENT}"/></linearGradient>'
        f'<radialGradient id="halo" cx="0.5" cy="0.5" r="0.5">'
        f'<stop offset="0" stop-color="{ACCENT}" stop-opacity="0.35"/>'
        f'<stop offset="1" stop-color="{ACCENT}" stop-opacity="0"/></radialGradient>'
        f'<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
        f'<feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#000" flood-opacity="0.45"/></filter>'
        f'<filter id="glow" x="-20%" y="-60%" width="140%" height="220%">'
        f'<feDropShadow dx="0" dy="0" stdDeviation="9" flood-color="{ACCENT}" flood-opacity="0.75"/></filter>'
        f"</defs>"
        f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>'
        f'<circle cx="980" cy="180" r="260" fill="url(#halo)"/>'
        f'<text x="60" y="92" font-size="58" font-weight="800" fill="#ffffff" letter-spacing="-1">'
        f'RAG With <tspan fill="{ACCENT}">Receipts</tspan></text>'
        f'<text x="60" y="132" font-size="20" fill="#b8c0d4">'
        f"Every design choice backed by a measured number, not a claim.</text>"
        f"{pill_row(60, 162, PILLS_ROW1)}{pill_row(60, 206, PILLS_ROW2)}"
        f"{cta}{receipt(summary, latency)}</svg>"
    )


if __name__ == "__main__":
    res = ROOT / "results"
    summary = json.loads((res / "eval_report.json").read_text())["summary"]
    latency = json.loads((res / "latency_report.json").read_text())
    OUT.write_text(build(summary, latency), encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT))
