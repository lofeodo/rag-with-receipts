# RAG With Receipts

A retrieval-augmented generation pipeline over a scoped slice of the [Old School RuneScape
Wiki](https://oldschool.runescape.wiki/), built to demonstrate retrieval accuracy, grounded
and cited answers, and latency-optimized inference — not just chat-with-a-doc.

**Status:** early scaffold. Architecture notes, setup instructions, and benchmark numbers
will land here as each pipeline stage is built — see [CLAUDE.md](CLAUDE.md) for the current
step and full plan.

## Why the OSRS Wiki

Chosen for a corpus that's deeply cross-referenced and numerically dense (exact XP values,
requirements, drop mechanics), which makes for a stronger eval set — both for multi-hop
retrieval questions and for the hallucination/grounding check. Content is CC BY-NC-SA 3.0;
used here for non-commercial, personal portfolio purposes.

## Architecture

_Coming as each step lands — see CLAUDE.md for the planned pipeline and current progress._

## Benchmarks

_Coming after the eval harness (Step 5) is built._
