# v1.7 review pack summary

> Research-only · not diagnostic. **Pixel-free summary** of the LOCAL-ONLY review pack
> (`review_packs/v1_7_hard_cases/`, gitignored). Reproduce: `mine_hard_cases_v1_7.py` then
> `build_relabel_review_pack_v1_7.py`.

## What was generated (local-only)

- **704 second-read panels** (PNG) — each shows the 2.5D foraminal crop's three adjacent
  parasagittal slices + the deployed severity bars + every candidate model's p_severe + the
  selection reason. Right-foraminal **338**, left-foraminal **366**.
- `index.html` — browsable viewer with a per-case label dropdown + ambiguity / insufficient /
  exclude flags + note field.
- `review_sheet.csv` — one row per case (metadata + blank review columns).
- `review_items.jsonl` — machine-readable items (+ allowed labels).

## Priority focus (re-annotation order)

1. right-foraminal **severe FN (87 in pack)**; 2. right-foraminal L4-L5 / L5-S1; 3. left-foraminal
hard cases; 4. moderate/severe borderline; 5. controls (179, to keep the pack unbiased).

## Status

**Awaiting expert labels.** No radiologist review file is present in this run, so the pack is the
deliverable + the exact human-review handoff (`v1_7_review_needed.md`). The pack is never committed
(imaging pixels); only this summary + the schema are committed. When a reviewed CSV is supplied,
`ingest_review_labels_v1_7.py` produces a versioned corrected-label set for Phase-6 retraining.
