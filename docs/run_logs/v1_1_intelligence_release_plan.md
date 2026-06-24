# v1.1 — Intelligent Evidence + Generalization + Main-Branch Release: plan

> Research-only · not diagnostic · not clinically validated. This is the Phase-0
> release/branch audit and merge strategy for the v1.1 milestone. **No merge is
> performed in this phase.**

## Branch / release audit (2026-06-23)

| Item | Value |
|---|---|
| Current / authoritative branch | `feature/spinescoutx-model-output-showcase` (HEAD `08b7d99`, pushed) |
| Default branch | `main` |
| `origin/main` tip | `150b9d1` — *"docs: real RSNA E0/E1/ablation results"* (v0.3-era) |
| Commits `main..HEAD` | **51** |
| Is `main` an ancestor of HEAD? | **YES** (strict ancestor → fast-forward / merge-commit possible) |
| Open PRs | none |
| Repo privacy | **private** (`gh repo view` → `isPrivate:true`) |
| Doctor | READY (RSNA + SPIDER data present, RTX 5080, torch 2.11 cu130) |

### What `main` is missing (everything since v0.3)
`origin/main` predates: robust auto-inference (v0.9), the locked-test protocol
(`splits_v1`), multi-condition oracle baselines, 5/5 five-finding auto, all Safety
Modes (v2–v4), the `finding_graph_v4` schema, the model-output showcase, the
output-first README, the gallery, and the trust/limitations docs. Confirmed:
`main` has **no** `docs/assets/showcase/` and its README is **not** output-first.

### Tag reachability (HARD requirement)
All milestone tags (`v0.6.0` … `v1.0.0-auto-robust-five-finding-research` = `13c346d`)
are **ancestors of HEAD** but **NOT reachable from `main`** today. This is a
hard-stop condition the v1.1 release must fix.

## Merge strategy (decided)

- **Authoritative branch to merge:** the v1.1 work branch (created off
  `feature/spinescoutx-model-output-showcase`), which carries the entire
  v0.9→v1.1 lineage as ancestors.
- **Merge method: MERGE COMMIT (`gh pr merge --merge`), NOT squash.**
  Rationale: `main` is a strict ancestor, so a merge commit (or fast-forward)
  keeps every tagged milestone commit reachable from `main`. A **squash** would
  create new commits and **orphan all v0.x/v1.0 tags** → violates the
  "tags reachable from main" hard-stop. Squash is therefore prohibited here.
- **PR:** open one PR `v1.1 branch → main` (none exists yet). Inspect the full
  diff for forbidden artifacts (DICOM/NIfTI/weights/runs/outputs/caches/large
  files) and confirm README is output-first **before** merging.

### Tags to keep / add
- Keep all existing tags (they become reachable from `main` after the merge).
- Add v1.1 tags **only if justified by real work**:
  - `v1.1.0-intelligent-evidence-research` — iff evidence-stability scoring +
    Safety Mode v5 + updated showcase are real and evaluated.
  - `v1.1.1-axial-stack-scorer` — iff the axial scorer v2 materially improves
    **or** is a rigorously documented negative.
  - `v1.1.2-right-foraminal-refinement` — iff right-foraminal improves **or** is
    rigorously diagnosed.
  - No model-improvement tag for docs-only changes.

## Before merge
1. Phase 1 baseline snapshot (freeze v1.0 numbers with provenance).
2. Phase 2 evidence-stability (the headline new intelligence) + evaluation vs errors.
3. Safety Mode v5 (consumes stability + condition-specific calibration).
4. Right-foraminal bounded refinement / honest diagnosis.
5. Internal domain-shift stress audit (external validation only if legally feasible).
6. Axial stack-scorer v2 — attempt or rigorous negative.
7. Showcase/README/model-card/trust updated to reflect new intelligence.
8. Full quality gates (pytest, ruff, build, doctor, forbidden-file/claim scans, image links).

## After merge verification
- `git checkout main && git pull` → confirm output-first README + `docs/assets/showcase/`.
- Confirm tags reachable from `main` (`git merge-base --is-ancestor <tag> main`).
- Push tags; create GitHub release notes (5/5 baseline, evidence stability, axial/
  right-foraminal status, limitations, research-only).

## Decision policy (recorded)
Real model improvement > documentation. Negatives are reported, not hidden. Evidence
stability must be **evaluated against errors**, not merely added as a field.
Similar-case retrieval must **not** change predictions. No clinical/diagnostic claims.
No hidden GT coordinates in auto inference. No locked-test tuning.
