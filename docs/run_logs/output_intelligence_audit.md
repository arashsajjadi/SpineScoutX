# Output intelligence & safety audit

> Research-only · not diagnostic. Automated checks that the model outputs are **real,
> derived, consistent, and safe** — not pretty templates. Enforced by
> `tests/test_finding_graph_schema.py`, `tests/test_report_outputs.py`,
> `tests/test_showcase_assets.py` (CI-gated).

## What is checked
1. **Derived, not hardcoded** — changing the input probabilities changes the
   `severity_estimate` (= argmax) and `P(severe)`; confidence = top-class probability.
2. **Probabilities are real** — `P(normal_mild)+P(moderate)+P(severe) ≈ 1`; severity equals
   the argmax of the *stored* probabilities (validator rejects tampering).
3. **View route matches condition** — canal → sagittal-T2, foraminal → sagittal-T1,
   subarticular → axial-T2; every finding carries a `crop_provenance`.
4. **Axial reports carry the level-scorer score** — subarticular findings populate
   `localizer.axial_level_scorer_score`; a low score raises `axial_level_uncertainty`.
5. **review_required ⇔ reasons present**, and reasons come only from the allowed set
   (low_confidence / high_entropy / model_disagreement / axial_level_uncertainty /
   near_severe_threshold / …). The flag is **selective** — the mostly-normal showcase case
   has 0 reviews; severe/busy cases have many. It is deliberately conservative (safety).
6. **Safety-mode vs balanced differ** — Safety Mode v4 lowers the severe threshold; the
   balanced (argmax) and safety operating points differ per condition (see `safety_mode_v4`).
7. **No diagnosis/treatment wording** anywhere in a generated report (validator strips the
   allowed negated/disclaimer phrases, then forbids positive-claim roots: `diagnos`,
   `treatment`, `prescrib`, `doctor replacement`, `fda-clear`).
8. **No identifiers leak** — `case_id` is `case_<sha1(study_id)[:10]>`; the JSON/MD filenames
   are the hashed case id; the every generated report re-validates and is anonymized.
9. **Markdown reflects JSON** — the rendered table contains each finding's severity,
   P(severe), and view route exactly as in the JSON (deterministic renderer).
10. **Committed assets are safe** — showcase cards are lightweight PNGs (no DICOM/.npy/.nii),
    rendering the structured output, not raw pixels.

## LLM / hallucination guard
The showcase pipeline is **fully deterministic** (no LLM). The optional local-Ollama
rewording path (`reporting/llm_report.py`) is fail-closed and may only rephrase the
deterministic graph — it cannot invent findings, severities, fields, or advice; it is not
used to generate any committed output here.

Reproduce: `python -m pytest tests/test_finding_graph_schema.py tests/test_report_outputs.py tests/test_showcase_assets.py -q`.
