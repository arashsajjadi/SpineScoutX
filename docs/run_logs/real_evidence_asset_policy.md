# Safe real-evidence asset policy (v1.3)

> Research-only · not diagnostic. Defines what evidence assets may be committed to the
> (private) repo, so the real-case viewer is convincing **without** redistributing medical
> imagery or leaking patient data.

## Context
- **RSNA LumbarDISC**: public **non-commercial research** competition data; redistribution of
  the imagery — including derived crops — is restricted by the competition terms.
- **SPIDER**: CC BY 4.0 (anatomy segmentation; different task; not the graded findings).
- Repo is **private**, but private ≠ a redistribution license.

## Decision (conservative — no-regret)
**Committed evidence assets are PIXEL-FREE.** We do not commit DICOMs, derived medical-image
crops, screenshots of pixels, masks, or any image-pixel arrays of patient anatomy.

- The **full real-pixel evidence viewer** (with actual derived crops) is generated **locally**
  under gitignored `outputs/real/evidence_case_viewer/` and is reproducible with one command.
- **Committed** cards show, for each real locked-test case:
  - the real **structured prediction** (severity estimate, P(severe), confidence, stability,
    route quality);
  - real **derived scalar signals** that are **non-reconstructive** and metadata-free
    (auto crop-centre x/y offset in px, slice/instance index within the stack, mean crop
    intensity, localizer/scorer confidence) — numbers, not pixels;
  - a **schematic** evidence-route diagram (boxes/arrows, no pixels);
  - the **held-out reference label**, explicitly marked *held-out reference, not model input*;
  - the **code-derived correctness** and the **review_required** reason.

## Hard rules (enforced by tests + the forbidden-file gate)
- **Never** commit DICOMs (`*.dcm`), NIfTI, `.npy/.npz` pixel arrays, or masks.
- **Never** commit DICOM metadata or headers.
- **Never** show `study_id` / `series_id` / patient id — use hashed `case_id` only.
- Strip all metadata; committed PNGs contain only schematic shapes + derived numbers + text.
- The reference label is always rendered as *held-out reference (not a model input)*.
- The full pixel viewer writes only under `outputs/` (gitignored).

## Why this still fixes "looks synthetic"
The committed cards are now unmistakably **derived from a specific real locked-test case**
(hashed id + real per-crop scalar signals + real prediction vs real held-out reference +
real correctness), even though no medical pixels are shipped. A reviewer with data access can
reproduce the full pixel viewer locally to see the actual crops.
