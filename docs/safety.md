# Safety & claims policy — SpineScoutX

**SpineScoutX is research-only, non-commercial, not diagnostic, not clinically
validated, and not for medical decision-making.** It studies disc-level lumbar
*degenerative finding grading* and anatomy-grounded evidence on public research
datasets. It is not a medical device.

## Language policy
- Outputs are **degenerative finding grades / severity estimates**, never a
  "diagnosis". Reports use "research finding", "severity estimate", or
  "non-diagnostic finding graph".
- Findings are limited to the five RSNA label types only: spinal canal stenosis;
  left/right neural foraminal narrowing; left/right subarticular stenosis. No
  disease category outside these labels is invented.
- SPIDER provides **anatomy** segmentation (vertebrae / discs / spinal canal). We
  never claim SPIDER gives pathology masks, nor that foraminal / lateral-recess
  regions are ground-truth segmentations — those evidence regions are
  **approximate** and flagged `evidence_region_source = "approximate"`.

## Every generated report includes
research-only · not diagnostic · not clinically validated · no medical
decision-making · no treatment recommendation.

## Optional LLM (Ollama) wording — fail-closed
A local Ollama model may **only** rephrase the deterministic structured finding
graph into readable prose. It must not invent findings, diagnoses, treatments, or
advice, and must quote the structured values. Every LLM output passes a
conservative safety filter (`reporting/llm_report.py`): it is rejected (fail
closed → deterministic report only) if it omits disclaimers, asserts
treatment/diagnosis/advice, mentions out-of-scope pathology, or states a severity
grade not present in the finding graph. The finding graph — never the model — is
authoritative.

## Forbidden claims
The following must never appear as claims about SpineScoutX (only inside this
list/disclaimers): "diagnostic", "clinically validated", "medical
decision-making", "treatment recommendation", "doctor replacement", "automatically
detects all abnormalities", "first AI system for lumbar disc disease", "approved".

## Allowed public statement
> SpineScoutX is a research-only prototype that studies anatomy-grounded lumbar MRI
> degenerative finding grading and evidence using public non-commercial research
> datasets. It is not diagnostic, not clinically validated, and not for medical
> decision-making.

## Data protection
No PHI is assumed or displayed; figures show no identifiers or raw metadata. No
data, DICOMs, masks, weights, or patient-like outputs are committed. No telemetry;
no hidden network calls (the only external calls are the explicit, user-initiated
Kaggle download and an optional localhost Ollama request).
