"""Optional, fail-closed LLM wording for finding-graph reports (local Ollama only).

The LLM may ONLY rephrase the deterministic structured finding graph into readable
prose. It must not invent findings, diagnoses, treatments, or advice. Every LLM
output passes a conservative safety filter; if it fails, we fall back to the
deterministic report (fail closed). The authoritative numbers always come from the
finding graph, never from the model.

Research-only. Not diagnostic. Not for medical decision-making.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..constants import SEVERITIES
from ..utils.logging import get_logger

log = get_logger()

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

REQUIRED_DISCLAIMERS = ("research-only", "not diagnostic")

# Positive claims that must never appear (negated forms like "not diagnostic" are allowed).
_FORBIDDEN_SUBSTRINGS = (
    "treatment",
    "prescrib",
    "medication",
    "surgery",
    "you should",
    "we recommend",
    "i recommend",
    "is recommended",
    "advise",
    "clinically validated",  # only allowed as "not clinically validated" (handled below)
    "decision-making",  # only allowed as "not for medical decision-making"
)
# Out-of-scope pathologies the model must not introduce (RSNA labels cover none of these).
_OUT_OF_SCOPE = (
    "cancer",
    "tumor",
    "tumour",
    "metasta",
    "fracture",
    "infection",
    "abscess",
    "malignan",
    "hemorrhage",
)
# Allowed negated contexts for otherwise-forbidden words.
_ALLOWED_NEGATIONS = (
    "not clinically validated",
    "not for medical decision-making",
    "no treatment",
    "not a diagnosis",
    "non-diagnostic",
    "not diagnostic",
)


def _strip_allowed(text: str) -> str:
    out = text
    for phrase in _ALLOWED_NEGATIONS:
        out = out.replace(phrase, " ")
    return out


def check_llm_safety(text: str, graph: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return (is_safe, reasons). Conservative / fail-closed.

    Rejects: missing disclaimers, treatment/diagnosis/advice claims, out-of-scope
    pathologies, severity grades not present in the finding graph, and (heuristic)
    positive uses of "diagnos*".
    """
    reasons: list[str] = []
    low = text.lower()

    for d in REQUIRED_DISCLAIMERS:
        if d not in low:
            reasons.append(f"missing disclaimer: {d!r}")

    scrubbed = _strip_allowed(low)
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad in scrubbed:
            reasons.append(f"forbidden claim substring: {bad!r}")
    for bad in _OUT_OF_SCOPE:
        if bad in scrubbed:
            reasons.append(f"out-of-scope pathology: {bad!r}")

    # "diagnos*" only allowed in negated/disclaimer contexts.
    if re.search(r"\bdiagnos", scrubbed):
        reasons.append("positive use of 'diagnos*'")

    # Severity grades mentioned must be a subset of grades present in the graph.
    graph_grades = {str(f.get("grade", "")).lower() for f in graph.get("findings", [])}
    for sev in SEVERITIES:
        mentioned = sev.replace("_", " ") in low or sev in low
        if mentioned and sev not in graph_grades:
            reasons.append(f"grade {sev!r} not present in finding graph")

    return (len(reasons) == 0, reasons)


def build_prompt(graph: dict[str, Any]) -> str:
    """Build a strict rewriting prompt from the structured finding graph."""
    findings_lines = []
    for f in graph.get("findings", []):
        side = f.get("side")
        side_txt = f" ({side})" if side else ""
        findings_lines.append(
            f"- level {f.get('level')}, {f.get('condition')}{side_txt}: "
            f"grade={f.get('grade')}, confidence={f.get('confidence')}, "
            f"calibrated_confidence={f.get('calibrated_confidence')}, "
            f"uncertainty={f.get('uncertainty_flag')}, "
            f"evidence_consistency(AEC)={f.get('evidence_consistency')}, "
            f"evidence_region={f.get('evidence_region')} ({f.get('evidence_region_source')})"
        )
    findings_block = "\n".join(findings_lines) if findings_lines else "(no findings)"
    return (
        "You are a careful research assistant. Rewrite the STRUCTURED model output below "
        "into a short, neutral, readable summary for a RESEARCH audience.\n"
        "STRICT RULES:\n"
        "1. Use ONLY the values given. Do NOT add, infer, or omit any finding.\n"
        "2. This is NOT a diagnosis. Do NOT diagnose, recommend treatment, or give advice.\n"
        "3. Quote the exact grade and confidence for each finding.\n"
        "4. Begin with: 'Research-only, not diagnostic, not clinically validated, "
        "not for medical decision-making.'\n"
        "5. Do NOT mention any disease or structure not listed.\n\n"
        f"STUDY: {graph.get('study_id')}  MODEL: {graph.get('model_version')}  "
        f"DATASET: {graph.get('dataset_source')}\n"
        f"STRUCTURED FINDINGS:\n{findings_block}\n\n"
        "Write 4-8 sentences. End with the disclaimer line again."
    )


def ollama_generate(
    prompt: str, model: str, host: str = DEFAULT_OLLAMA_HOST, timeout: float = 120.0
) -> str | None:
    """Call a local Ollama model via HTTP. Returns text or None on any failure."""
    payload = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.1}}
    ).encode()
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (localhost only)
            data = json.loads(resp.read().decode())
        return data.get("response")
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        log.warning("Ollama unavailable (%s); falling back to deterministic report.", exc)
        return None


def generate_safe_llm_report(
    graph: dict[str, Any], model: str, host: str = DEFAULT_OLLAMA_HOST
) -> dict[str, Any]:
    """Generate LLM-polished wording, gated by the safety filter (fail closed).

    Returns ``{ok, text, reasons, model}``. ``ok=False`` (with ``text=None``) when
    Ollama is unavailable or the output fails the safety check.
    """
    text = ollama_generate(build_prompt(graph), model, host)
    if text is None:
        return {"ok": False, "text": None, "reasons": ["ollama_unavailable"], "model": model}
    safe, reasons = check_llm_safety(text, graph)
    if not safe:
        log.warning("LLM output rejected by safety filter: %s", reasons)
        return {"ok": False, "text": None, "reasons": reasons, "model": model}
    return {"ok": True, "text": text.strip(), "reasons": [], "model": model}
