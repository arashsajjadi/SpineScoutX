# v1.9 — best model packaging

> Research-only · not diagnostic. All weights published as **GitHub Release assets** (not ordinary Git).

## Best raw model — `spinescoutx-best-raw-v1.9.tar.gz`

Size: 495.4 MiB  |  SHA-256: `452154208c346ea05529cd47f8913c54d22b150c87bd1b1ed1d612bffcedc9ba`

Contains: 5 grader `best.pt` + `config.json` + `metrics.json` for canal, foraminal (L+R share one run), subarticular (L+R share one run), and the foraminal localizer.

**Best raw model = v1.0 deployed reference**. Locked-test 5-route macro 0.752. None of v1.1–v1.8c improved raw argmax severe recall.

## Triage config — `spinescoutx-triage-config-v1.9.tar.gz`

Size: 0.0 MiB  |  SHA-256: `969254b2c881221b5a36bf1529247ab14b1c7d37e75c243cfdd106505a7bba0b`

Contains: v1.7 severe-FN triage summary + triage output JSON + docs. **Does NOT change argmax predictions.** At 15% review budget, effective foraminal severe recall improves 0.724 → 0.933 (22/29 FN captured).

## Upload command (after v1.9.0 tag exists)

```bash
python scripts/verify_release_assets_v1_9.py
gh release upload v1.9.0-research-story-best-model \
  outputs/real/v1_9_packages/spinescoutx-best-raw-v1.9.tar.gz \
  outputs/real/v1_9_packages/spinescoutx-triage-config-v1.9.tar.gz \
  docs/assets/v1_9/checksums.txt
```
