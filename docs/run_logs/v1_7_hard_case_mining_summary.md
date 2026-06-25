# v1.7 hard-case mining summary

> Research-only · not diagnostic. **Counts / IDs only — no imaging pixels.** Reproduce:
> `scripts/mine_hard_cases_v1_7.py`. Mined over `splits_v1` **train+dev** foraminal findings
> (16,740; locked-test never used for any cleaning decision). Models: deployed grader + v1.6
> baseline / LSS / joint / convnext-small (p_severe disagreement signal).

## Candidate groups (train+dev; total / right-for / left-for)

| group | total | R-for | L-for |
|---|---|---|---|
| A — severe false negatives (true severe, deployed ≠ severe) | 177 | **87** | 90 |
| B — confidently-normal severe miss (p_nm≥0.5, p_sev≤0.2) | 21 | 12 | 9 |
| C — moderate/severe borderline | 1550 | 745 | 805 |
| D — strong model disagreement (top 400) | 400 | 196 | 204 |
| F — high uncertainty (entropy/margin) | 4608 | 2204 | 2404 |
| G — control: correct severe | 494 | 239 | 255 |
| G — control: correct non-severe (top 80) | 80 | 41 | 39 |
| G — control: random easy | 60 | 34 | 26 |

(E — retrieval-conflict — deferred this run: requires the cached penultimate embeddings; the
disagreement signal D + borderline C cover the same conflicting-severity intent. Documented, not
silently dropped.)

## Curated review set (de-duplicated, priority-ranked)

- **704 cases** (right-foraminal **338**, left-foraminal **366**) — exceeds the targets (≥250
  R-for, ≥150 L-for, ≥100 control).
- By group: A 177 · C 189 · D 49 · F 110 · G-controls 179.
- **Right-foraminal severe FN in the pack: 87** (the highest-priority re-annotation target).
- Priority score upweights severe-FN, confidently-normal, disagreement, right-foraminal, and the
  weakest levels (L4-L5 / L5-S1).

## Key observation

The **confidently-normal severe-miss** group is small but decisive: only **21** train+dev foraminal
severe findings (12 right) are predicted normal_mild with p_nm≥0.5 and p_severe≤0.2 — these are the
prime "is this label right?" candidates. The far larger **borderline (1550)** and **uncertainty
(4608)** pools indicate the moderate/severe boundary is where most ambiguity lives — consistent with
the v1.6 autopsy that the ceiling is label-quality / ambiguity bound. The full curated set is the
input to the local-only review pack (`v1_7_review_pack_summary.md`).
