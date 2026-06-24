# Canal robust auto-inference — LOCKED-TEST confirmation (splits_v1)

> Research-only. Not diagnostic. Models retrained on splits_v1 `train`, selected on
> `dev` (auto, severe-aware), evaluated ONCE on the locked `test`. Cluster-bootstrap
> 95% CIs. Locked-test n=1480, severe=53.

## Severe recall [95% CI] | weighted log loss

| model | dev oracle | dev auto | **test oracle** | **test auto (real)** |
|---|---|---|---|---|
| oracle-trained control | 0.529 [0.400, 0.660] | 0.386 | 0.500 [0.379, 0.632] | 0.532 | 0.566 [0.420, 0.720] | 0.307 | 0.434 [0.306, 0.562] | 0.461 |
| **auto-trained robust** | 0.838 [0.727, 0.928] | 0.442 | 0.765 [0.649, 0.868] | 0.495 | 0.868 [0.772, 0.951] | 0.401 | 0.830 [0.725, 0.929] | 0.432 |

## Paired robust − control on locked test (auto, same nodes)
- severe recall Δ **+0.396** [+0.268, +0.529] (decisive=True)
- weighted log loss Δ -0.028 [-0.102, +0.037] (decisive=False; negative = better)
- McNemar severe (robust-catches / control-catches): 21 / 0, p=9.537e-07

Artifacts: `outputs/real/canal_locked_test.json`. Reproduce:
`python scripts/run_canal_locked_test.py`.
