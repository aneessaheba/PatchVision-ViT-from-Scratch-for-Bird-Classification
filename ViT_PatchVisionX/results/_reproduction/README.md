# Reproduction runs

These repeat an earlier run to check its result. They are kept here so result tables do not count them as extra seeds.

| Run | Original test top-1 | Rerun test top-1 | Paired test p-value |
|---|---|---|---|
| `ovf5_dp0.3_autoaug`, seed 42 | 53.00 % | 53.87 % | 0.082 |

Same settings and seed, on an RTX 5090 with PyTorch 2.11.0 under Docker on WSL2. The two runs make the same prediction on 69.6 % of test photos, so the same seed does not give an exact repeat.
