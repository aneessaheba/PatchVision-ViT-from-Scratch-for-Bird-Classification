# PatchVision experiment protocol

**Status: approved November 6, 2026.** Any change from here on goes in the deviation log with its date and reason.

## 1. Question

Does training-only guidance improve a small Vision Transformer trained from scratch on CUB-200-2011, and how much of its lead over a matched CNN comes from guidance and not from architecture?

## 2. Fixed setup

| Item | Rule |
|---|---|
| Data | Official CUB split, using the existing 5,394 train, 600 validation and 5,794 test files |
| Input | 224 x 224, with the existing normalization and augmentation tiers |
| Pretrained weights | None |
| Hardware and software | RTX 5090, Python 3.12, PyTorch 2.11.0, torchvision 0.26.0, one recorded Docker image |
| Schedule | 300 epochs, early stopping off, for every model |
| Best epoch | Highest validation top-1; the earliest epoch wins a tie |
| Seeds | 42, 43 and 44 |
| Traceability | Every run records its config, environment and git commit |

The same seed does not give an exact repeat. Two runs of `ovf5_dp0.3_autoaug` with seed 42 gave 53.00% and 53.87% test top-1.

## 3. Metrics

**Primary:** test top-1 accuracy, as mean and standard deviation over the three seeds.

**Secondary:** top-5, macro-F1, ECE, train-validation gap, parameters, GFLOPs, training time, and latency on one GPU.

**Look-alike metrics**, fixed from species names before seeing any new results. A group is all species whose name ends in the same word, such as Gull or Warbler, with at least 2 species. This gives 37 groups, 167 species and 4,833 test images.
- **Within-group accuracy:** top-1 on those test images.
- **Within-group confusion:** the share of those images predicted as a different species in the same group.

**Localization, for guided ViT models:** attention inside the bird box, and attention near part locations if parts are used.

## 4. How decisions are made during development

Development uses validation data only, with the default `train.py` command.

- **Score:** the best 10-epoch rolling mean of validation top-1, averaged over the seeds used.
- **A change is adopted** only if it raises the score by at least 1.0 point. One repeat run moved this score by about 0.4 points, and the validation set has only 600 images.
- **If it falls short,** the lower validation loss decides, and if that is also close, the simpler model wins.
- **Single-seed runs** can only screen ideas. A change entering the final comparison needs three seeds.

| Work | Seeds |
|---|---|
| Pilots, tuning, depth and patch size | 1 |
| Ablations | 3 |
| Final comparison and data-size sweep | 3 |

**Equal tuning:** the CNNs get the same search as the ViT, covering stochastic depth, weight decay, auto augmentation, dropout, and the same two combinations, with the same decision rule.

## 5. Test set rules

- `--final-test` is used only after the final configurations are frozen and the commit is tagged.
- Each frozen configuration is evaluated once per seed, and nothing is tuned afterward.
- If a bug forces a rerun, the bug and both results go in the deviation log and the paper.

**To disclose in the paper:**
- Test results for all earlier runs were seen, and the regularization recipe came from those runs.
- 21 same-species near-duplicate pairs cross the official splits, 19 of them involving test images.

## 6. Primary comparisons and claim rule

| # | Comparison |
|---|---|
| P1 | ViT with part guidance vs. ViT with box guidance |
| P2 | ViT with box guidance vs. ViT without guidance |
| P3 | ViT vs. matched CNN, both tuned and unguided |
| P4 | ViT vs. matched CNN, both tuned and guided |

P1 is removed if the literature review or Phase 5 rules out part guidance. All other comparisons are reported as descriptive.

**A result counts as a claim only if both hold:**
1. The difference has the same direction in all three seeds.
2. The Holm-adjusted p-value across the primary comparisons is below 0.05. It comes from a paired bootstrap that resamples test images 10,000 times, using each image's accuracy averaged over the seeds.

Three seeds give limited power, so every comparison reports the per-seed differences and the 95% bootstrap interval. Exact paired tests for each seed are reported as supporting detail. Negative and inconclusive results are reported.

## 7. Required before the first new run

- [x] Turn early stopping off for all models
- [x] Make saved predictions and reported accuracy use the same top-1 result
- [ ] Commit an environment file and record the Docker image
- [ ] Finish the GvT code comparison and literature review

## 8. Deviation log

| Date | Change | Reason | Effect on results |
|---|---|---|---|
| | | | |
