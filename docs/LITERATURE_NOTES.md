# Literature notes

Prior-art review from November 6, 2026. A search cannot prove something is new, so recheck these before submission. **Checked** means the source was re-read directly. **From review** means it comes from the review's reading, often through a summarizing tool. **Snippet** means only a search listing was seen. Verify every number before citing it.

## Verdicts

| Idea | Verdict | Position in the paper |
|---|---|---|
| Keypoints supervising an intermediate pooling step, from scratch | Partly anticipated | Main question, framed as a measurement |
| Multi-hop query and key neighbourhoods per head group | Largely anticipated | A tested component, not a novelty |
| Attention restricted per head | Already published | A tested variant, not a novelty |
| Flip consistency and attention supervision | Already published | Existing components, with citations |

No paper was found that compares keypoints with boxes head to head from scratch, or that measures look-alike species confusion under keypoint guidance.

## Closest work on keypoint guidance

| Paper | Status | What it does | How this project differs |
|---|---|---|---|
| [Spatially Grounded CBMs](https://arxiv.org/html/2606.04364), arXiv 2026 | Checked | Part queries over patches, with a training-only keypoint loss on attention centroids | It uses a frozen, pretrained DINOv3 and a concept bottleneck, with no pooled tokens entering later blocks |
| [Privileged Pooling](https://arxiv.org/html/2003.09168), arXiv | From review | Keypoints supervise attention maps on CUB, training only | Pretrained ResNet-101, no tokens |
| [Concept Transformers](https://iclr.cc/virtual/2022/poster/5962), ICLR 2022 | From review | Attention supervised on CUB | Concept attention, not pooling |
| [BPBreID](https://arxiv.org/abs/2211.03679), WACV 2023 | From review | Part attention supervised during training | CNN, person re-identification |
| [PFD](https://arxiv.org/abs/2112.02466), AAAI 2022 | From review | Pose information matched to decoder queries | Different task and mechanism |
| [Mask-CNN](https://arxiv.org/abs/1605.06878) | From review | Part annotations train masks that select descriptors | CNN |

**Part discovery without keypoints:** [PDiscoFormer](https://arxiv.org/abs/2407.04538), ECCV 2024 (from review); [PARTICLE](https://arxiv.org/abs/2309.13822), ICCVW 2023 (from review); [PART](https://arxiv.org/abs/2212.13685), TIP 2021 (snippet); [TCFormer](https://arxiv.org/abs/2204.08680), CVPR 2022 (snippet).

## Closest work on multi-scale heads

| Paper | Status | Relevance |
|---|---|---|
| [MNA-GT](https://ar5iv.labs.arxiv.org/html/2211.07970), arXiv 2022 | Checked | Queries and keys from k-hop neighbourhoods, with no adjacency mask stated, on general graphs and one attention block per hop |
| [Shunted Transformer](https://ar5iv.labs.arxiv.org/html/2111.15193), CVPR 2022 | From review | Different scale per head group |
| [MSCViT](https://arxiv.org/html/2501.06040v2), arXiv 2025 | Checked | Per-head key and value fusion on small datasets |
| [ViT-P](https://arxiv.org/pdf/2203.02358), arXiv 2022 | From review | Per-head receptive fields from 3x3 to global |
| [MaiT](https://arxiv.org/pdf/2207.03006), arXiv 2022 | From review | Masks on selected heads |

On an 8-connected grid, hops 1, 2 and 4 have 3x3, 5x5 and 9x9 support, so reviewers may read this as per-head multi-kernel smoothing.

## Attention supervision to cite

All snippet only: [GAIN](https://openaccess.thecvf.com/content_cvpr_2018/html/Li_Tell_Me_Where_CVPR_2018_paper.html), CVPR 2018; [Right for the Right Reasons](https://www.ijcai.org/proceedings/2017/371), IJCAI 2017; [Guo et al.](https://openaccess.thecvf.com/content_CVPR_2019/html/Guo_Visual_Attention_Consistency_Under_Image_Transforms_for_Multi-Label_Image_Classification_CVPR_2019_paper.html), CVPR 2019; [MixHop](https://arxiv.org/abs/1905.00067), ICML 2019. No exact paper for a box attention loss on ViTs was found; the nearest is [Object-Focused Attention](https://arxiv.org/abs/2504.08166), arXiv 2025.

## From-scratch results on CUB-200

| Model | [Cao et al.](https://ar5iv.labs.arxiv.org/html/2201.10728), ECCV 2022, 800 epochs | [Choi et al.](https://arxiv.org/pdf/2608.10723), arXiv 2026, vanilla |
|---|---|---|
| DeiT-Tiny | 24.3 | 17.69 |
| T2T-ViT-7 | 35.7 | 26.73 |
| PVTv2 (B0 in Cao et al.) | 66.6 | 47.15 |

The Choi et al. values are checked from its Table 1, and its best CUB result with a from-scratch ResNet-56 teacher is 54.59 (T2T-7). Its CUB epoch count was not confirmed. PatchVision-X reached 53.00 and 53.87 on one seed, using 5,394 training images and a tuned recipe. The large gaps between papers show that baselines depend heavily on the protocol, so report baselines on this project's own split.

## Evidence about the hypothesis

- In Spatially Grounded CBMs (checked), CUB accuracy is 86.4 with no keypoints, 88.5 with keypoints on 0.5% of images, and 88.9 with all of them. Pointing accuracy rises from 8.0 to 70.4, so keypoints mainly improve localization.
- Privileged Pooling (from review) reports larger gains with few training images and small gains on full CUB.
- All of this uses pretrained backbones, so it may not transfer to from-scratch training at about 50% accuracy.

## Disclosure

The project's public repositories already describe its architecture and results. Check the target venue's anonymity and preprint rules.
