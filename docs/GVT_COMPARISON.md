# PatchVision-X compared with GvT

Compared against the [GvT paper](https://www.nature.com/articles/s41598-025-10408-0) (Scientific Reports 15, article 24520, 2025) and the [authors' public code](https://github.com/cvcoding/GvT), `code4GvT/models/vit.py` and `code4GvT/pygcn/layers.py`.

**Summary:** PatchVision-X follows the paper's main attention equations, so it is a GvT-style model. It is not a reproduction of GvT: it is much larger, adds MLP blocks, and interprets two under-specified or inconsistent parts of the paper. The public code also differs from the paper, so "follows GvT" must say which one.

## Where the model matches the paper

| Part | Paper | `patchvision/models.py` |
|---|---|---|
| Graph | Eq. 3: A = (A_I + I) x E, with 8-neighbour A_I | Same for `base`; other head groups use wider masks |
| Queries and keys | Eq. 4: normalized graph convolution | Same |
| Attention | Eq. 5: softmax of Q K^T, with no mask | Same in the default model |
| Sparse gate | Y = C W_c, sigmoid shrinkage, S_hat = M S M | Same structure; see differences below |
| Head mixing | Eq. 7: learnable Phi | Same |
| Values | Eq. 8: graph convolution with R, plus x | Same |
| Optimizer | AdamW, weight decay 0.05, learning rate 5e-4, cosine, batch 64, 300 epochs | Same |

## Where it differs from the paper

| Part | Paper | PatchVision-X | Why it matters |
|---|---|---|---|
| Similarity E | Eq. 1 and 2 use exp(cosine distance), which gives similar tokens less weight, although the text calls E a similarity | Cosine similarity with a learned temperature | Deliberate interpretation; disclose it |
| Graph pooling | U is t x t and C = U X^T, whose dimensions do not work | t learned queries attend over tokens, C = softmax(U K^T), A' = C A C^T | Necessary interpretation; disclose it |
| MLP in each block | None in the complexity analysis, and the public code uses `nn.Identity` | MLP with ratio 3 | Major addition |
| Size | 7 to 12 blocks; hidden size 32 to 128; 87K and 173K parameters in the configurations checked; 64 tokens pooled to 16 | 8 blocks; hidden size 256; 4.27M parameters; 196 tokens pooled to 49 | Results are not comparable to the paper's |
| Gate input | No standardization | Standardized over tokens and divided by n | Our addition to the gate |
| Gate slope | Divides by u | Multiplies by a learned slope | Same start at 1; different training behaviour |
| Head mixing start and sign | Not specified | Starts as identity, with negatives clamped to 0 | Our choices |
| Output | Linear layer | Average of tokens, then linear layer | Paper does not specify the pooling |

The LFE block, multi-scale head groups, box and flip losses, EMA and box crop are not from GvT.

## Where the public code differs from the paper

| Part | Paper | Public code |
|---|---|---|
| Attention | Eq. 5 has no mask, and the text describes global attention | Masks non-neighbours before the softmax (`vit.py` line 279), so attention is limited to the graph |
| Order | Softmax, then gate, then mixing | Gate, then mixing, then mask, then softmax |
| Graph | Includes E and I | The current adjacency has neither, and the self-loop code is commented out |
| Pooling | Second-order formula | Gumbel-softmax assignment, also used for final token pooling |

This file may not be the exact version behind the paper. It is only one training script, for a lung dataset.

## What this means for the project

1. **Attention range:** the default model follows the paper's Eq. 5. The `restrict_attention` variant is closer to the public code, which limits attention to the graph. Test both instead of assuming either one is what GvT does.
2. **Sparsity:** the paper's sparse gate has no sparsity penalty, and the paper reports no measured sparsity level. A gate that stays near 0.5 here is therefore not, by itself, evidence of a bug.
3. **Pooling:** the paper describes aggregating semantic information, not discarding background.
4. **`base` is not GvT:** describe it as a GvT-style baseline with MLP blocks.
5. **Reported accuracy:** GvT reports the highest accuracy from each of five runs, averaged. That is more optimistic than choosing an epoch on validation, so do not compare the numbers directly. The paper also does not test CUB; Oxford-IIIT Pet is its closest fine-grained dataset.
