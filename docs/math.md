# Mathematical Formulation

This document collects the equations implemented by AVQA, indexed by
the corresponding section of the reference paper / spec.

## Standard Attention (spec §7.4)

$$
Y_i = \frac{\sum_{j=1}^{N} \exp(Q_i K_j^\top) V_j}{\sum_{l=1}^{N} \exp(Q_i K_l^\top)}
$$

Complexity: $\mathcal{O}(N^2 d)$.

## Vector Quantization (spec §7.5)

$$
a(K_j) = \arg\min_a \|K_j - C_a\|^2, \quad \hat{K}_j = C_{a(K_j)}
$$

Only keys are quantized; queries and values remain unchanged.

## VQ Attention (spec §7.7)

$$
Y_i \approx \frac{\sum_a \exp(Q_i C_a^\top) \bar V_a}{\sum_a \exp(Q_i C_a^\top) n_a}
$$

where $\bar V_a = \sum_{j: a(K_j) = a} V_j$ and $n_a = |\{j: a(K_j) = a\}|$.

## Hierarchical Codebook (spec §7.8)

Total size $M_{\text{total}} = M_0 (1 + \mathcal{C})$.

## Mean Constraint (spec §7.9)

$$
C_p = \frac{1}{\mathcal{C}} \sum_{c=1}^{\mathcal{C}} C_{p,c}
$$

## Importance (spec §7.10)

$$
w_j(I) = \sum_{i \in I} \frac{A_{ij} n_j}{\bar Z_i}
$$

## Online Softmax (spec §7.14)

Running state: $(m, \ell, a)$ updated as:

$$
m' = \max(m, \max(\text{tile logits})), \quad
\alpha = e^{m - m'}, \quad
\beta_{ij} = e^{S_{ij} - m'}
$$

$$
\ell' = \alpha \ell + \sum_j \beta_{ij}, \quad
a' = \alpha a + \sum_j \beta_{ij} v_j
$$

Final output:

$$
Y_i = \frac{a_i}{\ell_i}
$$

## Correcting Attention (spec §7.13)

Replace parent $p$ with its children:

$$
\Delta A = A_c - A_p
$$

where $A_p$ is the (rescaled) parent contribution and $A_c$ is the
children contribution, computed incrementally.

## Parent Logit Recovery (spec §7.12)

Under the mean constraint:

$$
S_p = \frac{1}{\mathcal{C}} \sum_c S_c
$$

## Complexity (spec §7.16)

$$
\mathcal{O}\!\left(N (M_0 + P\mathcal{C}) D\right)
$$

For fixed codebook parameters, linear in sequence length.

## Invariants (spec §7.17)

1. **Hierarchy**: $C_p = \tfrac{1}{\mathcal{C}} \sum_c C_{p,c}$ (always).
2. **Assignment**: every key maps to exactly one parent (or one of its
   children after refinement).
3. **Conservation**: $\sum_a \bar V_a = \sum_j V_j$.
4. **Count**: $\sum_a n_a = N$.
5. **Attention**: result remains normalized after correction.

## Two-Stage Hierarchical Assignment (spec §8.5)

Per-key cost $\mathcal{O}(M_0 + \mathcal{C})$ instead of $\mathcal{O}(M_0 \mathcal{C})$.

## Adaptive Refinement (spec §9.7, §9.11)

1. Compute parent attention via online softmax.
2. Estimate importance $w_j$ for each parent codeword.
3. Select top-$P$ parents.
4. Spawn the $P \cdot \mathcal{C}$ children.
5. Recompute attention for the children.
6. Correct parent contributions.
7. Produce the final output.