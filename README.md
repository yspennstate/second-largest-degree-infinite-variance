# The second largest outdegree of conditioned Galton–Watson trees with infinite variance

Companion material for the paper of the same title (Y. Shmalo, October 2026). [Read the paper](paper/main.pdf).

Let ξ be a random variable with values in {0, 1, 2, ...} and E ξ = 1, let Y_(1) ≥ Y_(2) ≥ ... be the outdegrees, in
decreasing order, of a Galton–Watson tree with offspring distribution ξ conditioned to have n vertices, and let
ξ_(1) ≥ ξ_(2) ≥ ... be the order statistics of n independent copies of ξ. By Theorem 19.7 of S. Janson, *Simply
generated trees, conditioned Galton–Watson trees, random allocations and condensation*, Probab. Surveys 9 (2012), if ξ
has finite variance then Y_(j) and ξ_(j) have asymptotically the same distribution for every fixed j. Problem 19.30 of
the survey (18.30 in the arXiv version) asks whether two consequences of that theorem remain true when the variance is
infinite:

- (ii) if n P(ξ ≥ h(n)) → 0, then Y_(1) < h(n) with high probability;
- (iii) if n P(ξ ≥ h(n)) → ∞, then Y_(j) ≥ h(n) with high probability, for every fixed j.

Kortchemski and Richier (*Condensation in critical Cauchy Bienaymé–Galton–Watson trees*, Ann. Appl. Probab. 29
(2019), Theorem 3 and the remark after it) showed that (ii) can fail. The paper shows that (iii) can fail as well.

- Theorem 1. There are an offspring distribution with mean 1 and infinite variance, and integers n_r → ∞ and L_r with
  n_r P(ξ ≥ L_r) ≥ r, such that with probability at least 1 − 1/r the conditioned tree with n_r vertices has exactly
  one vertex of outdegree at least L_r, of outdegree K_r = (r + 1/2) L_r, while every other vertex has outdegree at
  most L_r/(64 (log n_r)^2).
- Corollary 2. Along n_r each of the six conclusions (i)–(vi) of Janson's Theorem 19.7 fails, (iii) for every j ≥ 2.
  With probability tending to 1, the largest outdegree is larger than ξ_(1) by the factor r + 1/2 and the second
  largest is smaller than ξ_(2) by at least the factor 64 (log n_r)^2.

The same holds for the random allocation of n − 1 balls into n boxes. Whether (iii) can fail for j = 1 is not known.

The offspring distribution is built in stages. At stage r it has an atom at a frequent value L_r, of probability
r/n_r, and an atom at a rare value K_r, of probability 1/((2r + 1) n_r); every other value is at most
L_r/(64 (log n_r)^2) or larger than n_r. If a of n_r independent copies take the value L_r and c take the value K_r,
then on the event that the sum is n_r − 1 the small values must deviate from their mean by
((r + 1/2)(1 − c) − a) L_r, up to an error of order r. Because of the half in r + 1/2, this is at least L_r/4 in
absolute value unless (a, c) = (0, 1) (Lemma 6), and L_r is about log n_r standard deviations of the sum of the small
values. With a local limit estimate with explicit constants (Lemma 3), it follows that the conditioning removes every
copy of L_r and keeps exactly one copy of K_r.

## Contents

- `paper/` the LaTeX source and the compiled PDF.
- `code/checks.py` the computations of Section 6. For a one-stage version of the offspring distribution (values 0, 1,
  2, L, K) it computes the conditional law of (a, c) given that the sum is n − 1, cross-checked against a direct
  convolution, with a control in which L is of the order of the standard deviation. It checks the identity (4.3) in
  exact rational arithmetic and Lemma 6 on 346,500 cases, with a control showing that the bound of Lemma 6 fails for
  even L just below (88/7) r + 8; Lemma 3 against distributions computed by the fast Fourier transform for 48 random
  laws, with a control without the atom at 1; and the numerical constants of Sections 2–4. It also finds an admissible
  choice of n_1, ..., n_5 satisfying (C1)–(C5). `code/checks_output.txt` is its output.
- `code/CaseBound.lean` a Lean 4 proof, with Mathlib, of the identity (4.3), the bound (4.4) and Lemma 6. The comments
  in the file give the correspondence with the notation of the paper. `code/lean_output.txt` is the compiler's output
  and exit status.
- `SHA256SUMS.txt` checksums of the files in `paper/` and `code/`.

The computations test the identities and inequalities used in the proofs; they are not part of them.

## Running the checks

    cd code
    python checks.py > checks_output.txt

Python 3 with numpy, scipy and mpmath (run here with Python 3.14, numpy 2.4, scipy 1.17 and mpmath 1.3). The script
takes under a minute and ends with `all checks passed`; its exit status is 1 if any check fails.

The Lean file was checked with Lean 4.33.1 and Mathlib at commit 0df444a360eaa60ab8c11dca51a86af692955474. In a Lake
project that depends on that Mathlib,

    lake env lean CaseBound.lean

compiles it with no errors and no warnings.
