import Mathlib

/-!
The arithmetic of the deviation `D(a,c)` in Section 4 of the paper.

At stage `r` the frequent value is `L` and the rare value is `K = (r + 1/2) L`.  With `a`
boxes holding `L` and `c` boxes holding `K`, the deviation of the remaining boxes from their
mean is `D(a,c) = ((r + 1/2)(1 - c) - a) L + (a + c - 1) μ + D₀`, where `D₀ = D(0,1)`
(equations (4.3) and (4.4) of the paper).
-/

/-- The identity (4.3).  Here `n * (1 - r / n - 1 / ((2 r + 1) n) - x / n) = n θ`, where `θ` is the
mass of the low values, `x = n ζ_r` and `y = n ζ'_r`; the target is `m = n θ μ + K + y - f`
(the paper has `f = 1`) and `K = (r + 1/2) L`.  The deviation `m - a L - c K - (n - a - c) μ`
equals `((r + 1/2)(1 - c) - a) L + (a + c - 1) μ + D₀` with `D₀ = (1 - r - 1/(2r+1) - x) μ + y - f`. -/
theorem deviation_identity (n a c r L μ x y f : ℝ) (hn : n ≠ 0) (_hr : 2 * r + 1 ≠ 0) :
    (n * (1 - r / n - 1 / ((2 * r + 1) * n) - x / n) * μ + (r + 1 / 2) * L + y - f)
        - a * L - c * ((r + 1 / 2) * L) - (n - a - c) * μ
      = ((r + 1 / 2) * (1 - c) - a) * L + (a + c - 1) * μ
        + ((1 - r - 1 / (2 * r + 1) - x) * μ + y - f) := by
  field_simp
  ring

/-- The bound (4.4), `|D₀| ≤ 2r + 2`, with `t = 1/(2r+1) ≤ 1/3`, `x = n ζ_r ≤ 1/16`,
`y = n ζ'_r ≤ 1/16` and `0 ≤ f ≤ 1`. -/
theorem d01_bound (r μ t x y f : ℝ) (hr : 1 ≤ r) (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 8 / 7)
    (ht0 : 0 ≤ t) (ht1 : t ≤ 1 / 3) (hx0 : 0 ≤ x) (hx1 : x ≤ 1 / 16)
    (hy0 : 0 ≤ y) (hy1 : y ≤ 1 / 16) (hf0 : 0 ≤ f) (hf1 : f ≤ 1) :
    |(1 - r - t - x) * μ + y - f| ≤ 2 * r + 2 := by
  rw [abs_le]
  constructor <;> nlinarith [mul_nonneg hμ0 hx0, mul_nonneg hμ0 ht0, mul_nonneg hμ0 (sub_nonneg.mpr hr)]

/-- Lemma 6 of the paper: every allocation other than `(a,c) = (0,1)` forces a deviation of
at least `L/4`, provided `L ≥ 13 r + 8`, `0 ≤ μ ≤ 8/7` and `|D₀| ≤ 2r + 2`. -/
theorem case_bound (r a c : ℕ) (hr : 1 ≤ r) (L μ D0 : ℝ)
    (hL : 13 * (r : ℝ) + 8 ≤ L) (hμ0 : 0 ≤ μ) (hμ1 : μ ≤ 8 / 7) (hD : |D0| ≤ 2 * r + 2)
    (hac : ¬ (a = 0 ∧ c = 1)) :
    L / 4 ≤ |(((r : ℝ) + 1 / 2) * (1 - c) - a) * L + ((a : ℝ) + c - 1) * μ + D0| := by
  have hr' : (1 : ℝ) ≤ r := by exact_mod_cast hr
  obtain ⟨hD1, hD2⟩ := abs_le.mp hD
  have hLμ : 0 ≤ L - μ := by linarith
  have ha0 : (0 : ℝ) ≤ a := Nat.cast_nonneg a
  rcases Nat.lt_or_ge c 2 with hc | hc
  · interval_cases c
    · -- c = 0
      rcases Nat.lt_or_ge r a with hra | hra
      · -- a ≥ r + 1: the deviation is at most -L/4
        have ha : (r : ℝ) + 1 ≤ a := by exact_mod_cast hra
        have key : (((r : ℝ) + 1 / 2) * (1 - ((0 : ℕ) : ℝ)) - a) * L + ((a : ℝ) + ((0 : ℕ) : ℝ) - 1) * μ + D0
            = -((a : ℝ) - r - 1 / 2) * (L - μ) + ((r : ℝ) - 1 / 2) * μ + D0 := by push_cast; ring
        have h1 : (1 / 2 : ℝ) * (L - μ) ≤ ((a : ℝ) - r - 1 / 2) * (L - μ) :=
          mul_le_mul_of_nonneg_right (by linarith) hLμ
        have h2 : ((r : ℝ) - 1 / 2) * μ ≤ ((r : ℝ) - 1 / 2) * (8 / 7) :=
          mul_le_mul_of_nonneg_left hμ1 (by linarith)
        rw [key]
        apply le_abs.mpr
        right
        linarith
      · -- a ≤ r: the deviation is at least L/4
        have ha : (a : ℝ) ≤ r := by exact_mod_cast hra
        have key : (((r : ℝ) + 1 / 2) * (1 - ((0 : ℕ) : ℝ)) - a) * L + ((a : ℝ) + ((0 : ℕ) : ℝ) - 1) * μ + D0
            = ((r : ℝ) + 1 / 2 - a) * L + (a : ℝ) * μ - μ + D0 := by push_cast; ring
        have h1 : (1 / 2 : ℝ) * L ≤ ((r : ℝ) + 1 / 2 - a) * L :=
          mul_le_mul_of_nonneg_right (by linarith) (by linarith)
        have h2 : 0 ≤ (a : ℝ) * μ := mul_nonneg ha0 hμ0
        rw [key]
        apply le_abs.mpr
        left
        linarith
    · -- c = 1, so a ≥ 1
      have hne : a ≠ 0 := fun h => hac ⟨h, rfl⟩
      have ha : (1 : ℝ) ≤ a := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hne
      have key : (((r : ℝ) + 1 / 2) * (1 - ((1 : ℕ) : ℝ)) - a) * L + ((a : ℝ) + ((1 : ℕ) : ℝ) - 1) * μ + D0
          = -((a : ℝ) * (L - μ)) + D0 := by push_cast; ring
      have h1 : L - μ ≤ (a : ℝ) * (L - μ) := by
        have := mul_le_mul_of_nonneg_right ha hLμ
        linarith
      rw [key]
      apply le_abs.mpr
      right
      linarith
  · -- c ≥ 2
    have hc' : (2 : ℝ) ≤ c := by exact_mod_cast hc
    have hA : 0 ≤ ((r : ℝ) + 1 / 2) * L - μ := by nlinarith
    have h1 : ((r : ℝ) + 1 / 2) * L - μ ≤ ((c : ℝ) - 1) * (((r : ℝ) + 1 / 2) * L - μ) := by
      have := mul_le_mul_of_nonneg_right (show (1 : ℝ) ≤ (c : ℝ) - 1 by linarith) hA
      linarith
    have h2 : 0 ≤ (a : ℝ) * (L - μ) := mul_nonneg ha0 hLμ
    have h3 : (5 / 4 : ℝ) * L ≤ ((r : ℝ) + 1 / 2) * L := by nlinarith
    have key : (((r : ℝ) + 1 / 2) * (1 - c) - a) * L + ((a : ℝ) + c - 1) * μ + D0
        = -(((c : ℝ) - 1) * (((r : ℝ) + 1 / 2) * L - μ)) - (a : ℝ) * (L - μ) + D0 := by ring
    rw [key]
    apply le_abs.mpr
    right
    linarith
