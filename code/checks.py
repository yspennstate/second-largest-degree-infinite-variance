"""Numerical checks for the paper on part (iii) of Janson's Problem 19.30.

Run:  python checks.py > checks_output.txt      (numpy, scipy, mpmath; a few minutes)

1. A one-stage version of the offspring law (values 0, 1, 2, L, K), conditioned on S_n = n - 1,
   computed in double precision from the decomposition over the numbers a and c of boxes
   holding L and K balls; cross-checked against a direct convolution for small n; negative
   control with L of the order of the standard deviation of the bulk instead of log(n) times it.
2. The identity D(a,c) = E(a,c) L + (a+c-1) mu + D(0,1) in exact rational arithmetic, the bound
   |D(0,1)| <= 2r+2, the bound |D(a,c)| >= L/4 for (a,c) != (0,1) on a finite range, and a
   control showing that the constant in the condition L >= 13r+8 cannot be lowered much.
3. The local limit lemma (Lemma 3 of the paper) against distributions of T_N computed by FFT
   in double precision, with a control showing that the atom hypothesis is needed.
4. The numerical constants and elementary inequalities used in the proofs.
5. An explicit admissible choice of n_1, ..., n_5 (high-precision arithmetic).
Every check prints PASS or FAIL; the exit status is 1 if any check fails.
"""
import math
import random
import sys
from fractions import Fraction as F

import mpmath as mp
import numpy as np
from scipy import integrate
from scipy.special import gammaln, logsumexp

FAILS = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("   " + detail) if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------------------------------------
# 1. One-stage example
# ---------------------------------------------------------------------------------------------

def low_sum_logpmf(M, j, x0, x1, x2):
    """log P(T_M = j), T_M a sum of M i.i.d. variables with law (x0, x1, x2) on {0, 1, 2}.
    The trinomial terms are log-concave in k = number of 2s, so a window around the mode
    carries the sum; the window edges are checked to lie 200 nats below the maximum."""
    if j < 0 or j > 2 * M:
        return -np.inf
    klo, khi = max(0, j - M), j // 2
    l0, l1, l2 = math.log(x0), math.log(x1), math.log(x2)
    cst = gammaln(M + 1)

    def terms(k):
        zeros = M - j + k
        ones = j - 2 * k
        return cst - gammaln(zeros + 1) - gammaln(ones + 1) - gammaln(k + 1) + zeros * l0 + ones * l1 + k * l2

    lo, hi = klo, khi
    if khi - klo > 20000:
        grid = np.unique(np.linspace(klo, khi, 4001).astype(np.int64))
        kstar = int(grid[int(np.argmax(terms(grid)))])
        w = int((khi - klo) / 4000 + 30 * math.sqrt(M) + 100)
        lo, hi = max(klo, kstar - w), min(khi, kstar + w)
    t = terms(np.arange(lo, hi + 1, dtype=np.int64))
    tmax = float(t.max())
    if lo > klo:
        assert t[0] < tmax - 200, "window too narrow (left)"
    if hi < khi:
        assert t[-1] < tmax - 200, "window too narrow (right)"
    return float(logsumexp(t))


def stage_law(n, N, L):
    """P(2) = 1/4, P(1) = 1/2 - e, P(0) = 1/4 - p - q + e, P(L) = p = N/n,
    P(K) = q = 1/((2N+1)n), K = (2N+1)L/2, e = pL + qK, so that the mean is 1."""
    K = (2 * N + 1) * L // 2
    p = N / n
    q = 1 / ((2 * N + 1) * n)
    e = p * L + q * K
    P0, P1, P2 = 0.25 - p - q + e, 0.5 - e, 0.25
    assert min(P0, P1) > 0, "stage mass too large for this n"
    return K, p, q, (P0, P1, P2)


def one_stage(n, N, L, m=None):
    """Conditional law of (a, c) = (#boxes with L balls, #boxes with K balls) given S_n = m.
    Pairs whose Hoeffding bound lies 60 nats below the largest exact term are not evaluated;
    their total weight relative to the sum is at most (number of such pairs) * e^-60."""
    K, p, q, (P0, P1, P2) = stage_law(n, N, L)
    pi = P0 + P1 + P2
    x0, x1, x2 = P0 / pi, P1 / pi, P2 / pi
    mux = x1 + 2 * x2
    if m is None:
        m = n - 1
    pairs = [(a, c) for a in range(0, m // L + 1) for c in range(0, (m - a * L) // K + 1)]
    A = np.array([pr[0] for pr in pairs], dtype=np.int64)
    C = np.array([pr[1] for pr in pairs], dtype=np.int64)
    Mv = n - A - C
    Jv = m - A * L - C * K
    pref = (gammaln(n + 1) - gammaln(A + 1) - gammaln(C + 1) - gammaln(Mv + 1)
            + A * math.log(p) + C * math.log(q) + Mv * math.log(pi))
    bound = pref - (Jv - Mv * mux) ** 2 / (2.0 * Mv)        # Hoeffding: summands in [0, 2]
    order = np.argsort(-bound)
    logt = {}
    best = -np.inf
    pruned = 0
    for idx in order:
        if bound[idx] < best - 60:
            pruned = len(order) - len(logt)
            break
        a, c = pairs[idx]
        val = float(pref[idx]) + low_sum_logpmf(int(Mv[idx]), int(Jv[idx]), x0, x1, x2)
        logt[(a, c)] = val
        best = max(best, val)
    tot = float(logsumexp(list(logt.values())))
    cond = {key: math.exp(val - tot) for key, val in logt.items()}
    mean = P1 + 2 * P2 + p * L + q * K
    return dict(K=K, p=p, q=q, cond=cond, logt=logt, logtot=tot, mean=mean, law=(P0, P1, P2),
                pruned=pruned, pruned_rel=pruned * math.exp(-60), pairs=len(pairs))


def conv_power_at(law, n, m):
    """P(sum of n i.i.d. copies of a (sub-)probability vector equals m), by repeated squaring."""
    res = np.zeros(m + 1)
    res[0] = 1.0
    base = law[:m + 1].copy()
    k = n
    while k:
        if k & 1:
            res = np.convolve(res, base)[:m + 1]
        k >>= 1
        if k:
            base = np.convolve(base, base)[:m + 1]
    return res[m]


def section1():
    print("\n== 1. One-stage example (offspring law on {0,1,2,L,K}), conditioned on S_n = n-1 ==")
    for n, N, L in [(2000, 2, 300), (3000, 3, 300), (4000, 2, 600)]:
        res = one_stage(n, N, L)
        K, p, q, (P0, P1, P2) = res["K"], res["p"], res["q"], res["law"]
        m = n - 1
        law = np.zeros(m + 1)
        law[0], law[1], law[2] = P0, P1, P2
        law[L] += p
        no_rare_law = law.copy()
        law[K] += q
        full = conv_power_at(law, n, m)
        no_rare = conv_power_at(no_rare_law, n, m)
        dec_no_rare = sum(v for (a, c), v in res["cond"].items() if c == 0)
        ok = (abs(math.log(full) - res["logtot"]) < 1e-9 and abs(no_rare / full - dec_no_rare) < 1e-9
              and abs(res["mean"] - 1) < 1e-12)
        check(f"decomposition = convolution (n={n}, N={N}, L={L}, K={K}, mean 1)", ok,
              f"P(S=m) {full:.10e} vs {math.exp(res['logtot']):.10e};  P(no K-box | S=m) "
              f"{no_rare / full:.12f} vs {dec_no_rare:.12f}")
    print("  n        N  L       K        1-P(a=0,c=1|S)  P(a+c>=2|S)   iid P(>=2 boxes >= L)  "
          "n P(xi>=L)  pairs/pruned")
    rows = []
    for n in (10**4, 10**5, 10**6):
        s = math.sqrt(1.5 * n)
        for N in (2, 4):
            for tag, L in (("L = s log n", 2 * math.ceil(math.log(n) * s / 2)), ("control L = s", 2 * math.ceil(s / 2))):
                try:
                    res = one_stage(n, N, L)
                except AssertionError as exc:
                    print(f"  {n:<8d} {N:<2d} {L:<7d} skipped ({exc})")
                    continue
                cond = res["cond"]
                miss = sum(v for key, v in cond.items() if key != (0, 1))
                two = sum(v for (a, c), v in cond.items() if a + c >= 2)
                pb = res["p"] + res["q"]
                iid_two = 1 - (1 - pb) ** n - n * pb * (1 - pb) ** (n - 1)
                rows.append(dict(tag=tag, n=n, N=N, miss=miss, two=two, iid=iid_two, pruned_rel=res["pruned_rel"]))
                print(f"  {n:<8d} {N:<2d} {L:<7d} {res['K']:<8d} {miss:<15.3e} {two:<13.3e} {iid_two:<22.4f} "
                      f"{n * pb:<11.3f} {res['pairs']}/{res['pruned']}   [{tag}]", flush=True)
    check("pruned weight is negligible in every case (< 1e-20 relative)",
          all(r["pruned_rel"] < 1e-20 for r in rows), f"max {max(r['pruned_rel'] for r in rows):.1e}")
    big = [r for r in rows if r["tag"].startswith("L =") and r["n"] == 10**6]
    check("L ~ s log n, n = 1e6: the conditioned allocation has exactly one box >= L, of size K",
          len(big) == 2 and all(r["miss"] < 1e-3 for r in big),
          ", ".join(f"N={r['N']}: 1-P = {r['miss']:.2e} (iid: P(>=2 boxes >= L) = {r['iid']:.3f})" for r in big))
    trend = [r for r in rows if r["tag"].startswith("L =") and r["N"] == 2]
    check("1 - P(a=0,c=1|S) decreases in n along L ~ s log n (N=2)",
          len(trend) == 3 and all(trend[i]["miss"] > trend[i + 1]["miss"] for i in range(2)),
          " > ".join(f"{r['miss']:.2e}" for r in trend))
    ctrl = [r for r in rows if r["tag"].startswith("control")]
    check("negative control: with L ~ s a second box >= L survives the conditioning",
          len(ctrl) == 6 and all(r["two"] > 0.3 for r in ctrl),
          ", ".join(f"n={r['n']},N={r['N']}: {r['two']:.3f}" for r in ctrl))


# ---------------------------------------------------------------------------------------------
# 2. The deviation identity and the case bound
# ---------------------------------------------------------------------------------------------

def section2():
    print("\n== 2. D(a,c): identity, |D(0,1)| <= 2r+2, |D(a,c)| >= L/4 off (0,1) ==")
    rng = random.Random(20261013)
    worst = None
    for trial in range(4000):
        r = rng.randint(1, 20)
        n = rng.randint(10**6, 10**12)
        L = 2 * rng.randint(7 * r + 4, 10**5)
        mu = F(rng.randint(0, 8000), 7000)                       # mu in [0, 8/7]
        nqH = F(rng.randint(0, 1000), 1000) / 2 ** (r + 3)         # n q^H in [0, 2^{-r-3}]
        neH = F(rng.randint(0, 1000), 1000) / 2 ** (r + 3)         # n e^H in [0, 2^{-r-3}]
        f = F(rng.randint(0, 1))                                 # m = n - f
        p, q = F(r, n), F(1, (2 * r + 1) * n)
        pi = 1 - p - q - nqH / n
        K = (2 * r + 1) * F(L, 2)
        nun = n * pi * mu + n * (p * L + q * K) + neH             # n E(xi), split by value class
        m = nun - f
        D01 = (1 - r - F(1, 2 * r + 1) - nqH) * mu + neH - f
        for a, c in ((0, 0), (0, 1), (1, 0), (1, 1), (r, 0), (r + 1, 0), (0, 2), (3, 2), (2 * r + 3, 1)):
            direct = m - a * L - c * K - (n - a - c) * mu
            formula = ((r + F(1, 2)) * (1 - c) - a) * L + (a + c - 1) * mu + D01
            if direct != formula:
                worst = (r, n, L, a, c)
                break
        if worst:
            break
    check("identity D(a,c) = E(a,c)L + (a+c-1)mu + D(0,1), exact rationals, 4000 random cases",
          worst is None, "" if worst is None else f"counterexample {worst}")
    # D(0,1) is affine in each of mu, n q^H, n e^H, f separately, so its extremes over the box are at vertices
    ok = True
    for r in range(1, 200):
        for mu in (F(0), F(8, 7)):
            for nqH in (F(0), F(1, 2 ** (r + 3))):
                for neH in (F(0), F(1, 2 ** (r + 3))):
                    for f in (F(0), F(1)):
                        D01 = (1 - r - F(1, 2 * r + 1) - nqH) * mu + neH - f
                        ok &= abs(D01) <= 2 * r + 2
    check("|D(0,1)| <= 2r+2 for mu in [0,8/7], f in {0,1}, r < 200 (vertices)", ok)
    # |D(a,c)| >= L/4 off (0,1): D is affine in (mu, D01); if the four vertices of [0,8/7] x [-(2r+2), 2r+2]
    # give values of one sign and modulus >= L/4, the same holds on the whole rectangle
    bad = None
    count = 0
    for r in range(1, 31):
        L0 = 13 * r + 8
        L0 += L0 % 2
        for L in list(range(L0, L0 + 60, 2)) + [L0 + 1000, 10 * L0, 10**6]:
            for c in range(0, 6):
                for a in range(0, 3 * r + 12):
                    if (a, c) == (0, 1):
                        continue
                    E = (r + F(1, 2)) * (1 - c) - a
                    vals = [E * L + (a + c - 1) * mu + d0 for mu in (F(0), F(8, 7))
                            for d0 in (F(-(2 * r + 2)), F(2 * r + 2))]
                    count += 1
                    same_sign = all(v > 0 for v in vals) or all(v < 0 for v in vals)
                    if not same_sign or min(abs(v) for v in vals) < F(L, 4):
                        bad = (r, L, a, c)
                        break
                if bad:
                    break
            if bad:
                break
        if bad:
            break
    check(f"|D(a,c)| >= L/4 for all (a,c) != (0,1), L >= 13r+8 even, a <= 3r+11, c <= 5 ({count} cases)",
          bad is None, "" if bad is None else f"fails at r,L,a,c = {bad}")
    # control: below (88/7) r + 8 the vertex (a,c) = (r+1,0), mu = 8/7, D01 = 2r+2 violates the bound
    viol = 0
    for r in range(1, 31):
        Lb = 2 * math.floor(((F(88, 7) * r + 8) - F(1, 1000)) / 2)
        D = (r + F(1, 2) - (r + 1)) * Lb + r * F(8, 7) + (2 * r + 2)
        viol += abs(D) < F(Lb, 4)
    check("control: for even L just below (88/7)r + 8 the case c = 0 fails at a vertex (r = 1..30)", viol == 30,
          f"{viol}/30 violations")


# ---------------------------------------------------------------------------------------------
# 3. The local limit lemma
# ---------------------------------------------------------------------------------------------

def lclt_error(probs, N):
    probs = np.asarray(probs, dtype=float)
    b = len(probs) - 1
    vals = np.arange(b + 1)
    mu = float((vals * probs).sum())
    v = float(((vals - mu) ** 2 * probs).sum())
    s = math.sqrt(N * v)
    size = 1 << int(math.ceil(math.log2(N * b + 1)))
    padded = np.zeros(size)
    padded[:b + 1] = probs
    pmf = np.fft.ifft(np.fft.fft(padded) ** N).real
    j = np.arange(size)
    g = np.exp(-((j - N * mu) ** 2) / (2 * s * s)) / (s * math.sqrt(2 * math.pi))
    return b, s, s * float(np.max(np.abs(pmf - g))), float(pmf.sum())


def section3():
    print("\n== 3. Local limit lemma: s|P(T_N=j) - g(j)| <= (b/s)^(1/4) + 1/N + s exp(-2 g^2 N/(pi^2 b^2)) ==")
    rng = np.random.default_rng(1913)
    gamma = 1 / 8
    worst_ratio = worst_sharp = worst_main = 0.0
    n_cases = 0
    mass_ok = True
    for b in (2, 3, 4, 6, 10, 25):
        for trial in range(4):
            w = rng.dirichlet(np.ones(b + 1) * (0.3 if trial % 2 else 2.0))
            probs = gamma * np.eye(b + 1)[0] + gamma * np.eye(b + 1)[1] + (1 - 2 * gamma) * w
            vals = np.arange(b + 1)
            v = float(((vals - (vals * probs).sum()) ** 2 * probs).sum())
            N0 = int(math.ceil(256 * b * b / v)) + 4
            for N in (N0, 4 * N0):
                if N * b > 4_000_000:
                    continue
                bb, s, err, tot = lclt_error(probs, N)
                tail = s * math.exp(-2 * gamma ** 2 * N / (math.pi ** 2 * b ** 2))
                bound = (b / s) ** 0.25 + 1 / N + tail
                sharp = 0.125 * (b / s) ** 0.25 + 0.342 / N + tail
                n_cases += 1
                mass_ok &= abs(tot - 1) < 1e-9
                worst_ratio = max(worst_ratio, err / bound)
                worst_sharp = max(worst_sharp, err / sharp)
                worst_main = max(worst_main, err / (0.125 * (b / s) ** 0.25 + 0.342 / N))
    check("FFT distributions have total mass 1", mass_ok)
    check(f"bound holds on {n_cases} random laws (b = 2..25, P(0), P(1) >= 1/8, s >= 16b)", worst_ratio <= 1,
          f"max error/bound = {worst_ratio:.4f}")
    check("the sharper form 0.125 (b/s)^(1/4) + 0.342/N + tail also holds", worst_sharp <= 1,
          f"max error/sharper bound = {worst_sharp:.4f}")
    check("the error is below 0.125 (b/s)^(1/4) + 0.342/N even without the tail term", worst_main <= 1,
          f"max error/(first two terms) = {worst_main:.4f}")
    # control: a law on {0, 2} (P(1) = 0) has span 2 and the local error does not vanish
    b, s, err, _ = lclt_error([0.5, 0.0, 0.5], 20000)
    check("control: without the atom at 1 (law on {0,2}) the error stays near 1/sqrt(2 pi)",
          err > 0.39 and err > 0.125 * (b / s) ** 0.25 + 0.342 / 20000, f"s|P-g| = {err:.4f} at N = 20000")


# ---------------------------------------------------------------------------------------------
# 4. Constants
# ---------------------------------------------------------------------------------------------

def section4():
    print("\n== 4. Constants and elementary inequalities ==")
    u = np.linspace(1e-6, 1, 200001)
    rmin = float(np.min((1 - np.cos(u)) / u ** 2))
    check("1 - cos u >= (11/24) u^2 on (0,1]", rmin >= 11 / 24, f"min ratio {rmin:.6f} >= {11 / 24:.6f}")
    t = np.linspace(1e-6, math.pi, 200001)
    tmin = float(np.min((1 - np.cos(t)) / t ** 2))
    check("1 - cos t >= 2t^2/pi^2 on (0,pi] (equality at pi)", tmin >= 2 / math.pi ** 2 - 1e-12,
          f"min ratio {tmin:.6f} vs {2 / math.pi ** 2:.6f}")
    x = np.linspace(0, 0.5, 100001)
    check("1 - x >= exp(-x - x^2) on [0,1/2]", bool(np.all(1 - x >= np.exp(-x - x ** 2) - 1e-15)))
    x = np.linspace(0, 10, 100001)
    check("|e^(-x) - 1 + x| <= x^2/2 on [0,10]", bool(np.all(np.abs(np.exp(-x) - 1 + x) <= x ** 2 / 2 + 1e-15)))
    al = 23 / 64
    m3 = integrate.quad(lambda y: abs(y) ** 3 * math.exp(-al * y * y), -np.inf, np.inf)[0]
    m4 = integrate.quad(lambda y: y ** 4 * math.exp(-al * y * y), -np.inf, np.inf)[0]
    check("int |u|^3 e^{-a u^2} du = 1/a^2 (a = 23/64)", abs(m3 - 1 / al ** 2) < 1e-8, f"{m3:.6f}")
    check("int u^4 e^{-a u^2} du = (3/4) sqrt(pi) a^{-5/2}", abs(m4 - 0.75 * math.sqrt(math.pi) * al ** -2.5) < 1e-8,
          f"{m4:.6f}")
    ok = True
    for A in (2.0, 2.5, 3.0, 7.5):
        for a2 in (11 / 24, 0.5):
            tail = 2 * integrate.quad(lambda y: math.exp(-a2 * y * y), A, np.inf)[0]
            ok &= tail <= math.exp(-a2 * A * A) / (a2 * A)
    check("Gaussian tail int_(|u|>A) e^(-a u^2) du <= e^(-a A^2)/(a A)", ok)
    c_b = (1 / (2 * math.pi)) * ((1 / 6) * (64 / 23) ** 2 / 8 + (24 / 11) * math.exp(-11 / 6) + 2 * math.exp(-2))
    c_n = (1 / (2 * math.pi)) * (3 * math.sqrt(math.pi) / 32) * (64 / 23) ** 2.5
    check("Lemma 3 constants: coefficient of (b/s)^(1/4) <= 0.125 and of 1/N <= 0.342",
          c_b <= 0.125 and c_n <= 0.342, f"{c_b:.5f}, {c_n:.5f}")
    check("Bernstein exponent: 32 (8/7 + 1/768) <= 37", 32 * (8 / 7 + 1 / 768) <= 37, f"{32 * (8 / 7 + 1 / 768):.4f}")
    tgt = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.01) - 3 / 40
    check("target constant: ((2pi)^(-1/2) e^(-0.01) - 3/40)/sqrt(8/7) >= 0.29, and 2/0.29 <= 7",
          tgt / math.sqrt(8 / 7) >= 0.29 and 2 / 0.29 <= 7, f"{tgt / math.sqrt(8 / 7):.4f}")
    check("|D(0,1)| arithmetic: (r - 2/3 + 1/16)(8/7) + 1/16 + 1 <= 2r + 2 for r >= 1",
          all((r - 2 / 3 + 1 / 16) * 8 / 7 + 1 / 16 + 1 <= 2 * r + 2 for r in range(1, 10000)))
    check("case bounds: L >= 13r+8 gives L/2 - (22/7)r - 2, L - 8/7 - 2r - 2, (r+1/2)L - 8/7 - 2r - 2 >= L/4",
          all(L / 2 - 22 * r / 7 - 2 >= L / 4 and L - 8 / 7 - 2 * r - 2 >= L / 4
              and (r + .5) * L - 8 / 7 - 2 * r - 2 >= L / 4
              for r in range(1, 2000) for L in (13 * r + 8, 13 * r + 9, 40 * r + 100)))
    # stage-1 variance: Var X = x0 x1 + 4 x0 x2 + x1 x2 is concave on the plane x0 + x1 + x2 = 1
    # (second form -2(2d0 + d1)^2), so its minimum over x0 >= 1/8, x1 >= 3/8, x2 >= 1/4 is at a vertex
    verts = [(F(1, 8), F(3, 8), F(1, 2)), (F(1, 8), F(5, 8), F(1, 4)), (F(3, 8), F(3, 8), F(1, 4))]
    vmin = min(x0 * x1 + 4 * x0 * x2 + x1 * x2 for x0, x1, x2 in verts)
    check("stage-1 variance: min over the region = 23/64 >= V_1/6 = 1/4", vmin == F(23, 64) and vmin >= F(1, 4),
          f"min = {vmin}")
    check("variance for r >= 2: V - 1/8 - (8/7)^2 >= V/6 once V >= 2", 2 * 5 / 6 >= 1 / 8 + (8 / 7) ** 2)
    check("pi^(n-1) >= e^(-r-1): 1/3 + 1/16 + 1/2 <= 1", 1 / 3 + 1 / 16 + 1 / 2 <= 1)
    check("e^(-D^2/(2s'^2)) >= e^(-0.01) when s'^2 >= 200 (r+1)^2 and |D| <= 2r+2", (2 * 2) ** 2 / (2 * 200 * 4) <= 0.01)


# ---------------------------------------------------------------------------------------------
# 5. An explicit admissible sequence
# ---------------------------------------------------------------------------------------------

def section5(R=5):
    print("\n== 5. An admissible choice of n_1, ..., n_%d (n_r the first power of 2 above n_(r-1) "
          "that satisfies (C1)-(C5)) ==" % R)
    mp.mp.dps = 1200
    V = mp.mpf(3) / 2
    b = mp.mpf(2)
    nprev = mp.mpf(1)
    k = 2
    stages = []
    for r in range(1, R + 1):
        while True:
            n = mp.mpf(2) ** k
            s = mp.sqrt(n * V)
            Lam = mp.log(n)
            L = 2 * mp.ceil(Lam * s / 2)
            c1 = n >= 2 ** (r + 3) * (r + 1) * nprev and (r + mp.mpf(1) / 2) * L <= n / (2 ** (r + 3) * nprev)
            c2 = L > 2 * max(b, nprev) and L >= 13 * r + 8
            c3 = s >= 64 * b * Lam and n >= 800 * (r + 1) ** 2 + 1
            c4 = ((n - 1) * V >= 6 * mp.mpf(40) ** 8 * b ** 2
                  and 2 * s * mp.exp(-(n - 1) / (32 * mp.pi ** 2 * b ** 2)) <= mp.mpf(1) / 40)
            ratio = 7 * (2 * r + 1) * mp.e ** (r + 1) * s * mp.exp(-Lam ** 2 / 37)
            c5 = ratio <= mp.mpf(1) / r
            if c1 and c2 and c3 and c4 and c5:
                break
            k += 1
        K = (2 * r + 1) * L / 2
        p = mp.mpf(r) / n
        q = 1 / ((2 * r + 1) * n)
        stages.append(dict(r=r, k=k, n=n, L=L, K=K, p=p, q=q, e=p * L + q * K, V=V, b=b, ratio=ratio))
        V = V + p * L ** 2 + q * K ** 2
        b = K
        nprev = n
        k += 1
    sig_m = sum(st["p"] + st["q"] for st in stages)
    sig_e = sum(st["e"] for st in stages)
    for st in stages:
        later = [t for t in stages if t["r"] > st["r"]]
        print(f"  r={st['r']}: n_r = 2^{st['k']}, log10 L_r = {float(mp.log10(st['L'])):.2f}, "
              f"log10 V_r = {float(mp.log10(st['V'])):.2f}, (C5) ratio = 10^{float(mp.log10(st['ratio'])):.1f}, "
              f"n_r q^H = {float(st['n'] * sum(t['p'] + t['q'] for t in later)):.1e}, "
              f"n_r e^H = {float(st['n'] * sum(t['e'] for t in later)):.1e}, "
              f"log(p_r)/L_r = {float(mp.log(st['p']) / st['L']):.1e}")
    print("  (n_r q^H and n_r e^H above count the listed stages only; the bound 2^(-r-3) for the whole tail is (C1))")
    check("admissible n_r exist for r <= %d" % R, len(stages) == R)
    check("sum of stage masses <= 1/8 and sum of stage means <= 1/8", sig_m <= mp.mpf(1) / 8 and sig_e <= mp.mpf(1) / 8,
          f"{float(sig_m):.3e}, {float(sig_e):.3e}")
    check("L_r is even and K_r = (r+1/2) L_r is an integer",
          all(mp.floor(st["L"] / 2) * 2 == st["L"] and mp.floor(st["K"]) == st["K"] for st in stages))
    check("b_r < L_r, and every later atom exceeds n_r",
          all(st["b"] < st["L"] for st in stages)
          and all(stages[i + 1]["L"] > stages[i]["n"] for i in range(len(stages) - 1)))
    check("n_r P(xi >= L_r) >= r and n_r P(xi >= K_r) <= 1/(2r+1) + 2^(-r-3) (listed stages)",
          all(st["n"] * (st["p"] + st["q"]) >= st["r"]
              and st["n"] * (st["q"] + sum(t["p"] + t["q"] for t in stages if t["r"] > st["r"]))
              <= mp.mpf(1) / (2 * st["r"] + 1) + mp.mpf(2) ** (-st["r"] - 3) for st in stages))
    check("n_r q^H_r <= 2^(-r-3) and n_r e^H_r <= 2^(-r-3) (listed stages)",
          all(st["n"] * sum(t["p"] + t["q"] for t in stages if t["r"] > st["r"]) <= mp.mpf(2) ** (-st["r"] - 3)
              and st["n"] * sum(t["e"] for t in stages if t["r"] > st["r"]) <= mp.mpf(2) ** (-st["r"] - 3)
              for st in stages))
    check("p_r L_r^2 >= r (log n_r)^2 V_r, so Var xi = infinity",
          all(st["p"] * st["L"] ** 2 >= st["r"] * mp.log(st["n"]) ** 2 * st["V"] for st in stages))
    check("K_(r-1)/L_r <= 1/(64 (log n_r)^2), so Y(2)/xi(2) -> 0 on the good event",
          all(st["b"] / st["L"] <= 1 / (64 * mp.log(st["n"]) ** 2) for st in stages[1:]))


if __name__ == "__main__":
    section1()
    section2()
    section3()
    section4()
    section5()
    print("\n%d check(s) failed" % len(FAILS) if FAILS else "\nall checks passed")
    sys.exit(1 if FAILS else 0)
