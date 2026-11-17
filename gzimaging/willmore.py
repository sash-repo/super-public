#!/usr/bin/env python3
r"""The Willmore correlation of a registered pair, and the ceiling that comes with it.

What is checked, all of it for the Remark on the magnitude coherence (re:rhomag):

  1.  the normalization of eq. (hopf): for A = II in an orthonormal frame,
      alpha = (A11 - A22)/2 - i A12 has |alpha|^2 = H^2 - K exactly, so that
      int |alpha|^2 is the Willmore energy less its topological term;
  2.  Cauchy-Schwarz: rho_W := |int conj(alpha) beta| / sqrt(W1 W2) <= 1, and
      rho_mag := int |alpha||beta| / sqrt(W1 W2) <= 1, with equality in the second
      exactly when |beta| is proportional to |alpha|;
  3.  the factorization rho_W = rho . rho_mag, rho the paper's coherence;
  4.  rho = 1 iff arg(conj(alpha) beta) is constant where alpha beta =/= 0, and then
      beta = e^{i theta} h alpha with h >= 0 and theta constant;
  5.  the index claim behind the umbilic ceiling: with beta = e^{i theta} h alpha and h
      a non-negative function vanishing at an isolated point p where alpha(p) =/= 0, the
      winding of beta around a small circle about p is 0 -- so such a zero is of index 0
      and a generic field has none, whence rho = 1 makes the two umbilic sets coincide.

Exact where it can be (sympy, item 1), floating point with stated tolerances elsewhere.
"""
import cmath
import math
import sys

import numpy as np
import sympy as sp

LOG = []
FAILED = 0


def check(name, ok, detail=""):
    global FAILED
    LOG.append(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED += 1


def hopf(A):
    """(a, alpha) of eq. (hopf) for a symmetric 2x2 A."""
    return 0.5 * (A[0, 0] + A[1, 1]), 0.5 * (A[0, 0] - A[1, 1]) - 1j * A[0, 1]


def part1():
    LOG.append("PART 1.  |alpha|^2 = H^2 - K, symbolically")
    k1, k2, th = sp.symbols("k1 k2 th", real=True)
    R = sp.Matrix([[sp.cos(th), -sp.sin(th)], [sp.sin(th), sp.cos(th)]])
    A = sp.simplify(R * sp.diag(k1, k2) * R.T)
    a = (A[0, 0] + A[1, 1]) / 2
    al = (A[0, 0] - A[1, 1]) / 2 - sp.I * A[0, 1]
    mod2 = sp.simplify(sp.expand(sp.re(al) ** 2 + sp.im(al) ** 2))
    H, K = (k1 + k2) / 2, k1 * k2
    check("|alpha|^2 - (H^2 - K) = 0 for every principal frame and every angle",
          sp.simplify(mod2 - (H ** 2 - K)) == 0)
    check("a = H", sp.simplify(a - H) == 0)
    # the eigenvalues of the trace-free part are +- |alpha|
    tf = A - a * sp.eye(2)
    ev = [sp.simplify(e) for e in sp.Matrix(tf).eigenvals().keys()]
    check("the trace-free part has eigenvalues +- |alpha|",
          sp.simplify(sum(e ** 2 for e in ev) - 2 * mod2) == 0)


def part2(seed=20260919, draws=2000, n=400):
    LOG.append("")
    LOG.append("PART 2.  the two inequalities, the factorization, and the equality case")
    rng = np.random.default_rng(seed)
    worst_w = worst_m = 0.0
    worst_fac = 0.0
    for _ in range(draws):
        al = rng.normal(size=n) + 1j * rng.normal(size=n)
        be = rng.normal(size=n) + 1j * rng.normal(size=n)
        W1, W2 = np.sum(np.abs(al) ** 2), np.sum(np.abs(be) ** 2)
        num = abs(np.sum(np.conj(al) * be))
        den = np.sum(np.abs(al) * np.abs(be))
        rho = num / den
        rho_mag = den / math.sqrt(W1 * W2)
        rho_W = num / math.sqrt(W1 * W2)
        worst_w = max(worst_w, rho_W)
        worst_m = max(worst_m, rho_mag)
        worst_fac = max(worst_fac, abs(rho_W - rho * rho_mag))
    check("rho_W <= 1 on every draw", worst_w <= 1.0, f"max rho_W = {worst_w:.6f}")
    check("rho_mag <= 1 on every draw", worst_m <= 1.0, f"max rho_mag = {worst_m:.6f}")
    check("rho_W = rho . rho_mag on every draw", worst_fac < 1e-12,
          f"max |rho_W - rho rho_mag| = {worst_fac:.2e}")

    # equality in rho_mag exactly when |beta| = c |alpha|
    al = rng.normal(size=n) + 1j * rng.normal(size=n)
    for c in (0.3, 1.0, 2.7):
        be = c * np.abs(al) * np.exp(1j * rng.uniform(0, 2 * math.pi, n))
        W1, W2 = np.sum(np.abs(al) ** 2), np.sum(np.abs(be) ** 2)
        den = np.sum(np.abs(al) * np.abs(be))
        check(f"|beta| = {c} |alpha| gives rho_mag = 1",
              abs(den / math.sqrt(W1 * W2) - 1) < 1e-12)
    be = np.abs(al) ** 2 * np.exp(1j * rng.uniform(0, 2 * math.pi, n))
    W1, W2 = np.sum(np.abs(al) ** 2), np.sum(np.abs(be) ** 2)
    den = np.sum(np.abs(al) * np.abs(be))
    check("|beta| not proportional to |alpha| gives rho_mag < 1",
          den / math.sqrt(W1 * W2) < 1 - 1e-6,
          f"rho_mag = {den / math.sqrt(W1 * W2):.6f}")


def part3(seed=7, n=500):
    LOG.append("")
    LOG.append("PART 3.  the equality case of rho, and what it says about beta")
    rng = np.random.default_rng(seed)
    al = rng.normal(size=n) + 1j * rng.normal(size=n)
    h = rng.uniform(0.1, 3.0, n)
    for theta in (0.0, 0.7, 2.9):
        be = cmath.exp(1j * theta) * h * al
        num = abs(np.sum(np.conj(al) * be))
        den = np.sum(np.abs(al) * np.abs(be))
        check(f"beta = e^(i.{theta}) h alpha with h > 0 gives rho = 1",
              abs(num / den - 1) < 1e-12)
    # a non-constant argument strictly lowers it
    be = np.exp(1j * rng.uniform(0, 2 * math.pi, n)) * h * al
    num = abs(np.sum(np.conj(al) * be))
    den = np.sum(np.abs(al) * np.abs(be))
    check("a non-constant relative argument gives rho < 1",
          num / den < 0.2, f"rho = {num / den:.4f}")
    # rho = 1 forces the relative argument constant: recover it
    be = cmath.exp(1j * 0.7) * h * al
    rel = np.angle(np.conj(al) * be)
    check("and then arg(conj(alpha) beta) is one and the same number everywhere",
          float(np.max(np.abs(rel - rel[0]))) < 1e-12)


def part4():
    LOG.append("")
    LOG.append("PART 4.  the index of the extra zero: beta = e^(i theta) h alpha, h >= 0")
    # alpha non-vanishing near p = 0, h vanishing at p only.  Winding of beta on a small
    # circle about p, against the winding of alpha, which is 0 there.
    for hname, h in (("h = |z|^2", lambda z: abs(z) ** 2),
                     ("h = |z|", lambda z: abs(z)),
                     ("h = x^2 + 3y^4", lambda z: z.real ** 2 + 3 * z.imag ** 4)):
        for alname, al in (("alpha = 1", lambda z: 1 + 0j),
                           ("alpha = 2 + z", lambda z: 2 + z),
                           ("alpha = 1 + i - 0.3 z^2", lambda z: 1 + 1j - 0.3 * z ** 2)):
            r, N = 0.1, 4000
            wind, prev = 0.0, None
            for k in range(N + 1):
                z = r * cmath.exp(2j * math.pi * k / N)
                v = cmath.exp(1j * 0.7) * h(z) * al(z)
                if abs(v) < 1e-14:            # only at the center, not on the circle
                    wind = float("nan")
                    break
                ang = cmath.phase(v)
                if prev is not None:
                    d = ang - prev
                    while d > math.pi:
                        d -= 2 * math.pi
                    while d < -math.pi:
                        d += 2 * math.pi
                    wind += d
                prev = ang
            check(f"winding of beta about the zero of h is 0   ({hname}, {alname})",
                  abs(wind) < 1e-6, f"winding = {wind / (2 * math.pi):+.3e} turns")
    # and a genuine umbilic of alpha keeps its index
    for m, al in ((1, lambda z: z), (2, lambda z: z ** 2), (-1, lambda z: z.conjugate())):
        r, N = 0.1, 4000
        wind, prev = 0.0, None
        for k in range(N + 1):
            z = r * cmath.exp(2j * math.pi * k / N)
            v = cmath.exp(1j * 0.7) * (0.5 + abs(z)) * al(z)
            ang = cmath.phase(v)
            if prev is not None:
                d = ang - prev
                while d > math.pi:
                    d -= 2 * math.pi
                while d < -math.pi:
                    d += 2 * math.pi
                wind += d
            prev = ang
        check(f"a zero of alpha of index {m} keeps it under a positive factor",
              abs(wind / (2 * math.pi) - m) < 1e-6,
              f"winding = {wind / (2 * math.pi):+.4f} turns")


def part5():
    LOG.append("")
    LOG.append("PART 5.  kappa > 0 is eta_M + eta_B < 1, whatever the prevalence")
    pM, eM, eB = sp.symbols("pi_M eta_M eta_B", positive=True)
    pB = 1 - pM
    qM = pM * (1 - eM) + pB * eB
    qB = pB * (1 - eB) + pM * eM
    w = pM * (1 - eM) / qM
    wp = pB * (1 - eB) / qB
    kap = sp.simplify(w + wp - 1)
    check("kappa = pi_M pi_B (1 - eta_M - eta_B) / (q_M q_B)",
          sp.simplify(kap - pM * pB * (1 - eM - eB) / (qM * qB)) == 0,
          f"kappa = {sp.factor(sp.cancel(kap))}")
    check("and (1-eta_M)(1-eta_B) - eta_M eta_B is that same 1 - eta_M - eta_B",
          sp.simplify(sp.expand((1 - eM) * (1 - eB) - eM * eB - (1 - eM - eB))) == 0)
    # the equivalence, read off the numerator, and spot-checked
    bad = []
    for pv in (0.05, 0.3, 0.5, 0.9):
        for a in (0.02, 0.2, 0.45, 0.6, 0.9):
            for b in (0.02, 0.2, 0.45, 0.6, 0.9):
                k = float(kap.subs({pM: pv, eM: a, eB: b}))
                if (k > 0) != (a + b < 1):
                    bad.append((pv, a, b, k))
    check("kappa > 0 <=> eta_M + eta_B < 1 on 100 (prevalence, rate, rate) triples",
          not bad, f"{len(bad)} mismatches")
    worst = max(float(kap.subs({pM: pv, eM: a, eB: b}))
                for pv in (0.05, 0.3, 0.5, 0.9)
                for a in (0.0, 0.1, 0.5) for b in (0.0, 0.1, 0.5))
    check("kappa <= 1 on the same grid", worst <= 1 + 1e-12, f"max kappa = {worst:.6f}")


def main():
    part1()
    part2()
    part3()
    part4()
    part5()
    LOG.append("")
    LOG.append("=" * 65)
    n = sum(1 for l in LOG if l.strip().startswith("[ok]") or l.strip().startswith("[FAIL]"))
    LOG.append(f"{n} checks, {FAILED} failed")
    out = "\n".join(LOG)
    print(out)
    with open("willmore.log", "w", encoding="utf-8") as f:
        f.write(out + "\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
