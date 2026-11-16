"""How many weights does a density over k(1|n) carry?  Derived, not guessed.

A 1-dimensional-fiber module structure on the functions is

    L_f F = K_f(F) + rho(f) F,

and this is a representation iff rho is a 1-cocycle:  the defect

    L_f L_g F - (-1)^{p(f)p(g)} L_g L_f F - L_{{f,g}} F

is LINEAR in rho (the quadratic part rho(f)rho(g) - (-1)^{p(f)p(g)}rho(g)rho(f) vanishes,
the values being functions), so the admissible rho form a linear space, computed here.

Candidates: all operators  xi_S . d_xi^T . d_t^s  of weighted degree -2, which is what
the grading forces.  The answer is the number of weight parameters.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

from super import (mono, add, scal, mul, parity, d_t, d_xi, Kf, kbracket,  # noqa: E402
                   k_piece, XI)


def candidates(n, maxs=3):
    """Operators O: f -> function, of weighted degree -2:  |S| - |T| - 2s = -2."""
    out = []
    for s in range(maxs + 1):
        for tsize in range(n + 1):
            for T in combinations(range(1, n + 1), tsize):
                ssize = tsize + 2 * s - 2
                if ssize < 0 or ssize > n:
                    continue
                for S in combinations(range(1, n + 1), ssize):
                    out.append((S, T, s))
    return out


def apply_cand(cand, f):
    S, T, s = cand
    out = f
    for _ in range(s):
        out = d_t(out)
    for i in reversed(T):
        out = d_xi(i, out)
    if S:
        out = mul(mono(0, S), out)
    return out


def name(cand):
    S, T, s = cand
    bits = []
    if S:
        bits.append("xi_%s" % "".join(map(str, S)))
    for i in T:
        bits.append("d%d" % i)
    if s:
        bits.append("dt^%d" % s if s > 1 else "dt")
    return ".".join(bits) or "1"


def defect(n, cand, f, g, F):
    """The module defect with rho = cand, linear in rho."""
    pf, pg = parity(f), parity(g)

    def L(h, X):
        return add(Kf(n, h, X), mul(apply_cand(cand, h), X))

    lhs = add(L(f, L(g, F)), scal(-(-1) ** (pf * pg), L(g, L(f, F))))
    return add(lhs, scal(-1, L(kbracket(n, f, g), F)))


def defect_K(n, f, g, F):
    pf, pg = parity(f), parity(g)
    lhs = add(Kf(n, f, Kf(n, g, F)), scal(-(-1) ** (pf * pg), Kf(n, g, Kf(n, f, F))))
    return add(lhs, scal(-1, Kf(n, kbracket(n, f, g), F)))


def solve(n, verbose=True):
    cands = candidates(n)
    gens = []
    for i in range(-2, 3):
        gens += k_piece(n, i)
    Fs = [mono(0), mono(1), mono(2), mono(0, (1,)) if n >= 1 else mono(0)]
    if n >= 2:
        Fs.append(mono(1, (1, 2)))
    if n >= 3:
        Fs.append(mono(0, (1, 2, 3)))
    rows, keys = [], set()
    cols = []
    for cand in cands:
        col = []
        for f in gens:
            for g in gens:
                for F in Fs:
                    d = add(defect(n, cand, f, g, F),
                            scal(-1, defect_K(n, f, g, F)))
                    col.append(d)
                    keys |= set(d)
        cols.append(col)
    # assemble: one row per (position, monomial key)
    keylist = sorted(keys)
    nrowblocks = len(cols[0])
    mat = []
    for b in range(nrowblocks):
        for k in keylist:
            row = [c[b].get(k, Fr(0)) for c in cols]
            if any(row):
                mat.append(row)
    if not mat:
        basis = [[Fr(1) if i == j else Fr(0) for j in range(len(cands))]
                 for i in range(len(cands))]
    else:
        basis = nullspace(mat)
    if verbose:
        print("n=%d: %d candidate multipliers, %d conditions -> dim = %d"
              % (n, len(cands), len(mat), len(basis)))
        for v in basis:
            terms = [(c, name(cands[i])) for i, c in enumerate(v) if c]
            print("      rho(f) = " + " + ".join("%s*%s" % (c, nm) for c, nm in terms))
    return basis, cands


if __name__ == "__main__":
    # usage: python3 weights.py [nmax]   (default 3).  The cocycle condition is
    # imposed on k_{-2}+...+k_2 only; for n >= 4 a candidate supported in higher
    # degrees (e.g. xi_1..xi_4 d_t^3 for n = 4) vanishes there and passes the test
    # vacuously, so the count is reliable for n <= 3 only.
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    for n in range(nmax + 1):
        solve(n)
