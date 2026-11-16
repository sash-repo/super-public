"""Bilinear k(1|n)-invariant differential operators on weighted densities.

An operator invariant under k_{-2}+k_{-1} has constant coefficients in the frame
D_1,...,D_n,R (checked in test_super.py), and D_iD_j+D_jD_i = -2 delta_ij R, so a basis
of the ansatz of weighted order W is

    B(F,G) = sum c_{S1,s1,S2,s2} (D_{S1}R^{s1}F)(D_{S2}R^{s2}G),
    |S1| + |S2| + 2(s1+s2) = W,   S_i strictly increasing subsets of {1..n}.

Every term has parity |S1|+|S2| = W mod 2, so an operator of odd weighted order is an
ODD operator -- something the even contact case has no room for.  Invariance under the
grading element z = K_{2t} gives  lam3 = lam1 + lam2 + W/2, and what is left is k_0 and
k_1 (k_{<=1} generates k(1|n) -- checked in runsuper.py).

The invariance condition for a bilinear operator of parity p(B) is

    B(L_fF, G) + (-1)^{p(f)p(F)} B(F, L_fG) = (-1)^{p(B)p(f)} L_f(B(F,G)),

which is affine in the two weights; the conditions are assembled as a pencil
N_0 + lam N_1 + mu N_2 exactly as in affine.py, and affineness is asserted, not assumed.
"""

import os, sys, random
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rref  # noqa: E402

from super import (mono, add, scal, mul, parity, d_t, d_xi, Kf, kbracket,  # noqa: E402
                   k_piece, apply_frame, Dens)

SAMPLES = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(0), Fr(1)), (Fr(1), Fr(1))]


def subsets(n):
    out = []
    for k in range(n + 1):
        out += [S for S in combinations(range(1, n + 1), k)]
    return out


def term_basis(n, W):
    out = []
    subs = subsets(n)
    for S1 in subs:
        for S2 in subs:
            rest = W - len(S1) - len(S2)
            if rest < 0 or rest % 2:
                continue
            half = rest // 2
            for s1 in range(half + 1):
                out.append((S1, s1, S2, half - s1))
    return out


def apply_term(term, F, G):
    """(D_{S1}R^{s1} (x) D_{S2}R^{s2}) applied to F (x) G.

    The second operator is odd when |S2| is odd, so moving it past F costs the Koszul
    sign (-1)^{|S2| p(F)}: it is the OPERATOR that is the basis element, not the
    written product, and without this sign the contact bracket itself is not in the span.
    """
    S1, s1, S2, s2 = term
    a = apply_frame(S1, s1, F)
    if not a:
        return {}
    b = apply_frame(S2, s2, G)
    if not b:
        return {}
    sgn = (-1) ** (len(S2) * parity(F))
    return scal(sgn, mul(a, b))


def rand_fun(n, par, seed, maxdeg=3, terms=5):
    rnd = random.Random(seed)
    out = {}
    for _ in range(terms):
        a = rnd.randint(0, maxdeg)
        sizes = [k for k in range(n + 1) if k % 2 == par]
        k = rnd.choice(sizes)
        S = tuple(sorted(rnd.sample(range(1, n + 1), k)))
        out = add(out, mono(a, S, rnd.randint(-4, 4)))
    return out


def defect(n, term, f, F, G, W, lam, mu, rho2=(0, 0, 0)):
    """The invariance defect of one basis term, at the weights (lam, mu)."""
    nu = Fr(lam) + Fr(mu) + Fr(W, 2)
    n1, n2, n3 = rho2
    M1, M2, M3 = Dens(n, lam, n1), Dens(n, mu, n2), Dens(n, nu, n3)
    pf, pF, pB = parity(f), parity(F), W % 2
    out = apply_term(term, M1.act(f, F), G)
    out = add(out, scal((-1) ** (pf * pF), apply_term(term, F, M2.act(f, G))))
    out = add(out, scal(-(-1) ** (pB * pf), M3.act(f, apply_term(term, F, G))))
    return out


def pencil(n, W, nfield=2, seed=0, rho2=(0, 0, 0), gdeg=3):
    """(terms, V0, N0, N1, N2); V0 is the identity basis (no pre-reduction here).

    k_1 generates k_i (i>=2) for n>=3 only: for n = 1, 2 the higher pieces are NOT
    reached from k_1, so the conditions are imposed for k_0,...,k_{gdeg} and every
    operator found is verified afterwards against k_i for i up to 6 (runsuper.py).
    """
    terms = term_basis(n, W)
    if not terms:
        return terms, [], [], [], []
    V0 = [[Fr(1) if i == j else Fr(0) for j in range(len(terms))]
          for i in range(len(terms))]
    gens = [f for i in range(0, gdeg + 1) for f in k_piece(n, i)]
    Fs = [rand_fun(n, p, seed + 11 + i) for p in (0, 1) for i in range(nfield)]
    Gs = [rand_fun(n, p, seed + 71 + i) for p in (0, 1) for i in range(nfield)]
    N0, N1, N2 = [], [], []
    for f in gens:
        for F in Fs:
            if not F:
                continue
            for G in Gs:
                if not G:
                    continue
                blocks, keys = [], set()
                for (a, b) in SAMPLES:
                    cols = [defect(n, t, f, F, G, W, a, b, rho2) for t in terms]
                    for c in cols:
                        keys |= set(c)
                    blocks.append(cols)
                if not keys:
                    continue
                keys = sorted(keys)
                mats = [[[c.get(k, Fr(0)) for c in cols] for k in keys]
                        for cols in blocks]
                B00, B10, B01, B11 = mats
                A1 = [[x - y for x, y in zip(r1, r0)] for r1, r0 in zip(B10, B00)]
                A2 = [[x - y for x, y in zip(r1, r0)] for r1, r0 in zip(B01, B00)]
                for a0, a1, a2, bb in zip(B00, A1, A2, B11):
                    for c in range(len(terms)):
                        assert a0[c] + a1[c] + a2[c] == bb[c], "not affine in the weights"
                N0 += B00
                N1 += A1
                N2 += A2
    stacked = [r0 + r1 + r2 for r0, r1, r2 in zip(N0, N1, N2)]
    red = [r for r in rref(stacked)[0] if any(r)] if stacked else []
    m = len(terms)
    return (terms, V0, [r[:m] for r in red], [r[m:2 * m] for r in red],
            [r[2 * m:] for r in red])


def show_op(terms, vec):
    """Print an operator as a formula in D_S R^s."""
    def nm(S, s, X):
        bits = ""
        if S:
            bits += "D_%s" % "".join(map(str, S))
        if s:
            bits += "R^%d" % s if s > 1 else "R"
        return (bits + X) if bits else X
    out = []
    for c, (S1, s1, S2, s2) in zip(vec, terms):
        if not c:
            continue
        out.append("%s(%s)(%s)" % ("+" if c > 0 else "-",
                                   nm(S1, s1, "F"), nm(S2, s2, "G"))
                   if abs(c) == 1 else
                   "%+s (%s)(%s)" % (c, nm(S1, s1, "F"), nm(S2, s2, "G")))
    return " ".join(out) or "0"
