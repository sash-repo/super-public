"""Bilinear m(n)-invariant differential operators on weighted densities.

An operator invariant under m_{-2}+m_{-1} has constant coefficients in the frame
X_1..X_n, Th_1..Th_n, R (checked in test_mperi.py), so a basis of the ansatz is

    B(F,G) = sum c_{T1,T2} (T1 F)(T2 G),   T = X^a Th_S R^s,   s <= 1,

with the Koszul sign (-1)^{p(T2)p(F)}: it is the OPERATOR that is the basis element.

TWO integers grade the ansatz, not one.  The two diagonal elements of
m_0 = c(pe(n)) are  z = M_{2 tau}  (the grading element, eigenvalue = weighted degree)
and  h = M_{sum q_i xi_i} = (1/2)(E_q - E_xi)  (the gl(n)-trace).  Writing

    W      := w(T1) + w(T2)                          (the weighted order)
    kappa  := |S1| + |S2| - |a1| - |a2|              (the trace degree)

their two conditions read

    b_3 = b_1 + b_2 - kappa/(2n),        a_3 = a_1 + a_2 + kappa/(2n) - W/2,

and the parity of the operator is forced as well:

    p(B) = (|S1|+|S2|+s1+s2) mod 2 = ((W + kappa)/2) mod 2,

so W and kappa determine it -- an operator of the "wrong" parity does not exist.
Since |S1|, |S2| <= n and s1 + s2 <= 2 (the frame squares to zero in every odd
direction), the odd budget is bounded:

    (W + kappa)/2 <= 2n + 2,        kappa = W mod 2,      -W <= kappa <= W.

None of this is assumed: the two relations are what makes the calibration operators
(the multiplication at (W,kappa) = (0,0) and the pericontact bracket at (2,0)) come
out, and every operator found is verified independently in runm.py.

What is left after the ansatz is invariance under m_0 and m_1 (m_{<=1} generates
m(n) -- checked in runm.py).  The conditions are affine in the four free weights
(a1,b1,a2,b2), and affineness is asserted, not assumed.
"""

import random
import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from mperi import (  # noqa: E402
    add, apply_frame, Dens, frame_terms, m_piece, mbracket, mono, monomials,
    mul, parity, scal, sub, term_name, term_parity, term_wdeg,
)


def kappa_of(term):
    a, S, s = term
    return len(S) - sum(a)


def term_pairs(n, W, kappa):
    """Pairs (T1,T2) with w1+w2 = W and kappa1+kappa2 = kappa."""
    out = []
    for w1 in range(W + 1):
        for T1 in frame_terms(n, w1):
            for T2 in frame_terms(n, W - w1):
                if kappa_of(T1) + kappa_of(T2) == kappa:
                    out.append((T1, T2))
    return out


def target(n, W, kappa, a1, b1, a2, b2):
    a1, b1, a2, b2 = map(Fr, (a1, b1, a2, b2))
    k = Fr(kappa, 2 * n)
    return (a1 + a2 + k - Fr(W, 2), b1 + b2 - k)


def op_parity(W, kappa):
    return ((W + kappa) // 2) % 2


def apply_pair(n, pair, F, G):
    T1, T2 = pair
    A = apply_frame(n, T1, F)
    if not A:
        return {}
    B = apply_frame(n, T2, G)
    if not B:
        return {}
    sgn = (-1) ** (term_parity(T2) * parity(F))
    return scal(sgn, mul(A, B))


def rand_fun(n, par, rnd, maxdeg=4, terms=4):
    out = {}
    for _ in range(terms):
        d = rnd.randint(0, maxdeg)
        cs = [m for m in monomials(n, d) if parity(m) == par]
        if cs:
            out = add(out, scal(rnd.randint(-3, 3), rnd.choice(cs)))
    return out


def testdata(n, rnd, nrand=3, monodeg=-1):
    """Test functions: a few random ones of each parity (plus monomials if asked).

    A small set suffices to CUT OUT the space; every survivor is verified afterwards
    against fresh data and against m_i for i up to 6 (runm.py).
    """
    out = []
    for d in range(monodeg + 1):
        out += monomials(n, d)
    for p in (0, 1):
        for _ in range(nrand):
            f = rand_fun(n, p, rnd, maxdeg=4, terms=5)
            if f:
                out.append(f)
    return out


def rows_at(n, W, kappa, pairs, gens, Fs, Gs, a1, b1, a2, b2, cache=None):
    """The invariance system at the given weights, as a list of rows over the pairs.

    The actions L_f F and L_f G do not depend on the pair and are computed once;
    apply_pair(F,G) likewise.  Without that the loop is quadratic in the test data
    for nothing.
    """
    a3, b3 = target(n, W, kappa, a1, b1, a2, b2)
    M1, M2, M3 = Dens(n, a1, b1), Dens(n, a2, b2), Dens(n, a3, b3)
    pB = op_parity(W, kappa)
    act1 = [[M1.act(f, F) for F in Fs] for f in gens]
    act2 = [[M2.act(f, G) for G in Gs] for f in gens]
    if cache is None:
        cache = {}
    cols = {}
    vecs = []
    for pi, pair in enumerate(pairs):
        v = {}
        base = cache.setdefault(pi, [[apply_pair(n, pair, F, G) for G in Gs] for F in Fs])
        for gi, f in enumerate(gens):
            eps = (parity(f) + 1) % 2
            sg = -((-1) ** (pB * eps))
            for fi, F in enumerate(Fs):
                pF = parity(F)
                sf = (-1) ** (eps * pF)
                aF = act1[gi][fi]
                for gj, G in enumerate(Gs):
                    res = add(
                        apply_pair(n, pair, aF, G),
                        scal(sf, apply_pair(n, pair, F, act2[gi][gj])),
                        scal(sg, M3.act(f, base[fi][gj])),
                    )
                    for k, val in res.items():
                        key = (gi, fi, gj, k)
                        v[key] = val
                        cols[key] = None
        vecs.append(v)
    return vecs, sorted(cols, key=repr)


def pencil(n, W, kappa, seed=3, nrand=3, monodeg=-1, gdeg=1):
    """Return (pairs, gens, Fs, Gs, N0, N1, N2, N3, N4) with N affine in the weights."""
    pairs = term_pairs(n, W, kappa)
    if not pairs:
        return None
    rnd = random.Random(seed)
    gens = []
    for i in range(0, gdeg + 1):
        gens += m_piece(n, i)
    Fs = testdata(n, rnd, nrand, monodeg)
    Gs = testdata(n, random.Random(seed + 1), nrand, monodeg)

    cache = {}

    def R(a1, b1, a2, b2):
        return rows_at(n, W, kappa, pairs, gens, Fs, Gs, a1, b1, a2, b2, cache)

    # The key set must be COMMON to all evaluation points -- a key absent at one
    # point is a zero row there, not a missing equation.
    pts = [[Fr(0)] * 4] + [[Fr(1) if j == k else Fr(0) for j in range(4)]
                           for k in range(4)]
    chk = [Fr(2), Fr(-3), Fr(1, 2), Fr(5)]
    raw = [R(*pt) for pt in pts] + [R(*chk)]
    keys = sorted({k for _, ks in raw for k in ks}, key=repr)

    def mat(vecs):
        return [[v.get(k, Fr(0)) for v in vecs] for k in keys]

    mats = [mat(v) for v, _ in raw]
    N0 = mats[0]
    Ns = [[[x - y for x, y in zip(r1, r0)] for r1, r0 in zip(mats[k + 1], N0)]
          for k in range(4)]
    pred = [[N0[i][j] + sum(chk[k] * Ns[k][i][j] for k in range(4))
             for j in range(len(pairs))] for i in range(len(keys))]
    assert mats[5] == pred, "the invariance system is not affine in the weights"
    return pairs, gens, Fs, Gs, N0, Ns


def dim_at(N0, Ns, pt):
    pt = [Fr(x) for x in pt]
    if not N0:
        return None, None      # no equations: the caller knows the pair count
    M = [[N0[i][j] + sum(pt[k] * Ns[k][i][j] for k in range(4))
          for j in range(len(N0[0]))] for i in range(len(N0))]
    ns = nullspace(M)
    return len(ns), ns


def normalize(vec):
    """Scale to coprime integers with the first non-zero coefficient positive."""
    from math import gcd
    nz = [c for c in vec if c]
    if not nz:
        return vec
    den = 1
    for c in nz:
        den = den * c.denominator // gcd(den, c.denominator)
    ints = [int(c * den) for c in vec]
    g = 0
    for x in ints:
        g = gcd(g, abs(x))
    if g:
        ints = [x // g for x in ints]
    for x in ints:
        if x:
            if x < 0:
                ints = [-y for y in ints]
            break
    return [Fr(x) for x in ints]


def show_op(n, pairs, vec):
    vec = normalize(vec)
    bits = []
    for c, (T1, T2) in zip(vec, pairs):
        if not c:
            continue
        bits.append("%s (%s F)(%s G)" % (("+ " if c > 0 else "- ") +
                                         (str(abs(c)) if abs(c) != 1 else ""),
                                         term_name(n, T1), term_name(n, T2)))
    return " ".join(bits) if bits else "0"
