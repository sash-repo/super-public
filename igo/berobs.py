"""Berezin's two families of obstructions (Ch. 3 of "Vvedenie v superanaliz",
2nd ed., MCCME 2013) against the splitness obstruction of Green--Manin.

The book has, over the underlying manifold X_0 with conormal bundle N^*,

    n_k in H^1(X_0; T (x) Lambda^{2k} N^*)      (Secs. 4--5: RETRACTION),
    m_k in H^1(X_0; N (x) Lambda^{2k+1} N^*)    (Sec. 7:    the Z_+-grading),

whereas Green and Manin have ONE class, in H^1(X_0; Aut_{(2k)} Lambda N^*),
whose graded piece is H^1(X_0; T_{2k}) for the degree-2k part T_{2k} of the
tangent sheaf of the split model.  The relation is the extension

    0 -> N (x) Lambda^{q+1} N^*  ->  T_q  ->  T (x) Lambda^q N^*  ->  0,

i.e., Berezin's n is the IMAGE of the Green--Manin class in the quotient and
his m lives in the sub.  This engine checks that, and produces a supermanifold
which is retractable (n_k = 0 for every k, indeed the even coordinates are
glued with no odd correction at all) and NOT split (m_1 =/= 0).

Everything is over CP^1 with the two standard charts and exact rational
arithmetic; the frames, and hence the conventions, are those of `bashkin.py`:

    y = x^{-1},  eta_j = x^{-k_j} xi_j,   so   E = N^* = (+) O(-k_j),

    u_I     = xi_I d/dx        (|I| = q)      -- the quotient T (x) Lambda^q E,
    w_{J,t} = xi_J d/dxi_t     (|J| = q+1)    -- the sub N (x) Lambda^{q+1} E.
"""

from fractions import Fraction
from itertools import combinations
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank                                          # noqa: E402

import bashkin as B                                                # noqa: E402
import ns                                                          # noqa: E402


# --------------------------------------------------------- the three sheaves

def _keep(el, part):
    """Restrict a cochain to the u- or to the w-generators."""
    if part == "all":
        return dict(el)
    tag = "u" if part == "quot" else "w"
    return {(g, e): c for (g, e), c in el.items() if g[0] == tag}


def _gens(m, q, part):
    return [g for g in B.gens(m, q)
            if part == "all" or g[0] == ("u" if part == "quot" else "w")]


def _sections(ks, q, part, rmax, smax):
    """The two frames of the sheaf in question, in the U_0 coordinates.

    For the SUB one may not simply keep the w-part of every section of T_q:
    y^s eta_I d/dy is a section of T_q whose w-part is not a section of the
    sub.  The sub has its own frame (the second family of U1_sections), the
    quotient is the honest image of both families under the projection.
    """
    m = len(ks)
    kof = lambda S: sum(ks[i - 1] for i in S)
    U0, U1 = [], []
    for g in _gens(m, q, part):
        for r in range(rmax + 1):
            U0.append({(g, r): Fraction(1)})
    if part in ("all", "quot"):                      # y^s eta_I d/dy
        for I in (combinations(range(1, m + 1), q) if q >= 0 else []):
            kI = kof(I)
            for s in range(smax + 1):
                el = {(("u", I), 2 - s - kI): Fraction(-1)}
                if part == "all":
                    for i in range(1, m + 1):
                        if i in I:
                            continue
                        J, sgn = B.sign_sort(tuple(I) + (i,))
                        key = (("w", J, i), 1 - s - kI)
                        el[key] = el.get(key, Fraction(0)) - sgn * ks[i - 1]
                U1.append({k: v for k, v in el.items() if v})
    if part in ("all", "sub"):                       # y^s eta_J d/deta_t
        for J in (combinations(range(1, m + 1), q + 1) if q + 1 >= 0 else []):
            kJ = kof(J)
            for t in range(1, m + 1):
                for s in range(smax + 1):
                    U1.append({(("w", J, t), ks[t - 1] - s - kJ): Fraction(1)})
    return U0, U1


def _rows(elems, window, m, q, part):
    cols = {}
    for g in _gens(m, q, part):
        for e in range(-window, window + 1):
            cols[(g, e)] = len(cols)
    rows = []
    for el in elems:
        el = _keep(el, part)
        if not el:
            continue
        if any(abs(e) > window for (_, e) in el):
            continue
        row = [Fraction(0)] * len(cols)
        for (g, e), c in el.items():
            row[cols[(g, e)]] = c
        rows.append(row)
    return rows


def h0_h1(ks, q, part="all", window=None):
    """(dim H^0, dim H^1) of T_q ('all'), of its sub ('sub') or quotient
    ('quot'), computed from the honest frames in a truncated Laurent window."""
    m = len(ks)
    if not _gens(m, q, part):
        return 0, 0
    W = window or (2 * max(ks) * (m + 2) + 8)
    A, C = _sections(ks, q, part, W, W + 2 * sum(ks) + 4)
    rA, rC = _rows(A, W, m, q, part), _rows(C, W, m, q, part)
    ra, rc, rac = rank(rA), rank(rC), rank(rA + rC)
    h0 = ra + rc - rac
    W2 = W - (2 * sum(ks) + 6)
    whole = [{(g, e): Fraction(1)}
             for g in _gens(m, q, part) for e in range(-W2, W2 + 1)]
    h1 = rank(_rows(whole, W, m, q, part) + rA + rC) - rac
    return h0, h1


def is_coboundary(ks, q, coc, part="all", window=None):
    """Is the cocycle `coc` (a dict as in bashkin.py) a coboundary in T_q?"""
    m = len(ks)
    W = window or (2 * max(ks) * (m + 2) + 8)
    A, C = _sections(ks, q, part, W, W + 2 * sum(ks) + 4)
    base = _rows(A, W, m, q, part) + _rows(C, W, m, q, part)
    r0 = rank(base)
    return rank(base + _rows([coc], W, m, q, part)) == r0


# ---------------------------------------------- the retractable non-split X

def retractable_nonsplit(ks=(2, 2, 2), t=1, e=-1, c=1):
    """The supermanifold  y = 1/x,  eta_j = x^{-k_j} xi_j + delta_{jt} c x^e
    xi_1 xi_2 xi_3  --- Berezin's Example 1 of Sec. 6 with a THIRD odd
    coordinate, and with the even coordinates glued as on CP^1 itself."""
    m = len(ks)
    top = tuple(range(1, m + 1))
    z = {t: ns.g_from(top, ns.lp((e, c)))}
    return ns.SM.from_cocycle(list(ks), zx=None, zxi=z,
                              name=f"retractable-nonsplit{tuple(ks)}")


def cocycle_of(ks, t=1, e=-1, c=1):
    m = len(ks)
    return B.w(tuple(range(1, m + 1)), t, e, c)


# ------------------------------------------------------------------ the run

PASS = {"n": 0, "bad": 0}


def check(label, got, want):
    ok = got == want
    PASS["n"] += 1
    if not ok:
        PASS["bad"] += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got}"
          + ("" if ok else f"   (expected {want})"))
    return ok


def report(label, value):
    print(f"        {label}: {value}")
