"""An independent evaluator for the index expressions the text prints.

A term is written exactly as it appears in the text,

    coefficient * omega_{..}...  omega^{..}...  d_. ... chi_{...}  d_. ... A_{..}

and is given here as a tuple

    (coef, chi_derivs, chi_slots, A_derivs, A_slots, omegas_up, omegas_low)

with the letters 'i', 'j', 'k' standing for the three output slots and any other
letter for a dummy bound by one of the omega^{..}.  Nothing in this module
knows about `gen.py`: the evaluation is by the definition of the expression, so
comparing its output with the engine's answer is a genuine cross-check of the
printed formula, not of the code that found it.
"""

import os
import sys
from fractions import Fraction

from symp import pdiff, pmul, pscale, padd  # noqa: E402
from fiber import flat, unflat  # noqa: E402

OUT = ("i", "j", "k")


def deriv_table(S, T, k, order):
    """{sorted tuple of directions -> array of polynomials}, orders 0..order."""
    n = S.n
    tab = {(): T}
    cur = [()]
    for _ in range(order):
        nxt = []
        for al in cur:
            base = tab[al]
            for d in range(n):
                new = tuple(sorted(al + (d,)))
                if new in tab:
                    continue
                tab[new] = [pdiff(p, d) if p else {} for p in base]
                nxt.append(new)
        cur = nxt
    return tab


def evaluate(S, terms, chi, A, kchi=3, kA=2, kout=3):
    """sum of the terms, as an array of polynomials over {0..n-1}^kout."""
    n = S.n
    order = max([len(t[1]) + len(t[3]) for t in terms] or [0])
    dchi = deriv_table(S, chi, kchi, order)
    dA = deriv_table(S, A, kA, order)
    Omd = [[-S.Om[i][j] for j in range(n)] for i in range(n)]
    out = [dict() for _ in range(n ** kout)]
    for (coef, cd, cs, ad, asl, omup, omlow) in terms:
        coef = Fraction(coef)
        letters = sorted({x for x in list(cd) + list(cs) + list(ad) + list(asl)
                          if x not in OUT})
        bound = set()
        for (a, b) in omup:
            bound.add(a)
            bound.add(b)
        assert set(letters) == bound, \
            f"unbound letters in {(cd, cs, ad, asl, omup)}: {set(letters) ^ bound}"
        for fo in range(n ** kout):
            J = unflat(fo, n, kout)
            val = {OUT[a]: J[a] for a in range(kout)}
            c0 = coef
            for (o1, o2) in omlow:
                c0 *= Omd[val[o1]][val[o2]]
            if not c0:
                continue
            acc = {}
            for asg, w in _assignments(S, omup):
                v = dict(val)
                v.update(asg)
                da = tuple(sorted(v[x] for x in cd))
                db = tuple(sorted(v[x] for x in ad))
                pa = dchi[da][flat(tuple(v[x] for x in cs), n)]
                if not pa:
                    continue
                pb = dA[db][flat(tuple(v[x] for x in asl), n)]
                if not pb:
                    continue
                p = pscale(pmul(pa, pb), w * c0)
                for e, c in p.items():
                    q = acc.get(e, 0) + c
                    if q:
                        acc[e] = q
                    elif e in acc:
                        del acc[e]
            if acc:
                out[fo] = padd(out[fo], acc)
    return out


def _assignments(S, omup):
    combos = [({}, Fraction(1))]
    for (a, b) in omup:
        new = []
        for asg, w in combos:
            for u, v, c in S.Omnz:
                d = dict(asg)
                d[a] = u
                d[b] = v
                new.append((d, w * c))
        combos = new
    return combos


# ------------------------------------------------------------------ printing


def latex(term):
    coef, cd, cs, ad, asl, omup, omlow = term
    s = ""
    for (a, b) in omlow:
        s += f"\\omega_{{{a}{b}}}"
    for (a, b) in omup:
        s += f"\\omega^{{{a}{b}}}"
    if s:
        s += "\\,"
    s += "".join(f"\\partial_{a}" for a in cd) + "\\chi_{" + "".join(cs) + "}\\,"
    s += "".join(f"\\partial_{a}" for a in ad) + "A_{" + "".join(asl) + "}"
    return s
