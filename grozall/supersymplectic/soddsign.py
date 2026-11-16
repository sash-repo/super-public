"""The third odd operator in coordinates -- fitted, not transcribed by hand.

The obvious transcription of the third of the three invariant operators of
order 0 over a periplectic form is

    oGz_3(A,A')_{ij} = sym_{ij} sum_{a,b} omega^{ab} A_{ia} A'_{jb},

with the naive symmetrization sym_{ij}B_{ij} = (B_{ij} + B_{ji})/2 and with no
Koszul sign.  The operator itself lives one level down, on the polynomials:

    T3(A,A') = sum_{a,b} omega^{ab} (d_{u^a} A~)(d_{u^b} A~'),

and the transcription is what this script checks.  It does not survive: the two
halves of the symmetrization carry different signs, and neither is 1.

Method -- the one used for the order-2 shapes (soddexplicit.py): one unknown
per parity class, solved exactly over Q, then verified on fields the fit never saw.
The classes are indexed by

        (p(i), p(j), p(a), h)        h = 0 the term A_{ia}A'_{jb},
                                     h = 1 the term A_{ja}A'_{ib},

sixteen of them; p(b) = 1 - p(a) is not a coordinate, omega^{ab} vanishing unless
the two indices have opposite parities.  Splitting the two halves is what makes the
fit possible at all: lumped together, as a naive sym_{ij} lumps them, the system has
no solution whatever, which is the shape the defect of the naive formula takes.

Then the same fit with p(A) and p(A') adjoined -- 64 classes, fed fields of both
parities -- decides whether the coefficients depend on the parities of the arguments
as well.  On metrics they cannot be seen; off them they are what Remark rem:trap4 is
about.

Usage:  python3 soddsign.py [k ...]        (default 2 3)
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace                             # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sodd00 import T3                                      # noqa: E402
from soddexplicit import comps                             # noqa: E402

BASE = [(x, y, z, h) for x in (0, 1) for y in (0, 1)
        for z in (0, 1) for h in (0, 1)]
FULL = [c + (w, t) for c in BASE for w in (0, 1) for t in (0, 1)]


def blocks(S, P, Q, withpar):
    """For each (i,j), the polynomial each class contributes, and the target."""
    A = S.A
    C, Cp = comps(S, P), comps(S, Q)
    pA, pB = A.parity(P), A.parity(Q)
    tail = (pA, pB) if withpar else ()
    keys = FULL if withpar else BASE
    tgt = comps(S, T3(S, P, Q))
    for i in range(S.nc):
        for j in range(S.nc):
            acc = {c: [] for c in keys}
            for a, b, w in S.Omnz:
                for h, (ii, jj) in enumerate(((i, j), (j, i))):
                    x = C.get((ii, a))
                    y = Cp.get((jj, b))
                    if not x or not y:
                        continue
                    c = (S.upar[i], S.upar[j], S.upar[a], h) + tail
                    acc[c].append(A.scale(A.mul(x, y), w))
            yield ({c: A.summ(v) for c, v in acc.items()},
                   tgt.get((i, j), {}), keys)


def system(S, fields, withpar):
    rows, live = [], set()
    for P in fields:
        for Q in fields:
            for t, g, keys in blocks(S, P, Q, withpar):
                ks = set(g)
                for v in t.values():
                    ks |= set(v)
                for kk in sorted(ks):
                    row = [t[c].get(kk, Fraction(0)) for c in keys]
                    if any(row):
                        live |= {c for c, v in zip(keys, row) if v}
                    rows.append(row + [-g.get(kk, Fraction(0))])
    return rows, live


def solve(S, fields, withpar, label):
    keys = FULL if withpar else BASE
    rows, live = system(S, fields, withpar)
    ns = nullspace(rows)
    sol = [v for v in ns if v[-1]]
    print(f"\n  {label}: {len(rows)} equations, {len(live)}/{len(keys)} "
          f"classes live, nullspace dim {len(ns)}")
    if len(live) < len(keys):
        print(f"    {len(keys) - len(live)} classes never seen -- "
              f"a table with a hole in it is refused")
        return None
    if not sol:
        print("    NO formula of this shape reproduces the operator")
        return None
    if len(sol) > 1 or len(ns) > 1:
        print(f"    solution NOT unique (nullspace dim {len(ns)})")
        return None
    v = sol[0]
    co = {c: v[n] / v[-1] for n, c in enumerate(keys)}
    g = co[keys[0]]
    co = {c: x / g for c, x in co.items()}
    print(f"    unique, overall factor {g}; coefficients all +-1: "
          f"{all(x in (1, -1) for x in co.values())}")
    return co


def rule0(pi, pj, pa):
    """The closed rule for the term A_{ia}A'_{jb}."""
    return (-1) ** ((pj * (1 + pi) + pi * pa) % 2)


def rule1(pi, pj, pa):
    """The closed rule for the term A_{ja}A'_{ib}."""
    return (-1) ** ((pi + pj * pa) % 2)


def rulepar(pa, pjp, pA):
    """The factor the parity of the FIRST argument contributes.

    p(b) = 1 - p(a), and p(j') is the parity of the free index the SECOND
    argument carries -- j in the term A_{ia}A'_{jb}, i in the term A_{ja}A'_{ib}.
    The factor is 1 on metrics, which is why a formula on metrics need not
    show it; it matters only when the invariance test feeds the operator the
    odd fields L_X A.
    """
    return (-1) ** ((pA * ((1 - pa) + pjp)) % 2)


def check_rules(co, keys):
    bad = []
    for c in keys:
        pi, pj, pa, h = c[:4]
        want = rule0(pi, pj, pa) if h == 0 else rule1(pi, pj, pa)
        if len(c) > 4:
            want *= rulepar(pa, pj if h == 0 else pi, c[4])
        if co[c] != want:
            bad.append((c, co[c], want))
    print(f"    against the closed rules: "
          + ("AGREE everywhere" if not bad else f"{len(bad)} departures"))
    for b in bad[:8]:
        print(f"      class {b[0]}: fitted {b[1]}, rule {b[2]}")
    return not bad


def verify(S, fields):
    """The closed formula, on fields the fit never saw."""
    A = S.A
    bad = tot = 0
    for P in fields:
        for Q in fields:
            C, Cp = comps(S, P), comps(S, Q)
            tgt = comps(S, T3(S, P, Q))
            for i in range(S.nc):
                for j in range(S.nc):
                    out = []
                    for a, b, w in S.Omnz:
                        for h, (ii, jj) in enumerate(((i, j), (j, i))):
                            x = C.get((ii, a))
                            y = Cp.get((jj, b))
                            if not x or not y:
                                continue
                            r = (rule0 if h == 0 else rule1)(
                                S.upar[i], S.upar[j], S.upar[a])
                            out.append(A.scale(A.mul(x, y), w * r))
                    m = A.summ(out)
                    g = tgt.get((i, j), {})
                    tot += 1
                    if A.add(g, A.scale(m, -2)):
                        bad += 1
    print(f"    held-out check: {tot} components, {bad} mismatch "
          f"(the overall 2 is the normalization of comps)")
    return bad == 0


def run(k):
    S = SSetup(k, k, form="odd")
    A = S.A
    print(f"\n=== periplectic k = {k};  sdim M = ({k}|{k})")
    rng = random.Random(4242 + 17 * k)

    even = [T for T in (S.random_field(3, rng, nterms=3) for _ in range(3)) if T]
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    odd = []
    for f in hams:
        for P in even[:2]:
            L = S.lie(f, P)
            if L and A.parity(L) == 1:
                odd.append(L)
        if len(odd) >= 3:
            break
    odd = odd[:3]
    print(f"  fit fields: {len(even)} even, {len(odd)} odd")

    # 1) the naive transcription: can ANY per-class sign save it if the
    #    two halves of sym_{ij} are forced to share one coefficient?
    rows, live = [], set()
    for P in even:
        for Q in even:
            for t, g, keys in blocks(S, P, Q, False):
                merged = {}
                for c, v in t.items():
                    kk = c[:3]
                    merged[kk] = A.add(merged.get(kk, {}), v)
                ks = set(g)
                for v in merged.values():
                    ks |= set(v)
                mk = sorted(merged)
                for key in sorted(ks):
                    row = [merged[c].get(key, Fraction(0)) for c in mk]
                    rows.append(row + [-g.get(key, Fraction(0))])
    ns = nullspace(rows)
    print(f"  naive sym_{{ij}} (one coefficient per (p_i,p_j,p_a)): "
          f"{'solvable' if any(v[-1] for v in ns) else 'NO solution'}")

    # 2) the two halves separately, on metrics
    co = solve(S, even, False, "the two halves apart, on metrics")
    if co:
        check_rules(co, BASE)

    # 3) the same with the parities of the arguments adjoined
    if odd:
        co2 = solve(S, even + odd, True, "the same, fields of both parities")
        if co2:
            dep = sorted({(c[4], c[5]) for c in FULL
                          if co2[c] != co2[c[:4] + (0, 0)]})
            print(f"    does the table depend on (p(A), p(A'))? "
                  f"{'YES: ' + str(dep) if dep else 'NO -- one table serves both'}")
            print("    the ratio to the metric table, class by class:")
            for pA in (0, 1):
                for pB in (0, 1):
                    r = {c[:4]: co2[c[:4] + (pA, pB)] / co2[c[:4] + (0, 0)]
                         for c in BASE}
                    vals = sorted(set(r.values()))
                    if len(vals) == 1:
                        print(f"      p(A)={pA}, p(A')={pB}: uniformly {vals[0]}")
                    else:
                        neg = sorted(c for c in BASE if r[c] == -1)
                        print(f"      p(A)={pA}, p(A')={pB}: -1 on "
                              f"{len(neg)} classes {neg}")
            check_rules(co2, FULL)

    # 4) held-out verification of the closed rules
    rng2 = random.Random(777 + k)
    held = [T for T in (S.random_field(4, rng2, nterms=4) for _ in range(3)) if T]
    heldodd = []
    for f in hams:
        for P in held[:2]:
            L = S.lie(f, P)
            if L and A.parity(L) == 1:
                heldodd.append(L)
        if len(heldodd) >= 2:
            break
    print(f"  the closed rules on {len(held)} METRIC fields "
          f"the fit never saw (this is where the operator lives):")
    verify(S, held)


def main():
    for k in [int(x) for x in sys.argv[1:]] or [2, 3]:
        run(k)


if __name__ == "__main__":
    main()
