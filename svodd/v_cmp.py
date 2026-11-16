#!/usr/bin/env python3
"""
v_cmp.py -- the two routes side by side on one type: the enumeration with the
parametric certificate, and the same enumeration with the pool of parameters
empty (the family node then declines and another invariant subspace is
tried).

The point is not which prints more lines.  The point is whether the two
describe the SAME set of orbits: a family entry stands for a whole line of
classes, so a line that one route prints as one family and the other as a
family plus a further member of the same line is one and the same set.  The
script prints each entry with its family directions, so that this can be read
off, and it reports which entries of one list lie on a family of the other.

Usage:  python3 -u v_cmp.py III_1,4,3
"""

import os
import sys
import importlib
from fractions import Fraction as F

import stab
from probe import TYPES


def load(noparam):
    os.environ.pop("NOPARAM", None)
    if noparam:
        os.environ["NOPARAM"] = "1"
    import t3
    importlib.reload(t3)
    return t3


def classes_of(tname, n, m, D, noparam):
    t3 = load(noparam)
    p, q, terms = TYPES[tname](n, m)
    res = stab.analyse(tname, p, q, terms, D or 2 * n + 4, verbose=False)
    L, classes, cert, notes = t3.classify(res)
    return L, classes, cert


def on_line(L, v, base, dirs):
    """is v = base + sum t_i d_i for some scalars t_i?"""
    import ratfun
    n = L.n
    rows = [[d[i] for d in dirs] + [v[i] - base[i]] for i in range(n)]
    # solve the (overdetermined) system; exact
    k = len(dirs)
    r = 0
    for c in range(k):
        piv = next((i for i in range(r, n) if rows[i][c]), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]
        for i in range(n):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        r += 1
    return not any(rows[i][k] for i in range(r, n))


if __name__ == "__main__":
    spec = sys.argv[1] if len(sys.argv) > 1 else "III_1,4,3"
    head, _, dd = spec.partition(":")
    tn, n, m = head.split(",")
    D = int(dd) if dd else None
    for noparam in (True, False):
        L, cl, cert = classes_of(tn, int(n), int(m), D, noparam)
        tag = "WITHOUT the parametric certificate" if noparam else "WITH it"
        print(f"\n=== {tn}({n},{m}) {tag}: {len(cl)} entries, "
              f"{'certified' if cert else 'NOT certified'}")
        for c in cl:
            ds = "; ".join(L.show(d) for d in c["dirs"])
            print(f"    dim {c['dim']:3d}  {L.show(c['v'])}"
                  + (f"   + t.({ds})" if c["dirs"] else ""))
        if noparam:
            old = (L, cl)
    Lo, clo = old
    Ln, cln = L, cl
    print("\n--- every entry of the first list against the families of the "
          "second (a line of classes may be printed as one entry or as an "
          "entry plus a further member of the same line):")
    for c in clo:
        hit = None
        for d in cln:
            if d["dirs"] and all(not isinstance(x, type(None)) for x in d["v"]):
                try:
                    if on_line(Ln, c["v"], d["v"], d["dirs"]):
                        hit = d
                        break
                except Exception:
                    pass
        print(f"    dim {c['dim']:3d}  {Lo.show(c['v'])}  -->  "
              + (f"on the family {Ln.show(hit['v'])}" if hit
                 else "not on any family of the second list"))
