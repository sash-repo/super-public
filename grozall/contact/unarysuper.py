"""The LINEAR k(1|n)-invariant operators on weighted densities, F_lam -> F_{lam+W/2}.

Same ansatz, one argument: constant coefficients in the frame, so D_S R^s, and the
grading fixes the target weight.  Needed to tell an exceptional bilinear operator from
a composition.
"""

import os, sys, time
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rref, nullspace, rank            # noqa: E402

from super import (mono, add, scal, mul, parity, k_piece, apply_frame, Dens)  # noqa
from gensuper import subsets, rand_fun               # noqa: E402


def uterms(n, W):
    out = []
    for S in subsets(n):
        rest = W - len(S)
        if rest < 0 or rest % 2:
            continue
        out.append((S, rest // 2))
    return out


def apply_uterm(term, F):
    S, s = term
    return apply_frame(S, s, F)


def udefect(n, term, f, F, W, lam):
    nu = Fr(lam) + Fr(W, 2)
    M1, M2 = Dens(n, lam), Dens(n, nu)
    pf, pB = parity(f), W % 2
    out = apply_uterm(term, M1.act(f, F))
    return add(out, scal(-(-1) ** (pB * pf), M2.act(f, apply_uterm(term, F))))


def udim(n, W, lam, gdeg=3, nfield=3, verbose=False):
    terms = uterms(n, W)
    if not terms:
        return 0, [], []
    gens = [f for i in range(0, gdeg + 1) for f in k_piece(n, i)]
    Fs = [rand_fun(n, p, 7 + i, maxdeg=4) for p in (0, 1) for i in range(nfield)]
    rows = []
    for f in gens:
        for F in Fs:
            if not F:
                continue
            cols = [udefect(n, t, f, F, W, lam) for t in terms]
            keys = sorted(set().union(*[set(c) for c in cols]) if cols else [])
            for k in keys:
                row = [c.get(k, Fr(0)) for c in cols]
                if any(row):
                    rows.append(row)
    ns = nullspace(rows) if rows else [[Fr(1) if i == j else Fr(0)
                                        for j in range(len(terms))]
                                       for i in range(len(terms))]
    return len(ns), terms, ns


def show(terms, v):
    out = []
    for c, (S, s) in zip(v, terms):
        if not c:
            continue
        nm = ("D_%s" % "".join(map(str, S)) if S else "") + \
             ("R^%d" % s if s > 1 else ("R" if s else ""))
        out.append("%+s %s" % (c, nm or "id"))
    return " ".join(out) or "0"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    print("Linear k(1|%d)-invariant operators  F_lam -> F_{lam+W/2}" % n)
    vals = [Fr(a, 2) for a in range(-8, 9)] + [Fr(17, 13)]
    for W in range(0, 6):
        t0 = time.time()
        hits = []
        for lam in vals:
            d, terms, ns = udim(n, W, lam)
            if d:
                hits.append((lam, d, [show(terms, v) for v in ns]))
        if hits:
            print(" W=%d:" % W)
            for lam, d, ops in hits:
                print("     lam=%-6s dim=%d   %s" % (lam, d, "; ".join(ops)))
        else:
            print(" W=%d: none for any lam in the range  [%.0fs]" % (W, time.time() - t0))
        sys.stdout.flush()
