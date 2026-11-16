"""Census over the Buttin family b_{a,b}(n), at NON-singular values of the parameter.

b_{a,b}(n) = { M_f in m(n) | Div_lambda f = 0 },  lambda = 2a/(n(a-b)), and it
contains m_{-2} + m_{-1}, so the ansatz of genm.py is unchanged: what changes is the
set of generators the operator must commute with, and it is smaller.  Hence

    dim (invariant over b_{a,b})  >=  dim (invariant over m(n))

at the same three modules, and the excess is what this file computes.

Two diagonal elements of m_0 -- the grading element z = M_{2 tau} and the gl(n)-trace
h = M_{sum q_i xi_i} -- have  Div_lambda(2 tau) = 2bn  and  Div_lambda(h) = -an, so
NEITHER lies in b_{a,b} for a generic parameter: only the combination a.z + 2b.h does.
So over b_{a,b} one relation on the weights survives where m(n) has two, which is the
same statement as: a rank-1 module over b_{a,b} carries ONE weight, not two.  At the
modules of an m(n)-problem both relations hold anyway, so (W,kappa) still labels the
ansatz and the two censuses are directly comparable.

The singular values lambda = 0, 1, infinity -- where b_lambda(n) is not simple --
are excluded.
"""

import random
import os, sys
import time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from mperi import (  # noqa: E402
    add, b_piece, Dens, div_lambda, m_piece, mbracket, monomials, parity, scal,
)
from genm import (  # noqa: E402
    op_parity, rand_fun, show_op, target, term_pairs,
)
from runm import _combine, _compose, _ident, _system  # noqa: E402


def lam_of(n, a, b):
    """lambda = 2a/(n(a-b)); infinity when a = b."""
    a, b = Fr(a), Fr(b)
    if a == b:
        return "infinity"
    return Fr(2 * a, 1) / (n * (a - b))


def singular(n, a, b):
    l = lam_of(n, a, b)
    return l in ("infinity", Fr(0), Fr(1))


def generates_b(n, a, b, upto=4):
    """Does b_{<=1} generate b_{a,b}(n) up to degree `upto`?"""
    cur = {i: b_piece(n, i, a, b) for i in (-2, -1, 0, 1)}
    ok = True
    for d in range(2, upto + 1):
        basis = b_piece(n, d, a, b)
        keys = sorted({k for f in basis for k in f}, key=repr)
        idx = {k: i for i, k in enumerate(keys)}
        rows = []
        for i in (0, 1):
            for f in cur.get(i, []):
                for g in cur.get(d - i, []):
                    br = mbracket(n, f, g)
                    if br:
                        v = [Fr(0)] * len(keys)
                        for k, c in br.items():
                            v[idx[k]] = c
                        rows.append(v)
        got = rank(rows) if rows else 0
        cur[d] = basis
        print("   degree %d: generated %d of %d" % (d, got, len(basis)))
        if got != len(basis):
            ok = False
    return ok


def solve_b(n, W, kappa, pt, a, b, seed=23):
    pairs = term_pairs(n, W, kappa)
    if not pairs:
        return pairs, []
    rnd = random.Random(seed)
    g0 = b_piece(n, 0, a, b)
    Fs1 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 2, terms=5) for _ in range(2)] if f]
    Gs1 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 2, terms=5) for _ in range(2)] if f]
    M = _system(n, W, kappa, pairs, _ident(len(pairs)), g0, Fs1, Gs1, pt) if g0 else []
    vecs = nullspace(M) if M else _ident(len(pairs))
    if not vecs:
        return pairs, []
    gens = g0 + b_piece(n, 1, a, b)
    Fs2 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 3, terms=6) for _ in range(3)] if f]
    Gs2 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 3, terms=6) for _ in range(3)] if f]
    M2 = _system(n, W, kappa, pairs, vecs, gens, Fs2, Gs2, pt) if gens else []
    ker = nullspace(M2) if M2 else _ident(len(vecs))
    return pairs, (_compose(vecs, ker) if ker else [])


def verify_b(n, W, kappa, pairs, vec, pt, a, b, maxdeg=5, seed=4242):
    a1, b1, a2, b2 = map(Fr, pt)
    a3, b3 = target(n, W, kappa, a1, b1, a2, b2)
    M1, M2, M3 = Dens(n, a1, b1), Dens(n, a2, b2), Dens(n, a3, b3)
    pB = op_parity(W, kappa)
    rnd = random.Random(seed)
    gens = [f for i in range(-2, maxdeg + 1) for f in b_piece(n, i, a, b)]
    Fs = [f for p in (0, 1) for f in
          [rand_fun(n, p, rnd, maxdeg=W + 4, terms=6) for _ in range(3)] if f]
    Gs = [f for p in (0, 1) for f in
          [rand_fun(n, p, rnd, maxdeg=W + 4, terms=6) for _ in range(3)] if f]
    for f in gens:
        eps = (parity(f) + 1) % 2
        for F in Fs:
            sf = (-1) ** (eps * parity(F))
            for G in Gs:
                res = add(
                    _combine(n, pairs, vec, M1.act(f, F), G),
                    scal(sf, _combine(n, pairs, vec, F, M2.act(f, G))),
                    scal(-((-1) ** (pB * eps)),
                         M3.act(f, _combine(n, pairs, vec, F, G))),
                )
                if res:
                    return False
    return True


SAMPLES = [
    (Fr(1), Fr(0), Fr(1), Fr(0)),
    (Fr(3, 7), Fr(-2, 5), Fr(5, 3), Fr(1, 4)),
]

PARAMS = [(Fr(1), Fr(3)), (Fr(2), Fr(-1))]


def census_b(n, Wmax, log=None):
    def out(s):
        print(s)
        if log:
            log.write(s + "\n")
            log.flush()

    out("=" * 78)
    out("b_{a,b}(%d) at non-singular parameters: bilinear invariant operators" % n)
    for (a, b) in PARAMS:
        out("   parameter (a,b) = (%s,%s),  lambda = %s%s"
            % (a, b, lam_of(n, a, b), "  SINGULAR" if singular(n, a, b) else ""))
    for W in range(Wmax + 1):
        kk = [k for k in range(-W, W + 1) if (k - W) % 2 == 0
              and (W + k) // 2 <= 2 * n + 2]
        for kappa in kk:
            pairs = term_pairs(n, W, kappa)
            if not pairs:
                continue
            t0 = time.time()
            line = []
            shown = None
            for (a, b) in PARAMS:
                if singular(n, a, b):
                    line.append("-")
                    continue
                dims = []
                for pt in SAMPLES:
                    prs, vecs = solve_b(n, W, kappa, pt, a, b)
                    good = [v for v in vecs
                            if verify_b(n, W, kappa, prs, v, pt, a, b)]
                    dims.append(len(good))
                    if good and shown is None:
                        shown = (prs, good)
                line.append("/".join(map(str, dims)))
            out("  W=%d kappa=%+d  pairs=%3d   dim over b_lambda at the samples: %s"
                "   [%.1fs]" % (W, kappa, len(pairs), "  ".join(line),
                                time.time() - t0))
            if shown:
                prs, good = shown
                for v in good[:6]:
                    out("        %s" % show_op(n, prs, v))


def weights_b(n, a, b, log=None):
    """dim b_0/[b_0,b_0]: the number of weights a rank-1 b_{a,b}(n)-module carries."""
    g0 = b_piece(n, 0, a, b)
    keys = sorted({k for f in g0 for k in f}, key=repr)
    idx = {k: i for i, k in enumerate(keys)}
    rows = []
    for f in g0:
        for g in g0:
            br = mbracket(n, f, g)
            if br:
                v = [Fr(0)] * len(keys)
                for k, c in br.items():
                    v[idx[k]] = c
                rows.append(v)
    d1 = rank(rows) if rows else 0
    s = ("b_{%s,%s}(%d):  dim b_0 = %d, dim [b_0,b_0] = %d, quotient = %d "
         "(m(n) has 2)" % (a, b, n, len(g0), d1, len(g0) - d1))
    print(s)
    if log:
        log.write(s + "\n")
        log.flush()


def main():
    with open("runb.log", "w") as log:
        for n in (1, 2, 3):
            for (a, b) in PARAMS:
                weights_b(n, a, b, log)
        for n in (2, 1):
            for (a, b) in PARAMS[:1]:
                print("b_{%s,%s}(%d): does b_{<=1} generate?" % (a, b, n))
                log.write("b_{%s,%s}(%d): does b_{<=1} generate?\n" % (a, b, n))
                generates_b(n, a, b, upto=4)
            census_b(n, 4 if n == 1 else 3, log=log)


if __name__ == "__main__":
    main()
