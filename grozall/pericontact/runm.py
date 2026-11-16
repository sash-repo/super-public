"""Census: bilinear m(n)-invariant operators on the weighted densities F_{a,b}.

Two integers index the census, (W, kappa): the weighted order and the gl(n)-trace
degree; they determine the target weights and the parity of the operator (genm.py).

The solver works in three stages, which is what makes it affordable:

  1.  cut the ansatz by invariance under m_0 alone, on a small set of test functions;
  2.  refine the surviving (few) combinations by the full system -- m_0 and m_1, more
      test functions -- the column count now being tiny;
  3.  VERIFY every survivor independently: against m_i for i up to 6, on functions
      never used in finding it.

Stage 1 can only over-estimate (fewer equations, bigger kernel); stages 2 and 3 are
what make the printed dimension the true one.
"""

import random
import os, sys
import time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

from mperi import (  # noqa: E402
    add, apply_frame, Dens, m_piece, mbracket, monomials, mul, parity, scal,
    sub, term_name,
)
from genm import (  # noqa: E402
    apply_pair, kappa_of, op_parity, rand_fun, show_op, target, term_pairs,
)


def _combine(n, pairs, vec, F, G):
    out = {}
    for c, pair in zip(vec, pairs):
        if c:
            out = add(out, scal(c, apply_pair(n, pair, F, G)))
    return out


def _system(n, W, kappa, pairs, vecs, gens, Fs, Gs, pt):
    """Rows of the invariance system, columns = the given combinations vecs."""
    a1, b1, a2, b2 = map(Fr, pt)
    a3, b3 = target(n, W, kappa, a1, b1, a2, b2)
    M1, M2, M3 = Dens(n, a1, b1), Dens(n, a2, b2), Dens(n, a3, b3)
    pB = op_parity(W, kappa)
    act1 = [[M1.act(f, F) for F in Fs] for f in gens]
    act2 = [[M2.act(f, G) for G in Gs] for f in gens]
    cols, out = {}, []
    for vec in vecs:
        v = {}
        base = [[_combine(n, pairs, vec, F, G) for G in Gs] for F in Fs]
        for gi, f in enumerate(gens):
            eps = (parity(f) + 1) % 2
            sg = -((-1) ** (pB * eps))
            for fi, F in enumerate(Fs):
                sf = (-1) ** (eps * parity(F))
                for gj, G in enumerate(Gs):
                    res = add(
                        _combine(n, pairs, vec, act1[gi][fi], G),
                        scal(sf, _combine(n, pairs, vec, F, act2[gi][gj])),
                        scal(sg, M3.act(f, base[fi][gj])),
                    )
                    for k, val in res.items():
                        key = (gi, fi, gj, k)
                        v[key] = val
                        cols[key] = None
        out.append(v)
    keys = sorted(cols, key=repr)
    return [[v.get(k, Fr(0)) for v in out] for k in keys]


def _ident(m):
    return [[Fr(1) if i == j else Fr(0) for j in range(m)] for i in range(m)]


def _compose(vecs, ker):
    """ker is expressed over vecs; return the combinations over the original basis."""
    out = []
    for kv in ker:
        w = [Fr(0)] * len(vecs[0])
        for c, v in zip(kv, vecs):
            if c:
                w = [x + c * y for x, y in zip(w, v)]
        out.append(w)
    return out


def solve_at(n, W, kappa, pt, seed=17, verbose=False):
    """The space of invariant operators at the weight point pt = (a1,b1,a2,b2)."""
    pairs = term_pairs(n, W, kappa)
    if not pairs:
        return pairs, []
    rnd = random.Random(seed)
    # stage 1: m_0 only, few test functions
    g0 = m_piece(n, 0)
    Fs1 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 2, terms=5) for _ in range(2)] if f]
    Gs1 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 2, terms=5) for _ in range(2)] if f]
    M = _system(n, W, kappa, pairs, _ident(len(pairs)), g0, Fs1, Gs1, pt)
    vecs = nullspace(M) if M else _ident(len(pairs))
    if not vecs:
        return pairs, []
    # stage 2: the full system on the survivors
    gens = g0 + m_piece(n, 1)
    Fs2 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 3, terms=6) for _ in range(3)] if f]
    Gs2 = [f for p in (0, 1) for f in
           [rand_fun(n, p, rnd, maxdeg=W + 3, terms=6) for _ in range(3)] if f]
    M2 = _system(n, W, kappa, pairs, vecs, gens, Fs2, Gs2, pt)
    ker = nullspace(M2) if M2 else _ident(len(vecs))
    vecs = _compose(vecs, ker) if ker else []
    return pairs, vecs


def verify(n, W, kappa, pairs, vec, pt, maxdeg=5, seed=999):
    """Independent check: against m_i for i <= maxdeg, on fresh functions."""
    a1, b1, a2, b2 = map(Fr, pt)
    a3, b3 = target(n, W, kappa, a1, b1, a2, b2)
    M1, M2, M3 = Dens(n, a1, b1), Dens(n, a2, b2), Dens(n, a3, b3)
    pB = op_parity(W, kappa)
    rnd = random.Random(seed)
    gens = [f for i in range(-2, maxdeg + 1) for f in m_piece(n, i)]
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


def generates(n, upto=5):
    """Does m_{<=1} generate m(n) up to degree `upto`?"""
    cur = {i: list(m_piece(n, i)) for i in (-2, -1, 0, 1)}
    from exactlin import rank
    ok = True
    for d in range(2, upto + 1):
        basis = m_piece(n, d)
        keys = sorted({k for f in basis for k in f}, key=repr)
        idx = {k: i for i, k in enumerate(keys)}
        rows = []
        for i in (0, 1):
            for f in cur.get(i, []):
                for g in cur.get(d - i, []):
                    b = mbracket(n, f, g)
                    if b:
                        v = [Fr(0)] * len(keys)
                        for k, c in b.items():
                            v[idx[k]] = c
                        rows.append(v)
        got = rank(rows) if rows else 0
        cur[d] = basis
        if got != len(basis):
            ok = False
        print("   degree %d: generated %d of %d" % (d, got, len(basis)))
    return ok


SAMPLES = [
    (Fr(1), Fr(0), Fr(1), Fr(0)),          # the adjoint pair
    (Fr(0), Fr(0), Fr(0), Fr(0)),
    (Fr(3, 7), Fr(-2, 5), Fr(5, 3), Fr(1, 4)),   # generic
    (Fr(1, 2), Fr(1, 3), Fr(-1, 2), Fr(2, 3)),
]


def census(n, Wmax, kmin=None, log=None):
    def out(s):
        print(s)
        if log:
            log.write(s + "\n")
            log.flush()

    out("=" * 78)
    out("m(%d):  bilinear invariant operators on the densities F_{a,b}" % n)
    out("   b_3 = b_1+b_2 - kappa/(2n),  a_3 = a_1+a_2 + kappa/(2n) - W/2,"
        "  p(B) = (W+kappa)/2 mod 2")
    for W in range(Wmax + 1):
        kk = [k for k in range(-W, W + 1) if (k - W) % 2 == 0
              and (W + k) // 2 <= 2 * n + 2]
        for kappa in kk:
            t0 = time.time()
            dims = []
            found = None
            for pt in SAMPLES:
                pairs, vecs = solve_at(n, W, kappa, pt)
                good = [v for v in vecs if verify(n, W, kappa, pairs, v, pt)]
                dims.append(len(good))
                if good and found is None:
                    found = (pairs, good, pt)
            tag = "W=%d kappa=%+d" % (W, kappa)
            out("  %-14s pairs=%3d  dim at the %d sample weights: %s   [%.1fs]"
                % (tag, len(term_pairs(n, W, kappa)), len(SAMPLES),
                   " ".join(map(str, dims)), time.time() - t0))
            if found:
                pairs, good, pt = found
                for v in good:
                    out("        %s" % show_op(n, pairs, v))


def main():
    Wmax = {1: 6, 2: 4, 3: 3}
    with open("runm.log", "w") as log:
        for n in (1, 2, 3):
            log.write("m(%d): does m_{<=1} generate?\n" % n)
            print("m(%d): does m_{<=1} generate?" % n)
            generates(n, upto=4)
            census(n, Wmax[n], log=log)


if __name__ == "__main__":
    main()
