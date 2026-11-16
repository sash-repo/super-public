"""Linear m(n)- and b_{a,b}(n)-invariant differential operators on the densities.

Same bookkeeping as the bilinear census, with one argument:

    A(F) = sum_T c_T (T F),   T = X^a Th_S R^s,   W = w(T),  kappa = |S| - |a|,
    a_3 = a_1 + kappa/(2n) - W/2,   b_3 = b_1 - kappa/(2n),   p(A) = (W+kappa)/2 mod 2.

Why it is needed beyond its own interest: a bilinear invariant may be a COMPOSITION
of the multiplication (or of the bracket) with a linear invariant, and the linear
census is what tells the two apart.  Over b_{a,b}(n) there are more linear invariants
than over m(n) -- that is the whole of the excess in the bilinear census.
"""

import random
import os, sys
import time
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

from mperi import (  # noqa: E402
    add, apply_frame, b_piece, Dens, frame_terms, m_piece, parity, scal,
    term_name, term_parity,
)
from genm import kappa_of, normalize, op_parity, rand_fun  # noqa: E402


def target1(n, W, kappa, a1, b1):
    k = Fr(kappa, 2 * n)
    return (Fr(a1) + k - Fr(W, 2), Fr(b1) - k)


def terms_of(n, W, kappa):
    return [T for T in frame_terms(n, W) if kappa_of(T) == kappa]


def apply_op(n, terms, vec, F):
    out = {}
    for c, T in zip(vec, terms):
        if c:
            out = add(out, scal(c, apply_frame(n, T, F)))
    return out


def system(n, W, kappa, terms, gens, Fs, pt):
    a1, b1 = map(Fr, pt)
    a3, b3 = target1(n, W, kappa, a1, b1)
    M1, M3 = Dens(n, a1, b1), Dens(n, a3, b3)
    pA = op_parity(W, kappa)
    cols, vecs = {}, []
    for T in terms:
        v = {}
        for gi, f in enumerate(gens):
            eps = (parity(f) + 1) % 2
            sg = -((-1) ** (pA * eps))
            for fi, F in enumerate(Fs):
                res = add(apply_frame(n, T, M1.act(f, F)),
                          scal(sg, M3.act(f, apply_frame(n, T, F))))
                for k, val in res.items():
                    v[(gi, fi, k)] = val
                    cols[(gi, fi, k)] = None
        vecs.append(v)
    keys = sorted(cols, key=repr)
    return [[v.get(k, Fr(0)) for v in vecs] for k in keys]


def solve1(n, W, kappa, pt, gens, seed=31, maxdeg=None):
    terms = terms_of(n, W, kappa)
    if not terms:
        return terms, []
    rnd = random.Random(seed)
    md = maxdeg if maxdeg is not None else W + 3
    Fs = [f for p in (0, 1) for f in
          [rand_fun(n, p, rnd, maxdeg=md, terms=6) for _ in range(3)] if f]
    M = system(n, W, kappa, terms, gens, Fs, pt)
    if not M:
        return terms, [[Fr(1) if i == j else Fr(0) for j in range(len(terms))]
                       for i in range(len(terms))]
    return terms, nullspace(M)


def verify1(n, W, kappa, terms, vec, pt, gens, seed=8181):
    a1, b1 = map(Fr, pt)
    a3, b3 = target1(n, W, kappa, a1, b1)
    M1, M3 = Dens(n, a1, b1), Dens(n, a3, b3)
    pA = op_parity(W, kappa)
    rnd = random.Random(seed)
    Fs = [f for p in (0, 1) for f in
          [rand_fun(n, p, rnd, maxdeg=W + 5, terms=7) for _ in range(4)] if f]
    for f in gens:
        eps = (parity(f) + 1) % 2
        for F in Fs:
            res = add(apply_op(n, terms, vec, M1.act(f, F)),
                      scal(-((-1) ** (pA * eps)), M3.act(f, apply_op(n, terms, vec, F))))
            if res:
                return False
    return True


def show1(n, terms, vec):
    vec = normalize(vec)
    bits = []
    for c, T in zip(vec, terms):
        if c:
            bits.append(("+ " if c > 0 else "- ") +
                        (str(abs(c)) if abs(c) != 1 else "") + term_name(n, T))
    return " ".join(bits) or "0"


SAMPLES = [(Fr(0), Fr(0)), (Fr(1), Fr(0)), (Fr(0), Fr(1)),
           (Fr(3, 7), Fr(-2, 5)), (Fr(1, 2), Fr(1, 3))]


def census1(n, Wmax, a=None, b=None, log=None):
    def out(s):
        print(s)
        if log:
            log.write(s + "\n")
            log.flush()

    if a is None:
        gens_all = [f for i in range(-2, 6) for f in m_piece(n, i)]
        gens = m_piece(n, 0) + m_piece(n, 1)
        title = "m(%d)" % n
    else:
        gens_all = [f for i in range(-2, 6) for f in b_piece(n, i, a, b)]
        gens = b_piece(n, 0, a, b) + b_piece(n, 1, a, b)
        title = "b_{%s,%s}(%d)" % (a, b, n)
    out("=" * 78)
    out("%s:  LINEAR invariant operators on the densities F_{a,b}" % title)
    for W in range(Wmax + 1):
        for kappa in range(-W, W + 1):
            if (kappa - W) % 2 or (W + kappa) // 2 > n + 1:
                continue
            terms = terms_of(n, W, kappa)
            if not terms:
                continue
            t0 = time.time()
            dims, shown = [], None
            for pt in SAMPLES:
                ts, vecs = solve1(n, W, kappa, pt, gens)
                good = [v for v in vecs if verify1(n, W, kappa, ts, v, pt, gens_all)]
                dims.append(len(good))
                if good and shown is None:
                    shown = (ts, good, pt)
            out("  W=%d kappa=%+d  terms=%2d   dim at the %d sample weights: %s  [%.1fs]"
                % (W, kappa, len(terms), len(SAMPLES), " ".join(map(str, dims)),
                   time.time() - t0))
            if shown:
                ts, good, pt = shown
                for v in good[:6]:
                    out("        %s      (first at (a,b) = %s)" % (show1(n, ts, v), pt))


def main():
    with open("unarym.log", "w") as log:
        for n in (1, 2, 3):
            census1(n, 5 if n == 1 else 4, log=log)
        for n in (1, 2):
            census1(n, 4 if n == 1 else 3, a=Fr(1), b=Fr(3), log=log)


if __name__ == "__main__":
    main()
