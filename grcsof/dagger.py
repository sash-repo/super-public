#!/usr/bin/env python3
"""Is the list of singular vectors of [GLS, Sec.10] stable under the formal
adjoint?

An invariant operator D: T(A, lam) --> T(B, mu) has a formal adjoint
D^*: T(B^* (x) Vol) --> T(A^* (x) Vol), again invariant and of the same order.
On the weights of [GLS] -- lam = (lam_0; lam_1, lam_2, lam_3) with respect to
(K_t, K_{xi_i eta_i}) -- this is the involution

    dagger(lam) = (c - lam_0; w(lam_1, ..., lam_s)),      c = n - 2,

where w is the action of -w_0 on the o(n)-weight: the identity for o(3) = B_1
and o(4) = D_2, and the flip of the last coordinate for o(6) = D_3.  The
operator lam --> mu goes to mu^dagger --> lam^dagger.  So the LIST of all
singular vectors must be stable under

    (lam, mu)  |-->  (mu^dagger, lam^dagger)

-- as far as the fibres are finite-dimensional: for an infinite-dimensional
fibre the dual is not a highest weight module at all, and the pairing argument
says nothing.

This script encodes the three theorems of [GLS, Sec.10] (n = 3, 4, 6) verbatim,
applies the involution to every entry, and reports whether the image is again
an entry of the same list.  It also marks which entries have both fibres
finite-dimensional, i.e. where the closure is actually forced.

Usage: python3 dagger.py
"""
from fractions import Fraction as Fr
from itertools import product

HALF = [Fr(n, 2) for n in range(-12, 13)]        # the parameters we sample
WIDE = [Fr(n, 2) for n in range(-60, 61)]        # ... and where we look for the
                                                 # image, so that no entry is
                                                 # reported absent merely
                                                 # because the sample was cut


# ---------------------------------------------------------------- the lists
# every family is (name, npar, f) with f(params) -> (source, target),
# each weight a tuple (lam_0, lam_1, ..., lam_s).

LISTS = {
    3: [                                                    # k(1|3), s = 1
        ('1a',  1, lambda k, : ((k, -k), (k - 1, 1 - k))),
        ('1a*', 1, lambda k, : ((k + 1, k), (k, k - 1))),
        ('1b',  0, lambda: ((1, -1), (0, -1))),
        ('2a',  0, lambda: ((Fr(3, 2), Fr(1, 2)), (Fr(-1, 2), Fr(1, 2)))),
    ],
    4: [                                                    # k(1|4), s = 2
        ('1a',  1, lambda k, : ((k, -k, 0), (k - 1, 1 - k, 0))),
        ('1a*', 1, lambda k, : ((k + 1, -1, k), (k, -1, k - 1))),
        ('1b',  1, lambda k, : ((k, -1, 1 - k), (k - 1, -1, 2 - k))),
        ('1c',  1, lambda k, : ((k + 2, k, 0), (k + 1, k - 1, 0))),
        ('2a',  0, lambda: ((1, -1, -1), (-1, 0, 0))),
        ('2b',  0, lambda: ((2, -1, 1), (0, 0, 0))),
        ('2c',  0, lambda: ((2, -1, -1), (0, -1, -1))),
    ],
    6: [                                            # kas and k(1|6), s = 3
        ('1a',  2, lambda k, l: ((k, -k, l, l), (k - 1, 1 - k, l, l))),
        ('1a*', 2, lambda k, l: ((k, l, 1 - k, l + 1),
                                 (k - 1, l, 2 - k, l + 1))),
        ('1b',  2, lambda k, l: ((k, l, l, 2 - k), (k - 1, l, l, 3 - k))),
        ('1c',  2, lambda k, l: ((k, l, -l - 2, k - 2),
                                 (k - 1, l, -l - 2, k - 3))),
        ('2a',  0, lambda: ((3, -1, -1, -1), (1, -1, -1, -1))),
        ('2a*', 0, lambda: ((3, -1, 0, 0), (1, -1, 0, 0))),
        ('2b',  0, lambda: ((3, -1, -1, 1), (1, -1, -1, 1))),
    ],
}

FLIP = {3: False, 4: False, 6: True}       # is -w_0 non-trivial on the fibre?


def dag(n, lam):
    """the formal-adjoint involution on weights."""
    c = n - 2
    fib = list(lam[1:])
    if FLIP[n]:
        fib[-1] = -fib[-1]
    return (c - lam[0],) + tuple(fib)


def adjoint(n, pair):
    src, tgt = pair
    return (dag(n, tgt), dag(n, src))


def dominant(fib, n):
    """is the o(n)-module with this highest weight finite-dimensional?"""
    if n == 3:                                   # o(3) = B_1: lam_1 >= 0
        ok = fib[0] >= 0
    elif n == 4:                                 # o(4) = D_2: lam_1 >= |lam_2|
        ok = fib[0] >= abs(fib[1])
    else:                                        # o(6) = D_3
        ok = fib[0] >= fib[1] >= abs(fib[2])
    if not ok:
        return False
    ints = [x == int(x) for x in fib]
    return all(ints) or not any(ints)            # all integer or all half-int


def fdim(n, pair):
    return all(dominant(w[1:], n) for w in pair)


def flat(pair):
    return [Fr(x) for w in pair for x in w]


def match(npar, f, pair):
    """the parameters for which the family f gives this (source, target) pair,
    or None.  Every family is affine in its parameters, so we read off the
    affine map and solve; no search, hence no range to fall outside of."""
    v = flat(pair)
    if npar == 0:
        return () if flat(f()) == v else None
    zero = [Fr(0)] * npar
    b = flat(f(*zero))
    cols = []
    for j in range(npar):
        e = list(zero)
        e[j] = Fr(1)
        cols.append([x - y for x, y in zip(flat(f(*e)), b)])
    rhs = [x - y for x, y in zip(v, b)]
    # solve cols . p = rhs by elimination on the npar unknowns
    p = [None] * npar
    rows = list(range(len(rhs)))
    for j in range(npar):
        piv = next((i for i in rows if cols[j][i] != 0), None)
        if piv is None:
            p[j] = Fr(0)
            continue
        p[j] = rhs[piv] / cols[j][piv]
        for i in rows:
            rhs[i] -= p[j] * cols[j][i]
    if any(x != 0 for x in rhs):
        return None
    return tuple(p)


def in_list(n, pair):
    """the family this concrete (source, target) pair belongs to, or None."""
    for name, npar, f in LISTS[n]:
        p = match(npar, f, pair)
        if p is not None:
            return name
    return None


def fmt(p):
    return "(" + ", ".join(str(x) for x in p) + ")" if p else "()"


def report(n):
    print(f"\n=== [GLS, Sec.10], the list for "
          f"{'kas and k(1|6)' if n == 6 else f'k(1|{n})'} "
          f"(dagger: lam_0 -> {n-2} - lam_0"
          f"{', last fibre coordinate flipped' if FLIP[n] else ''}) ===")
    for name, npar, f in LISTS[n]:
        pars = ([()] if npar == 0 else
                [(k,) for k in HALF] if npar == 1 else
                list(product(HALF, HALF)))
        hit, miss, fd_hit, fd_miss = [], [], [], []
        for p in pars:
            pair = f(*p)
            a = adjoint(n, pair)
            where = in_list(n, a)
            fd = fdim(n, pair) and fdim(n, a)
            (hit if where else miss).append((p, where))
            if fd:
                (fd_hit if where else fd_miss).append((p, where))
        tot = len(pars)
        where = sorted({w for _, w in hit if w})
        print(f"  {name:4s}: adjoint in the list for {len(hit)}/{tot} sampled "
              f"parameter values"
              f"{'  --> ' + ', '.join(where) if where else ''}")
        if miss:
            ex = ", ".join(fmt(p) for p, _ in miss[:3])
            print(f"        NOT in the list for {len(miss)}/{tot}, e.g. {ex}")
        # the sector where the closure is forced
        fdp = [p for p in pars if fdim(n, f(*p))]
        print(f"        both fibres finite-dimensional: "
              f"{len(fdp)}/{tot} sampled values"
              f"{'  e.g. ' + ', '.join(fmt(p) for p in fdp[:3]) if fdp else ''}")
        if fd_miss:
            print(f"        *** finite-dimensional AND adjoint absent: "
                  f"{len(fd_miss)}/{tot}, e.g. "
                  f"{', '.join(fmt(p) for p, _ in fd_miss[:3])}")


if __name__ == '__main__':
    for n in (3, 4, 6):
        report(n)

    print("\n=== the two families missing from the n = 6 list ===")
    for nm, g in [('dagger of 1a',
                   lambda k, l: ((k, k - 4, l, l), (k - 1, k - 5, l, l))),
                  ('dagger of 1a*',
                   lambda k, l: ((k, -l - 1, k - 3, l),
                                 (k - 1, -l - 1, k - 4, l)))]:
        src, tgt = g(Fr(0), Fr(0))
        print(f"  {nm}:  {fmt(src)} -> {fmt(tgt)}  at k = l = 0")
        bad = sum(1 for k, l in product(HALF, HALF) if in_list(6, g(k, l)))
        print(f"     of {len(HALF)**2} sampled (k,l): {bad} lie in the list "
              f"of [GLS, Sec.10]")
        fd = [(k, l) for k, l in product(HALF, HALF) if fdim(6, g(k, l))]
        print(f"     with both fibres finite-dimensional: {len(fd)} sampled "
              f"(k,l)" + (f", e.g. {', '.join(fmt(p) for p in fd[:4])}"
                          if fd else ""))
    print("\n  (k, l) = (5, 0) of the first family is  (5,1,0,0) -> (4,0,0,0),"
          " i.e. the divergence.")
