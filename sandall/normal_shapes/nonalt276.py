#!/usr/bin/env python3
"""The (2,4)-structures: the non-alternating Hamiltonian algebras at p = 2.

Are there (2,4)-structures?

At p = 2 a symplectic (= non-degenerate closed) form need not be alternating,
and a non-alternating one lives in an ODD number of indeterminates as well; by
the classification of the non-alternating forms (run1.py, run3.py) every closed non-alternating
form is, after a linear change,

    omega = (du_1)^(2) + omega',      omega' closed alternating.

Here we take the standard shape in m = 2n+1 indeterminates,

    omega_I = (dx)^(2) + sum_{1<=i<=n} dp_i dq_i,

compute h(omega_I), and test

  1) the classical 2-structure   (is D^2 in the algebra for every D?),
  2) the (2,4)-structure of [BLLS, Subsec. 2.6.1, Example 2] for the
     Z/2-grading deg x = 1, deg p_i = deg q_i = 0: a 2-structure on g_+ and
     a map [4]: g_- -> g_+ with ad_{D^[4]} = (ad_D)^4,
  3) the additivity rule (add4) for [4] on all pairs of g_-.

Run:  python3 nonalt276.py
"""
import sys, os, time, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nalt2
from nalt2 import (Alg, d2, lie, act, bracket, gram, det, preserving_algebra,
                   reduce_basis, span_dim, derived, is_simple)


def omega_I(alg, n):
    """(dx)^(2) + sum dp_i dq_i in m = 2n+1 indeterminates, x = index 0."""
    om = {('b', 0): alg.one}
    for i in range(n):
        om[('a', 1 + 2 * i, 2 + 2 * i)] = alg.one
    return om


def ham_fields(alg, n):
    """H_f for f in O(m;1): iota_{H_f} omega = df, omega = omega_I."""
    m = 2 * n + 1
    out = {}
    for t in range(alg.dim):
        f = 1 << t
        D = [0] * m
        D[0] = alg.partial(0, f)                    # the x-direction: G_00 = 1
        for i in range(n):
            a, b = 1 + 2 * i, 2 + 2 * i
            D[a] = alg.partial(b, f)
            D[b] = alg.partial(a, f)
        out[f] = D
    return out


def square(alg, D):
    return [act(alg, D, D[i]) for i in range(len(D))]


def in_span(alg, basis, D):
    return len(reduce_basis(alg, basis + [D])) == len(reduce_basis(alg, basis))


def ad_inner(alg, basis, T):
    """is the derivation [T, -] of the algebra equal to ad_F for some F in it?"""
    m = len(T)
    def flat(vs):
        w = 0
        for D in vs:
            for i in range(m):
                w = (w << alg.dim) | D[i]
        return w
    piv = {}
    combos = []
    for c, F in enumerate(basis):
        v = flat([bracket(alg, F, E) for E in basis])
        cm = 1 << c
        while v:
            hb = v.bit_length() - 1
            if hb in piv:
                w, k = piv[hb]
                v ^= w
                cm ^= k
            else:
                piv[hb] = (v, cm)
                break
    v = flat([bracket(alg, T, E) for E in basis])
    while v:
        hb = v.bit_length() - 1
        if hb not in piv:
            return False
        v ^= piv[hb][0]
    return True


def xdeg(alg, f):
    """the parity of the degree in x = the 0th indeterminate (u = 1)."""
    degs = set()
    for t in range(alg.dim):
        if (f >> t) & 1:
            degs.add(alg.mons[t][0] % 2)
    return degs


def report(n):
    m = 2 * n + 1
    alg = Alg(m, (1,) * m)
    om = omega_I(alg, n)
    print("\n--- m = %d (n = %d) ---" % (m, n), flush=True)
    print("  closed: %s   non-degenerate: %s"
          % (not d2(alg, om), alg.const_term(det(alg, gram(alg, om))) == 1))

    H = ham_fields(alg, n)
    # sanity: the fields do preserve omega, and H is a Lie subalgebra
    bad = sum(1 for D in H.values() if lie(alg, D, om))
    hbasis = reduce_basis(alg, list(H.values()))
    pres = preserving_algebra(alg, om)
    print("  H_f preserve omega: %d failures;  dim span{H_f} = %d,  "
          "dim h(omega) = %d" % (bad, len(hbasis), len(pres)))
    print("  is span{H_f} simple: %s" % (is_simple(alg, hbasis)
                                         if len(hbasis) <= 9 else "not tested"))

    # 1) the classical 2-structure
    nosq = [D for D in hbasis if not in_span(alg, hbasis, square(alg, D))]
    notinner = [D for D in hbasis if not ad_inner(alg, hbasis, square(alg, D))]
    print("  1) basis fields D with D^2 outside the algebra: %d of %d;"
          "  with (ad_D)^2 not inner: %d"
          % (len(nosq), len(hbasis), len(notinner)))

    # 2) the (2,4)-structure for deg x = 1
    plus, minus = [], []
    for f, D in H.items():
        if D == [0] * m:
            continue
        d = xdeg(alg, f)
        if d == {0}:
            plus.append(D)
        elif d == {1}:
            minus.append(D)
    plus, minus = reduce_basis(alg, plus), reduce_basis(alg, minus)
    print("     grading: dim g_+ = %d, dim g_- = %d (total %d)"
          % (len(plus), len(minus), len(hbasis)))
    okplus = all(in_span(alg, plus, square(alg, D)) for D in plus)
    print("  2) g_+ closed under the square (a 2-structure on g_+): %s" % okplus)
    sq_minus_in = sum(1 for D in minus if in_span(alg, hbasis, square(alg, D)))
    print("     D in g_- with D^2 in the algebra: %d of %d"
          % (sq_minus_in, len(minus)))
    four = []
    ok4 = True
    for D in minus:
        D4 = square(alg, square(alg, D))
        four.append(D4)
        if not in_span(alg, plus, D4):
            ok4 = False
    print("     D^4 in g_+ for every D in g_-: %s;  all D^4 = 0: %s"
          % (ok4, all(D4 == [0] * m for D4 in four)))

    # 3) the additivity rule (add4) on all pairs of g_-
    fails = 0
    pairs = 0
    els = []
    if len(minus) <= 6:                       # the whole of g_-
        rng = range(1 << len(minus))
    else:                                     # basis vectors + a random sample
        import random
        random.seed(276)
        rng = list(range(len(minus) + 1)) + \
              [random.getrandbits(len(minus)) for _ in range(60)]
    for bits in rng:
        D = [0] * m
        for i in range(len(minus)):
            if (bits >> i) & 1:
                for t in range(m):
                    D[t] ^= minus[i][t]
        els.append(D)
    for X in els:
        for Y in els:
            pairs += 1
            lhs = square(alg, square(alg, [X[t] ^ Y[t] for t in range(m)]))
            rhs = [square(alg, square(alg, X))[t] ^ square(alg, square(alg, Y))[t]
                   for t in range(m)]
            # s(x,y) = [x,[x,[x,y]]] + [y,[y,[y,x]]] + [x,[y,[x,y]]] + [x,y]^{[2]}
            #   -- the additivity rule (add4) for the [4]-map
            br = lambda A, B: bracket(alg, A, B)
            s = br(X, br(X, br(X, Y)))
            s2 = br(Y, br(Y, br(Y, X)))
            s3 = br(X, br(Y, br(X, Y)))
            s4 = square(alg, br(X, Y))
            s = [s[t] ^ s2[t] ^ s3[t] ^ s4[t] for t in range(m)]
            if lhs != [rhs[t] ^ s[t] for t in range(m)]:
                fails += 1
    print("  3) additivity rule (add4) on %d pairs of g_-: %d failures"
          % (pairs, fails), flush=True)


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 78)
    print("The (2,4)-structure of the non-alternating Hamiltonian algebras, p=2")
    print("=" * 78)
    for n in (1, 2):
        report(n)
    print("\n[%.1f s]" % (time.time() - t0))
