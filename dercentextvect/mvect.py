#!/usr/bin/env python3
"""The pericontact ("odd contact") series m(n) and the Buttin family b_{a,b}(n).

The engine of dervect.py knows vect, svect, h, k and le.  The one series the
text names as untreated -- and the one where the le-phenomenon (an outer
derivation induced by an operator of ORDER 2, which no Der O can produce) might
recur -- is the pericontact series.  This module builds it.

Conventions (the text, formulas (8), (14), (17)):  indeterminates q_1..q_n (even)
and tau, xi_1..xi_n (ODD -- the pericontact form alpha_0 is even, so p(tau) = 1
and p(M_f) = p(f) + 1), weights  deg tau = 2, deg q_i = deg xi_i = 1, and

    Le_f = sum_i ( (d f / d q_i) d_{xi_i} + (-1)^{p(f)} (d f / d xi_i) d_{q_i} )
    M_f  = (2 - E)(f) d_tau - Le_f - (-1)^{p(f)} (d f / d tau) E,
    E    = sum_{y != tau} y d_y.

Since the overall sign conventions of the engine differ term by term from the
printed ones, the three signs are searched over and the closed combination is
reported; closure is the arbiter (check_closed).

The candidate outer derivation, the analogue of the odd Laplacian of le, is the
order-2 operator that the divergence of M_f is built from, eq. (2.4.1):

    Delta^m = (1 - E) d_tau - sum_i d^2 / (d xi_i d q_i),      Div M_f = 2 (-1)^{p(f)} Delta^m(f).
"""
import os
import sys
import time
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dervect import (Super, Echelon, check_closed, normalizer,      # noqa: E402
                     der_dim, ad_dim, derived)

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep


# --------------------------------------------------------------- the series
def pericontact(S, n, signs=(-1, -1)):
    """m(n) = span(M_f).  Returns (basis, {monomial f -> M_f}).

    signs = (s_Le, s_E): the coefficients of the Le_f term and of the
    (d_tau f) E term; see the module docstring.
    """
    s_le, s_e = signs
    TAU = ('xi', 0)
    fields = {}
    for u in S.mons:
        a, b = u
        X = {}
        sg = (-1) ** S.par_mon(u)
        # (2 - E)(f) d_tau,  E counting the degree in everything but tau
        e = sum(a) + sum(b[1:])
        c0 = (2 - e) % S.p
        if c0:
            X = S.add(X, {(u, TAU): c0})
        # s_le * Le_f
        for i in range(n):
            w, c = S.gen_act(('x', i, 0), u)               # d/d q_i
            if c:
                X = S.add(X, {(w, ('xi', 1 + i)): (s_le * c) % S.p})
            w, c = S.gen_act(('xi', 1 + i), u)             # d/d xi_i
            if c:
                X = S.add(X, {(w, ('x', i, 0)): (s_le * sg * c) % S.p})
        # s_e * (d_tau f) * E
        w, c = S.gen_act(TAU, u)
        if c:
            for i in range(S.m):
                z, cc = S.omul(w, ((0,) * i + (1,) + (0,) * (S.m - i - 1),
                                   (0,) * S.n))
                if cc:
                    X = S.add(X, {(z, ('x', i, 0)): (s_e * sg * c * cc) % S.p})
            for j in range(1, S.n):
                bb = [0] * S.n
                bb[j] = 1
                z, cc = S.omul(w, ((0,) * S.m, tuple(bb)))
                if cc:
                    X = S.add(X, {(z, ('xi', j)): (s_e * sg * c * cc) % S.p})
        if X:
            fields[u] = X
    E = Echelon([S.vec(X) for X in fields.values()], S.p)
    return [S.unvec(row) for row in E.rows], fields


def make(p, n, N=None):
    """the ambient O(n|n+1; N): q_1..q_n even, tau = xi_0 and xi_1..xi_n odd."""
    return Super(p, n, n + 1, N=N or [1] * n,
                 wt=[1] * n + [2] + [1] * n)


def find_signs(p, n):
    """which sign convention closes?  returns the list of closed ones."""
    S = make(p, n)
    ok = []
    for s in product((1, -1), repeat=2):
        L, _ = pericontact(S, n, s)
        closed, _ = check_closed(S, L)
        if closed:
            ok.append(s)
    return ok


# ------------------------------------------------------------ the candidate
def delta_m(S, n, u):
    """Delta^m(u) = (1 - E) d_tau (u) - sum_i d_{xi_i} d_{q_i} (u)."""
    out = {}
    w, c = S.gen_act(('xi', 0), u)
    if c:
        a, b = w
        e = sum(a) + sum(b[1:])
        co = (1 - e) * c % S.p
        if co:
            out[w] = (out.get(w, 0) + co) % S.p
    for i in range(n):
        w1, c1 = S.gen_act(('x', i, 0), u)
        if not c1:
            continue
        w2, c2 = S.gen_act(('xi', 1 + i), w1)
        if c2:
            out[w2] = (out.get(w2, 0) - c1 * c2) % S.p
    return {k: v for k, v in out.items() if v}


def buttin(S, n, a, b, fields, sdelta=1):
    """b_{a,b}(n) = {M_f in m(n) | Div_lambda(f) = 0}, eq. (2.7.3):

        Div_lambda = (b n - a E) d_tau - a Delta,   Delta = sum_i d^2/(d q_i d xi_i).

    Returns (basis of the subalgebra, {f -> M_f} restricted to the kernel).
    """
    keep = {}
    for u in S.mons:
        out = {}
        w, c = S.gen_act(('xi', 0), u)                  # d_tau
        if c:
            aa, bb = w
            e = sum(aa) + sum(bb[1:])
            co = (b * n - a * e) * c % S.p
            if co:
                out[w] = (out.get(w, 0) + co) % S.p
        for i in range(n):
            w1, c1 = S.gen_act(('x', i, 0), u)
            if not c1:
                continue
            w2, c2 = S.gen_act(('xi', 1 + i), w1)
            if c2:
                out[w2] = (out.get(w2, 0) - sdelta * a * c1 * c2) % S.p
        out = {k: v for k, v in out.items() if v}
        keep[u] = out
    # the kernel of a linear map on the span of the monomials
    from dervect import echelon as _ech
    mons = list(S.mons)
    idx = {u: i for i, u in enumerate(mons)}
    rows, cols = [], len(mons)
    for j, u in enumerate(mons):
        for w, c in keep[u].items():
            rows.append((idx[w], j, c))
    M = [[0] * cols for _ in range(cols)]
    for r, cc, v in rows:
        M[r][cc] = v % S.p
    from exactlin import nullspace as _ns
    ker = _ns(M, p=S.p) if any(any(r) for r in M) else \
        [[1 if i == k else 0 for i in range(cols)] for k in range(cols)]
    L = []
    for v in ker:
        X = {}
        for k, c in enumerate(v):
            c = int(c) % S.p
            if c and mons[k] in fields:
                X = S.add(X, fields[mons[k]], c)
        if X:
            L.append(X)
    E = Echelon([S.vec(X) for X in L], S.p)
    return [S.unvec(r) for r in E.rows]


def main():
    from laplace import test_sum                                   # noqa: E402
    log = open(HERE + "mvect.log", "w")

    def say(s):
        print(s)
        log.write(s + "\n")
        log.flush()

    say("=== which sign convention makes span(M_f) a subalgebra? ===")
    for p in (3, 5):
        for n in (1, 2):
            if p == 5 and n == 2:
                continue
            say(f"p={p} n={n}: closed for signs {find_signs(p, n)}")

    say("")
    say("=== the census for m(n) and m(n)^(1) ===")
    for p, n in ((3, 1), (5, 1), (3, 2)):
        S = make(p, n)
        L, fields = pericontact(S, n)
        closed, bad = check_closed(S, L)
        dim = Echelon([S.vec(X) for X in L], S.p).dim
        if dim > 130:
            say(f"p={p} m({n}): dim {dim} SKIPPED")
            continue
        for tag, M in (("m", L), ("m^(1)", derived(S, L))):
            t0 = time.time()
            d = Echelon([S.vec(X) for X in M], S.p).dim
            da, dz = ad_dim(S, M)
            dD, blocks = der_dim(S, M, want_blocks=True)
            NN = normalizer(S, M)
            dN = Echelon([S.vec(X) for X in NN], S.p).dim
            neg = {k: v for k, v in blocks.items() if k[0] < 0}
            say(f"p={p} {tag}({n}): dimL={d} closed={closed} centre={dz} "
                f"dim ad={da} dim Der={dD} out={dD - da} | dim N={dN} "
                f"gap={dD - dN} | neg blocks={neg} [{time.time() - t0:.0f}s]")

    say("")
    say("=== the family b_{a,b}(n), (a:b) over P^1(F_p) ===")
    say("    lambda = 2a/(n(a-b));  (0:1) is b(n) = b_0(n), which contains le(n)")
    for p, n in ((3, 1), (5, 1), (3, 2)):
        S = make(p, n)
        L, fields = pericontact(S, n)
        for (a, b) in [(1, x) for x in range(p)] + [(0, 1)]:
            B = buttin(S, n, a, b, fields)
            closed, _ = check_closed(S, B)
            d = Echelon([S.vec(X) for X in B], S.p).dim
            for tag, M in (("", B), ("^(1)", derived(S, B))):
                dd = Echelon([S.vec(X) for X in M], S.p).dim
                if dd == 0:
                    continue
                da, dz = ad_dim(S, M)
                dD, blocks = der_dim(S, M, want_blocks=True)
                NN = normalizer(S, M)
                dN = Echelon([S.vec(X) for X in NN], S.p).dim
                say(f"p={p} b_({a}:{b}){tag}({n}): dimL={dd} closed={closed} "
                    f"centre={dz} dim ad={da} dim Der={dD} | dim N={dN} "
                    f"gap={dD - dN}")

    say("")
    say("=== is Delta^m the missing derivation? ===")
    for p, n in ((3, 1), (5, 1), (3, 2)):
        S = make(p, n)
        L, fields = pericontact(S, n)
        if Echelon([S.vec(X) for X in L], S.p).dim > 130:
            continue
        test_sum(f"p={p} m({n}), Delta^m", S, fields,
                 lambda u, S=S, n=n: delta_m(S, n, u), log)
    log.close()


if __name__ == "__main__":
    main()
