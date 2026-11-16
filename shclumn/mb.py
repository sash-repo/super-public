#!/usr/bin/env python3
r"""mb(4|5) = (ba(4), c vect(0|3))^m_*  built from its non-positive part.

g_{-2} = Pi(C),  g_{-1} = Vol(0|3)^{1/2} = Lambda(3) (x) vvol^{1/2},  sdim (4|4),
g_0 = c vect(0|3) acting by T^{1/2},  and  [f,g] = beta(f,g) z, where beta is the
unique (up to a factor) T^{1/2}-invariant odd form on Lambda(3): it is solved for
rather than guessed.  Then g_- is realized by the left-invariant vector fields on
itself and the prolong is taken inside vect(4|5) with deg(t) = 2.
"""
import sys
from fractions import Fraction

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, solve as _solve                      # noqa: E402

from vfcore import Space                                            # noqa: E402
from excalg import (lam_basis, lam_mul, lam_der, vect0n_basis,       # noqa: E402
                    vect0n_act, vect0n_div, split_weight, echelon,
                    group_keys, key_wt, to_blocks, annihilator)

N = 3


def half_action(T, i, S):
    """T^{1/2}_{xi^T d_i} applied to xi^S; returns {S': coeff}."""
    out = dict(vect0n_act(N, T, i, S))
    pD = (len(T) + 1) % 2
    sgn = -1 if (len(S) % 2) * pD else 1
    for T1, c in vect0n_div(N, T, i).items():
        pr = lam_mul(S, T1)
        if pr is None:
            continue
        s, S2 = pr
        out[S2] = out.get(S2, 0) + Fraction(1, 2) * sgn * s * c
    return {k: v for k, v in out.items() if v}


def build(shift, top=1, verbose=True):
    """shift = 0 or 1: the parity of Lambda^k is k+shift."""
    basis = lam_basis(N)
    par = {S: (len(S) + shift) % 2 for S in basis}
    idx = {S: i for i, S in enumerate(basis)}
    n = len(basis)

    acts = []                       # (parity, matrix A[c][b] with A(e_c)=sum A[c][b] e_b)
    for (T, i) in vect0n_basis(N):
        A = [[Fraction(0)] * n for _ in range(n)]
        for S in basis:
            for S2, c in half_action(T, i, S).items():
                A[idx[S]][idx[S2]] += c
        if any(any(r) for r in A):
            acts.append(((len(T) + 1) % 2, A))

    # --- the invariant odd form beta
    var = {}
    for a in range(n):
        for b in range(n):
            var[(a, b)] = len(var)
    rows = []
    for a in range(n):
        for b in range(n):
            # antisymmetry: beta(b,a) = -(-1)^{p_a p_b} beta(a,b)
            r = [Fraction(0)] * len(var)
            s = -1 if (par[basis[a]] * par[basis[b]]) % 2 == 0 else 1
            r[var[(b, a)]] += 1
            r[var[(a, b)]] -= s
            rows.append(r)
    for (pD, A) in acts:
        for a in range(n):
            for b in range(n):
                # beta(Ae_a, e_b) + (-1)^{pD p_a} beta(e_a, A e_b) = 0
                r = [Fraction(0)] * len(var)
                for c in range(n):
                    if A[a][c]:
                        r[var[(c, b)]] += A[a][c]
                s = 1 if (pD * par[basis[a]]) % 2 == 0 else -1
                for c in range(n):
                    if A[b][c]:
                        r[var[(a, c)]] += s * A[b][c]
                if any(r):
                    rows.append(r)
    sol = nullspace(rows)
    if verbose:
        print('  shift=%d: invariant odd forms: %d' % (shift, len(sol)))
    if not sol:
        return None
    beta = [[sol[0][var[(a, b)]] for b in range(n)] for a in range(n)]
    # non-degeneracy
    from exactlin import rank as _rank
    if _rank(beta) != n:
        if verbose:
            print('  shift=%d: the invariant form is degenerate' % shift)
        return None
    # parity of z
    pz = None
    for a in range(n):
        for b in range(n):
            if beta[a][b]:
                q = (par[basis[a]] + par[basis[b]]) % 2
                pz = q if pz is None else pz
                assert pz == q
    if verbose:
        print('  shift=%d: p(z) = %d, sdim g_-1 = (%d|%d)'
              % (shift, pz, sum(1 for S in basis if par[S] == 0),
                 sum(1 for S in basis if par[S] == 1)))

    # --- the superspace: coordinates dual to g_{-1}, plus t dual to z
    order = [S for S in basis if par[S] == 0] + [S for S in basis if par[S] == 1]
    pos = {S: i for i, S in enumerate(order)}
    nev = sum(1 for S in basis if par[S] == 0)
    nod = n - nev
    assert pz == 1, 'z must be odd for m-type'
    wtv = []
    for S in order:
        wtv.append(tuple(-(1 if i in S else 0) for i in range(N)))
    # the weight of z is that of the top form
    wz = tuple(-1 for _ in range(N))
    Sp = Space(nev, nod + 1, deg=[1] * nev + [1] * nod + [2],
               wt=wtv[:nev] + wtv[nev:] + [wz])
    TT = Sp.n - 1                      # the generator index of t

    def coord(S):
        i = pos[S]
        e = [0] * Sp.nev
        if i < Sp.nev:
            e[i] = 1
            return (tuple(e), ())
        return ((0,) * Sp.nev, (i - Sp.nev,))

    dt = {(TT, Sp.one()): Fraction(1)}
    # X_S = d/d y_S + sum_V c_{S,V} y_V d/dt  with [X_S, X_V] = beta(S,V) d/dt
    cvar = {}
    for S in basis:
        for V in basis:
            cvar[(S, V)] = len(cvar)
    eqs, rhs = [], []
    for a, S in enumerate(basis):
        for b, V in enumerate(basis):
            r = [Fraction(0)] * len(cvar)
            # [d_S + f_S d_t, d_V + f_V d_t] = d_S(f_V) d_t - (-1)^{pS pV} d_V(f_S) d_t
            r[cvar[(V, S)]] += 1
            s = -1 if (par[S] * par[V]) % 2 == 0 else 1
            r[cvar[(S, V)]] += s
            eqs.append(r)
            rhs.append(beta[a][b])
    csol = _solve(eqs, rhs)
    assert csol is not None, 'no realization of g_-'
    gm1 = []
    for S in basis:
        D = {(pos[S], Sp.one()): Fraction(1)}
        for V in basis:
            c = csol[cvar[(S, V)]]
            if c:
                D[(TT, coord(V))] = D.get((TT, coord(V)), Fraction(0)) + c
        gm1.append({k: v for k, v in D.items() if v})
    # verify
    for a, S in enumerate(basis):
        for b, V in enumerate(basis):
            got = Sp.bracket(gm1[a], gm1[b])
            want = {(TT, Sp.one()): beta[a][b]} if beta[a][b] else {}
            assert got == want, ('g_- bracket', S, V, got, want)
        assert not Sp.bracket(gm1[a], dt), 'z is not central'

    # --- g_0: solve [D, X_S] = X_{A S},  [D, d_t] = chi * d_t
    cand = Sp.field_basis(0)
    g0 = []
    for (pD, A), chi in [(x, Fraction(0)) for x in acts] + \
                        [((0, [[Fraction(1) if a == b else Fraction(0) for b in range(n)]
                                for a in range(n)]), Fraction(2))]:
        target = []
        for a, S in enumerate(basis):
            T_ = {}
            for b, V in enumerate(basis):
                if A[a][b]:
                    for k, c in gm1[b].items():
                        T_[k] = T_.get(k, Fraction(0)) + A[a][b] * c
            target.append({k: v for k, v in T_.items() if v})
        target.append({(TT, Sp.one()): chi} if chi else {})
        keys = set()
        cols = []
        for D in cand:
            row = []
            for a in range(n):
                row.append(Sp.bracket(D, gm1[a]))
            row.append(Sp.bracket(D, dt))
            cols.append(row)
            for br in row:
                keys |= set(br)
        for T_ in target:
            keys |= set(T_)
        keys = sorted(keys)
        kp = {k: i for i, k in enumerate(keys)}
        M = []
        for row in cols:
            v = [Fraction(0)] * (len(keys) * (n + 1))
            for s, br in enumerate(row):
                for k, c in br.items():
                    v[s * len(keys) + kp[k]] = c
            M.append(v)
        b = [Fraction(0)] * (len(keys) * (n + 1))
        for s, T_ in enumerate(target):
            for k, c in T_.items():
                b[s * len(keys) + kp[k]] = c
        MT = [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]
        x = _solve(MT, b)
        assert x is not None, 'no degree-0 field realizing the g_0 action'
        D = {}
        for i, c in enumerate(x):
            if c:
                for k, cc in cand[i].items():
                    D[k] = D.get(k, Fraction(0)) + c * cc
        D = {k: v for k, v in D.items() if v}
        if D:
            g0.append(D)
    g0 = split_weight(Sp, g0)
    gm = {-2: [dt], -1: gm1}
    return Sp, gm, g0


def prolong_gen(S, gm, g0, top, gens):
    """Prolong with arbitrary generators of g_{-1} (not just the d/d(coordinate))."""
    comp = dict(gm)
    comp[0] = list(g0)
    for k in range(1, top + 1):
        prevkeys = group_keys(S, k - 1)
        prevblocks = to_blocks(S, comp[k - 1])
        ann = {}
        for w, keys in prevkeys.items():
            sub = prevblocks.get(w)
            ann[w] = ('all', keys) if sub is None else \
                     ('mat', keys, annihilator(keys, sub[0], sub[1]))
        newc = []
        for w, keys in group_keys(S, k).items():
            cands = [{kk: Fraction(1)} for kk in keys]
            rows = []
            for X in gens:
                wx = S.vf_wt(X)
                wa = tuple(p + q for p, q in zip(w, wx))
                tgt = prevkeys.get(wa, [])
                tpos = {kk: i for i, kk in enumerate(tgt)}
                T = [[Fraction(0)] * len(keys) for _ in range(len(tgt))]
                bad = False
                for j, D in enumerate(cands):
                    br = S.bracket(X, D)
                    for kk, c in br.items():
                        if kk not in tpos:
                            bad = True
                            break
                        T[tpos[kk]][j] += c
                    if bad:
                        break
                if bad:
                    rows = None
                    break
                A = ann.get(wa, ('all', []))
                if A[0] == 'all':
                    rows.extend(T)
                else:
                    for q in A[2]:
                        rows.append([sum(q[i] * T[i][j] for i in range(len(tgt)))
                                     for j in range(len(keys))])
            if rows is None:
                continue
            ker = nullspace(rows) if rows else \
                [[Fraction(1) if i == j else Fraction(0) for j in range(len(keys))]
                 for i in range(len(keys))]
            for sol in ker:
                D = {kk: c for kk, c in zip(keys, sol) if c}
                if D:
                    newc.append(D)
        comp[k] = newc
    return comp


if __name__ == '__main__':
    for shift in (0, 1):
        out = build(shift)
        if out is None:
            continue
        Sp, gm, g0 = out
        print('  shift=%d: dim g_0 = %d' % (shift, len(g0)))
        comp = prolong_gen(Sp, gm, g0, 3, gm[-1])
        for k in sorted(comp):
            ev = sum(1 for D in comp[k] if Sp.vf_par(D) == 0)
            print('     g_%2d: sdim (%d|%d)' % (k, ev, len(comp[k]) - ev))
