#!/usr/bin/env python3
"""The principal sl(2) and the symmetric-algebra invariants: Section 5 of the text.

Three jobs, all in exact rational arithmetic:

  1) the three pairs (sl(2), L^{2m}) of Subsec. 5.6, m = 2, 3, 5, i.e. the
     principal sl(2) in sl(3), o(5), g(2).  For each: the space of
     sl(2)-equivariant maps Lambda^2 L^{2m} --> L^{2m} (= the p-component of
     the bracket).  It vanishes exactly for m = 2, which is why sl(3)/o(3)
     is the only symmetric pair of the three.  Where it does not vanish we
     build the relative Chevalley-Eilenberg differential out of it, check
     d^2 = 0, and compute H^.(g, k) -- it is NOT (Lambda^. p)^k
     (Lemma 5.6.1, Statement 5.6.3).

  2) (Lambda^. L^{2m})^{sl(2)} by weight combinatorics (Statement 5.6.2).

  3) the symmetric-algebra invariants S^d(p)^k for a list of symmetric pairs,
     against the Chevalley--Kostant--Rallis prediction
        sum_d dim S^d(p)^k t^d = prod_i 1/(1 - t^{d_i}),
     d_i the degrees of the little Weyl group W(g, a).  The number of the d_i
     is the rank of the pair -- the Krull dimension proposed as the defect
     (Statement 5.3.3); and the same computation for gl(m|n) (Statement 5.3.4).

Usage:  python3 princ.py
"""
import sys, os, itertools
from fractions import Fraction
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from exactlin import rank as _rank, nullspace as _null, backend   # noqa: E402
import sym                                                        # noqa: E402


# --------------------------------------------------------------- sl(2)-modules
def module(m):
    """L^{2m}: basis v_0..v_{2m}, weight of v_j is 2m-2j.

    e v_j = j(2m-j+1) v_{j-1},  f v_j = v_{j+1},  h v_j = (2m-2j) v_j."""
    d = 2 * m + 1
    wt = [2 * m - 2 * j for j in range(d)]

    def e(j):
        return (j - 1, Fraction(j * (2 * m - j + 1))) if j > 0 else None

    def f(j):
        return (j + 1, Fraction(1)) if j < d - 1 else None

    return d, wt, e, f


def act_matrix(d, op):
    """matrix of a raising/lowering operator given as j -> (target, coeff)"""
    M = [[Fraction(0)] * d for _ in range(d)]
    for j in range(d):
        r = op(j)
        if r:
            i, c = r
            M[i][j] = c
    return M


def equivariant_maps(m, n):
    """basis of Hom_{sl(2)}(Lambda^2 L^{2m}, L^{2n}), as matrices
    rows = target basis, columns = the pairs (i<j)."""
    dm, wm, em, fm = module(m)
    dn, wn, en, fn = module(n)
    pairs = list(itertools.combinations(range(dm), 2))
    P = {s: t for t, s in enumerate(pairs)}
    unk = dn * len(pairs)                       # phi[a][s]

    def idx(a, s):
        return a * len(pairs) + P[s]

    rows = []
    # h-equivariance: weight of the pair must equal the weight of the target
    for a in range(dn):
        for s in pairs:
            if wm[s[0]] + wm[s[1]] != wn[a]:
                r = [Fraction(0)] * unk
                r[idx(a, s)] = Fraction(1)
                rows.append(r)
    # e-equivariance: phi(e.omega) = e.phi(omega)
    for s in pairs:
        # e . (v_i ^ v_j)
        img = defaultdict(Fraction)
        for pos in (0, 1):
            r = em(s[pos])
            if not r:
                continue
            t, c = r
            u = (t, s[1]) if pos == 0 else (s[0], t)
            if u[0] == u[1]:
                continue
            sgn = 1
            if u[0] > u[1]:
                u = (u[1], u[0]); sgn = -1
            img[u] += sgn * c
        for a in range(dn):
            r = [Fraction(0)] * unk
            for u, c in img.items():                     # phi(e.omega)
                r[idx(a, u)] += c
            re = en(a + 1) if a + 1 < dn else None       # e.phi(omega): which b -> a
            for b in range(dn):
                rb = en(b)
                if rb and rb[0] == a:
                    r[idx(b, s)] -= rb[1]
            rows.append(r)
    sol = _null(rows) if rows else []
    out = []
    for v in sol:
        M = [[Fraction(0)] * len(pairs) for _ in range(dn)]
        for a in range(dn):
            for s in pairs:
                M[a][P[s]] = v[idx(a, s)]
        out.append((pairs, M))
    return out


# ------------------------------------------- invariants in Lambda^n and S^n
def lam_invariants(wt, e_mat, n):
    """basis of (Lambda^n V^*)^{sl(2)}, as coefficient vectors on the increasing
    n-subsets of the dual basis.

    A weight-0 vector killed by the raising operator generates a highest weight
    module of highest weight 0, i.e. the trivial one, so it is an invariant:
    the two conditions below are enough.  The operator passed in must be the
    one acting on the DUAL basis (the transpose), which is what the relative
    differential below is written in."""
    idx = list(itertools.combinations(range(len(wt)), n))
    zero = [s for s in idx if sum(wt[i] for i in s) == 0]
    if not zero:
        return idx, []
    pos = {s: t for t, s in enumerate(idx)}
    # e acts on Lambda^n; restrict to weight-0 vectors, kernel of e
    rows = defaultdict(lambda: [Fraction(0)] * len(zero))
    for c, s in enumerate(zero):
        for p in range(n):
            for i in range(len(wt)):
                cf = e_mat[i][s[p]]
                if not cf:
                    continue
                ns = list(s); ns[p] = i
                if len(set(ns)) < n:
                    continue
                sgn = 1
                for u in range(len(ns)):
                    for v in range(len(ns) - 1):
                        if ns[v] > ns[v + 1]:
                            ns[v], ns[v + 1] = ns[v + 1], ns[v]; sgn = -sgn
                rows[tuple(ns)][c] += sgn * cf
    M = [r for r in rows.values()]
    ker = _null(M) if M else [[Fraction(1) if a == b else Fraction(0)
                              for b in range(len(zero))] for a in range(len(zero))]
    # lift back to the full Lambda^n coordinates
    out = []
    for v in ker:
        w = [Fraction(0)] * len(idx)
        for c, s in enumerate(zero):
            w[pos[s]] = v[c]
        out.append(w)
    return idx, out


def rel_cohomology(m):
    """H^.(g, k) for g = sl(2) + L^{2m} with the bracket-induced differential."""
    dm, wt, em, fm = module(m)
    e_dual = sym.transpose(act_matrix(dm, em))     # action on the dual basis
    maps = equivariant_maps(m, m)
    if not maps:
        return None, None, 0
    pairs, phi = maps[0]
    P = {s: t for t, s in enumerate(pairs)}
    # bracket [p_i, p_j]_p = sum_a phi[a][(i,j)] p_a
    br = {}
    for s in pairs:
        col = [phi[a][P[s]] for a in range(dm)]
        if any(col):
            br[s] = col
    dims_C, bases, idxs = [], [], []
    for n in range(dm + 1):
        idx, inv = lam_invariants(wt, e_dual, n)
        idxs.append(idx); bases.append(inv); dims_C.append(len(inv))
    ranks = []
    for n in range(dm + 1):
        if n + 1 > dm or not bases[n] or not bases[n + 1]:
            ranks.append(0); continue
        idx, idx1 = idxs[n], idxs[n + 1]
        pos = {s: t for t, s in enumerate(idx)}
        pos1 = {s: t for t, s in enumerate(idx1)}
        cols = []
        for v in bases[n]:
            img = [Fraction(0)] * len(idx1)
            for s1 in idx1:
                tot = Fraction(0)
                for a in range(len(s1)):
                    for b in range(a + 1, len(s1)):
                        key = (s1[a], s1[b])
                        t = br.get(key)
                        if t is None:
                            continue
                        rest = [s1[u] for u in range(len(s1)) if u not in (a, b)]
                        for k in range(dm):
                            if not t[k] or k in rest:
                                continue
                            ns = [k] + rest
                            sgn = 1
                            for u in range(len(ns)):
                                for w in range(len(ns) - 1):
                                    if ns[w] > ns[w + 1]:
                                        ns[w], ns[w + 1] = ns[w + 1], ns[w]; sgn = -sgn
                            tot += ((-1) ** (a + b)) * sgn * t[k] * v[pos[tuple(ns)]]
                img[pos1[s1]] = tot
            cols.append(img)
        A = sym.transpose(bases[n + 1])
        M = [sym.solve_ls(A, c) for c in cols]          # asserts d lands in the invariants
        ranks.append(_rank(sym.transpose(M)) if M else 0)
    H = [dims_C[n] - ranks[n] - (ranks[n - 1] if n else 0) for n in range(dm + 1)]
    return dims_C, H, sum(1 for r in ranks if r)


# ------------------------------------------------------ symmetric invariants
def sym_invariants(k, p, N, dmax):
    """dim S^d(p)^k for d <= dmax, p and k given by matrices"""
    Pb = sym.transpose([sym.flat(b) for b in p])
    d = len(p)
    act = []
    for a in range(len(k)):
        act.append([sym.solve_ls(Pb, sym.flat(sym.brk(k[a], p[i]))) for i in range(d)])
    out = []
    for deg in range(dmax + 1):
        mons = list(itertools.combinations_with_replacement(range(d), deg))
        pos = {s: t for t, s in enumerate(mons)}
        rows = []
        for a in range(len(k)):
            R = defaultdict(lambda: [Fraction(0)] * len(mons))
            for c, s in enumerate(mons):
                for t in range(deg):
                    i = s[t]
                    for j in range(d):
                        cf = act[a][i][j]
                        if not cf:
                            continue
                        ns = tuple(sorted(s[:t] + (j,) + s[t + 1:]))
                        R[ns][c] += cf
            rows.extend(R.values())
        ker = _null(rows) if rows else [[Fraction(1)]]
        out.append(len(ker))
    return out


def series_of_degrees(degs, dmax):
    """coefficients of prod 1/(1-t^{d_i}) up to dmax"""
    S = [1] + [0] * dmax
    for d in degs:
        for i in range(d, dmax + 1):
            S[i] += S[i - d]
    return S


# -------------------------------------------------------------------- main
def main():
    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)

    say("exactlin backend:", backend())

    say("\n=== 1. the p-component of the bracket for (sl(2), L^{2m}) ===")
    for m, name in ((2, "sl(3)"), (3, "o(5)"), (5, "g(2)")):
        mp = equivariant_maps(m, m)
        mk = equivariant_maps(m, 1)
        say(f"  {name} = (sl(2), L^{{{2*m}}}): "
            f"dim Hom(Lambda^2 L^{2*m}, L^{2*m}) = {len(mp)}   "
            f"dim Hom(Lambda^2 L^{2*m}, L^2) = {len(mk)}   "
            f"=> {'SYMMETRIC pair (Z/2-grading)' if not mp else 'NOT a Z/2-grading'}")

    say("\n=== 2. (Lambda^. L^{2m})^{sl(2)} and the true H^.(g, k) ===")
    for m, name in ((2, "sl(3)"), (3, "o(5)"), (5, "g(2)")):
        dm, wt, em, fm = module(m)
        P = sym.lam_inv_sl2(wt)
        C, H, nd = rel_cohomology(m)
        if C is None:
            say(f"  {name}: (Lambda^. p)^k = {sym.pp_str(P)}  (total {sum(P)});"
                f"  d = 0, so H^.(g,k) is the same")
        else:
            assert C == P, (C, P)
            say(f"  {name}: (Lambda^. p)^k = {sym.pp_str(P)}  (total {sum(P)})")
            say(f"          H^.(g,k)      = {sym.pp_str(H)}  (total {sum(H)});"
                f"  {nd} nonzero differential(s)")

    say("\n=== 3. S^d(p)^k against the Chevalley--Kostant--Rallis prediction ===")
    DMAX = 7
    cases = []
    B, N = sym.sl(3)
    alg = sym.Alg(B, N, "sl3")
    k, p = sym.eigenspaces(alg, lambda M: [[-M[j][i] for j in range(N)] for i in range(N)])
    cases.append(("sl(3)/o(3)", k, p, N, [2, 3]))
    E = [[Fraction(1) if i == j else Fraction(0) for j in range(3)] for i in range(3)]
    S = [[Fraction(-1 if i == j == 0 else (1 if i == j else 0)) for j in range(3)]
         for i in range(3)]
    cases.append(("sl(3)/s(gl(1)+gl(2))",) + sym.eigenspaces(
        alg, lambda M: sym.mmul(sym.mmul(S, M), S)) + (N, [2]))
    B, N = sym.sl(4)
    alg = sym.Alg(B, N, "sl4")
    J = sym.Jmat(2)
    Jin = [[-J[i][j] for j in range(4)] for i in range(4)]
    cases.append(("sl(4)/sp(4)",) + sym.eigenspaces(
        alg, lambda M: sym.mmul(sym.mmul(Jin, [[-M[j][i] for j in range(4)]
                                               for i in range(4)]), J)) + (N, [2]))
    B, N = sym.sp(2)
    alg = sym.Alg(B, N, "sp4")
    D = [[Fraction(1) if i == j and i < 2 else (Fraction(-1) if i == j else Fraction(0))
          for j in range(4)] for i in range(4)]
    cases.append(("sp(4)/gl(2)",) + sym.eigenspaces(
        alg, lambda M: sym.mmul(sym.mmul(D, M), D)) + (N, [2, 4]))
    B, N = sym.o(5)
    alg = sym.Alg(B, N, "o5")
    D5 = [[Fraction(1) if i == j and i < 2 else (Fraction(-1) if i == j else Fraction(0))
           for j in range(5)] for i in range(5)]
    cases.append(("o(5)/o(2)+o(3)",) + sym.eigenspaces(
        alg, lambda M: sym.mmul(sym.mmul(D5, M), D5)) + (N, [2, 4]))

    for name, k, p, N, degs in cases:
        got = sym_invariants(k, p, N, DMAX)
        want = series_of_degrees(degs, DMAX)
        ok = got == want
        rk = sym.rank_of_pair(k, p, N)[0]
        say(f"  {name:22s} dim k={len(k):3d} dim p={len(p):3d}  rank of the pair = {rk}"
            f"   #degrees = {len(degs)}")
        say(f"      dim S^d(p)^k, d<={DMAX}: {got}")
        say(f"      prod 1/(1-t^d), d in {degs}: {want}   {'MATCH' if ok else 'MISMATCH'}")
        assert ok, name

    say("\n=== 4. the same computation for gl(m|n): does it give the defect? ===")
    for m, n in ((1, 1), (2, 1), (2, 2), (3, 1)):
        N = m + n
        kk = [sym.E(N, i, j) for i in range(m) for j in range(m)]
        kk += [sym.E(N, m + i, m + j) for i in range(n) for j in range(n)]
        pp = [sym.E(N, i, m + j) for i in range(m) for j in range(n)]
        pp += [sym.E(N, m + j, i) for i in range(m) for j in range(n)]
        dmax = 6 if len(pp) <= 6 else 4
        got = sym_invariants(kk, pp, N, dmax)
        degs = [2 * i for i in range(1, min(m, n) + 1)]
        want = series_of_degrees(degs, dmax)
        ok = got == want
        say(f"  gl({m}|{n}): dim g_ev={len(kk):3d} sdim g_od=(0|{len(pp)})  "
            f"defect = min(m,n) = {min(m, n)}")
        say(f"      dim S^d(g_od)^{{g_ev}}, d<={dmax}: {got}")
        say(f"      prod 1/(1-t^d), d in {degs}: {want}   {'MATCH' if ok else 'MISMATCH'}")
        assert ok, (m, n)



if __name__ == "__main__":
    main()
