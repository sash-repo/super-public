#!/usr/bin/env python3
"""er^(1)(3; 1) without [GL3]: identification of the deform of k^(1)(3; 1),
the g_0-action on g_{-1}, and H^2(g_-; g).

Steps 1) and 2) (the Weisfeiler filtration of the deform and its associated
graded) are done in ermolaev.py; this file continues:

  3) gr_0 is identified as sl(2) (+) hei(2) by an explicit basis,
  4) gr_{-1} is identified as the gr_0-module O(1; 1) -- which produces the
     g_0-action on g_{-1},
  5) the Cartan prolong (O(1;1), sl(2) (+) hei(2))_{*,1} is built inside
     vect(3; 1) and compared with gr, closing the identification,
  6) H^2(g_-; g) is computed for g = gr, with the g_0-action and invariants.

Nothing here uses [KuJa], [GL3] or Ladilova: the only input is the deform.
"""
import numpy as np
from itertools import product as iproduct

from k3 import P, rref, nullspace
from deform import cocycles, deform, weisfeiler, unit, span_add
from ermolaev import red, quotient_basis, wfiltration, structure_on_quotient


# ------------------------------------------------------------------ gr g_lambda
class Graded:
    """an abstract graded Lie algebra given by a bracket table."""

    def __init__(self, deg, table, names=None):
        self.n = len(deg)
        self.deg = list(deg)
        self.c = table                      # c[i][j] = dict {k: coeff}
        self.names = names or ["b%d" % i for i in range(self.n)]
        self.wt = [(d,) for d in self.deg]

    def row(self, i):
        return self.c[i]

    def brk(self, i, j):
        return self.c[i][j]

    def bracket(self, u, v):
        out = [0] * self.n
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                for k, cc in self.c[i][j].items():
                    out[k] = (out[k] + a * b * cc) % P
        return out


def build_gr():
    """gr of the deform along the weight-(-3;3) cocycle, as a graded algebra."""
    idx = [i for i, (w, k, c) in enumerate(cocycles) if w == (-3, 3) and k == 2][0]
    coeffs = [0] * len(cocycles)
    coeffs[idx] = 1
    gl = deform(coeffs)
    n = gl.n

    subs = [L0 for cod, L0 in weisfeiler(gl, 3) if cod == 3]
    assert len(subs) == 1, "the Weisfeiler subalgebra is not unique"
    filt = wfiltration(gl, subs[0])
    layers = [[unit(n, i) for i in range(n)]] + filt      # L_{-1}, L_0, L_1, ...
    layers = [red(L, n) for L in layers]
    dims = [len(L) for L in layers] + [0]

    # basis of each gr_d = L_d / L_{d+1}, d = -1, 0, 1, ...
    reps, deg = [], []
    for t in range(len(layers)):
        nxt = layers[t + 1] if t + 1 < len(layers) else []
        r = quotient_basis(layers[t], nxt, n)
        reps.extend(r)
        deg.extend([t - 1] * len(r))
    assert len(reps) == n, (len(reps), n)

    # coordinates in the graded basis: express a vector of L_d modulo L_{d+1}
    piv_of = {}
    for d in sorted(set(deg)):
        rows = [reps[i] for i in range(n) if deg[i] == d]
        idxs = [i for i in range(n) if deg[i] == d]
        # solve inside L_d / L_{d+1}: append a basis of L_{d+1}
        tail = layers[d + 2] if d + 2 < len(layers) else []
        full = [r[:] for r in rows] + [r[:] for r in tail]
        aug = [full[j][:] + [1 if i == j else 0 for i in range(len(full))]
               for j in range(len(full))]
        Ra = rref(aug, n + len(full))[0]
        pv = [next(i for i, a in enumerate(r[:n]) if a) for r in Ra]
        piv_of[d] = (Ra, pv, idxs, len(rows))

    def coords(v, d):
        """coordinates of v in L_d, modulo L_{d+1}, in the graded basis."""
        Ra, pv, idxs, nr = piv_of[d]
        w = list(v)
        out = [0] * len(Ra)
        for i, pc in enumerate(pv):
            if w[pc]:
                out[i] = w[pc]
                w = [(a - out[i] * b) % P for a, b in zip(w, Ra[i][:n])]
        assert not any(w), "vector outside L_%d" % d
        acc = [0] * len(idxs)
        for i, c in enumerate(out):
            if c:
                for j in range(nr):
                    acc[j] = (acc[j] + c * Ra[i][n + j]) % P
        return {idxs[j]: c for j, c in enumerate(acc) if c}

    table = []
    for i in range(n):
        row = []
        for j in range(n):
            d = deg[i] + deg[j]
            br = gl.bracket(reps[i], reps[j])
            row.append(coords(br, d) if any(br) and d in piv_of else {})
        table.append(row)

    names = []
    for i in range(n):
        names.append("g%+d.%d" % (deg[i], sum(1 for t in range(i) if deg[t] == deg[i])))
    G = Graded(deg, table, names)

    # Jacobi, as a check that the quotient really is a Lie algebra
    for i in range(n):
        for j in range(n):
            for k in range(n):
                acc = {}
                for (a, b, c) in ((i, j, k), (j, k, i), (k, i, j)):
                    for m, co in G.c[a][b].items():
                        for t, co2 in G.c[m][c].items():
                            acc[t] = (acc.get(t, 0) + co * co2) % P
                assert not any(acc.values()), ("Jacobi", i, j, k)
    return gl, G, dims


# --------------------------------------------------------------- small-algebra ID
def subalg_table(G, idx):
    """the bracket table of the span of the basis vectors idx (must close)."""
    pos = {i: t for t, i in enumerate(idx)}
    tab = []
    for i in idx:
        row = []
        for j in idx:
            d = {}
            for k, c in G.c[i][j].items():
                assert k in pos, "not a subalgebra"
                d[pos[k]] = c
            row.append(d)
        tab.append(row)
    return tab


def subalg_table_raw(nn, tab, basis):
    """the bracket table of the subalgebra spanned by `basis` (rows of F_3^nn),
    written in that basis."""
    brk = bracket_of(nn, tab)
    aug = [basis[j][:] + [1 if i == j else 0 for i in range(len(basis))]
           for j in range(len(basis))]
    Ra = rref(aug, nn + len(basis))[0]
    pv = [next(i for i, a in enumerate(r[:nn]) if a) for r in Ra]

    def coords(v):
        w = v[:]
        co = [0] * len(Ra)
        for i, pc in enumerate(pv):
            if w[pc]:
                co[i] = w[pc]
                w = [(a - co[i] * b) % P for a, b in zip(w, Ra[i][:nn])]
        assert not any(w), "outside the subalgebra"
        out = [0] * len(basis)
        for i, c in enumerate(co):
            if c:
                for j in range(len(basis)):
                    out[j] = (out[j] + c * Ra[i][nn + j]) % P
        return out

    return [[{k: c for k, c in enumerate(coords(brk(basis[i], basis[j]))) if c}
             for j in range(len(basis))] for i in range(len(basis))]


def subspaces(nn, dim):
    """every dim-dimensional subspace of F_3^nn, once, as its RREF basis."""
    from itertools import combinations
    for piv in combinations(range(nn), dim):
        free = [(r, c) for r in range(dim) for c in range(piv[r] + 1, nn)
                if c not in piv]
        for vals in iproduct(range(P), repeat=len(free)):
            R = [[0] * nn for _ in range(dim)]
            for r, c in enumerate(piv):
                R[r][c] = 1
            for (r, c), v in zip(free, vals):
                R[r][c] = v
            yield R


def bracket_of(nn, tab):
    def brk(u, v):
        out = [0] * nn
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                for k, c in tab[i][j].items():
                    out[k] = (out[k] + a * b * c) % P
        return out
    return brk


def ideals(nn, tab, dim):
    """all ideals of the given dimension of a small F_3 Lie algebra."""
    brk = bracket_of(nn, tab)
    found = []
    for R in subspaces(nn, dim):
        ok = True
        for r in R:
            for i in range(nn):
                v = brk(r, unit(nn, i))
                test = [x[:] for x in R]
                if span_add(test, v[:]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            found.append([r[:] for r in R])
    return found


def subalgebras(nn, tab, dim, avoid=None):
    """dim-dimensional subalgebras; if `avoid` is given, only those complementary
    to the subspace `avoid`."""
    brk = bracket_of(nn, tab)
    out = []
    for R in subspaces(nn, dim):
        if avoid is not None:
            tot = [r[:] for r in R]
            if sum(1 for r in avoid if span_add(tot, r[:])) != len(avoid):
                continue
        ok = True
        for u in R:
            for v in R:
                w = brk(u, v)
                test = [x[:] for x in R]
                if any(w) and span_add(test, w[:]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            out.append([r[:] for r in R])
    return out


def main():
    print("=" * 70)
    print("3. gr of the deform, and the identification of gr_0")
    print("=" * 70)
    gl, G, dims = build_gr()
    n = G.n
    bydeg = {}
    for i in range(n):
        bydeg.setdefault(G.deg[i], []).append(i)
    print("   dim gr = %d,  graded dimensions %s"
          % (n, {d: len(bydeg[d]) for d in sorted(bydeg)}))

    g0 = bydeg[0]
    tab0 = subalg_table(G, g0)
    nn = len(g0)
    I3 = ideals(nn, tab0, 3)
    print("   ideals of gr_0 of dimension 3: %d" % len(I3))
    assert len(I3) == 1, "the nilradical is not unique"
    N = I3[0]

    def brk0(u, v):
        out = [0] * nn
        for i, a in enumerate(u):
            if not a:
                continue
            for j, b in enumerate(v):
                if not b:
                    continue
                for k, c in tab0[i][j].items():
                    out[k] = (out[k] + a * b * c) % P
        return out

    # N is Heisenberg?  [N, N] is 1-dimensional and central in N and in gr_0
    dN = []
    for u in N:
        for v in N:
            w = brk0(u, v)
            if any(w):
                span_add(dN, w[:])
    dN = red(dN, nn)
    print("   dim [N, N] = %d" % len(dN))
    zc = all(not any(brk0(dN[0], unit(nn, i))) for i in range(nn)) if dN else False
    print("   [N, N] is central in gr_0: %s" % zc)
    assert len(dN) == 1 and zc, "N is not the Heisenberg algebra hei(2)"

    # a complement: a 3-dimensional subalgebra S with S (+) N = gr_0, S simple
    comps = subalgebras(nn, tab0, 3, avoid=N)
    print("   3-dimensional subalgebras complementary to N: %d" % len(comps))
    assert comps, "no Levi complement"
    comp = comps[0]
    dS = []
    for u in comp:
        for v in comp:
            w = brk0(u, v)
            if any(w):
                span_add(dS, w[:])
    print("   dim [S, S] = %d  (sl(2) is perfect)" % len(red(dS, nn)))
    print("   S has no proper ideals: %s"
          % (len(ideals(3, subalg_table_raw(nn, tab0, comp), 1)) == 0
             and len(ideals(3, subalg_table_raw(nn, tab0, comp), 2)) == 0))

    print()
    print("=" * 70)
    print("4. gr_{-1} as a gr_0-module")
    print("=" * 70)
    gm = bydeg[-1]
    print("   dim gr_{-1} = %d" % len(gm))
    # the matrices of gr_0 on gr_{-1}
    mats = []
    for i in g0:
        M = np.zeros((len(gm), len(gm)), dtype=np.int64)
        for a, j in enumerate(gm):
            for k, c in G.c[i][j].items():
                M[gm.index(k), a] = c % P
        mats.append(M)
    faith = rref([[int(a) for a in M.reshape(-1)] for M in mats], len(gm) ** 2)[0]
    print("   the representation gr_0 -> gl(gr_{-1}) has image of dimension %d "
          "(faithful: %s)" % (len(faith), len(faith) == nn))
    # irreducible?
    print("   gr_{-1} is an irreducible gr_0-module: %s" % irreducible(mats, len(gm)))

    print()
    print("=" * 70)
    print("5. the Cartan prolong (gr_{-1}, gr_0)_{*,1} inside vect(3; 1)")
    print("=" * 70)
    pdims, pder = prolong_dims(mats)
    print("   graded dimensions of the prolong: %s,  total %d"
          % (pdims, sum(pdims.values())))
    got = {d: len(bydeg[d]) for d in sorted(bydeg)}
    print("   graded dimensions of gr:          %s,  total %d"
          % (got, sum(got.values())))
    print("   dim of the derived algebra of the prolong: %d" % pder)
    print("   so gr = (the prolong)^(1), which is what the superscript in (gl3) "
          "means: %s" % (pder == sum(got.values())))

    # does the centre of gr_0 act as a nonzero scalar?  (then id_V in gr_0, i.e.
    # the grading operator lies in gr_0 and Lemma modp applies)
    zc = dN[0]
    Mz = np.zeros((len(bydeg[-1]),) * 2, dtype=np.int64)
    for t, co in enumerate(zc):
        if co:
            Mz = (Mz + co * mats[t]) % P
    scal = all(Mz[a][b] % P == (Mz[0][0] if a == b else 0) % P
               for a in range(Mz.shape[0]) for b in range(Mz.shape[1]))
    print("   the centre of gr_0 acts on gr_{-1} as the scalar %d (so the grading "
          "operator lies in gr_0): %s" % (int(Mz[0][0]) % P, scal and Mz[0][0] % P))

    print()
    print("=" * 70)
    print("6. H^2(g_-; g) for g = er^(1)(3; 1)")
    print("=" * 70)
    from ceh import CE, module_data
    hidx = [i for i in range(n) if G.deg[i] < 0]
    g0idx = [i for i in range(n) if G.deg[i] == 0]
    ce = CE(G, hidx)
    H = ce.cohomology(2)
    print("   dim H^2(g_-; g) = %d" % sum(v[0] for v in H.values()))
    info = module_data(ce, H, g0idx, 2)
    for d in sorted(info):
        it = info[d]
        print("   degree %2d: dim %2d   g_0-invariants %d   submodule dims %s"
              % (d, it["dim"], it["inv"], sorted(set(it["subdims"]))))
    print("   total (H^2(g_-; g))^{g_0} = %d" % sum(v["inv"] for v in info.values()))
    H1 = ce.cohomology(1)
    print("   (control) dim H^1(g_-; g) = %d" % sum(v[0] for v in H1.values()))
    return G, bydeg, mats


def prolong_dims(mats):
    """the Cartan prolong of (V, g_0 subset gl(V)) inside vect(3; 1), V = F_3^3."""
    from gvect import Vect, rref as nrref, nullspace as nnull
    d = mats[0].shape[0]
    V = Vect([1] * d, [(0,) * d] * d, ["u%d" % (i + 1) for i in range(d)])
    Z = tuple([0] * d)

    def e(i):
        m = [0] * d
        m[i] = 1
        return tuple(m)

    gm1 = [{(Z, i): 1} for i in range(d)]
    g0 = []
    for M in mats:
        D = {}
        for i in range(d):
            for j in range(d):
                c = int(M[i, j]) % P
                if c:
                    D[(e(j), i)] = c
        g0.append(D)

    out = {-1: len(gm1), 0: len(g0)}
    layers = [gm1, g0]
    prev = g0
    k = 1
    while True:
        # a basis of vect(3;1)_k = O_{k+1} (x) V
        cand = [{key: 1} for key in V.basis_of_degree(k)]
        if not cand:
            break
        # solve: D in span(cand) with [D, d_i] in span(prev) for every i
        pkeys = sorted({key for D in prev for key in D})
        pm = np.array([[D.get(key, 0) % P for key in pkeys] for D in prev],
                      dtype=np.int64)
        R, piv = nrref(pm)
        cols = []
        for D in cand:
            col = []
            for i in range(d):
                br = V.bracket(D, gm1[i])
                v = np.array([br.get(key, 0) % P for key in pkeys], dtype=np.int64)
                w = v.copy()
                for t, pc in enumerate(piv):
                    if w[pc]:
                        w = (w - w[pc] * R[t]) % P
                extra = [int(x) for x in br.values() if False]      # placeholder
                # anything outside span(prev) must vanish
                col.extend(int(x) for x in w)
                # ... and any monomial not occurring in prev is a violation too
                col.extend(int(c) % P for key, c in br.items() if key not in pkeys)
            cols.append(col)
        width = max(len(c) for c in cols)
        Mx = np.array([c + [0] * (width - len(c)) for c in cols], dtype=np.int64)
        NS = nnull(Mx.T) if Mx.size else []
        cur = []
        for s in NS:
            D = {}
            for co, C in zip(s, cand):
                if co:
                    for key, c in C.items():
                        D[key] = (D.get(key, 0) + int(co) * c) % P
            D = {key: c for key, c in D.items() if c}
            if D:
                cur.append(D)
        if not cur:
            break
        out[k] = len(cur)
        layers.append(cur)
        prev = cur
        k += 1

    # the derived algebra of the prolong
    allb = [D for L in layers for D in L]
    keys = sorted({key for D in allb for key in D})
    rows = []
    for a in range(len(allb)):
        for b in range(a + 1, len(allb)):
            br = V.bracket(allb[a], allb[b])
            if br:
                rows.append([br.get(key, 0) % P for key in keys])
    R, _ = nrref(np.array(rows, dtype=np.int64)) if rows else ([], [])
    return out, len(R)


def irreducible(mats, d):
    """no proper nonzero invariant subspace (d is small, brute force over F_3)."""
    for k in range(1, d):
        for R in subspaces(d, k):
            ok = True
            for M in mats:
                for r in R:
                    v = (M @ np.array(r, dtype=np.int64)) % P
                    test = [x[:] for x in R]
                    if span_add(test, list(map(int, v))):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return False
    return True


if __name__ == "__main__":
    main()
