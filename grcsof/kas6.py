#!/usr/bin/env python3
"""n = 6: the exceptional pair as  subset  kas  subset  k(1|6).

k(1|6) is taken in the xi-eta realization of the paper:
    alpha_1 = dt + sum_{1<=i<=3} (xi_i d eta_i + eta_i d xi_i),
so the odd coordinates are ordered y = (xi1, eta1, xi2, eta2, xi3, eta3) and
the Gram matrix B is block-diagonal with the blocks antidiag(1,1).

Standard grading: deg t = 2, deg y_i = 1, deg K_f = deg f - 2, so

    g_{-2} = <K_1>,  g_{-1} = <K_{y_i}>,  g_0 = <K_t> + Lambda^2 = c o(6),
    g_1    = Lambda^3 + t Lambda^1.

Only for n = 6 does Lambda^3 split into two irreducible o(6)-modules,
Lambda^3_xi (the one containing xi1 xi2 xi3) and Lambda^3_eta; both are
10-dimensional, S^2(id) and S^2(id^*) for g_0 = gl(4).

    kas = (g_- + g_0, Lambda^3_xi + t Lambda^1)_*   (Shchepochkina),
    as  = <K_1, Lambda^1, Lambda^2, Lambda^3_xi>    (Sergeev's central
          extension of spe(4); 16|16-dimensional, K_1 spans its centre).

This module builds these algebras, checks their structure, and gives the
generating sets used in the classification of invariant operators.
"""
from fractions import Fraction as Fr
from itertools import combinations

from superN import Contact, add, smul, mono, parity, popcount

# ------------------------------------------------------------------ k(1|6)

NAMES = ['xi1', 'eta1', 'xi2', 'eta2', 'xi3', 'eta3']
B6 = [[0] * 6 for _ in range(6)]
for i in range(3):
    B6[2 * i][2 * i + 1] = 1
    B6[2 * i + 1][2 * i] = 1
ALG = Contact(6, B6, NAMES)

XI = [0, 2, 4]          # indices of xi1, xi2, xi3
ETA = [1, 3, 5]         # indices of eta1, eta2, eta3


def msk(*idx):
    m = 0
    for i in idx:
        m |= 1 << i
    return m


def fmono(*idx):
    """the superfunction y_{i1} ... y_{ir} (indices increasing)."""
    return mono(0, msk(*idx))


LAM1 = [fmono(i) for i in range(6)]
LAM2 = [fmono(i, j) for i, j in combinations(range(6), 2)]
LAM3 = [fmono(i, j, k) for i, j, k in combinations(range(6), 3)]
TT = mono(1, 0)                                   # t
ONE = mono(0, 0)                                  # 1
TLAM1 = [mono(1, 1 << i) for i in range(6)]       # t y_i

# --------------------------------------------- the split Lambda^3 = xi + eta


def _vec(f, basis_masks):
    return [f.get((0, m), 0) for m in basis_masks]


def _span(vectors):
    """row echelon basis of the span (exact, Fractions)."""
    rows = []
    for v in vectors:
        v = [Fr(x) for x in v]
        for r, p in rows:
            if v[p]:
                c = v[p] / r[p]
                v = [a - c * b for a, b in zip(v, r)]
        p = next((i for i, x in enumerate(v) if x), None)
        if p is not None:
            rows.append((v, p))
    return rows


def _in_span(rows, v):
    v = [Fr(x) for x in v]
    for r, p in rows:
        if v[p]:
            c = v[p] / r[p]
            v = [a - c * b for a, b in zip(v, r)]
    return not any(v)


def module_generated(seed, acting):
    """the smallest subspace of Lambda^3 containing seed and stable under
    ad K_f for f in acting (all of Lambda^2 here)."""
    masks = [msk(*c) for c in combinations(range(6), 3)]
    basis = [seed]
    rows = _span([_vec(seed, masks)])
    frontier = [seed]
    while frontier:
        new = []
        for g in frontier:
            for f in acting:
                h = ALG.kb(f, g)
                if not h:
                    continue
                v = _vec(h, masks)
                assert all(k[0] == 0 and popcount(k[1]) == 3 for k in h), h
                if not _in_span(rows, v):
                    rows = _span([r for r, _ in rows] + [v])
                    basis.append(h)
                    new.append(h)
        frontier = new
    return basis


CUB_XI = module_generated(fmono(*XI), LAM2)       # Lambda^3_xi,  10-dimensional
CUB_ETA = module_generated(fmono(*ETA), LAM2)     # Lambda^3_eta

# ------------------------------------------------------ the four subalgebras


def named(fs, tag):
    return {f"{tag}{i}": f for i, f in enumerate(fs)}


def gens_as():
    """ALL generating functions of as = <1, Lambda^1, Lambda^2, Lambda^3_xi>."""
    out = {'1': ONE}
    out.update({NAMES[i]: LAM1[i] for i in range(6)})
    for i, j in combinations(range(6), 2):
        out[NAMES[i] + NAMES[j]] = fmono(i, j)
    out.update(named(CUB_XI, 'c'))
    return out


def gens_asop():
    """the mirror algebra <1, Lambda^1, Lambda^2, Lambda^3_eta>: as with the
    two chiralities interchanged.  Used as a control experiment: everything
    that distinguishes the two half-spinors must change sides with it."""
    out = {'1': ONE}
    out.update({NAMES[i]: LAM1[i] for i in range(6)})
    for i, j in combinations(range(6), 2):
        out[NAMES[i] + NAMES[j]] = fmono(i, j)
    out.update(named(CUB_ETA, 'c'))
    return out


def gens_cas():
    """as + C K_t (the derivation algebra of as inside k(1|6))."""
    out = gens_as()
    out['t'] = TT
    return out


def gens_kas():
    """the generators of kas in degrees <= 1: g_-, g_0, Lambda^3_xi, t Lambda^1
    (g_1 generates g_+, so this is a generating set of kas)."""
    out = gens_cas()
    for i in range(6):
        out['t' + NAMES[i]] = TLAM1[i]
    return out


def gens_osp():
    """i(osp(6|2)) = g_- + g_0 + t Lambda^1 + C K_{t^2}."""
    out = {'1': ONE, 't': TT, 't2': mono(2, 0)}
    out.update({NAMES[i]: LAM1[i] for i in range(6)})
    for i, j in combinations(range(6), 2):
        out[NAMES[i] + NAMES[j]] = fmono(i, j)
    for i in range(6):
        out['t' + NAMES[i]] = TLAM1[i]
    return out


# small generating sets (Lie-generating, used to cut the number of equations;
# every solution found is re-verified against the FULL lists above)
SMALL = {
    'as': ['1'] + NAMES + ['c0'],
    'asop': ['1'] + NAMES + ['c0'],
    'cas': ['1'] + NAMES + ['c0', 't'],
    'kas': ['1'] + NAMES + ['c0', 't', 'txi1'],
    'osp': ['1'] + NAMES + ['t', 'txi1'],
}


def small_gens(which):
    F = {'as': gens_as, 'cas': gens_cas, 'kas': gens_kas,
         'osp': gens_osp}[which]()
    return [F[k] for k in SMALL[which]]


def full_gens(which):
    return {'as': gens_as, 'cas': gens_cas, 'kas': gens_kas,
            'osp': gens_osp}[which]()


# ------------------------------------------------------------- torus weights

def wt_of_index(i):
    """weight of y_i under the Cartan <K_{xi_i eta_i}>, as a 3-vector."""
    w = [0, 0, 0]
    w[i // 2] = 1 if i % 2 == 0 else -1
    return tuple(w)


def wt_of_subset(A):
    w = [0, 0, 0]
    for i in A:
        for k, x in enumerate(wt_of_index(i)):
            w[k] += x
    return tuple(w)


def zero_weight(*subsets):
    w = [0, 0, 0]
    for A in subsets:
        for k, x in enumerate(wt_of_subset(A)):
            w[k] += x
    return all(x == 0 for x in w)


# ------------------------------------------------------------------ checking

def closure(fs, maxit=8):
    """the Lie subalgebra of k(1|6) generated by the functions fs, as a list
    of superfunctions; returns None if it does not close within maxit rounds
    (i.e. is infinite-dimensional)."""
    keys = set()
    for f in fs:
        keys |= set(f.keys())
    all_keys = sorted(keys)

    def vec(f, ks):
        return [f.get(k, 0) for k in ks]

    cur = list(fs)
    for _ in range(maxit):
        ks = sorted({k for f in cur for k in f.keys()})
        rows = _span([vec(f, ks) for f in cur])
        newf = []
        for i, f in enumerate(cur):
            for g in cur[i:]:
                h = ALG.kb(f, g)
                if not h:
                    continue
                ks2 = sorted({k for f2 in cur + newf + [h] for k in f2.keys()})
                rows2 = _span([vec(f2, ks2) for f2 in cur + newf])
                if not _in_span(rows2, vec(h, ks2)):
                    newf.append(h)
        if not newf:
            return cur
        cur = cur + newf
    return None


def dims(fs):
    """(even, odd) dimensions of a list of superfunctions (as K_f's)."""
    ev = od = 0
    ks = sorted({k for f in fs for k in f.keys()})
    for f in fs:
        (ev, od) = (ev + 1, od) if parity(f) == 0 else (ev, od + 1)
    return ev, od


if __name__ == '__main__':
    print("dim Lambda^3_xi  =", len(CUB_XI))
    print("dim Lambda^3_eta =", len(CUB_ETA))
    print("Lambda^3_xi basis:")
    for f in CUB_XI:
        print("   ", ALG.show(f))
    A = gens_as()
    print("\ndim as =", len(A), " (even|odd) =", dims(list(A.values())))
    # closure of as
    cl = closure(list(A.values()))
    print("as closes:", cl is not None and len(cl) == len(A),
          "  dim of the closure:", len(cl) if cl else "inf")
    # [Lambda^3_xi, Lambda^3_xi] = 0 ?
    bad = [(i, j) for i, f in enumerate(CUB_XI) for j, g in enumerate(CUB_XI)
           if ALG.kb(f, g)]
    print("[Lambda^3_xi, Lambda^3_xi] = 0 :", not bad)
    bad2 = [(i, j) for i, f in enumerate(CUB_ETA) for j, g in enumerate(CUB_ETA)
            if ALG.kb(f, g)]
    print("[Lambda^3_eta, Lambda^3_eta] = 0 :", not bad2)
    # is K_1 central in as?
    print("K_1 central in as:",
          not any(ALG.kb(ONE, f) for f in A.values()))
    # the cocycle: {y_i, y_j} = const
    print("{Lambda^1, Lambda^1} =",
          {ALG.show(ALG.kb(LAM1[i], LAM1[j])) for i in range(6)
           for j in range(6)} - {'0'})
    # as generated by Lambda^1 and one cubic?
    gen = closure([ONE] + LAM1 + [CUB_XI[0]])
    print("as generated by <1, Lambda^1, xi1xi2xi3>:",
          gen is not None and len(gen) == 32, " dim =",
          len(gen) if gen else "inf")
