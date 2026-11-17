"""The same census, split in two so that a sweep in (lam1,lam2) is cheap.

k_0 = sp(2) + K z.  For f in sp(2) = <p^2,pq,q^2> one has f_t = 0, so the invariance
condition does NOT see any of the three weights; and z-invariance is what fixing the
weighted order W and setting lam3 = lam1+lam2+(W+k1+k2-k3)/2 already means.  So

  step 1 (once per (k1,k2,k3,W)):  the sp(2)-invariant subspace V_0 of the ansatz;
  step 2 (per weight pair):        invariance under k_1 inside V_0.

k_1 = S^3(W) + W as an sp(2)-module, so one generator of each component suffices --
p^3 and tp -- because f |-> I_f(B) is sp(2)-equivariant.  All six are used when the
run is cheap and in every verification.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [_HERE]
from exactlin import nullspace  # noqa: E402

from contact import Module, madd, mscal, pmono, pconst, k_piece  # noqa: E402
from gen import (  # noqa: E402
    term_basis, apply_term, defect, target_weight, rand_field, K0, K1, K2, K3, KMINUS,
)

SP2 = [pmono(0, 2, 0), pmono(0, 1, 1), pmono(0, 0, 2)]
Z = None  # the grading element, k_piece(0) contains t; z = 2t - pq
ZEL = {(1, 0, 0): Fr(2), (0, 1, 1): Fr(-1)}


def _conditions(vecs, terms, gens, M1, M2, M3, k1, k2, deg, nfield, seed=0):
    """Rows of the linear conditions on the coefficients of the given vectors.

    vecs=None means the identity, i.e. the coefficients of the terms themselves; then
    each term's defect is computed once instead of once per vector, which is the whole
    difference between minutes and hours as soon as the fibers are not trivial.
    """
    F_list = [rand_field(k1, deg, seed=seed + 11 + i) for i in range(nfield)]
    G_list = [rand_field(k2, deg, seed=seed + 71 + i) for i in range(nfield)]
    ncol = len(terms) if vecs is None else len(vecs)
    mat = []
    for f in gens:
        for F in F_list:
            for G in G_list:
                LfF, LfG = M1.act(f, F), M2.act(f, G)
                cols, keys = [], set()
                if vecs is None:
                    for term in terms:
                        acc = defect(term, f, F, G, M1, M2, M3, LfF, LfG)
                        cols.append(acc)
                        keys |= set(acc)
                else:
                    for vec in vecs:
                        acc = {}
                        for cf, term in zip(vec, terms):
                            if cf:
                                acc = madd(acc, mscal(
                                    cf, defect(term, f, F, G, M1, M2, M3, LfF, LfG)))
                        cols.append(acc)
                        keys |= set(acc)
                if not keys:
                    continue
                keys = sorted(keys)
                idx = {k: i for i, k in enumerate(keys)}
                block = [[Fr(0)] * ncol for _ in keys]
                for j, acc in enumerate(cols):
                    for k, v in acc.items():
                        block[idx[k]][j] = v
                mat.extend(block)
    return mat


def _compose(ns, vecs):
    """Express nullspace vectors (in the V_0 basis) back in the term basis."""
    out = []
    for c in ns:
        v = [Fr(0)] * len(vecs[0])
        for cf, base in zip(c, vecs):
            if cf:
                v = [a + cf * b for a, b in zip(v, base)]
        out.append(v)
    return out


_V0CACHE = {}


def sp2_invariants(k1, k2, k3, W, deg=None, nfield=2, check_z=True):
    """A basis of the sp(2)-invariant (equivalently k_{<=0}-invariant) ansatz."""
    key = (k1, k2, k3, W)
    if key in _V0CACHE:
        return _V0CACHE[key]
    terms = term_basis(k1, k2, k3, W)
    if not terms:
        return terms, []
    if deg is None:
        deg = W + 1
    ident = [[Fr(1) if i == j else Fr(0) for i in range(len(terms))]
             for j in range(len(terms))]
    M1, M2 = Module(k1, 0), Module(k2, 0)
    M3 = Module(k3, target_weight(k1, 0, k2, 0, k3, W))
    mat = _conditions(None, terms, SP2, M1, M2, M3, k1, k2, deg, nfield)
    ns = nullspace(mat) if mat else ident
    if check_z:
        # the fixed-W ansatz with lam3 as defined must already be z-invariant
        bad = _conditions(ns, terms, [ZEL], M1, M2, M3, k1, k2, deg, 1)
        assert not any(any(x for x in row) for row in bad), "z-invariance failed"
    _V0CACHE[key] = (terms, ns)
    return terms, ns


def census(k1, l1, k2, l2, k3, W, deg=None, nfield=2, gens=None, verify=True):
    """dim of the space of invariant operators, and a basis in the term basis."""
    terms, V0 = sp2_invariants(k1, k2, k3, W, deg=deg)
    l3 = target_weight(k1, l1, k2, l2, k3, W)
    if not V0:
        return l3, terms, []
    if deg is None:
        deg = W + 1
    if gens is None:
        gens = [pmono(0, 3, 0), pmono(1, 1, 0)]  # p^3 and tp
    M1, M2, M3 = Module(k1, l1), Module(k2, l2), Module(k3, l3)
    mat = _conditions(V0, terms, gens, M1, M2, M3, k1, k2, deg, nfield)
    ns = nullspace(mat) if mat else [
        [Fr(1) if i == j else Fr(0) for i in range(len(V0))] for j in range(len(V0))]
    ops = _compose(ns, V0)
    if verify and ops:
        bad = _conditions(ops, terms, K0 + K1 + K2 + K3 + KMINUS, M1, M2, M3,
                          k1, k2, deg + 2, nfield + 1, seed=500)
        assert not any(any(x for x in row) for row in bad), \
            "an operator failed the independent verification"
    return l3, terms, ops


def alive(terms, ops, k1, k2, deg=4):
    """Those that are non-zero as operators (a scheme can be the zero operator)."""
    out = []
    F = [rand_field(k1, deg, seed=7 + i) for i in range(2)]
    G = [rand_field(k2, deg, seed=97 + i) for i in range(2)]
    for vec in ops:
        ok = False
        for f in F:
            for g in G:
                acc = {}
                for cf, term in zip(vec, terms):
                    if cf:
                        acc = madd(acc, mscal(cf, apply_term(term, f, g)))
                if acc:
                    ok = True
        if ok:
            out.append(vec)
    return out
