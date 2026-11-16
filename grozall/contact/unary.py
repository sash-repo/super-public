"""Linear (unary) k(3)-invariant differential operators T(k,lam) -> T(k',lam').

Same ansatz as the bilinear one with one argument:  u^{(c)} (D F^{(m)}),  D = e_1^i e_2^j R^s,
weighted order W = i+j+2s, and z-invariance ties lam' = lam + (W + k - k')/2.

Needed to tell a genuinely new bilinear operator from a composition: a bilinear operator
is "new" only if it is not a composition of a unary one with a lower one.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

from contact import Module, madd, mscal, component, from_components, pmono  # noqa: E402
from gen import apply_D, rand_field, K0, K1, K2, K3, KMINUS  # noqa: E402
from fast import SP2  # noqa: E402


def u_target(k, lam, k3, W):
    return Fr(lam) + Fr(W + k - k3, 2)


def u_terms(k, k3, W):
    out = []
    for s in range(W // 2 + 1):
        E = W - 2 * s
        for i in range(E + 1):
            j = E - i
            for m in range(k + 1):
                for c in range(k3 + 1):
                    out.append((m, (i, j, s), c))
    return out


def u_apply(term, F):
    m, D, c = term
    a = apply_D(D, component(F, m))
    return from_components({c: a}) if a else {}


def u_defect(term, f, F, M1, M3, LfF=None):
    if LfF is None:
        LfF = M1.act(f, F)
    return madd(M3.act(f, u_apply(term, F)), mscal(-1, u_apply(term, LfF)))


def u_rows(terms, gens, M1, M3, k, deg, nfield, seed=0):
    mat = []
    for f in gens:
        for i in range(nfield):
            F = rand_field(k, deg, seed=seed + 13 + i)
            LfF = M1.act(f, F)
            cols, keys = [], set()
            for term in terms:
                d = u_defect(term, f, F, M1, M3, LfF)
                cols.append(d)
                keys |= set(d)
            if not keys:
                continue
            keys = sorted(keys)
            idx = {kk: ii for ii, kk in enumerate(keys)}
            block = [[Fr(0)] * len(terms) for _ in keys]
            for jj, d in enumerate(cols):
                for kk, v in d.items():
                    block[idx[kk]][jj] = v
            mat.extend(block)
    return mat


def u_census(k, lam, k3, W, deg=None, nfield=2, verify=True):
    terms = u_terms(k, k3, W)
    if not terms:
        return None, terms, []
    l3 = u_target(k, lam, k3, W)
    if deg is None:
        deg = W + 1
    M1, M3 = Module(k, lam), Module(k3, l3)
    mat = u_rows(terms, K0 + K1, M1, M3, k, deg, nfield)
    ns = nullspace(mat) if mat else [
        [Fr(1) if i == j else Fr(0) for i in range(len(terms))] for j in range(len(terms))]
    # drop those that are the zero operator, then verify the rest independently
    keep = []
    F = [rand_field(k, deg + 1, seed=301 + i) for i in range(2)]
    for vec in ns:
        if any(any(mscal(cf, u_apply(term, f)) for cf, term in zip(vec, terms) if cf)
               for f in F):
            acc_any = False
            for f in F:
                acc = {}
                for cf, term in zip(vec, terms):
                    if cf:
                        acc = madd(acc, mscal(cf, u_apply(term, f)))
                if acc:
                    acc_any = True
            if acc_any:
                keep.append(vec)
    if verify and keep:
        for f in K0 + K1 + K2 + K3 + KMINUS:
            for i in range(3):
                FF = rand_field(k, deg + 2, seed=901 + i)
                LfF = M1.act(f, FF)
                for vec in keep:
                    acc = {}
                    for cf, term in zip(vec, terms):
                        if cf:
                            acc = madd(acc, mscal(cf, u_defect(term, f, FF, M1, M3, LfF)))
                    assert not acc, "a unary operator failed verification"
    return l3, terms, keep
