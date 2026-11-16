#!/usr/bin/env python3
r"""
skry3.py --- the 3-structures of the Skryabin algebras over F_3.

A vector field D in vect(n; \One) over a field of characteristic p is a derivation
of O(n; \One), hence so is D^p (composition p times): vect(n; \One) is restricted,
with D^{[p]} = D^p.  Every Skryabin algebra of \cite{Sk,GL} is realized inside
vect(n; \One) as a (generalized) Cartan prolong, so the only question is whether
the realization is CLOSED under D |--> D^3.

This module supplies
  * act(V, D, f)     --- the field D applied to a polynomial f in O(n; \One);
  * cube(V, D)       --- D^3, read off its values on the coordinates;
  * is_derivation    --- the check that D^3 really is a vector field;
  * padic tests      --- ad_{D^3} = (ad_D)^3 on the whole algebra.

The models of my(6;\One), smy(6;\One), by(7;\One), sby(7;\One) are the complete
Cartan prolongs of prolong.py; dy(10;\One) is built here from
the explicit fields of [KLS] = arXiv:2309.16370.

Output: runs/skry3.log
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
from gvect import P, Vect, rref, nullspace                            # noqa: E402

assert P == 3


# ----------------------------------------------------------------- the cube map
def act(V, D, f):
    """the vector field D applied to the polynomial f (both dicts over F_p)."""
    out = {}
    for (m, i), c in D.items():
        for mm, cc in V.pmul({m: 1}, V.dpart(f, i)).items():
            co = (out.get(mm, 0) + c * cc) % P
            if co:
                out[mm] = co
            else:
                out.pop(mm, None)
    return out


def power(V, D, k):
    r"""D^k as a derivation of O(n; \One), read off its values on the coordinates.

    O(n; \One) is generated as an algebra by u_1, ..., u_n when p >= 3 (u^{(2)} =
    2u^2 for p = 3), so a derivation is determined by the images of the u_a; and
    D^p is a derivation because p is the characteristic.  For k = p the result is
    therefore an honest element of vect(n; \One)."""
    out = {}
    for a in range(V.n):
        f = {tuple(1 if t == a else 0 for t in range(V.n)): 1}
        for _ in range(k):
            f = act(V, D, f)
            if not f:
                break
        for m, c in f.items():
            if c % P:
                out[(m, a)] = c % P
    return out


def cube(V, D):
    return power(V, D, P)


def is_derivation(V, D, trials=None):
    """check on all products of two coordinates that D is a derivation of O."""
    cs = [{tuple(1 if t == a else 0 for t in range(V.n)): 1} for a in range(V.n)]
    for a in range(V.n):
        for b in range(V.n):
            fg = V.pmul(cs[a], cs[b])
            lhs = act(V, D, fg)
            rhs = {}
            for part in (V.pmul(act(V, D, cs[a]), cs[b]),
                         V.pmul(cs[a], act(V, D, cs[b]))):
                for m, c in part.items():
                    rhs[m] = (rhs.get(m, 0) + c) % P
            rhs = {m: c for m, c in rhs.items() if c}
            if lhs != rhs:
                return False
    return True


# --------------------------------------------------------------- linear algebra
def coords_in(basis_mat, keys, D):
    """coordinates of the field D in the rows of basis_mat, or None."""
    v = np.zeros(len(keys), dtype=np.int64)
    kpos = {k: i for i, k in enumerate(keys)}
    for k, c in D.items():
        if k not in kpos:
            return None
        v[kpos[k]] = c % P
    aug = np.concatenate([basis_mat, np.eye(len(basis_mat), dtype=np.int64)],
                         axis=1)
    Ra, _ = rref(aug)
    R, piv = rref(basis_mat)
    T = Ra[:len(R), len(keys):]
    co = np.zeros(len(R), dtype=np.int64)
    w = v.copy() % P
    for i, c in enumerate(piv):
        if w[c]:
            co[i] = w[c] % P
            w = (w - co[i] * R[i]) % P
    if w.any():
        return None
    return (co @ T) % P


class Model:
    r"""a graded subalgebra of vect(n; \One) given by a basis of fields."""

    def __init__(self, V, basis, deg, wt, name):
        self.V, self.basis, self.name = V, list(basis), name
        self.deg, self.wt = list(deg), list(wt)
        self.n = len(self.basis)
        self.keys = sorted({k for D in self.basis for k in D})
        self.M = np.zeros((self.n, len(self.keys)), dtype=np.int64)
        self.kpos = {k: i for i, k in enumerate(self.keys)}
        for r, D in enumerate(self.basis):
            for k, c in D.items():
                self.M[r, self.kpos[k]] = c % P
        self.R, self.piv = rref(self.M)
        aug = np.concatenate([self.M, np.eye(self.n, dtype=np.int64)], axis=1)
        Ra, _ = rref(aug)
        self.T = Ra[:len(self.R), len(self.keys):]
        assert len(self.R) == self.n, "the given basis is dependent"

    def coords(self, D):
        """coordinates of the field D in self.basis, or None if D is outside."""
        v = np.zeros(len(self.keys), dtype=np.int64)
        for k, c in D.items():
            i = self.kpos.get(k)
            if i is None:
                return None
            v[i] = c % P
        co = np.zeros(len(self.R), dtype=np.int64)
        for i, c in enumerate(self.piv):
            if v[c]:
                co[i] = v[c] % P
                v = (v - co[i] * self.R[i]) % P
        if v.any():
            return None
        return (co @ self.T) % P

    def field(self, vec):
        out = {}
        for c, D in zip(vec, self.basis):
            if c % P:
                for k, cc in D.items():
                    out[k] = (out.get(k, 0) + int(c) * cc) % P
        return {k: c for k, c in out.items() if c}

    def admat(self, D):
        """the matrix of ad_D on the algebra, or None if ad_D leaves it."""
        A = np.zeros((self.n, self.n), dtype=np.int64)
        for j, E in enumerate(self.basis):
            co = self.coords(self.V.bracket(D, E))
            if co is None:
                return None
            A[:, j] = co
        return A


def report(model, log, torus=None):
    """cube every basis element; decide restrictedness; return the p-map."""
    V = model.V
    say = log
    inside, outside, inner_only = [], [], []
    pmap = {}
    for i, D in enumerate(model.basis):
        C = cube(V, D)
        co = model.coords(C) if C else np.zeros(model.n, dtype=np.int64)
        if co is not None:
            pmap[i] = co
            (inside if C else inside).append(i)
            if not C:
                pass
        else:
            outside.append(i)
    say("  cubes inside the algebra: %d of %d" % (len(inside), model.n))
    if outside:
        say("  cubes OUTSIDE: %d  (indices %s)" % (len(outside), outside[:10]))
        # is ad^3 nevertheless inner?
        for i in outside:
            A = model.admat(model.basis[i])
            if A is None:
                inner_only.append((i, None))
                continue
            A3 = (A @ A @ A) % P
            cols = []
            for j in range(model.n):
                cols.append(model.admat(model.basis[j]))
            big = np.stack([c.reshape(-1) for c in cols], axis=1) % P
            sol = np.concatenate([big, A3.reshape(-1, 1)], axis=1)
            R1, p1 = rref(big)
            R2, p2 = rref(sol)
            inner_only.append((i, len(p2) == len(p1)))
        say("  ad^3 inner anyway: %s" % inner_only)
    nz = [i for i in pmap if pmap[i].any()]
    say("  non-zero p-map on %d of %d basis elements" % (len(nz), model.n))
    return pmap, outside


def pretty(model, pmap, log, limit=40):
    V = model.V
    shown = 0
    for i in sorted(pmap):
        if not pmap[i].any():
            continue
        shown += 1
        if shown > limit:
            log("  ... (%d more)" % (len([1 for j in pmap if pmap[j].any()])
                                     - limit))
            break
        log("    (%s)^[3] = %s     [deg %d -> %d, wt %s -> %s]"
            % (V.show(model.basis[i]), V.show(model.field(pmap[i])),
               model.deg[i], 3 * model.deg[i],
               model.wt[i], tuple(3 * a for a in model.wt[i])))
