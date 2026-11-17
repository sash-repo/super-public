#!/usr/bin/env python3
"""
models.py -- the structure functions of an arbitrary Minkowski-superspace model
of "From Supergravity to Ballbearings" (Grozman & Leites).

A model is a parity string Par of length n = N+4 (four positions of one parity
carry the spacetime sl(4), the other N carry the internal sl(N)).  The marking
of the simple roots -- which is what actually fixes the Z-grading -- is read off
Par by the rule stated in the paper: cut Par into consecutive blocks of size
<= 2, a change of parity forcing a cut.  Blocks B_1,...,B_r give

    deg E_ij = b(j) - b(i),      b(i) = index of the block holding i,

so g_- is the strictly lower block-triangular part, of depth r-1, and g_0 is the
block diagonal:  "Riemannian"  = its semisimple part, (+) sl(2) over the size-2
blocks;  "conformal" = the whole of it inside sl(N|4) (mod the centre if N = 4).

The structure functions are the Spencer cohomology

    H^2(g_-; (g_-,g_0)_*),        (g_-,g_0)_* = the generalized Cartan prolong,

graded by the Z-degree; each H^{2,k} is a g_0-module, reported as a sum of
irreducible sl(2)^{(number of size-2 blocks)}-modules R(w_1,...,w_k), which is
the format of the paper's Table.  The parity of a class (the Table's subscript
1 = odd) is read off its weight: for a weight vector of weight sum(c_i eps_i)
the parity is sum(c_i p_i) mod 2.

Everything is exact and weight-homogeneous: every space is split by the full
diagonal torus of gl(N|4) before any elimination, so the linear algebra happens
in small blocks and the answer comes out as a character.

Conventions.
  * [E_ij, E_kl] = d_jk E_il - (-1)^{p(ij)p(kl)} d_li E_kj.
  * a q-cochain is a super-antisymmetric q-linear map g_- -> P, i.e.
    c(x,y) = -(-1)^{p(x)p(y)} c(y,x);  a basis is given by its values on the
    non-decreasing tuples whose repetitions occur at ODD arguments only.
  * (dc)(x,y) = (-1)^{p(c)p(x)}[x,c(y)] - (-1)^{p(y)(p(c)+p(x))}[y,c(x)]
                - c([x,y]),
    (dc)(x,y,z) = (-1)^{p(c)p(x)}[x,c(y,z)] - (-1)^{p(y)(p(c)+p(x))}[y,c(x,z)]
                + (-1)^{p(z)(p(c)+p(x)+p(y))}[z,c(x,y)]
                - c([x,y],z) + (-1)^{p(y)p(z)}c([x,z],y)
                - (-1)^{p(x)(p(y)+p(z))}c([y,z],x);
    d o d = 0 is verified in test_models.py, not assumed.
"""

import os
import sys
from fractions import Fraction
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from exactlin import rank, nullspace, backend        # noqa: E402,F401


# ==========================================================================
#  the model
# ==========================================================================


def blocks_of(par):
    """Cut Par into consecutive blocks of size <= 2; a parity change forces a
    cut.  Returns a list of (size, parity)."""
    out, i = [], 0
    while i < len(par):
        if i + 1 < len(par) and par[i + 1] == par[i]:
            out.append((2, int(par[i])))
            i += 2
        else:
            out.append((1, int(par[i])))
            i += 1
    return out


class Model:
    """gl(n) with the parities of Par and the Z-grading of its block cut."""

    def __init__(self, par):
        self.par = par
        self.p = [int(c) for c in par]
        self.n = len(par)
        self.blocks = blocks_of(par)
        self.r = len(self.blocks)
        b, pos = [], []
        for k, (s, _) in enumerate(self.blocks):
            pos.append(tuple(range(len(b), len(b) + s)))
            b += [k] * s
        self.b = b
        self.bpos = pos
        self.sl2 = [q for q in pos if len(q) == 2]
        c0, c1 = self.p.count(0), self.p.count(1)
        self.Nsusy = c1 if c0 == 4 else (c0 if c1 == 4 else None)

    def deg(self, i, j):
        return self.b[j] - self.b[i]

    def par_of(self, i, j):
        return (self.p[i] + self.p[j]) % 2

    def wt(self, i, j):
        w = [0] * self.n
        w[i] += 1
        w[j] -= 1
        return tuple(w)

    def parity_of_weight(self, w):
        return sum(c * q for c, q in zip(w, self.p)) % 2

    def sl2_weight(self, w):
        return tuple(w[u] - w[v] for (u, v) in self.sl2)

    def unit_bracket(self, i, j, k, l):
        """[E_ij, E_kl] as a dict {(a,b): coeff}."""
        out = {}
        if j == k:
            out[(i, l)] = out.get((i, l), 0) + 1
        if l == i:
            s = -(-1) ** (self.par_of(i, j) * self.par_of(k, l))
            out[(k, j)] = out.get((k, j), 0) + s
        return {kk: v for kk, v in out.items() if v}


# ==========================================================================
#  the prolong P = (g_-, g_0)_*
# ==========================================================================


class Prolong:
    MAXDEG = 8

    def __init__(self, par, kind="riem", verbose=False, keep_identity=False):
        self.M = Model(par)
        self.kind = kind
        self.verbose = verbose
        self.keep_identity = keep_identity
        self.basis = {}          # m -> list of basis items
        self.wts = {}            # m -> list of weights (tuples)
        self.pars = {}           # m -> list of parities
        self.rb = {}             # m -> list over idx of {a: {t: coeff}}
        self.unit_index = {}     # m<0 -> {(i,j): idx}
        self._negative()
        self._zero()
        self._prolong()

    # ------------------------------------------------------------------ g_-

    def _negative(self):
        M = self.M
        neg = {}
        for i in range(M.n):
            for j in range(M.n):
                d = M.deg(i, j)
                if d < 0:
                    neg.setdefault(d, []).append((i, j))
        for d, items in neg.items():
            self.basis[d] = items
            self.unit_index[d] = {u: k for k, u in enumerate(items)}
            self.wts[d] = [M.wt(*u) for u in items]
            self.pars[d] = [M.par_of(*u) for u in items]
        self.depth = -min(neg)
        self.gm = []                       # flat basis of g_-
        for d in sorted(neg, reverse=True):
            for k, u in enumerate(neg[d]):
                self.gm.append((d, k, u))
        self.ngm = len(self.gm)
        self.gm_deg = [x[0] for x in self.gm]
        self.gm_wt = [M.wt(*x[2]) for x in self.gm]
        self.gm_par = [M.par_of(*x[2]) for x in self.gm]
        self.gm_at = {(d, k): a for a, (d, k, _) in enumerate(self.gm)}
        # structure constants
        self.f = []
        for a, (da, _, ua) in enumerate(self.gm):
            row = []
            for b, (db, _, ub) in enumerate(self.gm):
                br = M.unit_bracket(*ua, *ub)
                tgt = da + db
                out = {}
                for u, c in br.items():
                    if tgt in self.unit_index and u in self.unit_index[tgt]:
                        out[self.gm_at[(tgt, self.unit_index[tgt][u])]] = c
                    elif c:
                        raise RuntimeError("g_- is not closed")
                row.append(out)
            self.f.append(row)
        for d in sorted(neg):
            self._rb_matrix(d)

    def _rb_matrix(self, m):
        """Right brackets [X, v_a] for a matrix-realized degree m <= 0."""
        M = self.M
        tab = []
        for item in self.basis[m]:
            terms = item if isinstance(item, dict) else {item: 1}
            row = {}
            for a in range(self.ngm):
                tgt = m + self.gm_deg[a]
                va = self.gm[a][2]
                out = {}
                for u, c in terms.items():
                    for v, c2 in M.unit_bracket(*u, *va).items():
                        if tgt in self.unit_index and v in self.unit_index[tgt]:
                            t = self.unit_index[tgt][v]
                            out[t] = out.get(t, 0) + c * c2
                        elif c * c2:
                            raise RuntimeError(
                                f"[g_{m}, g_-] left the prolong at degree {tgt}")
                out = {t: v for t, v in out.items() if v}
                if out:
                    row[a] = out
            tab.append(row)
        self.rb[m] = tab

    # ------------------------------------------------------------------ g_0

    def _zero(self):
        M = self.M
        bas = []
        if self.kind in ("riem", "conf"):
            for (u, v) in M.sl2:
                bas += [{(u, v): 1}, {(v, u): 1}, {(u, u): 1, (v, v): -1}]
            if self.kind == "conf":
                # + the grading element z, [z, E_ij] = deg(E_ij) E_ij:
                # this is the superization of co(4) = o(4) (+) C.Id
                bas.append({(i, i): -M.b[i] for i in range(M.n) if M.b[i]})
        else:
            for q in M.bpos:
                for i in q:
                    for j in q:
                        if i != j:
                            bas.append({(i, j): 1})
            sgn = [(-1) ** M.p[i] for i in range(M.n)]
            if sum(sgn) != 0 or self.keep_identity:
                # a basis of {d : sum sgn_i d_i = 0}
                for i in range(1, M.n):
                    bas.append({(0, 0): sgn[i], (i, i): -sgn[0]})
            else:
                # N = 4: the identity is supertraceless -- take a complement of
                # C.1 inside it, i.e. the supertraceless diagonals with d_0 = 0
                for i in range(2, M.n):
                    bas.append({(1, 1): sgn[i], (i, i): -sgn[1]})
        self.basis[0] = bas
        self.wts[0], self.pars[0] = [], []
        for X in bas:
            ks = list(X)
            w = M.wt(*ks[0])
            for k in ks[1:]:
                if M.wt(*k) != w:
                    raise RuntimeError("g_0 basis vector is not a weight vector")
            self.wts[0].append(w)
            self.pars[0].append(M.par_of(*ks[0]))
        self._rb_matrix(0)

    # -------------------------------------------------------------- prolong

    def dim(self, m):
        return len(self.basis.get(m, []))

    def _prolong(self):
        k = 1
        while k <= self.MAXDEG:
            got = self._degree(k)
            if not got:
                break
            acts, wts = got
            self.basis[k] = acts
            self.wts[k] = wts
            self.pars[k] = [self.M.parity_of_weight(w) for w in wts]
            # [X, v_a] = X(v_a): the act table IS the right-bracket table
            self.rb[k] = [{a: dict(d) for a, d in enumerate(act) if d}
                          for act in acts]
            if self.verbose:
                print(f"    g_{k}: dim {len(acts)}")
            k += 1
        else:
            raise RuntimeError("prolong did not terminate")
        self.top = max(self.basis)

    def _degree(self, k):
        """g_k as the space of degree-k derivations g_- -> P."""
        slots = []
        for a in range(self.ngm):
            m = k + self.gm_deg[a]
            for idx in range(self.dim(m)):
                slots.append((a, idx, m))
        if not slots:
            return None
        byw = {}
        for s, (a, idx, m) in enumerate(slots):
            w = tuple(x - y for x, y in zip(self.wts[m][idx], self.gm_wt[a]))
            byw.setdefault(w, []).append(s)

        acts, wts = [], []
        for w in sorted(byw):
            cols = byw[w]
            colmap = {(slots[s][0], slots[s][1]): c for c, s in enumerate(cols)}
            rows = self._equations(k, colmap, len(cols))
            sols = nullspace(rows) if rows else \
                [[Fraction(int(i == j)) for i in range(len(cols))]
                 for j in range(len(cols))]
            for sol in sols:
                act = [dict() for _ in range(self.ngm)]
                for c, val in enumerate(sol):
                    if val:
                        a, idx, _ = slots[cols[c]]
                        act[a][idx] = val
                acts.append(act)
                wts.append(w)
        return (acts, wts) if acts else None

    def _equations(self, k, colmap, ncol):
        """X([v_a,v_b]) - [X(v_a),v_b] + (-1)^{p_a p_b}[X(v_b),v_a] = 0."""
        eqs = {}

        def add(key, col, val):
            if not val:
                return
            row = eqs.get(key)
            if row is None:
                row = eqs[key] = [0] * ncol
            row[col] += val

        for a in range(self.ngm):
            for b in range(a, self.ngm):
                sgn = (-1) ** (self.gm_par[a] * self.gm_par[b])
                # LHS  X([v_a,v_b])
                for c, lam in self.f[a][b].items():
                    for idx in range(self.dim(k + self.gm_deg[c])):
                        col = colmap.get((c, idx))
                        if col is not None:
                            add((a, b, idx), col, lam)
                # RHS  -[X(v_a),v_b] + (-1)^{p_a p_b} [X(v_b),v_a]
                for (x, y, sg) in ((a, b, -1), (b, a, sgn)):
                    m = k + self.gm_deg[x]
                    for idx in range(self.dim(m)):
                        col = colmap.get((x, idx))
                        if col is None:
                            continue
                        for t, val in self.rb[m][idx].get(y, {}).items():
                            add((a, b, t), col, sg * val)
        return [r for r in eqs.values() if any(r)]

    # ------------------------------------------------------ module structure

    def left_bracket(self, a, m, idx):
        """[v_a, X] for X the idx-th basis vector of P[m]; lands in
        P[m + deg v_a].  Returns a coefficient dict."""
        img = self.rb[m][idx].get(a, {})
        if not img:
            return {}
        s = -(-1) ** (self.gm_par[a] * self.pars[m][idx])
        return {t: s * v for t, v in img.items()}

    def summary(self):
        return {m: self.dim(m) for m in sorted(self.basis)}
