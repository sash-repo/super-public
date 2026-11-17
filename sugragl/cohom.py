#!/usr/bin/env python3
"""
cohom.py -- H^2(g_-; P) for P = (g_-,g_0)_* the prolong built by models.py,
split by Z-degree and by the weight of the full diagonal torus, then assembled
into sl(2)^k-characters and decomposed into irreducibles.

Cochains (conventions in models.py):

    C^1 basis : (a; m,idx)             a an index of the g_- basis
    C^2 basis : (a,b; m,idx)           a < b, or a = b with v_a odd
    C^3 coords: (x,y,z; m',t)          x <= y <= z, repetitions at odd v's only

A cochain is stored by its values on the admissible non-decreasing tuples; the
value on an arbitrary tuple is recovered by super-antisymmetry.  The two
differentials are built column by column: for one basis cochain only O(dim g_-)
triples receive a contribution, so nothing of the size of C^3 is ever formed.
"""

from fractions import Fraction
from itertools import product

from models import Prolong, rank


class Complex:
    def __init__(self, P):
        self.P = P
        self.ngm = P.ngm
        self.gp = P.gm_par
        self.gd = P.gm_deg
        self.gw = P.gm_wt

    # ------------------------------------------------------------- symmetry

    def ok2(self, x, y):
        return x < y or (x == y and self.gp[x] == 1)

    def ok3(self, x, y, z):
        return (x <= y <= z
                and not (x == y and self.gp[x] == 0)
                and not (y == z and self.gp[y] == 0))

    def val1(self, vals, x):
        return vals.get(x, {})

    def val2(self, vals, x, y):
        """c(v_x,v_y) from the values stored on ordered pairs."""
        if x < y:
            return vals.get((x, y), {})
        if x == y:
            return vals.get((x, y), {}) if self.gp[x] else {}
        v = vals.get((y, x))
        if not v:
            return {}
        s = -(-1) ** (self.gp[x] * self.gp[y])
        return {k: s * c for k, c in v.items()}

    # -------------------------------------------------------------- d1, d2
    # values are dicts {(m, idx): coeff}; left_bracket needs (m, idx) split.

    def _lb(self, x, valdict, s):
        """s * [v_x, sum coeff * X_{m,idx}] as a dict {(m',t): coeff}."""
        out = {}
        for (m, idx), cf in valdict.items():
            if not cf:
                continue
            tgt = m + self.gd[x]
            for t, v in self.P.left_bracket(x, m, idx).items():
                key = (tgt, t)
                out[key] = out.get(key, 0) + s * cf * v
        return out

    def eval_d1(self, vals, pc, x, y):
        """(dc)(v_x,v_y) = (-1)^{p(c)p(x)}[x,c(y)]
                         - (-1)^{p(y)(p(c)+p(x))}[y,c(x)] - c([x,y])."""
        gp = self.gp
        out = {}

        def add(d):
            for k, v in d.items():
                out[k] = out.get(k, 0) + v

        add(self._lb(x, self.val1(vals, y), (-1) ** (pc * gp[x])))
        add(self._lb(y, self.val1(vals, x),
                     -(-1) ** (gp[y] * (pc + gp[x]))))
        for c, lam in self.P.f[x][y].items():
            for k, cf in self.val1(vals, c).items():
                out[k] = out.get(k, 0) - lam * cf
        return {k: v for k, v in out.items() if v}

    def eval_d2(self, vals, pc, x, y, z):
        """(dc)(v_x,v_y,v_z), the six terms of the Chevalley-Eilenberg
        formula."""
        gp = self.gp
        out = {}

        def add(d):
            for k, v in d.items():
                out[k] = out.get(k, 0) + v

        add(self._lb(x, self.val2(vals, y, z), (-1) ** (pc * gp[x])))
        add(self._lb(y, self.val2(vals, x, z),
                     -(-1) ** (gp[y] * (pc + gp[x]))))
        add(self._lb(z, self.val2(vals, x, y),
                     (-1) ** (gp[z] * (pc + gp[x] + gp[y]))))
        for (u, v, w, s) in ((x, y, z, -1),
                             (x, z, y, (-1) ** (gp[y] * gp[z])),
                             (y, z, x, -(-1) ** (gp[x] * (gp[y] + gp[z])))):
            for c, lam in self.P.f[u][v].items():
                for k, cf in self.val2(vals, c, w).items():
                    out[k] = out.get(k, 0) + s * lam * cf
        return {k: val for k, val in out.items() if val}

    # ------------------------------------------------------------- columns

    def d1_column(self, a0, m, idx):
        pc = (self.P.pars[m][idx] + self.gp[a0]) % 2
        vals = {a0: {(m, idx): 1}}
        cand = set()
        for w in range(self.ngm):
            for pr in ((w, a0), (a0, w)):
                if self.ok2(*pr):
                    cand.add(pr)
        for x in range(self.ngm):
            for y in range(self.ngm):
                if a0 in self.P.f[x][y] and self.ok2(x, y):
                    cand.add((x, y))
        out = {}
        for (x, y) in sorted(cand):
            v = self.eval_d1(vals, pc, x, y)
            if v:
                out[(x, y)] = v
        return out

    def d2_column(self, a0, b0, m, idx):
        pc = (self.P.pars[m][idx] + self.gp[a0] + self.gp[b0]) % 2
        vals = {(a0, b0): {(m, idx): 1}}
        cand = set()
        for w in range(self.ngm):
            t = tuple(sorted((w, a0, b0)))
            if self.ok3(*t):
                cand.add(t)
        partner = {a0: b0, b0: a0}
        for x in range(self.ngm):
            fx = self.P.f[x]
            for y in range(self.ngm):
                r = fx[y]
                if not r:
                    continue
                for q in (a0, b0):
                    if q in r:
                        t = tuple(sorted((x, y, partner[q])))
                        if self.ok3(*t):
                            cand.add(t)
        out = {}
        for tri in sorted(cand):
            v = self.eval_d2(vals, pc, *tri)
            if v:
                out[tri] = v
        return out


# ==========================================================================
#  the graded, weight-split computation
# ==========================================================================


def c2_items(P):
    """(a, b, m, idx, degree, weight) for every C^2 basis vector."""
    C = Complex(P)
    for a in range(P.ngm):
        for b in range(a, P.ngm):
            if a == b and P.gm_par[a] == 0:
                continue
            for m in sorted(P.basis):
                for idx in range(P.dim(m)):
                    deg = m - P.gm_deg[a] - P.gm_deg[b]
                    w = tuple(u - v - t for u, v, t in
                              zip(P.wts[m][idx], P.gm_wt[a], P.gm_wt[b]))
                    yield (a, b, m, idx, deg, w)


def c1_items(P):
    for a in range(P.ngm):
        for m in sorted(P.basis):
            for idx in range(P.dim(m)):
                deg = m - P.gm_deg[a]
                w = tuple(u - v for u, v in zip(P.wts[m][idx], P.gm_wt[a]))
                yield (a, m, idx, deg, w)


def h2(P, verbose=False, only_degrees=None):
    """Returns {degree: {weight: dim H^{2,deg}_weight}} and the C^2 dimensions."""
    C = Complex(P)
    c2 = {}
    for it in c2_items(P):
        a, b, m, idx, deg, w = it
        if only_degrees is not None and deg not in only_degrees:
            continue
        c2.setdefault((deg, w), []).append((a, b, m, idx))
    c1 = {}
    for it in c1_items(P):
        a, m, idx, deg, w = it
        if only_degrees is not None and deg not in only_degrees:
            continue
        c1.setdefault((deg, w), []).append((a, m, idx))

    H, c2dim = {}, {}
    for (deg, w), cols in sorted(c2.items()):
        c2dim[deg] = c2dim.get(deg, 0) + len(cols)
        colpos = {(a, b, m, idx): i for i, (a, b, m, idx) in enumerate(cols)}
        # --- ker d2 ---
        rows = {}
        for i, (a, b, m, idx) in enumerate(cols):
            for tri, val in C.d2_column(a, b, m, idx).items():
                for key, cf in val.items():
                    rows.setdefault((tri, key), {})[i] = cf
        mat = [[r.get(i, 0) for i in range(len(cols))] for r in rows.values()]
        ker = len(cols) - (rank(mat) if mat else 0)
        # --- im d1 ---
        img = []
        for (a, m, idx) in c1.get((deg, w), []):
            col = C.d1_column(a, m, idx)
            vec = [0] * len(cols)
            for (x, y), val in col.items():
                for (mm, t), cf in val.items():
                    j = colpos.get((x, y, mm, t))
                    if j is None:
                        raise RuntimeError(
                            f"d1 image left the block: {(x, y, mm, t)}")
                    vec[j] += cf
            if any(vec):
                img.append(vec)
        im = rank(img) if img else 0
        if ker - im:
            H.setdefault(deg, {})[w] = ker - im
        if ker - im < 0:
            raise RuntimeError("negative cohomology dimension")
    return H, c2dim


# ==========================================================================
#  characters
# ==========================================================================


def character(P, Hdeg):
    """{sl(2)^k-weight: (dim_even, dim_odd)} from {gl-weight: dim}."""
    M = P.M
    out = {}
    for w, d in Hdeg.items():
        key = M.sl2_weight(w)
        par = M.parity_of_weight(w)
        ev, od = out.get(key, (0, 0))
        out[key] = (ev + d, od) if par == 0 else (ev, od + d)
    return out


def decompose(char):
    """Peel a character of sl(2)^k into irreducibles.  `char` maps a weight
    tuple to a multiplicity (an int).  Returns a list of (weight, mult)."""
    ch = {w: m for w, m in char.items() if m}
    out = []
    while ch:
        hi = max(ch, key=lambda w: (sum(w), w))
        if any(x < 0 for x in hi):
            raise RuntimeError(f"not a character: highest weight {hi}")
        mult = ch[hi]
        out.append((hi, mult))
        for shift in product(*[range(h + 1) for h in hi]):
            w = tuple(h - 2 * s for h, s in zip(hi, shift))
            ch[w] = ch.get(w, 0) - mult
            if ch[w] == 0:
                del ch[w]
        if any(v < 0 for v in ch.values()):
            raise RuntimeError("negative multiplicity: not a character")
    return sorted(out, key=lambda t: (-sum(t[0]), t[0]))


def irreps(P, Hdeg):
    """[(weight, parity, multiplicity)] for one degree."""
    ch = character(P, Hdeg)
    res = []
    for par in (0, 1):
        sub = {w: d[par] for w, d in ch.items() if d[par]}
        if sub:
            res += [(w, par, m) for w, m in decompose(sub)]
    return sorted(res, key=lambda t: (-sum(t[0]), t[1], t[0]))


def fmt(res):
    """The Table's notation: (w1w2...) with a subscript 1 for odd, an exponent
    for a multiplicity > 1."""
    out = []
    for w, par, m in res:
        s = "(" + (",".join(map(str, w)) if max(w, default=0) > 9
                   else "".join(map(str, w))) + ")"
        if par:
            s += "_1"
        if m > 1:
            s += f"^{m}"
        out.append(s)
    return ", ".join(out) if out else "---"
