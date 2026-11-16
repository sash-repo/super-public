"""Linear k(2n+1)-invariant operators T(k,lam) -> T(k',lam'), any n.

The same scheme calculus as genn.py with one argument: every k_{<=0}-invariant
expression is a monomial in

    Xi = u_3^a d_{u^a}          W = 0     (the fiber contraction)
    Ee = u_3^a e_a              W = 1
    Dv = varpi^{ab} e_a d_{u^b} W = 1     (the divergence along the distribution)
    R                           W = 2

with Xi^x Ee^y Dv^z R^s admissible for (k,k',W) when x+z = k, x+y = k', y+z+2s = W;
and the weight is forced:  lam' = lam + (W + k - k')/2.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from contactn import Contact, Module, pmul, fiber_monomials  # noqa: E402
from genn import _bump, _fact, apply_word, _gen_exps, rand_field  # noqa: E402


def uschemes(k, kp, W):
    out = []
    for x in range(min(k, kp) + 1):
        z, y = k - x, kp - x
        left = W - y - z
        if left < 0 or left % 2:
            continue
        out.append((x, y, z, left // 2))
    return out


def uscheme_latex(s):
    x, y, z, sp = s
    parts = []
    for sym, m in ((r"\Xi", x), (r"\Ee", y), (r"\Dvr", z), ("R", sp)):
        if m == 1:
            parts.append(sym)
        elif m > 1:
            parts.append("%s^{%d}" % (sym, m))
    return "".join(parts) or "1"


def uexpand(alg, s):
    """One monomial -> dict {(w, s, A, C): coefficient}."""
    x, y, z, sp = s
    dim = alg.dim
    zt = (0,) * dim
    cur = {((), 0, zt, zt): Fr(1)}
    wp = [(a, b, alg.varpi_up(a, b))
          for a in range(1, dim + 1) for b in range(1, dim + 1) if alg.varpi_up(a, b)]

    def step(cur, kind):
        nxt = {}

        def put(key, val):
            v = nxt.get(key, Fr(0)) + val
            if v:
                nxt[key] = v
            elif key in nxt:
                del nxt[key]
        for (w, sr, A, C), c in cur.items():
            if kind == "Xi":
                for a in range(1, dim + 1):
                    put((w, sr, _bump(A, a), _bump(C, a)), c)
            elif kind == "Ee":
                for a in range(1, dim + 1):
                    put(((a,) + w, sr, A, _bump(C, a)), c)
            elif kind == "Dv":
                for a, b, v in wp:
                    put(((a,) + w, sr, _bump(A, b), C), c * v)
            elif kind == "R":
                put((w, sr + 1, A, C), c)
        return nxt

    for kind, m in (("Ee", y), ("Dv", z), ("Xi", x), ("R", sp)):
        for _ in range(m):
            cur = step(cur, kind)
            if not cur:
                return {}
    return cur


def ueval(alg, terms, F):
    Fc = {}
    for (e, mu), v in F.items():
        Fc.setdefault(mu, {})[e] = Fc.setdefault(mu, {}).get(e, Fr(0)) + v
    out = {}
    cache = {}
    for (w, sr, A, C), c in terms.items():
        fa = Fc.get(A)
        if not fa:
            continue
        key = (w, sr, A)
        if key not in cache:
            cache[key] = apply_word(alg, w, sr, fa)
        a = cache[key]
        if not a:
            continue
        coef = c * _fact(A)
        for ke, ve in a.items():
            kk = (ke, C)
            nv = out.get(kk, Fr(0)) + coef * ve
            if nv:
                out[kk] = nv
            elif kk in out:
                del out[kk]
    return out


class ULin:
    def __init__(self, n, k, kp, W):
        self.alg = Contact(n)
        self.k, self.kp, self.W = k, kp, W
        ss = uschemes(k, kp, W)
        ex = [uexpand(self.alg, s) for s in ss]
        keep = [i for i, e in enumerate(ex) if e]
        self.schemes = [ss[i] for i in keep]
        self.expanded = [ex[i] for i in keep]

    def target_weight(self, lam):
        return Fr(lam) + Fr(self.W + self.k - self.kp, 2)

    def _rows(self, f, F, M1, M2):
        cols, keys = [], set()
        for terms in self.expanded:
            d = M2.act(f, ueval(self.alg, terms, F))
            for key, v in ueval(self.alg, terms, M1.act(f, F)).items():
                d[key] = d.get(key, Fr(0)) - v
            d = {a: b for a, b in d.items() if b}
            cols.append(d)
            keys |= set(d)
        if not keys:
            return []
        keys = sorted(keys)
        idx = {k: i for i, k in enumerate(keys)}
        mat = [[Fr(0)] * len(cols) for _ in keys]
        for j, d in enumerate(cols):
            for kk, v in d.items():
                mat[idx[kk]][j] = v
        return mat

    def solve(self, lam, pieces=(0, 1, 2), nsolve=3, deg=None):
        alg = self.alg
        if not self.schemes:
            return 0
        lp = self.target_weight(lam)
        M1, M2 = Module(alg, self.k, lam), Module(alg, self.kp, lp)
        if deg is None:
            deg = self.W + 2
        gens = []
        for i in pieces:
            gens += alg.k_piece(i)
        Fs = [rand_field(alg, self.k, deg, 41 + i) for i in range(nsolve)]
        mat = []
        for f in gens:
            for F in Fs:
                mat.extend(self._rows(f, F, M1, M2))
        ns = nullspace(mat) if mat else [
            [Fr(1) if i == j else Fr(0) for i in range(len(self.schemes))]
            for j in range(len(self.schemes))]
        if not ns:
            return 0
        return self._opdim(ns)

    def _opdim(self, ns, deg=2):
        alg = self.alg
        tests = [{(e, mu): Fr(1)} for e in _gen_exps(alg.nv, deg)
                 for mu in fiber_monomials(alg.dim, self.k)]
        vals, keys = [], set()
        for vec in ns:
            d = {}
            for i, F in enumerate(tests):
                acc = {}
                for cf, terms in zip(vec, self.expanded):
                    if not cf:
                        continue
                    for kk, v in ueval(alg, terms, F).items():
                        acc[kk] = acc.get(kk, Fr(0)) + cf * v
                for kk, v in acc.items():
                    if v:
                        d[(i,) + kk] = v
            vals.append(d)
            keys |= set(d)
        keys = sorted(keys)
        rows = [[d.get(kk, Fr(0)) for kk in keys] for d in vals]
        rows = [r for r in rows if any(r)]
        return rank(rows) if rows else 0


def run(n, KMAX=2, WMAX=3, say=print, weights=None):
    if weights is None:
        weights = [Fr(a, 2) for a in range(-4, 7)]
    for k in range(KMAX + 1):
        for kp in range(KMAX + 3):
            for W in range(WMAX + 1):
                if not uschemes(k, kp, W):
                    continue
                hits = []
                for lam in weights:
                    U = ULin(n, k, kp, W)
                    d = U.solve(lam)
                    if d:
                        hits.append((str(lam), d))
                if hits:
                    say("  T(%d,l) -> T(%d,l+%s)  W=%d : %s"
                        % (k, kp, Fr(W + k - kp, 2), W, hits))
