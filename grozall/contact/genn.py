"""Bilinear k(2n+1)-invariant operators T(k1,l1) x T(k2,l2) -> T(k3,l3), ANY n.

The naive ansatz of gen.py -- all products (D_1 F^{(m)})(D_2 G^{(n)}) with D a word in
the frame -- has, for n >= 2, thousands of terms, and the invariance under sp(2n) is
then imposed by brute force.  Here that step is done first and once: by the first
fundamental theorem for Sp(2n), every sp(2n)-invariant expression is a linear
combination of complete contractions by varpi, and the complete contractions are the
monomials in the ten operators

    Om   = varpi^{ab} d_{u_F^a} d_{u_G^b}          W = 0     (the fiber pairing)
    Xi_F = u_3^a d_{u_F^a},  Xi_G = u_3^a d_{u_G^a}  W = 0
    Th_F = varpi^{ab} e^F_a d_{u_G^b},  Th_G = varpi^{ab} d_{u_F^a} e^G_b     W = 1
    Dv_F = varpi^{ab} e^F_a d_{u_F^b},  Dv_G = varpi^{ab} e^G_a d_{u_G^b}     W = 1
    Ee_F = u_3^a e^F_a,  Ee_G = u_3^a e^G_a                                   W = 1
    Pi   = varpi^{ab} e^F_a e^G_b                                             W = 2

together with the Reeb derivatives R_F, R_G of W = 2.  For n = 1 these are not
independent (the second fundamental theorem), but they still SPAN, which is all that is
needed: the span is the space of k_{<=0}-invariant expressions, and the classification
is the same computation on a basis of a few units instead of a few thousands.  This is
what makes the census possible for n >= 2, and it is uniform in n: only the numerical
values of the k_1-conditions depend on the dimension.

A monomial in these operators is admissible for (k1,k2,k3,W) when it consumes exactly
k1 indices of u_F, k2 of u_G, produces exactly k3 of u_3, and has weighted order W.

Each monomial is expanded into "naive" terms

    coefficient * u_3^C * ( e_{w1} R^{s1} d_{u_F}^A F ) * ( e_{w2} R^{s2} d_{u_G}^B G ),

the word w = (a_1,...,a_r) meaning e_{a_1} ... e_{a_r} applied after R^s.  Since the
e_a do not commute ([e_a,e_b] = varpi_{ab} R), the value of one monomial depends on the
order in which its factors are written; the correction is always another monomial of the
same list (two e's collapse to an R), so the SPAN does not depend on the convention, and
the operators found are verified afterwards against the algebra itself.

Everything is exact rational arithmetic.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import product

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from contactn import (Contact, Module, padd, pscal, pmul, pzero, pconst,  # noqa: E402
                      pvar, fiber_monomials)

# --------------------------------------------------------------- the generators
# name -> (W, consumes u_F, consumes u_G, produces u_3)
GENS = {
    "Om":   (0, 1, 1, 0),
    "XiF":  (0, 1, 0, 1),
    "XiG":  (0, 0, 1, 1),
    "ThF":  (1, 0, 1, 0),
    "ThG":  (1, 1, 0, 0),
    "DvF":  (1, 1, 0, 0),
    "DvG":  (1, 0, 1, 0),
    "EeF":  (1, 0, 0, 1),
    "EeG":  (1, 0, 0, 1),
    "Pi":   (2, 0, 0, 0),
    "RF":   (2, 0, 0, 0),
    "RG":   (2, 0, 0, 0),
}
ORDER = ["Pi", "ThF", "ThG", "DvF", "DvG", "EeF", "EeG", "Om", "XiF", "XiG", "RF", "RG"]


def schemes(k1, k2, k3, W):
    """All admissible monomials, as dicts name -> exponent (in the order ORDER)."""
    out = []
    names = ORDER

    def rec(i, left, u1, u2, u3, acc):
        if i == len(names):
            if left == 0 and u1 == 0 and u2 == 0 and u3 == 0:
                out.append(dict(acc))
            return
        nm = names[i]
        w, c1, c2, c3 = GENS[nm]
        m = 0
        while True:
            if m * w > left or (c1 and m * c1 > u1) or (c2 and m * c2 > u2) \
                    or (c3 and m * c3 > u3):
                break
            acc[nm] = m
            rec(i + 1, left - m * w, u1 - m * c1, u2 - m * c2, u3 - m * c3, acc)
            del acc[nm]
            m += 1
            if w == 0 and c1 == 0 and c2 == 0 and c3 == 0:
                break
    rec(0, W, k1, k2, k3, {})
    return out


def scheme_latex(s):
    sym = {"Pi": r"\Pip", "ThF": r"\ThA_F", "ThG": r"\ThA_G", "DvF": r"\Dvr_F",
           "DvG": r"\Dvr_G", "EeF": r"\Ee_F", "EeG": r"\Ee_G", "Om": r"\Om",
           "XiF": r"\Xi_F", "XiG": r"\Xi_G", "RF": "R_F", "RG": "R_G"}
    parts = []
    for nm in ORDER:
        m = s.get(nm, 0)
        if m == 1:
            parts.append(sym[nm])
        elif m > 1:
            parts.append("%s^{%d}" % (sym[nm], m))
    return "".join(parts) or "1"


def scheme_name(s):
    parts = []
    for nm in ORDER:
        m = s.get(nm, 0)
        if m == 1:
            parts.append(nm)
        elif m > 1:
            parts.append("%s^%d" % (nm, m))
    return "*".join(parts) or "1"


# ------------------------------------------------------- expansion into terms
# A naive term is (w1, s1, A, w2, s2, B, C):  words w as tuples of frame indices,
# A,B,C exponent tuples of length 2n.

def _bump(tup, a, d=1):
    t = list(tup)
    t[a - 1] += d
    return tuple(t)


def expand(alg, s):
    """One scheme monomial -> dict {naive term: Fraction}."""
    dim = alg.dim
    z = (0,) * dim
    cur = {((), 0, z, (), 0, z, z): Fr(1)}
    wp = [(a, b, alg.varpi_up(a, b))
          for a in range(1, dim + 1) for b in range(1, dim + 1)
          if alg.varpi_up(a, b)]
    for nm in ORDER:
        for _ in range(s.get(nm, 0)):
            nxt = {}

            def put(key, val):
                if not val:
                    return
                v = nxt.get(key, Fr(0)) + val
                if v:
                    nxt[key] = v
                elif key in nxt:
                    del nxt[key]

            for (w1, s1, A, w2, s2, B, C), c in cur.items():
                if nm == "RF":
                    put((w1, s1 + 1, A, w2, s2, B, C), c)
                elif nm == "RG":
                    put((w1, s1, A, w2, s2 + 1, B, C), c)
                elif nm == "Pi":
                    for a, b, v in wp:
                        put(((a,) + w1, s1, A, (b,) + w2, s2, B, C), c * v)
                elif nm == "Om":
                    for a, b, v in wp:
                        put((w1, s1, _bump(A, a), w2, s2, _bump(B, b), C), c * v)
                elif nm == "ThF":
                    for a, b, v in wp:
                        put(((a,) + w1, s1, A, w2, s2, _bump(B, b), C), c * v)
                elif nm == "ThG":
                    for a, b, v in wp:
                        put((w1, s1, _bump(A, a), (b,) + w2, s2, B, C), c * v)
                elif nm == "DvF":
                    for a, b, v in wp:
                        put(((a,) + w1, s1, _bump(A, b), w2, s2, B, C), c * v)
                elif nm == "DvG":
                    for a, b, v in wp:
                        put((w1, s1, A, (a,) + w2, s2, _bump(B, b), C), c * v)
                elif nm == "XiF":
                    for a in range(1, dim + 1):
                        put((w1, s1, _bump(A, a), w2, s2, B, _bump(C, a)), c)
                elif nm == "XiG":
                    for a in range(1, dim + 1):
                        put((w1, s1, A, w2, s2, _bump(B, a), _bump(C, a)), c)
                elif nm == "EeF":
                    for a in range(1, dim + 1):
                        put(((a,) + w1, s1, A, w2, s2, B, _bump(C, a)), c)
                elif nm == "EeG":
                    for a in range(1, dim + 1):
                        put((w1, s1, A, (a,) + w2, s2, B, _bump(C, a)), c)
                else:
                    raise KeyError(nm)
            cur = nxt
            if not cur:
                return {}
    return cur


# ------------------------------------------------------------- evaluation
def _fact(t):
    out = 1
    for e in t:
        for i in range(2, e + 1):
            out *= i
    return out


def apply_word(alg, w, s, poly):
    out = poly
    for _ in range(s):
        out = alg.R(out)
        if not out:
            return {}
    for a in reversed(w):
        out = alg.e(a, out)
        if not out:
            return {}
    return out


def eval_terms(alg, terms, F, G):
    """terms: dict {naive term: coef}.  F, G in the Module representation."""
    Fc, Gc = {}, {}
    for (e, mu), v in F.items():
        Fc.setdefault(mu, {})[e] = Fc.setdefault(mu, {}).get(e, Fr(0)) + v
    for (e, mu), v in G.items():
        Gc.setdefault(mu, {})[e] = Gc.setdefault(mu, {}).get(e, Fr(0)) + v
    out = {}
    cache1, cache2 = {}, {}
    for (w1, s1, A, w2, s2, B, C), c in terms.items():
        fa = Fc.get(A)
        if not fa:
            continue
        gb = Gc.get(B)
        if not gb:
            continue
        k1 = (w1, s1, A)
        if k1 not in cache1:
            cache1[k1] = apply_word(alg, w1, s1, fa)
        a = cache1[k1]
        if not a:
            continue
        k2 = (w2, s2, B)
        if k2 not in cache2:
            cache2[k2] = apply_word(alg, w2, s2, gb)
        b = cache2[k2]
        if not b:
            continue
        coef = c * _fact(A) * _fact(B)
        for ke, ve in pmul(a, b).items():
            key = (ke, C)
            nv = out.get(key, Fr(0)) + coef * ve
            if nv:
                out[key] = nv
            elif key in out:
                del out[key]
    return out


# --------------------------------------------------------------- the solver
def target_weight(k1, l1, k2, l2, k3, W):
    return Fr(l1) + Fr(l2) + Fr(W + k1 + k2 - k3, 2)


def rand_field(alg, k, maxdeg, seed, nmon=14):
    import random
    rnd = random.Random(seed)
    fm = fiber_monomials(alg.dim, k)
    out = {}
    for _ in range(nmon):
        e = [0] * alg.nv
        for _ in range(rnd.randint(0, maxdeg)):
            e[rnd.randrange(alg.nv)] += 1
        mu = fm[rnd.randrange(len(fm))]
        key = (tuple(e), mu)
        out[key] = out.get(key, Fr(0)) + Fr(rnd.randint(1, 6))
    return {k_: v for k_, v in out.items() if v}


class Problem:
    def __init__(self, n, k1, k2, k3, W):
        self.alg = Contact(n)
        self.k1, self.k2, self.k3, self.W = k1, k2, k3, W
        self.schemes = schemes(k1, k2, k3, W)
        self.expanded = [expand(self.alg, s) for s in self.schemes]
        keep = [i for i, e in enumerate(self.expanded) if e]
        self.schemes = [self.schemes[i] for i in keep]
        self.expanded = [self.expanded[i] for i in keep]

    def dim_ansatz(self):
        return len(self.schemes)

    def _rows(self, f, F, G, M1, M2, M3):
        alg = self.alg
        LfF, LfG = M1.act(f, F), M2.act(f, G)
        cols, keys = [], set()
        for terms in self.expanded:
            d = M3.act(f, eval_terms(alg, terms, F, G))
            for key, v in eval_terms(alg, terms, LfF, G).items():
                d[key] = d.get(key, Fr(0)) - v
            for key, v in eval_terms(alg, terms, F, LfG).items():
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
            for k, v in d.items():
                mat[idx[k]][j] = v
        return mat

    def solve(self, l1, l2, deg=None, nsolve=2, verify=True, quiet=True,
              vpieces=(0, 1, 2, 3)):
        alg = self.alg
        l3 = target_weight(self.k1, l1, self.k2, l2, self.k3, self.W)
        M1 = Module(alg, self.k1, l1)
        M2 = Module(alg, self.k2, l2)
        M3 = Module(alg, self.k3, l3)
        if deg is None:
            deg = self.W + 1
        gens = alg.k_piece(0) + alg.k_piece(1)
        Fs = [rand_field(alg, self.k1, deg, 17 + i) for i in range(nsolve)]
        Gs = [rand_field(alg, self.k2, deg, 93 + i) for i in range(nsolve)]
        mat = []
        for f in gens:
            for F in Fs:
                for G in Gs:
                    mat.extend(self._rows(f, F, G, M1, M2, M3))
        if not mat:
            ns = [[Fr(1) if i == j else Fr(0) for i in range(len(self.schemes))]
                  for j in range(len(self.schemes))]
        else:
            ns = nullspace(mat)
        if verify and ns:
            ns = self.verify(ns, l1, l2, deg + 2, quiet=quiet, pieces=vpieces)
        return l3, ns

    def verify(self, ns, l1, l2, deg, ngen=2, quiet=True, pieces=(0, 1, 2, 3)):
        """Independent check on denser fields and on k_2, k_3 as well."""
        alg = self.alg
        l3 = target_weight(self.k1, l1, self.k2, l2, self.k3, self.W)
        M1, M2 = Module(alg, self.k1, l1), Module(alg, self.k2, l2)
        M3 = Module(alg, self.k3, l3)
        gens = []
        for i in pieces:
            gens += alg.k_piece(i)
        Fs = [rand_field(alg, self.k1, deg, 5000 + i) for i in range(ngen)]
        Gs = [rand_field(alg, self.k2, deg, 6000 + i) for i in range(ngen)]
        for _ in range(4):
            bad = []
            for f in gens:
                for F in Fs:
                    for G in Gs:
                        LfF, LfG = M1.act(f, F), M2.act(f, G)
                        for vec in ns:
                            acc = {}
                            for cf, terms in zip(vec, self.expanded):
                                if not cf:
                                    continue
                                d = M3.act(f, eval_terms(alg, terms, F, G))
                                for k, v in eval_terms(alg, terms, LfF, G).items():
                                    d[k] = d.get(k, Fr(0)) - v
                                for k, v in eval_terms(alg, terms, F, LfG).items():
                                    d[k] = d.get(k, Fr(0)) - v
                                for k, v in d.items():
                                    if v:
                                        acc[k] = acc.get(k, Fr(0)) + cf * v
                            if any(acc.values()):
                                bad.append((f, F, G))
                                break
            if not bad:
                return ns
            if not quiet:
                print("      verification added %d conditions" % len(bad))
            mat = []
            for f, F, G in bad:
                mat.extend(self._rows(f, F, G, M1, M2, M3))
            # re-solve on the whole ansatz with the extra conditions
            old = []
            for f in alg.k_piece(0) + alg.k_piece(1):
                for F in Fs:
                    for G in Gs:
                        old.extend(self._rows(f, F, G, M1, M2, M3))
            ns = nullspace(mat + old)
            if not ns:
                return ns
        raise RuntimeError("verification did not converge")

    def fingerprints(self, ns, deg=2):
        """Each solution vector as a VECTOR OF VALUES on a spanning set of arguments.

        The schemes are not independent for small n, so the dimension of the space of
        operators is the rank of these fingerprints, not the number of solutions.
        """
        alg = self.alg
        tests1 = [{(e, mu): Fr(1)} for e in _gen_exps(alg.nv, deg)
                  for mu in fiber_monomials(alg.dim, self.k1)]
        tests2 = [{(e, mu): Fr(1)} for e in _gen_exps(alg.nv, deg)
                  for mu in fiber_monomials(alg.dim, self.k2)]
        vals, keys = [], set()
        for vec in ns:
            d = {}
            for i, F in enumerate(tests1):
                for j, G in enumerate(tests2):
                    acc = {}
                    for cf, terms in zip(vec, self.expanded):
                        if not cf:
                            continue
                        for k, v in eval_terms(alg, terms, F, G).items():
                            acc[k] = acc.get(k, Fr(0)) + cf * v
                    for k, v in acc.items():
                        if v:
                            d[(i, j) + k] = v
            vals.append(d)
            keys |= set(d)
        keys = sorted(keys)
        return [[d.get(k, Fr(0)) for k in keys] for d in vals]

    def op_dim(self, ns, deg=2):
        fp = self.fingerprints(ns, deg)
        fp = [r for r in fp if any(r)]
        if not fp:
            return 0
        return rank(fp)

def _small_exps(alg, maxdeg):
    out = []
    for e in _gen_exps(alg.nv, maxdeg):
        out.append(e)
    return out


def _gen_exps(nv, maxdeg):
    if nv == 0:
        yield ()
        return
    for d in range(maxdeg + 1):
        for rest in _gen_exps(nv - 1, maxdeg - d):
            yield (d,) + rest


def operator_dimension(n, k1, l1, k2, l2, k3, W, fpdeg=2, **kw):
    """The dimension of the space of invariant operators, as OPERATORS."""
    P = Problem(n, k1, k2, k3, W)
    if not P.schemes:
        return 0, P
    l3, ns = P.solve(l1, l2, **kw)
    return P.op_dim(ns, deg=fpdeg), P
