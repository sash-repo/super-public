#!/usr/bin/env python3
"""Exact Chevalley-Eilenberg cohomology of pe(n), spe(n) (trivial coefficients).

Why: in the rigidity proof for sle'(n) the Hochschild-Serre term E_2^{p,0} is
H^p(spe(n)), and n=4 is the only value for which H^2(spe(n)) != 0 (Sergeev's
central extension as).  This engine computes H^p(spe(n)) and H^p(pe(n)) from
scratch, over Q, with no library and no SuperLie, and prints the exceptional
cocycle explicitly.

Method.  The Chevalley-Eilenberg complex of a Lie superalgebra g is the free
supercommutative algebra C = C[x^a] on generators dual to a basis {e_a} of g,
with the *shifted* parity  p(x^a) = p(e_a) + 1,  and the odd derivation

    Q x^c = -1/2 sum_{a,b} (-1)^{p_a p'_b} f^c_{ab} x^a x^b,   p' = shifted p,

whose square is zero precisely by the Jacobi identity (checked here).  The
Cartan subalgebra h of g_ev acts on C by L_h = Q i_h + i_h Q, so every non-zero
weight subcomplex is acyclic and H^.(g) = H^.(C_0), C_0 the weight-0 part.
That is what makes the computation small enough to be exact.

Usage:  python3 spe.py [nmax] [degmax]
"""
import sys
from fractions import Fraction
from itertools import combinations


# --------------------------------------------------------------- the algebras
# An element of gl(n|n) is a 2n x 2n matrix over Q, stored as a dict
# {(i,j): coeff}.  pe(n) = {[[A,B],[C,-A^t]] : B = B^t, C = -C^t}; the even part
# is the A-block, the odd part is B (in S^2 V) and C (in Lambda^2 V^*).

def mat(entries):
    return {k: Fraction(v) for k, v in entries.items() if v}


def matmul(X, Y, N):
    Z = {}
    rows = {}
    for (i, j), v in X.items():
        rows.setdefault(j, []).append((i, v))
    for (j, k), w in Y.items():
        for (i, v) in rows.get(j, ()):
            Z[(i, k)] = Z.get((i, k), 0) + v * w
    return {k: v for k, v in Z.items() if v}


def superbracket(X, pX, Y, pY, N):
    A = matmul(X, Y, N)
    B = matmul(Y, X, N)
    sgn = -1 if (pX * pY) % 2 == 0 else 1     # [X,Y] = XY - (-1)^{pX pY} YX
    Z = dict(A)
    for k, v in B.items():
        Z[k] = Z.get(k, 0) + sgn * v
    return {k: v for k, v in Z.items() if v}


def basis_pe(n, special=True):
    """Basis of spe(n) (special=True) or pe(n).  Returns list of dicts with
    keys name, parity, weight (in Z^n), mat."""
    B = []
    # even part: sl(n) (or gl(n)), embedded as diag(A, -A^t)
    def even(A, name, weight):
        M = {}
        for (i, j), v in A.items():
            M[(i, j)] = Fraction(v)
            M[(n + j, n + i)] = -Fraction(v)
        B.append(dict(name=name, parity=0, weight=weight, mat=M))
    for i in range(n):
        for j in range(n):
            if i != j:
                w = [0] * n
                w[i] += 1
                w[j] -= 1
                even({(i, j): 1}, "E%d%d" % (i + 1, j + 1), tuple(w))
    for i in range(n - 1):
        even({(i, i): 1, (i + 1, i + 1): -1}, "H%d" % (i + 1), tuple([0] * n))
    if not special:
        even({(0, 0): 1}, "Z", tuple([0] * n))      # completes sl(n) to gl(n)
    # odd part: B = B^t in the upper right block
    for i in range(n):
        for j in range(i, n):
            M = {(i, n + j): Fraction(1)}
            if i != j:
                M[(j, n + i)] = Fraction(1)
            w = [0] * n
            w[i] += 1
            w[j] += 1
            B.append(dict(name="b%d%d" % (i + 1, j + 1), parity=1,
                          weight=tuple(w), mat=M))
    # odd part: C = -C^t in the lower left block
    for i in range(n):
        for j in range(i + 1, n):
            M = {(n + i, j): Fraction(1), (n + j, i): Fraction(-1)}
            w = [0] * n
            w[i] -= 1
            w[j] -= 1
            B.append(dict(name="c%d%d" % (i + 1, j + 1), parity=1,
                          weight=tuple(w), mat=M))
    return B


def decompose(M, B, N):
    """Coordinates of a matrix M in the basis B (linear solve on the support)."""
    if not M:
        return {}
    # build once per call: the pivot table is small, so a direct solve is fine
    keys = sorted({k for e in B for k in e["mat"]} | set(M))
    idx = {k: t for t, k in enumerate(keys)}
    rows = []
    for e in B:
        v = [Fraction(0)] * len(keys)
        for k, c in e["mat"].items():
            v[idx[k]] = c
        rows.append(v)
    tgt = [Fraction(0)] * len(keys)
    for k, c in M.items():
        tgt[idx[k]] = c
    # solve  sum_a x_a rows[a] = tgt   by Gaussian elimination on the transpose
    m = len(keys)
    aug = [[rows[a][r] for a in range(len(B))] + [tgt[r]] for r in range(m)]
    piv = []
    r = 0
    for c in range(len(B)):
        p = next((rr for rr in range(r, m) if aug[rr][c]), None)
        if p is None:
            continue
        aug[r], aug[p] = aug[p], aug[r]
        pv = aug[r][c]
        aug[r] = [x / pv for x in aug[r]]
        for rr in range(m):
            if rr != r and aug[rr][c]:
                f = aug[rr][c]
                aug[rr] = [x - f * y for x, y in zip(aug[rr], aug[r])]
        piv.append(c)
        r += 1
        if r == m:
            break
    for rr in range(r, m):
        assert not aug[rr][-1], "matrix not in the span of the basis"
    out = {}
    for k, c in enumerate(piv):
        if aug[k][-1]:
            out[c] = aug[k][-1]
    return out


def structure_constants(B, n):
    """f[(a,b)] = {c: coeff} with [e_a, e_b] = sum_c coeff e_c.

    Basis vectors flagged "drop" (a central element we quotient by, as in
    psl(2|2) = sl(2|2)/C.1) are used to decompose brackets but are then removed
    both from the coefficients and from the index set."""
    N = max(max(i, j) for e in B for (i, j) in e["mat"]) + 1
    f = {}
    for a in range(len(B)):
        for b in range(len(B)):
            Z = superbracket(B[a]["mat"], B[a]["parity"],
                             B[b]["mat"], B[b]["parity"], N)
            if Z:
                f[(a, b)] = decompose(Z, B, N)
    keep = [a for a, e in enumerate(B) if not e.get("drop")]
    if len(keep) == len(B):
        return B, f
    ren = {a: t for t, a in enumerate(keep)}
    f2 = {}
    for (a, b), co in f.items():
        if a not in ren or b not in ren:
            continue
        c2 = {ren[c]: v for c, v in co.items() if c in ren}
        if c2:
            f2[(ren[a], ren[b])] = c2
    return [B[a] for a in keep], f2


# ------------------------------------------------------- the CE algebra C[x^a]
# A monomial is (E, O): E = tuple of exponents of the *commuting* generators
# (those dual to odd basis vectors), O = sorted tuple of indices of the
# *anticommuting* generators (dual to even basis vectors).  Coefficients are
# Fractions; a polynomial is a dict {monomial: coeff}.

class CE:
    def __init__(self, B, f):
        self.B = B
        self.f = f
        self.ncom = [a for a, e in enumerate(B) if e["parity"] == 1]  # x even
        self.nanti = [a for a, e in enumerate(B) if e["parity"] == 0]  # x odd
        self.com_pos = {a: t for t, a in enumerate(self.ncom)}
        self.anti_pos = {a: t for t, a in enumerate(self.nanti)}
        self.Q1 = {}                     # Q on generators
        for c in range(len(B)):
            self.Q1[c] = self._Qgen(c)

    # ---- monomial arithmetic
    def gen_mono(self, a):
        if self.B[a]["parity"] == 1:
            E = [0] * len(self.ncom)
            E[self.com_pos[a]] = 1
            return (tuple(E), ())
        return (tuple([0] * len(self.ncom)), (self.anti_pos[a],))

    def mono_mul(self, m1, m2):
        E1, O1 = m1
        E2, O2 = m2
        if set(O1) & set(O2):
            return None
        E = tuple(x + y for x, y in zip(E1, E2))
        merged = list(O1) + list(O2)
        # sign of sorting
        sgn = 1
        arr = merged[:]
        for i in range(1, len(arr)):
            j = i
            while j > 0 and arr[j - 1] > arr[j]:
                arr[j - 1], arr[j] = arr[j], arr[j - 1]
                sgn = -sgn
                j -= 1
        return ((E, tuple(arr)), sgn)

    def poly_mul(self, p1, p2):
        out = {}
        for m1, c1 in p1.items():
            for m2, c2 in p2.items():
                r = self.mono_mul(m1, m2)
                if r is None:
                    continue
                m, s = r
                out[m] = out.get(m, 0) + s * c1 * c2
        return {m: c for m, c in out.items() if c}

    def _Qgen(self, c):
        """Q x^c = -1/2 sum_{a,b} (-1)^{p_a (p_b+1)} f^c_{ab} x^a x^b."""
        out = {}
        for (a, b), coeffs in self.f.items():
            if c not in coeffs:
                continue
            pa = self.B[a]["parity"]
            pb1 = (self.B[b]["parity"] + 1) % 2
            sgn = -1 if (pa * pb1) % 2 else 1
            r = self.mono_mul(self.gen_mono(a), self.gen_mono(b))
            if r is None:
                continue
            m, s = r
            out[m] = out.get(m, 0) + Fraction(-1, 2) * sgn * s * coeffs[c]
        return {m: v for m, v in out.items() if v}

    def Q(self, poly):
        out = {}
        for (E, O), coeff in poly.items():
            # commuting factors first (all of even CE-parity: no sign)
            for t, e in enumerate(E):
                if not e:
                    continue
                a = self.ncom[t]
                E2 = list(E)
                E2[t] -= 1
                rest = ((tuple(E2), O))
                for m, v in self.poly_mul(self.Q1[a], {rest: Fraction(1)}).items():
                    out[m] = out.get(m, 0) + coeff * e * v
            # anticommuting factors, in their sorted order
            for i, o in enumerate(O):
                a = self.nanti[o]
                left = (E, O[:i])
                right = (tuple([0] * len(E)), O[i + 1:])
                sgn = (-1) ** i
                pr = self.poly_mul({left: Fraction(1)}, self.Q1[a])
                pr = self.poly_mul(pr, {right: Fraction(1)})
                for m, v in pr.items():
                    out[m] = out.get(m, 0) + coeff * sgn * v
        return {m: c for m, c in out.items() if c}

    # ---- weights and parities
    def mono_weight(self, m, n):
        E, O = m
        w = [0] * n
        for t, e in enumerate(E):
            if e:
                ww = self.B[self.ncom[t]]["weight"]
                for i in range(n):
                    w[i] -= e * ww[i]        # x^a has weight -wt(e_a)
        for o in O:
            ww = self.B[self.nanti[o]]["weight"]
            for i in range(n):
                w[i] -= ww[i]
        return tuple(w)

    def mono_degree(self, m):
        E, O = m
        return sum(E) + len(O)

    def mono_form_parity(self, m):
        """Parity of the cochain as a multilinear form (paper's convention)."""
        E, O = m
        return sum(E) % 2                     # the x's dual to odd e's

    def monomials(self, deg, n, zero_weight_only=True, full_torus=False):
        """All monomials of the given total degree with weight 0 (mod the
        relation sum e_i = 0 imposed by sl(n) <-> gl(n))."""
        res = []
        ncom, nanti = len(self.ncom), len(self.nanti)

        def rec(t, left, E, O):
            if t == ncom:
                for k in range(left + 1):
                    pass
                for O2 in combinations(range(nanti), left):
                    res.append((tuple(E), O2))
                return
            for e in range(left + 1):
                rec(t + 1, left - e, E + [e], O)

        rec(0, deg, [], ())
        if not zero_weight_only:
            return res
        out = []
        for m in res:
            w = self.mono_weight(m, n)
            s = sum(w)
            if full_torus:                        # gl(n)-Cartan: weight = 0
                ok = not any(w)
            else:                                 # sl(n)-Cartan: w ~ (1,...,1)
                ok = all(n * x == s for x in w)
            if ok:
                out.append(m)
        return out


# ------------------------------------------------------------------ linear alg
def rank_and_kernel(rows, ncols):
    """rows: list of dicts {col: coeff}.  Returns (rank, kernel basis)."""
    M = [dict(r) for r in rows if r]
    piv = {}
    order = []
    for r in M:
        r = dict(r)
        while r:
            c = min(r)
            if c in piv:
                f = r[c] / piv[c][c]
                for k, v in piv[c].items():
                    r[k] = r.get(k, 0) - f * v
                    if not r[k]:
                        del r[k]
            else:
                piv[c] = r
                order.append(c)
                break
    return len(piv), piv


def kernel_dim(rows, ncols):
    """Dimension of the kernel of the map given by columns = coordinates."""
    # rows are indexed by source basis vectors: row i = image of basis vector i
    r, _ = rank_and_kernel(rows, ncols)
    return len(rows) - r, r


def kernel_basis(rows, ncols):
    """Basis of {v : sum_i v_i rows[i] = 0} by Gaussian elimination."""
    m = len(rows)
    aug = []
    for i, r in enumerate(rows):
        d = dict(r)
        d[("id", i)] = Fraction(1)
        aug.append(d)
    piv = {}
    for d in aug:
        d = dict(d)
        while True:
            cols = [c for c in d if not (isinstance(c, tuple) and c[0] == "id")]
            if not cols:
                break
            c = min(cols)
            if c in piv:
                f = d[c] / piv[c][c]
                for k, v in piv[c].items():
                    d[k] = d.get(k, 0) - f * v
                    if not d[k]:
                        del d[k]
            else:
                piv[c] = d
                break
        else:
            pass
        cols = [c for c in d if not (isinstance(c, tuple) and c[0] == "id")]
        if not cols:
            yield {c[1]: v for c, v in d.items() if isinstance(c, tuple)}


# ---------------------------------------------------------------------- driver
def cohomology_of(B, name, n, degmax, show=(), full_torus=False):
    B, f = structure_constants(B, n)
    dim_ev = sum(1 for e in B if e["parity"] == 0)
    dim_od = sum(1 for e in B if e["parity"] == 1)
    ce = CE(B, f)
    # Jacobi <=> Q^2 = 0 on generators
    for c in range(len(B)):
        assert not ce.Q(ce.Q1[c]), "Q^2 != 0 on generator %d (Jacobi fails)" % c
    print("%-8s sdim = (%d|%d)   Q^2 = 0 on all generators: OK"
          % (name, dim_ev, dim_od))

    mons = {}
    for d in range(degmax + 2):
        mons[d] = ce.monomials(d, n, full_torus=full_torus)
    result = {}
    ranks = {}
    for d in range(degmax + 1):
        src = mons[d]
        tgt = {m: i for i, m in enumerate(mons[d + 1])}
        rows = []
        for m in src:
            q = ce.Q({m: Fraction(1)})
            rows.append({tgt[k]: v for k, v in q.items() if k in tgt})
            assert all(k in tgt for k in q), "target monomial missing"
        r, _ = rank_and_kernel(rows, len(tgt))
        ranks[d] = r
    for d in range(degmax + 1):
        dimC = len(mons[d])
        h = dimC - ranks[d] - (ranks[d - 1] if d >= 1 else 0)
        result[d] = h
    print("   weight-0 cochains:", {d: len(mons[d]) for d in range(degmax + 1)})
    print("   rank d_k        :", {d: ranks[d] for d in range(degmax + 1)})
    print("   H^k(%s)      : %s" % (name, {d: result[d] for d in range(degmax + 1)}))

    # parities and explicit cocycles in the requested degrees
    for d in show:
        if d > degmax:
            continue
        src = mons[d]
        tgt = {m: i for i, m in enumerate(mons[d + 1])}
        rows = []
        for m in src:
            q = ce.Q({m: Fraction(1)})
            rows.append({tgt[k]: v for k, v in q.items()})
        ker = list(kernel_basis(rows, len(tgt)))
        # image of d_{k-1}
        img = []
        if d >= 1:
            tgt2 = {m: i for i, m in enumerate(src)}
            for m in mons[d - 1]:
                q = ce.Q({m: Fraction(1)})
                img.append({tgt2[k]: v for k, v in q.items()})
        # reduce kernel modulo image
        piv = {}
        for r in img:
            r = dict(r)
            while r:
                c = min(r)
                if c in piv:
                    fc = r[c] / piv[c][c]
                    for k, v in piv[c].items():
                        r[k] = r.get(k, 0) - fc * v
                        if not r[k]:
                            del r[k]
                else:
                    piv[c] = r
                    break
        reps = []
        for v in ker:
            r = dict(v)
            while r:
                c = min(r)
                if c in piv:
                    fc = r[c] / piv[c][c]
                    for k, val in piv[c].items():
                        r[k] = r.get(k, 0) - fc * val
                        if not r[k]:
                            del r[k]
                else:
                    piv[c] = r
                    reps.append(dict(v))
                    break
        for v in reps:
            terms = []
            par = set()
            for i, c in sorted(v.items()):
                m = src[i]
                terms.append("%s*%s" % (c, mono_name(ce, m)))
                par.add((ce.mono_form_parity(m) + d) % 2)   # parity in H^.(g)
            print("   H^%d(%s) cocycle [%s]: %s"
                  % (d, name, "Latin/even" if par == {0} else
                     ("Greek/odd" if par == {1} else "mixed"), " + ".join(terms)))
    return result


def mono_name(ce, m):
    E, O = m
    parts = []
    for t, e in enumerate(E):
        if e:
            nm = ce.B[ce.ncom[t]]["name"] + "*"
            parts.append(nm if e == 1 else "%s^%d" % (nm, e))
    for o in O:
        parts.append(ce.B[ce.nanti[o]]["name"] + "*")
    return "".join(parts) if parts else "1"


def cohomology(n, special, degmax, show=()):
    return cohomology_of(basis_pe(n, special),
                         ("spe" if special else "pe") + "(%d)" % n,
                         n, degmax, show=show, full_torus=not special)


# ------------------------------------------------------------------- test beds
def basis_sl(n):
    """sl(n) as a purely even Lie algebra (a control case)."""
    B = []
    for i in range(n):
        for j in range(n):
            if i != j:
                w = [0] * n
                w[i] += 1
                w[j] -= 1
                B.append(dict(name="E%d%d" % (i + 1, j + 1), parity=0,
                              weight=tuple(w), mat={(i, j): Fraction(1)}))
    for i in range(n - 1):
        B.append(dict(name="H%d" % (i + 1), parity=0, weight=tuple([0] * n),
                      mat={(i, i): Fraction(1), (i + 1, i + 1): Fraction(-1)}))
    return B


def basis_sl11():
    """sl(1|1): even center z = 1_{1|1}, odd x, y with [x,y] = z.

    The only torus inside sl(1|1) is central, so no weight restriction may be
    imposed: all weights are recorded as 0."""
    B = [dict(name="z", parity=0, weight=(0, 0),
              mat={(0, 0): Fraction(1), (1, 1): Fraction(1)}),
         dict(name="x", parity=1, weight=(0, 0), mat={(0, 1): Fraction(1)}),
         dict(name="y", parity=1, weight=(0, 0), mat={(1, 0): Fraction(1)})]
    return B


def basis_psl22():
    """psl(2|2) = sl(2|2)/C.1_{2|2}, built inside gl(2|2)."""
    B = [dict(name="one", parity=0, weight=(0, 0), drop=True,
              mat={(i, i): Fraction(1) for i in range(4)})]
    def add(name, parity, weight, M):
        B.append(dict(name=name, parity=parity, weight=weight,
                      mat={k: Fraction(v) for k, v in M.items()}))
    # even: sl(2) + sl(2), weights (a; b) recorded as (a, b)
    add("e1", 0, (2, 0), {(0, 1): 1})
    add("f1", 0, (-2, 0), {(1, 0): 1})
    add("h1", 0, (0, 0), {(0, 0): 1, (1, 1): -1})
    add("e2", 0, (0, 2), {(2, 3): 1})
    add("f2", 0, (0, -2), {(3, 2): 1})
    add("h2", 0, (0, 0), {(2, 2): 1, (3, 3): -1})
    # odd: the four E_{i,2+j} and the four E_{2+j,i}
    for i in range(2):
        for j in range(2):
            add("u%d%d" % (i + 1, j + 1), 1, (1 - 2 * i, -(1 - 2 * j)),
                {(i, 2 + j): 1})
            add("v%d%d" % (j + 1, i + 1), 1, (-(1 - 2 * i), 1 - 2 * j),
                {(2 + j, i): 1})
    return B


def selftest():
    print("=== control cases (answers known independently) ===")
    for n in (2, 3):
        cohomology_of(basis_sl(n), "sl(%d)" % n, n, 6 if n == 2 else 5)
    cohomology_of(basis_sl11(), "sl(1|1)", 2, 5)
    cohomology_of(basis_psl22(), "psl(2|2)", 2, 4, full_torus=True)
    print()


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    degmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print("Chevalley-Eilenberg cohomology with trivial coefficients,")
    print("computed on the weight-0 subcomplex over Q.\n")
    if "--test" in sys.argv:
        selftest()
    for n in range(2, nmax + 1):
        cohomology(n, True, degmax, show=(1, 2, 3))
        print()
    for n in range(3, min(nmax, 4) + 1):
        cohomology(n, False, degmax, show=(1, 2))
        print()
