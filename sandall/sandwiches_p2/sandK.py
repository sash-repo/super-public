#!/usr/bin/env python3
"""
The sandwich variety over the algebraically closed K = \bar F_2, not just over
F_2 and F_4 (this settles the proviso that the sandwiches might appear only
over an extension of F_2).

A sandwich is an x with

    (san)   (ad_x)^2 = 0,     (san2)  [[L, x], [L, x]] = 0.

For x = sum_i a_i e_i both conditions are HOMOGENEOUS QUADRATIC in the
coordinates a_i, with coefficients in F_2:

    (ad_x)^2 (e_j)  = sum_{i<k} a_i a_k (ad_i ad_k + ad_k ad_i)(e_j)
                      + sum_i a_i^2 (ad_i)^2 (e_j),
    [[e_p,x],[e_q,x]] = sum_{i<k} a_i a_k ([[e_p,e_i],[e_q,e_k]]
                                          + [[e_p,e_k],[e_q,e_i]])
                      + sum_i a_i^2 [[e_p,e_i],[e_q,e_i]].

(Beware: in char 2 over K the map a -> a^2 is NOT the identity, so a_i^2 is a
genuine quadratic monomial and must be kept as such; over F_2 it collapses to
a_i, which is why the F_2-point count alone proves nothing over K.)

So the sandwiches form a cone S(L) in L (x) K, cut out by quadrics defined over
F_2, i.e. a projective variety V in P(L (x) K) = P^{n-1}.

What we prove here, for each algebra L:

  (A) V is EMPTY  (hence s(L) = 0 over K as well), or
  (B) V lies in the F_2-span S of the F_2-rational sandwiches (hence
      s(L) (x) K = s(L)_{F_2} (x) K and the whole table 3.7 survives), or
  (C) neither -- then a new sandwich exists over some extension and is hunted.

Method.  Both (A) and (B) are of the form "V misses the open set {l != 0}" for
each l in a basis of the annihilator Ann(S) of S in the dual space (for (A),
S = 0 and Ann(S) is the whole dual, so the n coordinate charts cover P^{n-1};
for (B) it is enough to run over a BASIS of Ann(S), because the linear forms
vanishing on V form a subspace, so if a basis of Ann(S) vanishes on V then all
of Ann(S) does, and then V is contained in P(S)).

In the chart l = 1 the quadrics become inhomogeneous polynomials of degree <= 2
in n-1 variables, and "no K-point" is, by the weak Nullstellensatz, equivalent
to 1 lying in the ideal they generate.  We certify that by exact linear algebra
over F_2 on the Macaulay matrix in degrees <= D: if the constant polynomial 1
lies in the F_2-span of {monomial * generator} with deg <= D, the chart is
empty over EVERY extension of F_2, K included.  (The rank of a matrix over F_2
does not change under field extension, so working over the prime field is no
loss.)  A positive answer is a proof; a negative one at degree D is
inconclusive and we raise D.

Usage:  python3 sandK.py [file.txt ...]     (default: dim15 + dim16 + thin15)
"""

import glob
import itertools
import os
import sys

from css import LieF2, rref, in_span, load
from thin import algebra_from_thin_table, load_table


# --------------------------------------------------------------- the algebras

def build(path):
    name = os.path.basename(path)[:-4]
    data = load_table(path)
    if isinstance(data[0][0], list):
        dim = len(data[0])
        if len(data) == dim:
            return LieF2.from_adjoint_basis(data, name)
        return LieF2.from_generators(data, name, dim)
    return algebra_from_thin_table(data, name)


# ------------------------------------------------- the quadrics cutting out V
# A polynomial is a set of monomials; a monomial is a sorted tuple of variable
# indices with repetitions (so (i, i) = a_i^2 is kept distinct from (i,)).

def quadrics(L):
    """the list of quadratic forms whose common zeros are the sandwiches"""
    n = L.n
    e = [1 << i for i in range(n)]
    polys = []

    # (san): coefficient of the coordinate t in (ad_x)^2 (e_j)
    for j in range(n):
        coef = {}                      # monomial -> bit vector of the value
        for i in range(n):
            for k in range(n):
                v = L.mul(e[i], L.mul(e[k], e[j]))
                if not v:
                    continue
                mon = (i, k) if i <= k else (k, i)
                coef[mon] = coef.get(mon, 0) ^ v
        polys.extend(_split(coef, n))

    # (san2): coefficient of the coordinate t in [[e_p, x], [e_q, x]]
    for p in range(n):
        for q in range(p + 1, n):
            coef = {}
            for i in range(n):
                for k in range(n):
                    v = L.mul(L.mul(e[p], e[i]), L.mul(e[q], e[k]))
                    if not v:
                        continue
                    mon = (i, k) if i <= k else (k, i)
                    coef[mon] = coef.get(mon, 0) ^ v
            polys.extend(_split(coef, n))
    return [p for p in polys if p]


def _split(coef, n):
    """monomial -> vector of values  ==>  one polynomial per coordinate"""
    out = []
    for t in range(n):
        p = frozenset(m for m, v in coef.items() if (v >> t) & 1)
        if p:
            out.append(p)
    return out


def eval_poly_f2(poly, a):
    """value at the F_2-point a (a bit vector); over F_2, a_i^2 = a_i"""
    s = 0
    for mon in poly:
        v = 1
        for i in mon:
            v &= (a >> i) & 1
        s ^= v
    return s


# ------------------------------------------------------ linear algebra over F_2

def rank_rows(rows):
    piv = {}
    for v in rows:
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            piv[v.bit_length() - 1] = v
    return piv


def reduce_by(piv, v):
    for p in sorted(piv, reverse=True):
        if (v >> p) & 1:
            v ^= piv[p]
    return v


# ------------------------------------------------------------- chart l = 1

def substitute(polys, n, ell):
    """set the linear form ell (a bit mask of variables) equal to 1.

    Pick j in ell; then a_j = 1 + sum_{i in ell, i != j} a_i.  Returns the
    polynomials in the remaining variables, re-indexed 0..n-2, as sets of
    monomials of degree <= 2, and the index map."""
    j = (ell & -ell).bit_length() - 1
    rest = [i for i in range(n) if i != j]
    idx = {i: r for r, i in enumerate(rest)}
    # a_j = 1 + sum of these variables (in NEW indices)
    subst = frozenset([()] + [(idx[i],) for i in rest if (ell >> i) & 1])

    def lin(i):
        return subst if i == j else frozenset([(idx[i],)])

    out = []
    for poly in polys:
        acc = frozenset()
        for mon in poly:
            (i, k) = mon
            acc = _xor(acc, _mulpoly(lin(i), lin(k)))
        if acc:
            out.append(acc)
    return out, n - 1


def _xor(p, q):
    return frozenset(p) ^ frozenset(q)


def _mulmon(m1, m2):
    return tuple(sorted(m1 + m2))


def _mulpoly(p, q):
    acc = set()
    for m1 in p:
        for m2 in q:
            m = _mulmon(m1, m2)
            acc ^= {m}
    return frozenset(acc)


# ------------------------------------------------- Macaulay: is 1 in the ideal?

def monomials_upto(m, d):
    """all monomials of degree <= d in m variables, as sorted tuples"""
    out = [()]
    for k in range(1, d + 1):
        out.extend(itertools.combinations_with_replacement(range(m), k))
    return out


def one_in_ideal(polys, m, D, verbose=False):
    """True if the constant 1 lies in the F_2-span of {mu * f : deg(mu*f) <= D},
    which certifies that the system has no solution over any extension of F_2."""
    mons = monomials_upto(m, D)
    index = {mu: i for i, mu in enumerate(mons)}
    multipliers = monomials_upto(m, max(0, D - 2))

    def vec(poly):
        v = 0
        for mon in poly:
            v |= 1 << index[mon]
        return v

    piv = {}
    target = 1 << index[()]
    for mu in multipliers:
        for f in polys:
            g = frozenset(_mulmon(mu, mon) for mon in f)
            if any(len(mon) > D for mon in g):
                continue
            v = reduce_by(piv, vec(g))
            if v:
                piv[v.bit_length() - 1] = v
                if reduce_by(piv, target) == 0:
                    if verbose:
                        print("        1 in the ideal at degree <= %d" % D)
                    return True
    return False


def reduce_generators(polys):
    """drop F_2-linear combinations: keep a basis of the span of the polys"""
    mons = sorted({mon for p in polys for mon in p})
    index = {mu: i for i, mu in enumerate(mons)}
    piv, keep = {}, []
    for p in polys:
        v = 0
        for mon in p:
            v |= 1 << index[mon]
        w = reduce_by(piv, v)
        if w:
            piv[w.bit_length() - 1] = w
            keep.append(p)
    return keep


# --------------------------------------------------------------------- report

def annihilator_basis(S, n):
    """basis of the space of linear forms (bit masks) vanishing on span(S)"""
    R = rref(list(S))
    pivots = [r.bit_length() - 1 for r in R]
    free = [i for i in range(n) if i not in pivots]
    out = []
    for f in free:
        # the form  a_f + sum over pivots of the corresponding coefficients
        mask = 1 << f
        for r in R:
            p = r.bit_length() - 1
            if (r >> f) & 1:
                mask |= 1 << p
        out.append(mask)
    return out


def analyse(L, maxdeg=4, verbose=True):
    n = L.n
    polys = quadrics(L)
    polys = reduce_generators(polys)

    # ---- validation: the F_2-points of the quadrics must be exactly the
    # sandwiches found by brute force in css.py
    brute = set(L.sandwiches())
    fromeq = {a for a in range(1, 1 << n)
              if all(eval_poly_f2(p, a) == 0 for p in polys)}
    ok = (brute == fromeq)
    S = rref(list(brute)) if brute else []
    if verbose:
        print("%-24s dim %2d | %4d independent quadrics | F_2-sandwiches %3d, "
              "span %d | equations reproduce them: %s"
              % (L.name, n, len(polys), len(brute), len(S), ok), flush=True)
    if not ok:
        return None

    charts = annihilator_basis(S, n)
    if verbose:
        print("      charts to kill: %d (= codim of the F_2-span)" % len(charts),
              flush=True)
    results = []
    for ell in charts:
        aff, m = substitute(polys, n, ell)
        aff = reduce_generators(aff)
        proved, deg = False, None
        for D in (2, 3, maxdeg):
            if one_in_ideal(aff, m, D):
                proved, deg = True, D
                break
        results.append((ell, proved, deg if proved else None))
        if verbose:
            names = "+".join("a%d" % i for i in range(n) if (ell >> i) & 1)
            print("      chart %-18s : %s"
                  % (names, "empty over K (degree <= %d)" % deg if proved
                     else "NOT decided up to degree %d" % maxdeg), flush=True)
    allgood = all(r[1] for r in results)
    if verbose:
        if allgood and not brute:
            print("      ==> the sandwich variety is EMPTY over K: s(L) = 0.",
                  flush=True)
        elif allgood:
            print("      ==> every K-sandwich lies in the F_2-span: "
                  "s(L) is the same over K.", flush=True)
        else:
            print("      ==> INCONCLUSIVE; hunt for a point in the open charts.",
                  flush=True)
    return allgood, results


def main(paths):
    for path in paths:
        L = build(path)
        okj, msg = L.check_jacobi()
        if not okj:
            print("%s: not a Lie algebra (%s)" % (L.name, msg))
            continue
        analyse(L)
        print(flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = (sorted(glob.glob("css-bases/thin15/*.txt"))
                + sorted(glob.glob("css-bases/dim16/*.txt")))
    main(args)
