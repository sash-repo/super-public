"""Even elements of the free Volichenko algebra.

An element u of a Volichenko algebra h in g is EVEN when u lies in g_ev.  Two
consequences are identities of the whole class:

        u o x = 0        for all x            (u is in the anticenter)
        u(xy) = (ux)y + x(uy)                 (L_u is a derivation)

Volichenko's "Theorem" V asserts that the even elements one gets from the
Jordan elements a o b and the Jacobi elements J(a,b,c) -- that is, the
subalgebra A^(JJ) they generate -- account for ALL the identities.  Baranov
disproved that.  This module measures the gap directly: it compares

        Z_n     = ALL multilinear even elements of degree n
        E_n     = those coming from A^(JJ)

and a new even element in degree n produces a new identity in degree n+1.
"""

import itertools
import sys
from fractions import Fraction as Fr

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from exactlin import rank as ex_rank, nullspace as ex_null  # noqa: E402

import volid
import videal
from volid import bracketings, evaluate, par
from videal import (mmul, mcirc, mjac, mevaluate, msubst, relabel_tree,
                    set_partitions_ordered, MSpace)


def monomial_exprs(S):
    """All multilinear monomials on the index tuple S, as elements of M."""
    return [{t: Fr(1)} for t in bracketings(tuple(S))]


def known_even(n, cache):
    """A spanning set of the degree-n part of A^(JJ), as formal expressions."""
    if n in cache:
        return cache[n]
    idx = tuple(range(1, n + 1))
    out = []
    # Jordan elements  u o v  with u, v arbitrary monomials on complementary sets
    for k in range(1, n // 2 + 1):
        for S in itertools.combinations(idx, k):
            T = tuple(x for x in idx if x not in S)
            if k == n - k and S[0] != idx[0]:
                continue                      # o is symmetric: one split only
            for u in monomial_exprs(S):
                for v in monomial_exprs(T):
                    out.append(mcirc(u, v))
    # Jacobi elements  J(u,v,w)
    for blocks in set_partitions_ordered(idx, 3):
        for u in monomial_exprs(blocks[0]):
            for v in monomial_exprs(blocks[1]):
                for w in monomial_exprs(blocks[2]):
                    out.append(mjac(u, v, w))
    # products of two even elements
    for k in range(2, n - 1):
        for S in itertools.combinations(idx, k):
            T = tuple(x for x in idx if x not in S)
            if len(T) < 2:
                continue
            for u in relabelled(known_even(k, cache), S):
                for v in relabelled(known_even(len(T), cache), T):
                    out.append(mmul(u, v))
                    out.append(mmul(v, u))
    cache[n] = out
    return out


def relabelled(exprs, S):
    """Rename the variables 1..k of each expression to the tuple S."""
    sub = {i + 1: S[i] for i in range(len(S))}
    return [msubst(f, sub) for f in exprs]


def analyse(n, verbose=True):
    ms = MSpace(n)
    trees = ms.trees
    vals = [evaluate(t) for t in trees]
    sp = volid.Space(vals)
    rows = sp.rows()

    # dim W_n and dim I_n
    dW = ex_rank(rows)
    dM = len(trees)
    dI = dM - dW

    # even expressions: combinations of trees whose value has no odd part
    okeys = [k for k in sp.keys if par(k) == 1]
    opos = {k: i for i, k in enumerate(okeys)}
    orows = []
    for e in vals:
        r = [Fr(0)] * len(okeys)
        for k, c in e.items():
            if par(k) == 1:
                r[opos[k]] = Fr(c)
        orows.append(r)
    mat = [[orows[j][i] for j in range(len(orows))] for i in range(len(okeys))]
    ker = ex_null(mat) if mat else [[Fr(1) if i == j else Fr(0)
                                     for j in range(dM)] for i in range(dM)]
    Zexprs = []
    for v in ker:
        f = {t: c for t, c in zip(trees, v) if c}
        if f:
            Zexprs.append(f)
    Zvals = [mevaluate(f) for f in Zexprs]
    dZ = ex_rank(volid.Space(Zvals).rows() or [[]]) if Zvals else 0

    # the even elements Volichenko's theorem accounts for
    Eexprs = known_even(n, {})
    Evals = [mevaluate(f) for f in Eexprs]
    Evals = [e for e in Evals if e]
    dE = ex_rank(volid.Space(Evals).rows()) if Evals else 0

    both = ex_rank(volid.Space(Zvals + Evals).rows())

    if verbose:
        print("degree %d:  dim M = %-6d dim W = %-5d dim I = %-5d"
              % (n, dM, dW, dI))
        print("           even elements: dim Z = %-4d  from A^(JJ): dim E = %-4d"
              "  new = %d   (Z + E has dim %d)"
              % (dZ, dE, dZ - dE, both))
    return dict(n=n, dM=dM, dW=dW, dI=dI, dZ=dZ, dE=dE,
                Zexprs=Zexprs, Zvals=Zvals, Evals=Evals, ms=ms)


if __name__ == "__main__":
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    for n in range(lo, hi + 1):
        analyse(n)
