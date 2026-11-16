#!/usr/bin/env python3
r"""
run_pairs.py -- the case 3^o run done backwards, as item 2) of Subsec. 9.3 says.

A subalgebra u as in problem (PROB) which is infinite-dimensional is itself a
simple W-graded vectorial Lie superalgebra with polynomial coefficients, or a
relative of one; in the case 3^o it has u_{-1} = g_{-1}.  So BOTH members of
every row of table (96) are entries of table (65) sharing their g_{-1},
and the run is a finite one: build every depth-1 entry of table (65) that lives
on a given g_{-1} = V, together with its relatives g + C.E, and test
every ordered pair for inclusion, layer by layer, exactly.

The Hasse diagram of that inclusion order is the answer: u is a maximal graded
subalgebra of g exactly when u < g with nothing of the list in between --- and,
since the list is complete by [LS1], nothing at all in between, provided the
intermediate algebra would again be infinite-dimensional and of depth 1.

Output: printed to stdout.
Usage:  run_pairs.py [--kmax K]
"""

import sys
from fractions import Fraction as F

import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from wgrad import VF, bracket, Span, to_vec, divergence              # noqa: E402
from run_cartan import euler_field                                   # noqa: E402
from run_case3b import (independent, layers_vect, layers_svect,      # noqa: E402
                        layers_le, layers_ham)

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------------ inclusion

def span_of(fields):
    keys = sorted({k for X in fields for k in X.t}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    S = Span(idx)
    for X in fields:
        v = to_vec(X, idx)
        if v is not None:
            S.add(v)
    return S, idx


def contained(small, big):
    """is span(small) inside span(big)?  exact."""
    if not small:
        return True
    if not big:
        return False
    keys = sorted({k for X in list(small) + list(big) for k in X.t}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    S = Span(idx)
    for X in big:
        S.add(to_vec(X, idx))
    d0 = S.dim()
    for X in small:
        S.add(to_vec(X, idx))
    return S.dim() == d0


def layer_dims(L, kmax):
    return [len(L.get(k, [])) for k in range(-1, kmax + 1)]


def includes(U, G, kmax):
    """u_k < g_k for every k; returns (verdict, first offending k)."""
    for k in range(-1, kmax + 1):
        if not contained(U.get(k, []), G.get(k, [])):
            return False, k
    return True, None


# --------------------------------------------------------------- the algebras

def add_E(L, ne, no):
    """g + C.E, the relative got by adjoining the Euler operator."""
    E = euler_field(ne, no)
    out = dict(L)
    out[0] = independent(list(L.get(0, [])) + [E], ne, no)
    return out


def derived_layers(L, ne, no, kmax):
    """the derived algebra, layer by layer."""
    out = {}
    for k in range(-1, kmax + 1):
        acc = []
        for i in range(-1, k + 2):
            j = k - i
            if j < -1 or j > kmax:
                continue
            for X in L.get(i, []):
                for Y in L.get(j, []):
                    C = bracket(X, Y)
                    if C:
                        acc.append(C)
        out[k] = independent(acc, ne, no) if acc else []
    return out


def intersect_layers(A, B, ne, no, kmax):
    out = {}
    for k in range(-1, kmax + 1):
        a, b = A.get(k, []), B.get(k, [])
        if not a or not b:
            out[k] = []
            continue
        keys = sorted({kk for X in a + b for kk in X.t}, key=str)
        idx = {kk: i for i, kk in enumerate(keys)}

        def vec(X):
            v = [F(0)] * len(keys)
            for kk, c in X.t.items():
                v[idx[kk]] = c
            return v
        from exactlin import nullspace
        M = [[vec(a[i])[t] for i in range(len(a))]
             + [-vec(b[j])[t] for j in range(len(b))] for t in range(len(keys))]
        acc = []
        for vv in nullspace(M):
            X = VF({}, ne, no)
            for i in range(len(a)):
                if vv[i]:
                    X = X.add(a[i], F(vv[i]))
            if X:
                acc.append(X)
        out[k] = independent(acc, ne, no) if acc else []
    return out


def build(m, n, kmax):
    """every depth-1 entry of table (65) living on g_{-1} = (m|n), plus relatives."""
    algs = {}
    algs["vect"] = layers_vect(m, n, kmax)
    algs["svect"] = layers_svect(m, n, kmax)
    algs["svect+CE"] = add_E(algs["svect"], m, n)
    if m % 2 == 0:
        algs["h"] = layers_ham(m // 2, n, kmax)
        algs["h+CE"] = add_E(algs["h"], m, n)
    if m == n and m > 0:
        algs["le"] = layers_le(n, kmax)
        algs["le+CE"] = add_E(algs["le"], m, n)
        # sle(n) is the divergence-free part of le(n), i.e. le cap svect; its
        # 0th component is spe^a(n), of dim 2n^2-1, per eq. (38).  It is
        # NOT the derived algebra of le(n): le(n) is simple, hence perfect.
        algs["sle"] = intersect_layers(algs["le"], algs["svect"], m, n, kmax)
        algs["sle+CE"] = add_E(algs["sle"], m, n)
    return algs


# ------------------------------------------------------------------- the run

def run(m, n, kmax):
    say("=" * 78)
    say(f"g_-1 = ({m}|{n}):  the depth-1 entries of table (65) on it, "
        f"and their inclusion order")
    say("=" * 78)
    algs = build(m, n, kmax)
    names = list(algs)
    for nm in names:
        say(f"    dim {nm:<14} in degrees -1..{kmax}: "
            + ", ".join(str(d) for d in layer_dims(algs[nm], kmax)))
    say("")
    inc = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            ok, bad = includes(algs[a], algs[b], kmax)
            if ok:
                same = layer_dims(algs[a], kmax) == layer_dims(algs[b], kmax)
                inc[(a, b)] = "=" if same else "<"
    say("    strict inclusions found (u < g):")
    strict = [(a, b) for (a, b), r in inc.items() if r == "<"]
    for a, b in sorted(strict):
        say(f"        {a:<14} < {b}")
    eq = [(a, b) for (a, b), r in inc.items() if r == "="]
    if eq:
        say("    coincidences (same dimensions in every degree, mutual inclusion):")
        for a, b in sorted(eq):
            if a < b:
                say(f"        {a} = {b}")
    say("")
    say("    MAXIMAL pairs (nothing of the list strictly in between):")
    for a, b in sorted(strict):
        mids = [c for c in names
                if c not in (a, b)
                and inc.get((a, c)) == "<" and inc.get((c, b)) == "<"]
        if not mids:
            say(f"        {a:<14} is maximal in {b}")
        else:
            say(f"        {a:<14} < {b}   -- but {', '.join(mids)} sits in between")
    say("")


def main():
    global LOG
    args = sys.argv[1:]
    kmax = 3
    if "--kmax" in args:
        kmax = int(args[args.index("--kmax") + 1])
    LOG = None
    say("The case 3^o run done backwards: the inclusion order of the depth-1")
    say(f"entries of table (65) sharing a g_-1  (exact over Q, degrees -1..{kmax})\n")
    for (m, n) in [(2, 2), (3, 3), (2, 1), (2, 0), (4, 0), (1, 2), (2, 4)]:
        run(m, n, kmax)
    say("Done.")


if __name__ == "__main__":
    main()
