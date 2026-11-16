#!/usr/bin/env python3
"""H^*(witt_-; witt) --- exact Chevalley-Eilenberg cohomology of the negative
part of the Witt algebra with coefficients in the whole of it.

The problem (Appendix A of the text): compute H^2(witt_-; witt), where
witt_- = \\oplus_{i<0} witt_i.  The obstruction: the spaces of cochains of any
given degree are infinite-dimensional.

The way out implemented here.  Put E_i := L_{-i} (i >= 1), so that

    [E_i, E_j] = (j - i) E_{i+j},                       witt_- = <E_1, E_2, ...>

i.e. witt_- is the positive Witt algebra L_1 = <e_i | i >= 1> under i <-> -i,
and the coefficient module witt becomes p_m := L_{-m}, m in Z, with

    E_i . p_m = (m - i) p_{m+i},

which is the density module F_lambda at lambda = -1, but LAURENT: its weight
components sit in every degree m in Z, not only in m >= 0.  That is exactly why
each weight component of the cochain space is infinite-dimensional.

Now grade cochains by WEIGHT: c has weight w if c(E_{i_1},...,E_{i_q}) lies in
the component of degree (i_1 + ... + i_q) + w.  The differential preserves w.
Inside a fixed weight, filter by the total degree a = i_1 + ... + i_q of the
ARGUMENT.  In

    (dc)(E_{j_0},..,E_{j_q}) = sum_k (-1)^k E_{j_k} . c(E_{J - j_k})
                             + sum_{k<l} (-1)^{k+l} c([E_{j_k},E_{j_l}], E_{J-j_k-j_l})

every c on the right is evaluated on arguments of total degree <= a, since
deg E_{j_k} >= 1 and [E_{j_k},E_{j_l}] has degree j_k + j_l.  Hence

    F_N := {c | c vanishes on all arguments of total degree <= N}

is a SUBcomplex, the quotient Q_N := C_w/F_N -- "the cochains, truncated to
arguments of degree <= N" -- is FINITE-DIMENSIONAL, the transition maps
Q_{N+1} ->> Q_N are onto, and C_w = lim_N Q_N.  All H^q(Q_N) being finite-
dimensional, Mittag-Leffler gives lim^1 = 0 and

    H^q(witt_-; witt)_w = lim_N H^q(Q_N^w).

So the whole cochain space (no finiteness assumption on cochains whatever: the
identity map IS a cochain here) is reached by a sequence of finite exact
computations, one per (w, N).  This module does them.

Everything is over Q with exact arithmetic (../lib/exactlin.py -> FLINT).

Modules supported (flag --module):
    adj      p_m, m in Z, E_i.p_m = (m - i)p_{m+i}          = witt itself
    F<l>     p_m, m in Z, E_i.p_m = (m + l*i)p_{m+i}        = Laurent F_lambda
    triv     the trivial module K                            (Goncharova check)
    adjneg   p_m, m >= 1 only: the adjoint module witt_-      (control)
"""
import sys, os, argparse
from fractions import Fraction
from itertools import combinations

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, nullspace, backend            # noqa: E402


# ----------------------------------------------------------------- modules

class Module:
    """A Z-graded witt_- module with 1-dimensional homogeneous components."""

    def __init__(self, name, exists, mu):
        self.name, self.exists, self.mu = name, exists, mu


def make_module(spec):
    if spec == "adj":
        return Module("adj", lambda n: True, lambda n, i: n - i)
    if spec == "triv":
        return Module("triv", lambda n: n == 0, lambda n, i: 0)
    if spec == "adjneg":
        return Module("adjneg", lambda n: n >= 1, lambda n, i: n - i)
    if spec.startswith("F"):
        lam = Fraction(spec[1:])
        return Module("F%s" % lam, lambda n: True, lambda n, i, l=lam: n + l * i)
    raise SystemExit("unknown module %r" % spec)


# ------------------------------------------------------- the graded pieces

def basis(q, N, w, mod):
    """q-subsets {i_1<...<i_q} of Z_{>0} with sum <= N whose value lives."""
    out = []
    if q == 0:
        return [()] if (N >= 0 and mod.exists(w)) else []
    for I in _subsets(q, N):
        if mod.exists(sum(I) + w):
            out.append(I)
    return out


def _subsets(q, N):
    """All {i_1<...<i_q}, i_k >= 1, with sum <= N -- in a fixed order."""
    res = []

    def rec(start, left, budget, acc):
        if left == 0:
            res.append(tuple(acc))
            return
        i = start
        while True:
            # the remaining left-1 entries are at least i+1, ..., i+left-1
            need = i + sum(range(i + 1, i + left))
            if need > budget:
                break
            acc.append(i)
            rec(i + 1, left - 1, budget - i, acc)
            acc.pop()
            i += 1
    rec(1, q, N, [])
    return sorted(res)


def differential(q, N, w, mod):
    """Matrix of d: C^q -> C^{q+1}, rows = C^{q+1} basis, cols = C^q basis."""
    src, tgt = basis(q, N, w, mod), basis(q + 1, N, w, mod)
    pos = {I: k for k, I in enumerate(src)}
    M = [[0] * len(src) for _ in tgt]
    for r, J in enumerate(tgt):
        a = sum(J)
        for k, jk in enumerate(J):
            rest = J[:k] + J[k + 1:]
            if rest in pos:                       # module term
                coef = mod.mu(a - jk + w, jk)
                if coef:
                    M[r][pos[rest]] += (-1) ** k * coef
        for k in range(len(J)):
            for l in range(k + 1, len(J)):
                jk, jl = J[k], J[l]
                s = jk + jl
                rest = tuple(x for x in J if x != jk and x != jl)
                if s in rest:                     # E_s already present: 0
                    continue
                new = tuple(sorted((s,) + rest))
                if new not in pos:
                    continue
                shift = sum(1 for x in rest if x < s)   # move E_s into place
                sign = (-1) ** (k + l) * (-1) ** shift
                M[r][pos[new]] += sign * (jl - jk)
    return M, src, tgt


def _rank(M):
    if not M or not M[0]:
        return 0
    return rank(M)


def cohomology(q, N, w, mod):
    """dim H^q(Q_N) in weight w = dim ker d_q - rank d_{q-1}."""
    dq, src, _ = differential(q, N, w, mod)
    rq = _rank(dq)
    dq1, _, _ = differential(q - 1, N, w, mod) if q >= 1 else (None, None, None)
    rq1 = _rank(dq1) if q >= 1 else 0
    return len(src) - rq - rq1, len(src)


# ------------------------------------------- stable part under truncation

def _restrict(vecs, src_big, src_small):
    idx = {I: k for k, I in enumerate(src_big)}
    return [[v[idx[I]] for I in src_small] for v in vecs]


def stable_H(q, N, Nbig, w, mod):
    """dim of the image of H^q(Q_{Nbig}) -> H^q(Q_N): the part of the class
    that survives every further truncation, i.e. the honest inverse limit."""
    dq_b, src_b, _ = differential(q, Nbig, w, mod)
    Zb = nullspace(dq_b) if (src_b and dq_b and dq_b[0]) else \
        ([[1 if i == j else 0 for j in range(len(src_b))] for i in range(len(src_b))]
         if src_b else [])
    _, src_s, _ = differential(q, N, w, mod)
    Zres = _restrict(Zb, src_b, src_s)
    dq1_s, _, _ = differential(q - 1, N, w, mod)
    B = []
    if dq1_s and dq1_s[0]:
        # columns of d_{q-1} span B^q
        B = [[dq1_s[r][c] for r in range(len(dq1_s))] for c in range(len(dq1_s[0]))]
        B = [list(col) for col in zip(*dq1_s)] if dq1_s else []
    B = [row for row in B if any(row)]
    rB = _rank(B) if B else 0
    both = (Zres if Zres else []) + B
    both = [row for row in both if any(row)]
    rBoth = _rank(both) if both else 0
    return rBoth - rB


# --------------------------------------------------------------- reporting

def scan(mod, wlo, whi, Ns, qs=(1, 2)):
    print("backend: %s   module: %s" % (backend(), mod.name))
    for q in qs:
        print("\n=== H^%d,  truncation levels N = %s" % (q, list(Ns)))
        print("  w  " + "".join("  N=%-3d" % N for N in Ns) + "   dim C^%d(N=%d)" % (q, Ns[-1]))
        for w in range(wlo, whi + 1):
            row, dimC = [], 0
            for N in Ns:
                h, dimC = cohomology(q, N, w, mod)
                row.append(h)
            if any(row):
                print("%4d " % w + "".join("  %-5d" % h for h in row) + "   %d" % dimC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="adj")
    ap.add_argument("--wlo", type=int, default=-20)
    ap.add_argument("--whi", type=int, default=20)
    ap.add_argument("--N", type=int, nargs="+", default=[12, 16, 20, 24])
    ap.add_argument("--q", type=int, nargs="+", default=[1, 2])
    args = ap.parse_args()
    scan(make_module(args.module), args.wlo, args.whi, args.N, args.q)


if __name__ == "__main__":
    main()
