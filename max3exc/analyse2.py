#!/usr/bin/env python3
"""Maximal graded subalgebras of osp_a(4|2)=D(2,1;a): final computation.

For each of the two Z-gradings and each candidate u (built as a Cartan prolong
of a prescribed non-positive part, as in the Leites--Shchepochkina method), we
  * certify maximality either by Burnside (g/u an irreducible u-module) or by
    an exhaustive-on-root-vectors + randomised search for an intermediate
    subalgebra <u,x>;
  * print structural invariants (graded pieces, derived series, dim(u cap h),
    ideal decomposition) so the answer can be identified.
Run for several values of the parameter a to see what is a-independent.
"""
import sys, random
from fractions import Fraction as F
from itertools import product
from lin import span, dim, inspan, kernel, rref, zero
from d21a import (N, PAR, build, bracket, check_jacobi, xi, sl2, grading,
                  sdim, subalg_generated, stab)

random.seed(20260803)
NAMES = ["E1", "H1", "F1", "E2", "H2", "F2", "E3", "H3", "F3"] + \
        ["x%d%d%d" % I for I in product((0, 1), repeat=3)]
E = [[F(1) if i == k else F(0) for i in range(N)] for k in range(N)]
CARTAN = [E[sl2(m, "H")] for m in range(3)]


class G:
    def __init__(self, a):
        self.a = a
        self.B = build((F(1), a, -1 - a))
        assert check_jacobi(self.B) == 0

    def br(self, x, y):
        return bracket(self.B, x, y)


def prolong(g, comps, neg, u0):
    u = {k: span([list(v) for v in b]) for k, b in neg.items()}
    u[0] = span([list(v) for v in u0])
    for k in range(1, max(comps) + 1):
        cur = comps[k]
        rows = []
        for j in sorted(x for x in u if x < 0):
            tgt = u.get(j + k, [])
            ann = kernel(span(tgt), N) if tgt else [list(e) for e in E]
            for y in u[j]:
                for phi in ann:
                    rows.append([sum(phi[t] * g.br(x, y)[t] for t in range(N))
                                 for x in cur])
        if rows:
            out = []
            for co in kernel(rows, len(cur)):
                v = zero(N)
                for cf, x in zip(co, cur):
                    if cf:
                        v = [p + cf * q for p, q in zip(v, x)]
                out.append(v)
            u[k] = span(out)
        else:
            u[k] = list(cur)
    return u


def flat(u):
    return [v for k in sorted(u) for v in u[k]]


def is_subalg(g, b):
    return all(inspan(g.br(x, y), b) for x in b for y in b)


def quotient_ops(g, ub):
    m, piv = rref(ub)
    comp = [i for i in range(N) if i not in piv]
    d = len(comp)

    def proj(v):
        w = list(v)
        for row, pc in zip(m, piv):
            if w[pc] != 0:
                f = w[pc]
                w = [p - f * q for p, q in zip(w, row)]
        return [w[i] for i in comp]
    ops = [[proj(g.br(x, E[c])) for c in comp] for x in ub]
    return d, comp, proj, ops


def burnside(g, ub):
    d, comp, proj, ops = quotient_ops(g, ub)
    if d == 0:
        return None
    fl = lambda M: [t for r in M for t in r]
    unfl = lambda v: [[v[i * d + j] for j in range(d)] for i in range(d)]
    mul = lambda P, Q: [[sum(P[i][k] * Q[k][j] for k in range(d))
                         for j in range(d)] for i in range(d)]
    ident = [[F(1) if i == j else F(0) for j in range(d)] for i in range(d)]
    basis = span([fl(ident)] + [fl(o) for o in ops])
    changed = True
    while changed and len(basis) < d * d:
        changed = False
        for p in list(basis):
            for o in ops:
                pr = fl(mul(unfl(p), o))
                if any(pr) and not inspan(pr, basis):
                    basis = span(basis + [pr]); changed = True
    return len(basis) == d * d


def find_intermediate(g, u, comps, trials=80):
    """search for x outside u with <u,x> != g."""
    ub = span([list(x) for x in flat(u)])
    for k in sorted(comps):
        tests = [list(v) for v in comps[k]]
        for _ in range(trials):
            v = zero(N)
            for b in comps[k]:
                cf = F(random.randint(-3, 3))
                v = [p + cf * q for p, q in zip(v, b)]
            if any(v):
                tests.append(v)
        for x in tests:
            if inspan(x, ub):
                continue
            gen = subalg_generated(g.B, ub + [x])
            if len(gen) != N:
                return (k, x, gen)
    return None


def derived(g, b):
    out = [len(b)]
    cur = b
    for _ in range(4):
        nxt = span([g.br(x, y) for x in cur for y in cur])
        out.append(len(nxt))
        if len(nxt) == len(cur) or not nxt:
            break
        cur = nxt
    return out


def ideals_split(g, b):
    """crude: dimension of the centre and of [u,u]."""
    ctr = []
    rows = []
    for x in b:
        pass
    # centre = {x in u : [x,u]=0}
    conds = []
    for y in b:
        for t in range(N):
            conds.append([g.br(x, y)[t] for x in b])
    ker = kernel(conds, len(b))
    return len(ker)


def describe(g, tag, u, comps):
    b = span([list(x) for x in flat(u)])
    if len(b) == N:
        return ("whole", None)
    if not is_subalg(g, b):
        print("   %-42s NOT A SUBALGEBRA (bug)" % tag)
        return None
    sd = sdim(b)
    pieces = " ".join("%+d:%d|%d" % (k, *sdim(u[k])) for k in sorted(u) if u[k])
    bur = burnside(g, b)
    if bur:
        verdict = "MAXIMAL (Burnside)"
    else:
        bad = find_intermediate(g, u, comps)
        verdict = ("MAXIMAL (no intermediate found)" if bad is None
                   else "not maximal (inside %d|%d)" % sdim(bad[2]))
    inter_h = dim([list(h) for h in CARTAN] + list(b)) - len(b)
    torus = 3 - inter_h        # dim (u cap h)
    print("   %-42s %d|%d  %-30s d.s.%s  h∩u=%d  z=%d  %s"
          % (tag, sd[0], sd[1], pieces, derived(g, b), torus,
             ideals_split(g, b), verdict))
    return (sd, verdict)


def run(g, D, title, Vlist, negshapes=()):
    comps = grading(g.B, D)
    print("\n" + "-" * 118)
    print("%s   [a=%s]   %s" % (title, g.a,
          "  ".join("g_%+d=%d|%d" % (d, *sdim(comps[d])) for d in sorted(comps))))
    print("-" * 118)
    neg_all = {k: comps[k] for k in comps if k < 0}
    # 4o : u = g_{<=0}  (u_1 = 0 by fiat, not a prolong)
    u = {k: comps[k] for k in comps if k <= 0}
    describe(g, "4o  u = g_{<=0}", u, comps)
    # 5o
    u = {k: comps[k] for k in comps if k >= 0}
    describe(g, "5o  u = g_{>=0}", u, comps)
    for name, V in Vlist:
        Vb = span([list(v) for v in V])
        neg = {-1: Vb}
        for k in comps:
            if k < -1:
                neg[k] = span([g.br(x, y) for x in Vb for y in Vb]) or []
        st = stab(g.B, comps[0], Vb)
        for k in sorted(neg):
            if k < -1 and neg[k]:
                st = stab(g.B, st, span(neg[k]))
        describe(g, "2o  V=%s" % name, prolong(g, comps, neg, st), comps)
        st0 = stab(g.B, comps[0], Vb)
        describe(g, "3o  u_0=stab(%s)=%d|%d" % (name, *sdim(st0)),
                 prolong(g, comps, neg_all, st0), comps)
    for name, neg in negshapes:
        st = comps[0]
        for k in sorted(neg):
            if neg[k]:
                st = stab(g.B, st, span(neg[k]))
        describe(g, "6o  %s" % name, prolong(g, comps, neg, st), comps)


for a in [F(5), F(1), F(1, 3)]:
    g = G(a)
    D1 = [F(0)] * N
    D1[sl2(0, "H")] = F(1, 2); D1[sl2(1, "H")] = F(1, 2)
    F1, F2 = E[sl2(0, "F")], E[sl2(1, "F")]
    x110, x111 = E[xi(1, 1, 0)], E[xi(1, 1, 1)]
    FF = [p + q for p, q in zip(F1, F2)]
    V1 = [("<F1> isotr. even line", [F1]),
          ("<F1+F2> non-isotr. even line", [FF]),
          ("<x110> odd line", [x110]),
          ("(1|1) isotr. even line", [F1, x110]),
          ("(1|1) non-isotr. even line", [FF, x110]),
          ("(2|0)=(g_-1)_ev", [F1, F2]),
          ("(0|2)=(g_-1)_od", [x110, x111]),
          ("(2|1)", [F1, F2, x110]),
          ("(1|2) isotr. even line", [F1, x110, x111]),
          ("(1|2) non-isotr. even line", [FF, x110, x111])]
    run(g, D1, "DEPTH-1 GRADING", V1)
    D2 = [F(0)] * N
    D2[sl2(0, "H")] = F(1)
    c2 = grading(g.B, D2)
    o = c2[-1]
    V2 = [("<v> odd line", [o[0]]),
          ("(0|2) isotropic", [o[0], o[1]]),
          ("(0|2) non-degenerate", [o[0], o[3]]),
          ("(0|3)", [o[0], o[1], o[2]]),
          ("(0|4)=g_-1 : case 3o only", [o[0], o[1], o[2], o[3]])]
    run(g, D2, "CONTACT GRADING (depth 2)", V2,
        negshapes=[("u_-1=0, u_-2=g_-2", {-2: c2[-2], -1: []})])
