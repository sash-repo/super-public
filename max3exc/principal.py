#!/usr/bin/env python3
"""Statement 9.5: the subalgebras u of ag(2), ab(3)
whose even part contains the principal sl(2) of s = g(2), resp. o(7).
For every even part u_ev in the finite list below and every u_ev-submodule
generator x of g_od (highest vectors, single and paired, generic combinations
where multiplicity > 1), A = <u_ev, x> is computed; if A != g, its sdim, whether it
contains a maximal torus of g (rank), and the multiplicity-free maximality test."""
import sys, os, random
from fractions import Fraction as F
HERE = os.path.dirname(os.path.abspath(__file__))
import run3
from sweep3 import Ech, closure, sdim_of, ev
from lin import kernel
from irrbranch import principal_sl2, root_index, cartan_indices
from casmax import test

def vec_of(g, i): return ev(g, i)
def add(a, b): return [x + y for x, y in zip(a, b)]
def sm(c, a): return [c * x for x in a]

def toral_rank(g, A):
    """dimension of a maximal toral subalgebra of the even part of A,
    estimated as rank of ad on g of generic semisimple... we use: dim of
    centralizer of a generic element of A_ev intersected with A_ev (rank of A_ev)."""
    ev_b = [v for v in Ech(g.N, [[x if g.PAR[i] == 0 else F(0) for i, x in enumerate(v)] for v in A.basis()]).basis() if any(v)]
    rnd = random.Random(5)
    x = [F(0)] * g.N
    for v in ev_b: x = add(x, sm(F(rnd.randint(1, 50)), v))
    rows = []
    for t in range(g.N):
        rows.append([g.br(x, v)[t] for v in ev_b])
    return len(kernel(rows, len(ev_b)))

def witness(g, A):
    """a basis vector y not in A with A < <A,y> < g, if any: A is then not maximal."""
    best = None
    for i in range(g.N):
        y = vec_of(g, i)
        if A.contains(y): continue
        B = closure(g, A.basis() + [y])
        if B.dim() < g.N:
            sd = sdim_of(g, B.basis())
            if best is None or B.dim() > best[0]: best = (B.dim(), sd)
    return best[1] if best else None

def setup(which):
    g = run3.model(which)
    if which == "G3":
        simple = [(0, 1, 0), (0, -1, 1)]; dl = (2, 0, 0)
    else:
        simple = [(0, 1, -1, 0), (0, 0, 1, -1), (0, 0, 0, 1)]; dl = (1, 0, 0, 0)
    E, H, Fv = principal_sl2(g, simple)
    ie = root_index(g, dl); jf = root_index(g, tuple(-x for x in dl))
    e, f = vec_of(g, ie), vec_of(g, jf)
    h = g.br(e, f)
    assert any(h)
    return g, (E, H, Fv), (e, h, f)

def highest(g, ops_raise, odd_only=True):
    """vectors of g_od killed by all raising operators."""
    idx = [i for i in range(g.N) if g.PAR[i] == 1]
    rows = []
    for R in ops_raise:
        for t in range(g.N):
            rows.append([g.br(R, vec_of(g, i))[t] for i in idx])
    out = []
    for co in kernel(rows, len(idx)):
        v = [F(0)] * g.N
        for c, i in zip(co, idx):
            if c: v[i] = c
        out.append(v)
    return out

def run(which):
    g, (E, H, Fv), (e, h, f) = setup(which)
    s = "g(2)" if which == "G3" else "o(7)"
    print("==", g.name, " N =", g.N)
    cases = []
    pr = [E, H, Fv]
    cases.append(("sl2_pr", pr, [E]))
    cases.append(("sl2_pr + Ce", pr + [e], [E, e]))
    cases.append(("sl2_pr + Ch", pr + [h], [E]))
    cases.append(("sl2_pr + b", pr + [h, e], [E, e]))
    cases.append(("sl2_pr + sl2", pr + [e, h, f], [E, e]))
    D = [add(E, e), add(H, h), add(Fv, f)]
    cases.append(("sl2_diag", D, [D[0]]))
    sfull = [vec_of(g, i) for i in range(g.N) if g.PAR[i] == 0]
    rows = []
    for t in range(g.N):
        rows.append([g.br(x, e)[t] for x in sfull]); rows.append([g.br(x, f)[t] for x in sfull])
    cen = []
    for co in kernel(rows, len(sfull)):
        v = [F(0)] * g.N
        for c, x in zip(co, sfull):
            if c: v = add(v, sm(c, x))
        cen.append(v)
    sfull = cen                       # centralizer of the sl2 factor in g_ev = s
    # s itself (g(2) or o(7)): the raising operators of s = positive root vectors of s
    pos = [vec_of(g, i) for i in range(g.N) if g.PAR[i] == 0 and any(g.WT[i]) and g.WT[i][0] == 0
           and next(x for x in g.WT[i][1:] if x != 0) > 0]
    for nm, kk, rr in (("", [], []), (" + Ce", [e], [e])):
        cases.append(("s" + nm, sfull + kk, pos + rr))
    if which == "F4":
        # g(2) = stabilizer in o(7) of the sl2_pr-invariant spinor
        hv = highest(g, [E, Fv])          # killed by E and F: invariants (+ e-part)
        # choose an odd vector killed by E, F and e (spinor (x) highest line)
        inv = [v for v in highest(g, [E, Fv, e])]
        assert inv, "no invariant spinor"
        w = inv[0]
        o7 = [vec_of(g, i) for i in range(g.N) if g.PAR[i] == 0]
        # restrict to o(7): even part minus sl2 factor -> find X in g_ev with [X,w]=0 and [X,e]=0,[X,f]=0
        rows = []
        for t in range(g.N):
            rows.append([g.br(x, w)[t] for x in o7])
            rows.append([g.br(x, e)[t] for x in o7])
            rows.append([g.br(x, f)[t] for x in o7])
        G2 = []
        for co in kernel(rows, len(o7)):
            v = [F(0)] * g.N
            for c, x in zip(co, o7):
                if c: v = add(v, sm(c, x))
            G2.append(v)
        print("   stabilizer of the invariant spinor in o(7): dim", len(G2))
        assert len(G2) == 14
        # raising ops of g2: we use E (principal, in g2) -- highest vectors wrt full g2 are
        # those killed by g2's positive part; use all of g2 and ask for invariants/irreducibles via E only
        cases.append(("g2", G2, [E]))
        cases.append(("g2 + Ch", G2 + [h], [E]))
        cases.append(("g2 + b", G2 + [h, e], [E, e]))
        cases.append(("g2 + sl2", G2 + [e, h, f], [E, e]))
    rnd = random.Random(11)
    seen = {}
    for name, uev, rais in cases:
        assert closure(g, uev).dim() == Ech(g.N, uev).dim(), ("u_ev not closed", name)
        hv = highest(g, rais)
        gens = []
        # every nonempty subset of the highest vectors, and generic combinations of pairs
        for mask in range(1, 1 << len(hv)):
            gens.append([hv[i] for i in range(len(hv)) if mask >> i & 1])
        for i in range(len(hv)):
            for j in range(i + 1, len(hv)):
                gens.append([add(hv[i], sm(F(rnd.randint(2, 30)), hv[j]))])
        if hv:
            z = [F(0)] * g.N
            for x in hv: z = add(z, sm(F(rnd.randint(1, 40)), x))
            gens.append([z])
        res = {}
        for gg in gens:
            if not gg: continue
            A = closure(g, uev + gg)
            if A.dim() == g.N:
                res.setdefault("g", 0); res["g"] += 1; continue
            sd = sdim_of(g, A.basis())
            rk = toral_rank(g, A)
            key = (sd, rk)
            if key in res: continue
            try:
                v, info = test(g, A.basis())
            except AssertionError:
                v = "test inconclusive (repeated eigenvalue)"
            if v != "MAX":
                w = witness(g, A)
                if w: v += "; contained in a proper subalgebra of sdim %s" % (w,)
            res[key] = v
        print("  u_ev = %-14s  #highest odd = %d" % (name, len(hv)))
        for k, v in res.items():
            if k == "g": print("      -> generates g  (%d generators)" % v)
            else: print("      -> A of sdim %s, rank %d : %s" % (k[0], k[1], v))

if __name__ == "__main__":
    for w in sys.argv[1:] or ["G3", "F4"]:
        run(w)
