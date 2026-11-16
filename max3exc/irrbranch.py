#!/usr/bin/env python3
"""The irreducible branch of case 3o for the two CONTACT gradings:
   g_0 = c(g(2)), g_{-1} = Pi(V_7)   [ag(2) < k(1|7)]
   g_0 = c(o(7)), g_{-1} = Pi(spin_8) [ab(3) < k(1|8)]

The only proper almost simple candidate for an irreducible u_0 is the principal
sl(2) of s (all other maximal subalgebras of g(2), o(7) are reducible on V_7,
resp. spin_8 -- Dynkin).  We construct it, check irreducibility on g_{-1} and
compute the first prolongation (g_{-1}, u_0)_1: if it vanishes then
u = g_{-1} + u_0 is contained in g_{<=0} and hence is not maximal.
"""
from fractions import Fraction as F
import run3
from lin import kernel
from sweep3 import Ech, Grading, ev, sdim_of, closure

SIMPLE_ROOTS = {
    # in the coordinates (delta, e_1, e_2) of ag(2) and (delta,e1,e2,e3) of ab(3)
    "G3": [(0, 1, 0), (0, -1, 1)],
    "F4": [(0, 1, -1), (0, 0, 1), (0, 0, 0)],   # filled in below for o(7)
}


def root_index(g, wt):
    wt = tuple(F(x) for x in wt)
    for i in range(g.N):
        if tuple(g.WT[i]) == wt:
            return i
    return None


def cartan_indices(g):
    return [i for i in range(g.N) if all(x == 0 for x in g.WT[i])]


def solve_H(g, Es):
    """H in the Cartan with [H, E] = 2E for each E in Es."""
    cart = cartan_indices(g)
    rows = []
    for E in Es:
        for t in range(g.N):
            rows.append([g.br(ev(g, c), E)[t] - 2 * E[t] for c in cart])
    # inhomogeneous: solve exactly by adding a slack column
    rows2 = []
    for E in Es:
        for t in range(g.N):
            rows2.append([g.br(ev(g, c), E)[t] for c in cart] + [-2 * E[t]])
    for sol in kernel(rows2, len(cart) + 1):
        if sol[-1] != 0:
            sc = F(1) / sol[-1]
            H = [F(0)] * g.N
            for cf, c in zip(sol[:-1], cart):
                if cf:
                    H = [x + cf * sc * y for x, y in zip(H, ev(g, c))]
            return H
    return None


def principal_sl2(g, simple):
    """(E,H,F) of the principal sl(2) of s, from the simple root vectors."""
    Eidx = [root_index(g, a) for a in simple]
    assert all(i is not None for i in Eidx), ("no root vector", simple, Eidx)
    Fidx = [root_index(g, tuple(-x for x in a)) for a in simple]
    E = [F(0)] * g.N
    for i in Eidx:
        E = [x + y for x, y in zip(E, ev(g, i))]
    H = solve_H(g, [ev(g, i) for i in Eidx])
    assert H is not None, "no principal H"
    # F = sum a_i F_i with [E,F] = H
    rows = []
    for t in range(g.N):
        rows.append([g.br(E, ev(g, i))[t] for i in Fidx] + [-H[t]])
    Fv = None
    for sol in kernel(rows, len(Fidx) + 1):
        if sol[-1] != 0:
            sc = F(1) / sol[-1]
            Fv = [F(0)] * g.N
            for cf, i in zip(sol[:-1], Fidx):
                if cf:
                    Fv = [x + cf * sc * y for x, y in zip(Fv, ev(g, i))]
            break
    assert Fv is not None, "no F"
    # checks
    assert g.br(E, Fv) == H
    assert g.br(H, E) == [2 * x for x in E]
    assert g.br(H, Fv) == [-2 * x for x in Fv]
    return E, H, Fv


def irreducible_on(g, ops, space):
    """is span(space) irreducible under the sl(2) with raising operator ops[0]?
       (for sl(2): irreducible <=> the kernel of E on the space is 1-dim)"""
    Sb = Ech(g.N, space).basis()
    rows = []
    for t in range(g.N):
        rows.append([g.br(ops[0], x)[t] for x in Sb])
    return len(kernel(rows, len(Sb)))


def prolong1(g, gr, u0):
    """u_1 = {x in g_1 : [x, g_{-1}] subset u_0}."""
    U = Ech(g.N, u0)
    cur = gr.vecs(1)
    conds = []
    for y in gr.vecs(-1):
        for x in cur:
            red = U.reduce(g.br(x, y))
            conds.append(red)
    # rows: for each (y,t) a linear condition on the coefficients of x
    rows = []
    for y in gr.vecs(-1):
        red = [U.reduce(g.br(x, y)) for x in cur]
        for t in range(g.N):
            rows.append([red[k][t] for k in range(len(cur))])
    ker = kernel(rows, len(cur))
    out = []
    for co in ker:
        v = [F(0)] * g.N
        for cf, x in zip(co, cur):
            if cf:
                v = [p + cf * q for p, q in zip(v, x)]
        out.append(v)
    return Ech(g.N, out)


def run(which, h, simple):
    g = run3.model(which)
    gr = Grading(g, h)
    print("\n%s  contact grading h=%s : %s" % (g.name, h, gr.shape()))
    E, H, Fv = principal_sl2(g, simple)
    sl2 = [E, H, Fv]
    print("   principal sl(2) of s: built and checked ([E,F]=H, [H,E]=2E)")
    k = irreducible_on(g, sl2, gr.vecs(-1))
    print("   dim ker(E) on g_{-1} = %d  ==> g_{-1} is %s"
          % (k, "IRREDUCIBLE over the principal sl(2)" if k == 1
             else "reducible (%d highest vectors)" % k))
    z = grading_element(g, gr)
    for name, u0 in (("u_0 = sl(2)", sl2), ("u_0 = c(sl(2))", sl2 + [z])):
        u1 = prolong1(g, gr, u0)
        print("   %-16s : dim u_1 = %d %s" % (name, u1.dim(),
              "==> u = g_{-1}+u_0 is inside g_{<=0}, NOT maximal"
              if u1.dim() == 0 else "<== nonzero, needs a closer look"))
    return g, gr, sl2


def grading_element(g, gr):
    """the element of the Cartan defining the grading."""
    cart = cartan_indices(g)
    rows = []
    for i in range(g.N):
        d = gr.deg[i]
        for c in cart:
            pass
    # z = sum c_k h_k with [z, x] = deg(x) x for all basis x
    rows = []
    for i in range(g.N):
        row = []
        for c in cart:
            v = g.br(ev(g, c), ev(g, i))
            row.append(v[i])
        rows.append(row + [-gr.deg[i]])
    for sol in kernel(rows, len(cart) + 1):
        if sol[-1] != 0:
            sc = F(1) / sol[-1]
            z = [F(0)] * g.N
            for cf, c in zip(sol[:-1], cart):
                if cf:
                    z = [x + cf * sc * y for x, y in zip(z, ev(g, c))]
            return z
    raise RuntimeError("no grading element")


if __name__ == "__main__":
    # ag(2): simple roots of g(2) in the (e_1,e_2)-coordinates: alpha = e_1
    # (short), beta = e_2 - e_1 (long)
    run("G3", (1, 0, 0), [(0, 1, 0), (0, -1, 1)])
    # ab(3): simple roots of o(7): e_1-e_2, e_2-e_3, e_3
    run("F4", (2, 0, 0, 0), [(0, 1, -1, 0), (0, 0, 1, -1), (0, 0, 0, 1)])
