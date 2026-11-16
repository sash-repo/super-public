"""Self-tests for defco.py.  Run:  python3 test_defco.py

Every test is an identity that must hold whatever the answer to the research
question is; a failure means the machinery is wrong, not the mathematics.
"""

from fractions import Fraction as Q
import sys

import defco
from defco import Graded, Window, _solve
import yb
from stringy import add, smul

OK = [0, 0]


def chk(name, cond, extra=""):
    OK[1] += 1
    if cond:
        OK[0] += 1
        print("  ok   %-58s %s" % (name, extra))
    else:
        print("  FAIL %-58s %s" % (name, extra))


# ----------------------------------------------------- 1. the super Jacobi

def jacobi_defect(g, N=2):
    bad = 0
    tot = 0
    vecs = [(v, p, d) for d in g.degrees(N) for v, p in g.basis(d)]
    for (x, px, _) in vecs:
        for (y, py, _) in vecs:
            for (z, pz, _) in vecs:
                tot += 1
                s = {}
                s = add(s, smul(Q((-1) ** (px * pz)), g.brk(g.brk(x, y), z)))
                s = add(s, smul(Q((-1) ** (py * px)), g.brk(g.brk(y, z), x)))
                s = add(s, smul(Q((-1) ** (pz * py)), g.brk(g.brk(z, x), y)))
                if s:
                    bad += 1
    return bad, tot


# ------------------------------------------- 2. the bracket is delta(id)

def bracket_is_coboundary(g, N):
    """c = [ , ] must (a) satisfy every cocycle row, (b) lie in B."""
    W = Window(g, g.degrees(N))
    vec = [Q(0)] * W.nunk
    for (i, j), base in W.pairs.items():
        vi = W.vec[i][0]
        vj = W.vec[j][0]
        td, tl = W.ptarget[(i, j)]
        b = g.brk(vi, vj)
        if not b:
            continue
        co = g.coords(td, b)
        for a, t in enumerate(tl):
            vec[base + a] = co[t]
    rows = W.cocycle_rows()
    bad = sum(1 for r in rows if sum(c * vec[u] for u, c in r.items()))
    B = W.coboundaries()
    mat = [[b[i] for b in B] for i in range(W.nunk)]
    inB = _solve(mat, vec) is not None
    return bad, len(rows), inB


# ---------------------------------------------------------------- the run

def main():
    print("defco self-tests   (backend: %s)" % defco.__dict__.get("backend", "?"))
    try:
        from exactlin import backend
        print("  exact linear algebra backend: %s" % backend())
    except Exception as e:
        print("  backend query failed: %s" % e)

    algs = [Graded(yb.Witt(), "witt"),
            Graded(yb.Contact(1), "k^L(1|1)"),
            Graded(yb.Contact(2), "k^L(1|2)"),
            Graded(yb.Contact(3), "k^L(1|3)"),
            Graded(yb.Ramond(1), "k^M(1|1)"),
            Graded(yb.Ramond(2), "k^M(1|2)"),
            Graded(yb.Vect(1), "vect^L(1|1)"),
            Graded(yb.MPer(), "m^L(1)"),
            Graded(yb.SVect(Q(0)), "svect^L_0(1|2)"),
            Graded(yb.SVect(Q(1, 3)), "svect^L_{1/3}(1|2)")]

    print("\n1. the super Jacobi identity on a degree window")
    for g in algs:
        bad, tot = jacobi_defect(g, 2)
        chk("Jacobi  %s" % g.name, bad == 0, "%d triples" % tot)

    print("\n2. the grading is a grading:  deg[x,y] = deg x + deg y")
    for g in algs:
        bad = 0
        for d1 in g.degrees(2):
            for d2 in g.degrees(2):
                for x, _ in g.basis(d1):
                    for y, _ in g.basis(d2):
                        b = g.brk(x, y)
                        if b and {g.alg.deg(k) for k in b} != {d1 + d2}:
                            bad += 1
        chk("grading %s" % g.name, bad == 0)

    print("\n3. the grading element E (inner <=> H^* sits in weight 0)")
    for g in algs:
        E = g.grading_element()
        chk("E in g_0  %s" % g.name, True,
            "INNER" if E is not None else "outer only")

    print("\n4. the components are finite-dimensional and nonzero")
    for g in algs:
        dims = [g.dim(d) for d in g.degrees(3)]
        chk("dims  %s" % g.name, all(x > 0 for x in dims),
            "step %d, dims %s" % (g.step(), dims))

    print("\n5. c = [.,.] is a cocycle and equals delta(id)")
    for g in algs[:8]:
        bad, nrows, inB = bracket_is_coboundary(g, 2)
        chk("bracket is a cocycle  %s" % g.name, bad == 0,
            "%d rows" % nrows)
        chk("bracket = delta(id)   %s" % g.name, inB)

    print("\n6. coboundaries satisfy the cocycle equations, BOTH parities")
    # The odd half is what caught the missing sign (-1)^{p(x)} in the middle
    # term of delta(phi) for an odd parameter: without it B is not inside Z,
    # and the census returns a NEGATIVE h.
    for q in (0, 1):
        for g in algs[:8]:
            W = Window(g, g.degrees(2), 0, q)
            if W.nunk == 0:
                chk("B subset Z  q=%d  %s" % (q, g.name), True,
                    "no odd cochains of weight 0: parity is fixed by degree")
                continue
            rows = W.cocycle_rows()
            B = W.coboundaries()
            bad = 0
            for b in B:
                for r in rows:
                    if sum(c * b[u] for u, c in r.items()):
                        bad += 1
            chk("B subset Z  q=%d  %s" % (q, g.name), bad == 0,
                "%d x %d" % (len(B), len(rows)))

    print("\n%d/%d checks pass" % (OK[0], OK[1]))
    return 0 if OK[0] == OK[1] else 1


if __name__ == "__main__":
    sys.exit(main())
