#!/usr/bin/env python3
"""runrest.py -- the checks of record for the restricted cochain complex in ALL degrees, p = 2.

For every sample:
   the axioms (Jacobi, the squaring axioms, innerness of the power maps);
   d^2 = 0 on the resolution in all degrees computed;
   H^n of the explicit complex == Ext^n_{u(g)}(K;M), the latter computed
   independently from a free resolution over the finite dimensional u (resco.py).

Sections:
   A) 2-structures on Lie algebras;
   B) 2|4-structures on Lie superalgebras (and u_{2|2} = u_{2|4});
   C) (2,4)-structures, through the restricted closure u_{2,4}(g) = u_2(L);
   D) a (2,-)-structure, where u is infinite dimensional;
   E) the counterexample to the completeness of the complex of [EM] in degrees >= 4.

Usage:  python runrest.py      (writes runrest.log)
"""

import os
import sys
import time

from restcx import RLA, Closure
import dpco
import resco

LOG = []


def out(s=""):
    print(s)
    LOG.append(s)
    sys.stdout.flush()


def mkbr(dim, pairs):
    br = [[0] * dim for _ in range(dim)]
    for (i, j), v in pairs.items():
        br[i][j] = v
        br[j][i] = v
    return br


def bit(*ks):
    v = 0
    for k in ks:
        v |= 1 << k
    return v


def report(name, g, maxdeg=5, ext_len=5, do_ext=True):
    out("-" * 78)
    out(name)
    bad = g.check()
    out("   axioms: %s" % ("OK" if not bad else "FAILED " + str(bad[:3])))
    nb = g.dsquared(maxdeg)
    out("   d^2 = 0 on the resolution, degrees <= %d: %s"
        % (maxdeg, "OK" if nb == 0 else "FAILED (%d monomials)" % nb))
    for (mn, M) in [("K", g.trivial()), ("g", g.adjoint())]:
        rb = g.check_restricted(M)
        h = g.cohom(M, maxdeg - 1)
        line = "   M = %-2s  H^n = %s" % (mn, h)
        if do_ext and g.nfree == 0:
            u, e = g.ext(M, ext_len)
            ok = list(h[:len(e)]) == list(e[:len(h)])
            line += "   Ext^n_u = %s (dim u = %d)  ==> %s" % (
                e, len(u.basis), "AGREE" if ok else "MISMATCH")
        if rb:
            line += "   [module NOT restricted: %s]" % rb[:1]
        out(line)


# ============================================================ A) 2-structures
def part_A():
    out("=" * 78)
    out("A)  2-structures on Lie algebras:  C = Gamma(g),  delta(gamma_2(x)) = x^{[2]}")
    out("=" * 78)
    report("A1  abelian dim 1, x^[2] = 0   (u = K[x]/(x^2))",
           RLA(1, mkbr(1, {}), 0, [0]))
    report("A2  abelian dim 2, x^[2] = 0   (u = Lambda(x,y))",
           RLA(2, mkbr(2, {}), 0, [0, 0]))
    report("A3  toral line, x^[2] = x",
           RLA(1, mkbr(1, {}), 0, [bit(0)]))
    report("A4  [x,y] = y, x^[2] = x, y^[2] = 0",
           RLA(2, mkbr(2, {(0, 1): bit(1)}), 0, [bit(0), 0]))
    report("A5  Heisenberg dim 3, [x,y] = z, all [2] = 0",
           RLA(3, mkbr(3, {(0, 1): bit(2)}), 0, [0, 0, 0]))
    report("A6  sl(2) for p = 2 = o^(1)(3): [h,x]=x, [h,y]=y, [x,y]=h; "
           "h^[2]=h, x^[2]=y^[2]=0",
           RLA(3, mkbr(3, {(0, 1): bit(1), (0, 2): bit(2), (1, 2): bit(0)}),
               0, [bit(0), 0, 0]), maxdeg=4, ext_len=4)


# ====================================================== B) 2|4-structures
def part_B():
    out()
    out("=" * 78)
    out("B)  2|4-structures on Lie superalgebras: the same complex, the squaring")
    out("    s on g_od and the [2] on g_ev being the one map of the theorem")
    out("=" * 78)
    report("B1  sdim (1|1), s(f) = e, e^[2] = 0",
           RLA(2, mkbr(2, {}), 0, [0, bit(0)]))
    report("B2  sdim (1|2), [f1,f2] = e, s(f_i) = 0, e^[2] = 0",
           RLA(3, mkbr(3, {(1, 2): bit(0)}), 0, [0, 0, 0]))
    report("B3  gl(1|1): e0 central, [e1,f0]=f0, [e1,f1]=f1, [f0,f1]=e0, "
           "e1^[2]=e1",
           RLA(4, mkbr(4, {(1, 2): bit(2), (1, 3): bit(3), (2, 3): bit(0)}),
               0, [0, bit(1), 0, 0]), maxdeg=4, ext_len=4)
    # u_{2|2} = u_{2|4}
    out("-" * 78)
    out("u_{2|2}(g) = u_{2|4}(g): the relation x^2 = x^{[2]} of [BLLS, (5)] holds")
    out("in u_{2|4}(g) for EVERY x, homogeneous or not")
    cases = [("sdim (1|1), s(f)=e", 2, mkbr(2, {}), [0, bit(0)]),
             ("gl(1|1)", 4, mkbr(4, {(1, 2): bit(2), (1, 3): bit(3),
                                     (2, 3): bit(0)}), [0, bit(1), 0, 0]),
             ("sdim (1|2), [f1,f2]=e", 3, mkbr(3, {(1, 2): bit(0)}), [0, 0, 0])]
    for (nm, dim, br, pmap) in cases:
        u = resco.RestAlg(dim, br, [2] * dim, pmap)
        bad = 0
        for x in range(1, 1 << dim):
            sq = 0
            for i in dpco.bits(x):
                for j in dpco.bits(x):
                    for w in u.reduce((i, j)):
                        sq ^= 1 << u.index[w]
            v = 0
            bs = dpco.bits(x)
            for i in bs:
                v ^= pmap[i]
            for a in range(len(bs)):
                for b in range(a + 1, len(bs)):
                    v ^= br[bs[a]][bs[b]]
            pw = 0
            for k in dpco.bits(v):
                pw ^= 1 << u.index[(k,)]
            if sq != pw:
                bad += 1
        out("   %-24s dim u = %4d   all %4d nonzero x: %s"
            % (nm, len(u.basis), (1 << dim) - 1,
               "OK" if bad == 0 else "FAILED (%d)" % bad))


# ================================================== C) (2,4)-structures
def part_C():
    out()
    out("=" * 78)
    out("C)  (2,4)-structures: the restricted closure  L = g_+ (+) g_- (+) g_-^{(2)}")
    out("    computed inside u_{2,4}(g), where the (2,4)-structure becomes a")
    out("    2-structure and  u_{2,4}(g) = u_2(L)")
    out("=" * 78)
    cases = []
    cases.append(("C1  abelian: g_+ = <x>, g_- = <y>, all power maps 0",
                  1, 1, mkbr(2, {}), [0], [0], 5))
    cases.append(("C2  abelian: g_+ = <x1,x2>, g_- = <y>",
                  2, 1, mkbr(3, {}), [0, 0], [0], 4))
    # o^(1)(3) = [BLLS, 2.6.1 Ex. 1], n = 1: H in g_+, X, Y in g_-
    cases.append(("C3  o^(1)(3): [H,X]=X, [H,Y]=Y, [X,Y]=H; H^[2]=H, X^[4]=Y^[4]=0",
                  1, 2, mkbr(3, {(0, 1): bit(1), (0, 2): bit(2),
                                 (1, 2): bit(0)}), [bit(0)], [0, 0], 4))
    for (nm, dimp, dimm, br, p2, p4, md) in cases:
        out("-" * 78)
        out(nm)
        cl = Closure(dimp, dimm, br, p2, p4)
        out("   dim u_{2,4}(g) = %d  (expected 2^%d 4^%d = %d);  dim L = %d"
            % (len(cl.u.basis), dimp, dimm, 2 ** dimp * 4 ** dimm, cl.dimL))
        out("   L closed under bracket and square: %s"
            % ("OK" if not cl.problems else "FAILED " + str(cl.problems[:3])))
        L = cl.rla()
        bad = L.check()
        out("   L is a Lie algebra with a 2-structure: %s"
            % ("OK" if not bad else "FAILED " + str(bad[:3])))
        out("   d^2 = 0 on the resolution over u_2(L), degrees <= %d: %s"
            % (md, "OK" if L.dsquared(md) == 0 else "FAILED"))
        for (mn, Mdim, act_g) in [("K", 1, lambda k, v: 0),
                                  ("g", dimp + dimm,
                                   lambda k, v: _adj(br, k, v))]:
            ML = cl.module_L(Mdim, act_g)
            h = L.cohom(ML, md - 1)
            e = cl.ext_over_u24(Mdim, act_g, md)
            ok = list(h[:len(e)]) == list(e[:len(h)])
            out("   M = %-2s  H^n(Gamma(L)-complex) = %s   "
                "Ext^n_{u_{2,4}(g)} = %s   ==> %s"
                % (mn, h, e, "AGREE" if ok else "MISMATCH"))


def _adj(br, k, v):
    out_ = 0
    for i in dpco.bits(v):
        out_ ^= br[k][i]
    return out_


def part_C_hI():
    """h_I(3;1): the smallest algebra with a (2,4)-structure and no
    2-structure.  dim u_{2,4} = 2048, so no independent Ext here; the numbers
    are those of the complex, whose correctness is what parts A-C check."""
    out()
    out("=" * 78)
    out("C4  h_I(3;One), dim 7: the restricted cohomology of a (2,4)-structure")
    out("=" * 78)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hI
    dim, br0, grading, two_map, four_map, names = hI.build(3)
    plus = [i for i in range(dim) if grading[i] == 0]
    minus = [i for i in range(dim) if grading[i] == 1]
    order = plus + minus
    pos = {old: new for new, old in enumerate(order)}

    def relab(v):
        w = 0
        for i in dpco.bits(v):
            w |= 1 << pos[i]
        return w
    br = [[0] * dim for _ in range(dim)]
    for a, i in enumerate(order):
        for b, j in enumerate(order):
            br[a][b] = relab(br0[i][j])
    p2 = [relab(two_map[i]) for i in plus]
    p4 = [relab(four_map[i]) for i in minus]
    out("   dim g_+ = %d, dim g_- = %d;  basis: %s"
        % (len(plus), len(minus), ", ".join(names[i] for i in order)))
    cl = Closure(len(plus), len(minus), br, p2, p4)
    out("   dim u_{2,4} = %d = 2^%d 4^%d;  dim L = %d;  L closed: %s"
        % (len(cl.u.basis), len(plus), len(minus), cl.dimL,
           "OK" if not cl.problems else "FAILED"))
    L = cl.rla()
    bad = L.check()
    out("   L is a Lie algebra with a 2-structure: %s"
        % ("OK" if not bad else "FAILED " + str(bad[:3])))
    t0 = time.time()
    out("   d^2 = 0 on the resolution, degrees <= 3: %s  [%.1fs]"
        % ("OK" if L.dsquared(3) == 0 else "FAILED", time.time() - t0))
    for (mn, Mdim, act_g) in [("K", 1, lambda k, v: 0),
                              ("g", dim, lambda k, v: _adj(br, k, v))]:
        ML = cl.module_L(Mdim, act_g)
        t0 = time.time()
        h = L.cohom(ML, 2)
        out("   M = %-2s  H^n_res(h_I(3;One);M) = %s   [%.0fs]"
            % (mn, h, time.time() - t0))


# ============================================== D) a (2,-)-structure
def part_D():
    out()
    out("=" * 78)
    out("D)  a (2,-)-structure, [BLLS, 2.6.1 Ex. 3]: g = <a_n (n in Z), b>,")
    out("    [b,a_n] = a_{n-1}, deg a_n = n mod 2, deg b = 1; the [2] on g_+ is")
    out("    zero and b has no [4].  Here u is INFINITE dimensional, and the")
    out("    complex is Gamma(g_+) (x) Lambda(g_-).")
    out("=" * 78)
    N = 3                       # the quotient by <a_n | n < -N>
    idx = {}
    order = []
    # height-infinity part first: g_- = <a_n, n odd> + <b>
    for n in range(-N, N + 1):
        if n % 2 != 0:
            idx[("a", n)] = len(order)
            order.append(("a", n))
    idx[("b",)] = len(order)
    order.append(("b",))
    nfree = len(order)
    for n in range(-N, N + 1):
        if n % 2 == 0:
            idx[("a", n)] = len(order)
            order.append(("a", n))
    dim = len(order)
    br = [[0] * dim for _ in range(dim)]
    ib = idx[("b",)]
    for n in range(-N, N + 1):
        if ("a", n - 1) in idx:
            i, j = ib, idx[("a", n)]
            br[i][j] ^= 1 << idx[("a", n - 1)]
            br[j][i] = br[i][j]
    g = RLA(dim, br, nfree, [0] * (dim - nfree))
    report("D1  the truncation at N = %d, dim = %d, dim g_+ = %d"
           % (N, dim, dim - nfree), g, maxdeg=4, do_ext=False)


# =============================== E) the complex of [EM] is not complete
def part_E():
    out()
    out("=" * 78)
    out("E)  the restricted complex of [EM] = arXiv:2504.05929, Subsec. 4.4:")
    out("    an n-cochain there is a pair (phi, omega), phi in Lambda^n(g)^* (x) M")
    out("    and omega: g x Lambda^{n-2}(g) -> M.  Its dimension in degree n is")
    out("    dim Lambda^n(g) + dim g . dim Lambda^{n-2}(g), which for g of")
    out("    dimension 1 vanishes for n >= 4, whereas Ext^n_{u(g)}(K;K) = K for")
    out("    every n.  So that complex, complete in degrees <= 3, cannot compute")
    out("    Hochschild's restricted cohomology in higher degrees.")
    out("=" * 78)
    from math import comb
    for (nm, d) in [("dim g = 1", 1), ("dim g = 2", 2), ("dim g = 3", 3)]:
        g = RLA(d, mkbr(d, {}), 0, [0] * d)
        h = g.cohom(g.trivial(), 6)
        em = [comb(d, n) + (d * comb(d, n - 2) if n >= 2 else 0)
              for n in range(7)]
        ours = [len(g.res.ce.basis(n)) for n in range(7)]
        out("   %-10s (abelian, [2]=0)   Ext^n = H^n = %s" % (nm, h))
        out("              dim C^n(ours) = %s" % ours)
        out("              dim C^n([EM])  = %s" % em)


def main():
    out("runrest.py -- the restricted cochain complex in all degrees, p = 2")
    out("17 August 2026;  everything exact over F_2")
    out()
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    part_C_hI()
    with open("runrest.log", "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
