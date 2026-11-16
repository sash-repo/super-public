"""runres.py -- the checks of record for the restricted half.

  A) the Ext machinery over a finite dimensional algebra, against values known
     in closed form (exterior algebra, a toral generator, a truncated
     polynomial algebra of height 4);
  B) Ext^1_{u(g)}(K,M) against Hochschild's restricted 1-cocycles;
  C) the (2,4)-enveloping algebra u_{2,4}(g) of h_I(3;1): centrality of the
     defining elements in U(g), the size 2^{dim g_+} 4^{dim g_-}, sampled
     associativity, and -- the point -- basis-independence of the defining
     ideal, i.e. the additivity rule (add4) holding in the quotient.
"""

import random

from dpco import SLA2, Env, bits, span_int, in_span, rank2_int
from resco import RestAlg, resolution, ext_dims, module_action
import hI

LOG = open("runres.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ------------------------------------------------------------------- part A

def ext_trivial(A, length=5):
    maps = resolution(A, length)

    def Mact(a, v):
        # the trivial module: only the empty word acts as the identity
        return v if A.basis[a] == () else 0
    return ext_dims(A, maps, Mact, 1, length)


def partA():
    say("=" * 74)
    say("A) Ext^n_A(K,K) over finite dimensional algebras, against closed forms")
    say("=" * 74)
    # Lambda(x,y): a 2-structure on the abelian 2-dim Lie algebra, [2] = 0
    A = RestAlg(2, [[0, 0], [0, 0]], [2, 2], [0, 0], "Lambda(x,y)")
    got = ext_trivial(A, 5)
    say("   u = Lambda(x,y)  (abelian, x^{[2]} = 0):  dim u =", len(A.basis),
        " Ext^n =", got, " expected [1,2,3,4,5]",
        "OK" if got == [1, 2, 3, 4, 5] else "MISMATCH")
    # a toral generator: u = K[x]/(x^2+x) = K x K
    A = RestAlg(1, [[0]], [2], [1], "K[x]/(x^2+x)")
    got = ext_trivial(A, 4)
    say("   u = K[x]/(x^2+x)  (x^{[2]} = x):          dim u =", len(A.basis),
        " Ext^n =", got, " expected [1,0,0,0]",
        "OK" if got == [1, 0, 0, 0] else "MISMATCH")
    # a height-4 generator: the (2,4)-case in its simplest form
    A = RestAlg(1, [[0]], [4], [0], "K[y]/(y^4)")
    got = ext_trivial(A, 5)
    say("   u = K[y]/(y^4)    (y^{[4]} = 0):          dim u =", len(A.basis),
        " Ext^n =", got, " expected [1,1,1,1,1]",
        "OK" if got == [1, 1, 1, 1, 1] else "MISMATCH")
    # the abelian (2,4)-structure: one even, one "minus" generator
    A = RestAlg(2, [[0, 0], [0, 0]], [2, 4], [0, 0], "Lambda(x)(x)K[y]/(y^4)")
    got = ext_trivial(A, 5)
    say("   u = Lambda(x) (x) K[y]/(y^4):             dim u =", len(A.basis),
        " Ext^n =", got, " expected [1,2,3,4,5]",
        "OK" if got == [1, 2, 3, 4, 5] else "MISMATCH")


# ------------------------------------------------------------------- part B

def restricted_cocycles(dim, br, caps, pmap, Mdim, act):
    """{c: g -> M linear | c([x,y]) = x.c(y)+y.c(x),
                           c(x^{[cap]}) = x^{cap-1}. c(x) }, and the inner ones.
    act(k, v): the action of the k-th generator on M."""
    # coordinates: c(g_i) in M, i.e. dim*Mdim unknowns
    def coord(i, t):
        return i * Mdim + t
    rows = []
    for i in range(dim):
        for j in range(i + 1, dim):
            for t in range(Mdim):
                pass
    # build the conditions as linear forms on the unknowns, over F_2
    conds = []
    N = dim * Mdim
    for i in range(dim):
        for j in range(i + 1, dim):
            # c([g_i,g_j]) + g_i.c(g_j) + g_j.c(g_i) = 0, one condition per
            # coordinate of M
            mat = [[0] * N for _ in range(Mdim)]
            for k in bits(br[i][j]):
                for t in range(Mdim):
                    mat[t][coord(k, t)] ^= 1
            for (a, b) in [(i, j), (j, i)]:
                for t in range(Mdim):
                    img = act(a, 1 << t)
                    for s in bits(img):
                        mat[s][coord(b, t)] ^= 1
            conds += mat
    for i in range(dim):
        # c(g_i^{[cap]}) + g_i^{cap-1} . c(g_i) = 0
        mat = [[0] * N for _ in range(Mdim)]
        for k in bits(pmap[i]):
            for t in range(Mdim):
                mat[t][coord(k, t)] ^= 1
        for t in range(Mdim):
            img = 1 << t
            for _ in range(caps[i] - 1):
                img = act(i, img)
            for s in bits(img):
                mat[s][coord(i, t)] ^= 1
        conds += mat
    rowints = []
    for row in conds:
        v = 0
        for j, x in enumerate(row):
            if x:
                v |= 1 << j
        rowints.append(v)
    rk = rank2_int(rowints)
    Z = N - rk
    # inner: c(x) = x.m for m in M
    inner = []
    for t in range(Mdim):
        v = 0
        for i in range(dim):
            img = act(i, 1 << t)
            for s in bits(img):
                v |= 1 << coord(i, s)
        inner.append(v)
    B = rank2_int(inner)
    return Z, B, Z - B


def partB():
    say("")
    say("=" * 74)
    say("B) Ext^1_{u(g)}(K,M) against Hochschild's restricted 1-cocycles")
    say("=" * 74)
    cases = []
    # abelian 2-dim, [2] = 0
    cases.append(("abelian (2-dim), x^{[2]} = 0", 2, [[0, 0], [0, 0]], [2, 2],
                  [0, 0]))
    # 1-dim toral
    cases.append(("toral (1-dim), x^{[2]} = x", 1, [[0]], [2], [1]))
    # the 2-dim non-abelian Lie algebra: [x,y] = y, x^{[2]} = x, y^{[2]} = 0
    cases.append(("non-abelian (2-dim), [x,y]=y", 2, [[0, 2], [2, 0]], [2, 2],
                  [1, 0]))
    # the abelian (2,4)-structure
    cases.append(("(2,4) abelian, caps (2,4)", 2, [[0, 0], [0, 0]], [2, 4],
                  [0, 0]))
    for (name, dim, br, caps, pmap) in cases:
        A = RestAlg(dim, br, caps, pmap, name)
        maps = resolution(A, 3)

        def ad(k, v):
            out = 0
            for i in bits(v):
                out ^= br[k][i]
            return out

        for (mname, Mdim, act) in [("K", 1, lambda k, v: 0),
                                   ("g", dim, ad)]:
            Mact = module_action(A, act, Mdim)
            ext = ext_dims(A, maps, Mact, Mdim, 3)
            Z, B, H = restricted_cocycles(dim, br, caps, pmap, Mdim, act)
            say("   %-32s M = %-1s  dim u = %-4d Ext^1 = %-2d  "
                "restricted Z^1/B^1 = %d/%d = %d  %s"
                % (name, mname, len(A.basis), ext[1], Z, B, H,
                   "OK" if ext[1] == H else "MISMATCH"))


# ------------------------------------------------------------------- part C

def partC():
    say("")
    say("=" * 74)
    say("C) the (2,4)-enveloping algebra of h_I(3;1)")
    say("=" * 74)
    dim, br, grading, two_map, four_map, names = hI.build(3)
    say("   h_I(3;1): dim = %d, basis = %s" % (dim, names))
    say("   grading (0 = g_+, 1 = g_-): %s -- dim g_+ = %d, dim g_- = %d"
        % (grading, grading.count(0), grading.count(1)))
    # is it a Z/2-grading of the bracket?
    bad = 0
    for i in range(dim):
        for j in range(dim):
            want = (grading[i] + grading[j]) % 2
            for k in bits(br[i][j]):
                if grading[k] != want:
                    bad += 1
    say("   the x-degree IS a Z/2-grading of the bracket:",
        "OK" if bad == 0 else "%d violations" % bad)

    caps = [2 if grading[i] == 0 else 4 for i in range(dim)]
    pmap = [two_map.get(i, 0) if grading[i] == 0 else four_map.get(i, 0)
            for i in range(dim)]

    # (i)-(ii) centrality in U(g) of the defining elements
    gg = SLA2(dim, 0, {(i, j): br[i][j] for i in range(dim)
                       for j in range(i + 1, dim) if br[i][j]}, [], "h_I(3;1)")
    env = Env(gg)
    noncentral = 0
    for i in range(dim):
        c = caps[i]
        elt = set(env.reduce((i,) * c))
        for k in bits(pmap[i]):
            elt ^= set(env.reduce((k,)))
        for k in range(dim):
            lhs = set()
            rhs = set()
            for w in elt:
                lhs ^= set(env.reduce(w + (k,)))
                rhs ^= set(env.reduce((k,) + w))
            if lhs != rhs:
                noncentral += 1
    say("   (i)-(ii) x^2+x^{[2]} (x in g_+) and y^4+y^{[4]} (y in g_-) are")
    say("            central in U(g):",
        "OK" if noncentral == 0 else "%d failures" % noncentral)

    # (iv) the size of the quotient
    A = RestAlg(dim, br, caps, pmap, "u_{2,4}(h_I(3;1))")
    expect = 2 ** grading.count(0) * 4 ** grading.count(1)
    say("   (iv) dim u_{2,4} = %d, expected 2^%d 4^%d = %d  %s"
        % (len(A.basis), grading.count(0), grading.count(1), expect,
           "OK" if len(A.basis) == expect else "MISMATCH"))

    # sampled associativity of the quotient
    rng = random.Random(17)
    short = [i for i, w in enumerate(A.basis) if len(w) <= 2]
    bad = 0
    trials = 400
    for _ in range(trials):
        i, j, k = (rng.choice(short) for _ in range(3))
        if A.mul(A.mul(1 << i, 1 << j), 1 << k) != \
           A.mul(1 << i, A.mul(1 << j, 1 << k)):
            bad += 1
    say("   associativity of u_{2,4} on %d random triples of short monomials:"
        % trials, "OK" if bad == 0 else "%d failures" % bad)

    # (iii) basis-independence: (y+z)^4 = (y+z)^{[4]} in u for all y,z in g_-
    def lie_br(x, y):
        out = 0
        for i in bits(x):
            for j in bits(y):
                out ^= br[i][j]
        return out

    def four_of_vector(v):
        """(sum of basis vectors)^{[4]} by iterating the additivity rule"""
        acc, done = 0, 0
        for i in bits(v):
            x, y = done, 1 << i
            s = (lie_br(x, lie_br(x, lie_br(x, y)))
                 ^ lie_br(y, lie_br(y, lie_br(y, x)))
                 ^ lie_br(x, lie_br(y, lie_br(x, y))))
            b = lie_br(x, y)                      # [x,y] lies in g_+
            sq_b = 0
            for t in bits(b):
                sq_b ^= two_map.get(t, 0)
            # the polarization of the 2-structure on g_+
            bl = bits(b)
            for a in range(len(bl)):
                for c in range(a + 1, len(bl)):
                    sq_b ^= br[bl[a]][bl[c]]
            acc = acc ^ four_map.get(i, 0) ^ s ^ sq_b
            done ^= y
        return acc

    minus = [i for i in range(dim) if grading[i] == 1]
    bad = 0
    tested = 0
    for mask in range(1, 1 << len(minus)):
        v = 0
        for a, i in enumerate(minus):
            if mask >> a & 1:
                v |= 1 << i
        # (v)^4 in u
        word_elt = {(): 1}
        elt = {0: None}
        acc = 0
        # expand (sum)^4 = sum over 4-tuples of the summands
        terms = bits(v)
        prod = 0
        for a in terms:
            for b in terms:
                for c in terms:
                    for d in terms:
                        for w in A.reduce((a, b, c, d)):
                            prod ^= 1 << A.index[w]
        rhs = 0
        for t in bits(four_of_vector(v)):
            for w in A.reduce((t,)):
                rhs ^= 1 << A.index[w]
        tested += 1
        if prod != rhs:
            bad += 1
    say("   (iii) (y+z+...)^4 = (y+z+...)^{[4]} in u_{2,4} for all %d non-zero"
        % tested)
    say("         elements of g_-:", "OK" if bad == 0 else "%d failures" % bad)
    say("   => the defining ideal does not depend on the basis of g_-, so")
    say("      u_{2,4}(g) and H^n_{2,4}(g;M) := Ext^n_{u_{2,4}(g)}(K,M) are")
    say("      well defined.")


# ------------------------------------------------------------------- part D

def partD():
    """the two theories side by side, and the kernel Hochschild's sequence
    predicts for H^2_res -> H^2:

        0 -> H^1_res -> H^1 -> Hom_Fr(g;M^g) -> H^2_res -> H^2

    (Hom_Fr = the p-semilinear maps; over F_2 they are the linear ones), so
        dim ker(H^2_res -> H^2) = dim Hom_Fr - dim H^1 + dim H^1_res.
    """
    from dpco import Resolution, cohomology_dims
    say("")
    say("=" * 74)
    say("D) restricted (Ext over u) against ordinary (Ext over U), and the")
    say("   kernel of H^2_res -> H^2 predicted by Hochschild's sequence")
    say("=" * 74)
    cases = [("abelian (2-dim), x^{[2]}=0", 2, [[0, 0], [0, 0]], [2, 2], [0, 0]),
             ("toral (1-dim), x^{[2]}=x", 1, [[0]], [2], [1]),
             ("non-abelian (2-dim), [x,y]=y", 2, [[0, 2], [2, 0]], [2, 2],
              [1, 0])]
    for (name, dim, br, caps, pmap) in cases:
        A = RestAlg(dim, br, caps, pmap, name)
        maps = resolution(A, 3)
        Mact = module_action(A, lambda k, v: 0, 1)
        res_dims = ext_dims(A, maps, Mact, 1, 3)
        g = SLA2(dim, 0, {(i, j): br[i][j] for i in range(dim)
                          for j in range(i + 1, dim) if br[i][j]}, [], name)
        ord_dims = cohomology_dims(Resolution(g), g.trivial_module(), 2)
        semilinear = dim                      # dim Hom_Fr(g; K) over F_2
        ker = semilinear - ord_dims[1] + res_dims[1]
        say("   %-30s H^n_res = %s   H^n = %s" % (name, res_dims, ord_dims))
        say("   %-30s dim Hom_Fr(g;K) = %d, so Hochschild's sequence gives "
            "dim ker(H^2_res -> H^2) = %d" % ("", semilinear, ker))


def main():
    partA()
    partB()
    partC()
    partD()
    LOG.close()


if __name__ == "__main__":
    main()
