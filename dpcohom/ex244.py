#!/usr/bin/env python3
"""ex244.py -- the (2,4|4)-structures, exact over F_2.

1) The restricted closure of a (2,4|4)-structure (extension of Theorem Tclos):
   L = Span(g u {y^2 | y in (g_-)_ev}) inside u_{2,4|4}(g); checked: L closed under
   bracket and square, a squared Lie algebra with L' = L, dim L = dim g + dim (g_-)_ev,
   u_2(L) = u_{2,4|4}(g) (dimensions), and H^n(Gamma(L^*) (x) M) = Ext^n_u(K;M).
2) oo^{(1)}_{I Pi}(1|2) and oo^{(1)}_{I Pi}(3|2): ALL Z/2-gradings defined over F_2
   (= even derivations D with D^2 = D, D|_{g_-} = id, D|_{g_+} = 0), and for each:
   the (2,4)-structure on g_ev, the 2-structure (i.e. a 2|4-structure), etc.

Usage: python ex244.py [quick]
Single log writer: main() writes ex244.log.
"""
import itertools
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), HERE]
sys.setrecursionlimit(100000)

import dpco
import resco
from restcx import RLA
from dpco import bits

LOG = []


def out(s=""):
    print(s)
    sys.stdout.flush()
    LOG.append(s)


# ------------------------------------------------------------ F_2 linear algebra
def solve(cols, target):
    """find c with sum c_i cols[i] = target (ints); None if none."""
    piv = {}
    for a, v in enumerate(cols):
        cur, comb = v, 1 << a
        while cur:
            h = cur.bit_length() - 1
            if h in piv:
                cur ^= piv[h][0]
                comb ^= piv[h][1]
            else:
                piv[h] = (cur, comb)
                break
    cur, comb = target, 0
    while cur:
        h = cur.bit_length() - 1
        if h not in piv:
            return None
        cur ^= piv[h][0]
        comb ^= piv[h][1]
    return comb


def nullspace(rows, ncols):
    """rows: ints over ncols bits; returns basis of {v | row.v = 0 for all rows}"""
    piv = {}
    for r in rows:
        cur = r
        while cur:
            h = cur.bit_length() - 1
            if h in piv:
                cur ^= piv[h]
            else:
                piv[h] = cur
                break
    # full reduction
    keys = sorted(piv)
    for h in keys:
        for h2 in keys:
            if h2 != h and (piv[h2] >> h) & 1:
                piv[h2] ^= piv[h]
    free = [j for j in range(ncols) if j not in piv]
    basis = []
    for f in free:
        v = 1 << f
        for h, r in piv.items():
            if (r >> f) & 1:
                v |= 1 << h
        basis.append(v)
    return basis


def inverse(vecs, n):
    """vecs: n ints (columns = new basis in old coords). returns function old->new coords"""
    def conv(v):
        c = solve(vecs, v)
        assert c is not None
        return c
    return conv


# ------------------------------------------------------------ a Lie superalgebra
class SA:
    """F_2 Lie superalgebra: basis 0..n-1, parity list, br[i][j] bitmask,
    sq[i] bitmask for odd i (None for even)."""

    def __init__(self, par, br, sq, names):
        self.n = len(par)
        self.par, self.br, self.sq, self.names = par, br, sq, names

    def bracket(self, x, y):
        o = 0
        for i in bits(x):
            for j in bits(y):
                o ^= self.br[i][j]
        return o

    def square(self, x):
        """x odd vector"""
        b = bits(x)
        o = 0
        for i in b:
            o ^= self.sq[i]
        for a in range(len(b)):
            for c in range(a + 1, len(b)):
                o ^= self.br[b[a]][b[c]]
        return o

    def ad(self, x):
        """ad_x as list of column images"""
        return [self.bracket(x, 1 << j) for j in range(self.n)]

    def evmask(self):
        return sum(1 << i for i in range(self.n) if self.par[i] == 0)

    def check(self):
        bad = []
        n = self.n
        for i in range(n):
            for j in range(n):
                if self.br[i][j] != self.br[j][i]:
                    bad.append("sym")
                for k in bits(self.br[i][j]):
                    if self.par[k] != (self.par[i] + self.par[j]) % 2:
                        bad.append("parity")
            if self.br[i][i]:
                bad.append("diag")
        for i, j, k in itertools.combinations(range(n), 3):
            if (self.bracket(1 << i, self.br[j][k]) ^ self.bracket(1 << j, self.br[i][k])
                    ^ self.bracket(1 << k, self.br[i][j])):
                bad.append("Jacobi %d%d%d" % (i, j, k))
        odd = [i for i in range(n) if self.par[i]]
        for i in odd:
            for k in bits(self.sq[i]):
                if self.par[k]:
                    bad.append("s odd")
            for y in range(n):
                if self.bracket(self.sq[i], 1 << y) != self.bracket(1 << i, self.br[i][y]):
                    bad.append("[s(x),y]")
        return bad

    def rebase(self, newb, newpar, names):
        """newb: list of ints (old coords) forming a homogeneous basis"""
        conv = inverse(newb, self.n)
        n = self.n
        br = [[conv(self.bracket(newb[i], newb[j])) for j in range(n)] for i in range(n)]
        sq = [conv(self.square(newb[i])) if newpar[i] else None for i in range(n)]
        return SA(newpar, br, sq, names)

    def center(self):
        rows = []
        # z = sum c_i e_i with [z, e_j] = 0 for all j: coefficient matrix
        cols = [self.ad(1 << i) for i in range(self.n)]
        # [z,e_j] = sum_i c_i [e_i,e_j]; constraint per (j,k)
        for j in range(self.n):
            for k in range(self.n):
                r = 0
                for i in range(self.n):
                    if (cols[i][j] >> k) & 1:
                        r |= 1 << i
                rows.append(r)
        return nullspace(rows, self.n)

    def ops(self):
        """the parities and the polynomial data, for printing"""
        lines = []
        nm = self.names
        def vec(v):
            return "+".join(nm[k] for k in bits(v)) if v else "0"
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.br[i][j]:
                    lines.append("[%s,%s]=%s" % (nm[i], nm[j], vec(self.br[i][j])))
        for i in range(self.n):
            if self.par[i]:
                lines.append("s(%s)=%s" % (nm[i], vec(self.sq[i])))
        return lines


def flat(mat):
    """list of column ints (n bits each) -> one int"""
    n = len(mat)
    v = 0
    for j, c in enumerate(mat):
        v |= c << (j * n)
    return v


def matmul(A, B):
    """compose column-lists: (A o B)"""
    out = []
    for c in B:
        o = 0
        for k in bits(c):
            o ^= A[k]
        out.append(o)
    return out


def inner(g, M, cand):
    """M (column list) = ad_z for z in span(cand)? return z or None"""
    cols = [flat(g.ad(z)) for z in cand]
    c = solve(cols, flat(M))
    if c is None:
        return None
    z = 0
    for a in bits(c):
        z ^= cand[a]
    return z


# ------------------------------------------------------------ the gradings
def derivations(g):
    """even derivations D (respecting the squaring): basis of D as column lists"""
    n = g.n
    # unknowns D[k][j] = coefficient of e_k in D(e_j), only same parity
    unk = [(k, j) for j in range(n) for k in range(n) if g.par[k] == g.par[j]]
    uidx = {u: a for a, u in enumerate(unk)}
    N = len(unk)
    rows = []

    def Dvec(j):
        """D(e_j) as dict k -> unknown bitmask (linear forms)"""
        return {k: 1 << uidx[(k, j)] for k in range(n) if (k, j) in uidx}

    def lin_bracket(form_vec, y):
        """[sum_k form_k e_k, e_y] -> dict comp -> form"""
        res = {}
        for k, f in form_vec.items():
            for t in bits(g.br[k][y]):
                res[t] = res.get(t, 0) ^ f
        return res

    def add(a, b):
        r = dict(a)
        for k, f in b.items():
            r[k] = r.get(k, 0) ^ f
        return r

    for i in range(n):
        for j in range(i + 1, n):
            lhs = {}
            for t in bits(g.br[i][j]):
                lhs = add(lhs, Dvec(t))
            rhs = add(lin_bracket(Dvec(i), j), lin_bracket(Dvec(j), i))
            for k, f in add(lhs, rhs).items():
                if f:
                    rows.append(f)
    for i in range(n):
        if g.par[i]:
            lhs = {}
            for t in bits(g.sq[i]):
                lhs = add(lhs, Dvec(t))
            rhs = lin_bracket(Dvec(i), i)
            for k, f in add(lhs, rhs).items():
                if f:
                    rows.append(f)
    sols = nullspace(rows, N)
    Ds = []
    for s in sols:
        cols = [0] * n
        for a in bits(s):
            k, j = unk[a]
            cols[j] |= 1 << k
        Ds.append(cols)
    return Ds


def gradings(g):
    Ds = derivations(g)
    found = []
    for c in range(1 << len(Ds)):
        D = [0] * g.n
        for a in bits(c):
            D = [x ^ y for x, y in zip(D, Ds[a])]
        if matmul(D, D) == D:
            found.append(D)
    return Ds, found


def analyse_grading(g, D):
    """returns dict with the data of the grading"""
    n = g.n
    ev = [1 << i for i in range(n) if g.par[i] == 0]
    od = [1 << i for i in range(n) if g.par[i] == 1]
    # g_- = Im D, g_+ = Ker D, per parity
    def img(vs):
        piv = dpco.span_int([sum((D[k] if (v >> k) & 1 else 0) for k in []) for v in vs])
        return piv
    def image_basis(vs):
        im = []
        piv = {}
        for v in vs:
            w = 0
            for k in bits(v):
                w ^= D[k]
            x = w
            while x:
                h = x.bit_length() - 1
                if h in piv:
                    x ^= piv[h]
                else:
                    piv[h] = x
                    im.append(w)
                    break
        return im
    def kernel_basis(vs):
        # vectors v in span(vs) with D v = 0
        rows = []
        m = len(vs)
        # D(sum c_a vs_a) = 0
        Dv = []
        for v in vs:
            w = 0
            for k in bits(v):
                w ^= D[k]
            Dv.append(w)
        for t in range(n):
            r = 0
            for a in range(m):
                if (Dv[a] >> t) & 1:
                    r |= 1 << a
            rows.append(r)
        ker = []
        for c in nullspace(rows, m):
            w = 0
            for a in bits(c):
                w ^= vs[a]
            ker.append(w)
        return ker
    res = {"pe": kernel_basis(ev), "me": image_basis(ev),
           "po": kernel_basis(od), "mo": image_basis(od)}
    pe = res["pe"]
    # 2-structure on (g_+)_ev:  ad_x^2 = ad_z, z in (g_+)_ev
    two = {}
    for x in pe:
        A = g.ad(x)
        two[x] = inner(g, matmul(A, A), pe)
    four = {}
    sq_minus = {}
    for y in res["me"]:
        A = g.ad(y)
        A2 = matmul(A, A)
        four[y] = inner(g, matmul(A2, A2), pe)
        sq_minus[y] = inner(g, A2, ev)        # is (ad_y)^2 inner at all (in g_ev)?
    # 2|4: a 2-structure on all of g_ev
    full2 = {}
    for x in ev:
        A = g.ad(x)
        full2[x] = inner(g, matmul(A, A), ev)
    res["two"], res["four"], res["sqminus"], res["full2"] = two, four, sq_minus, full2
    # mixed odd brackets: does s(x) for every odd x land in (g_+)_ev?
    mixed = [g.bracket(a, b) for a in res["po"] for b in res["mo"]]
    res["mixed"] = [m for m in mixed if m]
    return res


# ------------------------------------------------------------ the closure
class Closure244:
    """u = u_{2,4|4}(g) for g with basis ordered (g_+)_ev | (g_-)_ev | g_od;
    L = Span(g u {y^2}) computed inside u."""

    def __init__(self, g, npe, nme, p2, p4):
        self.g = g
        n = g.n
        self.npe, self.nme = npe, nme
        caps = [2] * npe + [4] * nme + [2] * (n - npe - nme)
        pmap = list(p2) + list(p4) + [g.sq[i] for i in range(npe + nme, n)]
        self.u = resco.RestAlg(n, g.br, caps, pmap)
        self.Lb = [self._word((i,)) for i in range(n)]
        for j in range(nme):
            self.Lb.append(self._word((npe + j, npe + j)))
        self.dimL = len(self.Lb)
        self.problems = []
        self.lin_indep = dpco.rank2_int(self.Lb) == self.dimL
        dL = self.dimL
        self.br = [[0] * dL for _ in range(dL)]
        for a in range(dL):
            for b in range(a + 1, dL):
                v = self.u.mul(self.Lb[a], self.Lb[b]) ^ self.u.mul(self.Lb[b], self.Lb[a])
                c = solve(self.Lb, v)
                if c is None:
                    self.problems.append("[L%d,L%d] leaves L" % (a, b))
                    c = 0
                self.br[a][b] = self.br[b][a] = c
        self.pmap = []
        for a in range(dL):
            c = solve(self.Lb, self.u.mul(self.Lb[a], self.Lb[a]))
            if c is None:
                self.problems.append("L%d^2 leaves L" % a)
                c = 0
            self.pmap.append(c)

    def _word(self, w):
        o = 0
        for r in self.u.reduce(w):
            o ^= 1 << self.u.index[r]
        return o

    def check_relations(self):
        """central elements x^2+x^[2] (x in (g+)ev), y^4+y^[4] (y in (g-)ev);
        odd x^4 = x^[4] := (s(x))^[2] for x homogeneous AND inhomogeneous odd"""
        return None

    def rla(self):
        return RLA(self.dimL, self.br, 0, self.pmap, "L")

    def module_L(self, Mdim, act_g):
        n, npe = self.g.n, self.npe

        def act(k, v):
            if k < n:
                return act_g(k, v)
            j = k - n
            return act_g(npe + j, act_g(npe + j, v))
        return dpco.Module(Mdim, [[act(k, 1 << i) for k in range(self.dimL)]
                                  for i in range(Mdim)], [0] * Mdim, "M")


def L_axioms(br, pmap, d):
    """squared Lie algebra axioms on a basis: Jacobi, [x^[2],y]=[x,[x,y]]
    (polarization is then automatic for an associative square; checked on pairs)"""
    def bra(x, y):
        o = 0
        for i in bits(x):
            for j in bits(y):
                o ^= br[i][j]
        return o
    bad = 0
    for i, j, k in itertools.combinations(range(d), 3):
        if bra(1 << i, br[j][k]) ^ bra(1 << j, br[i][k]) ^ bra(1 << k, br[i][j]):
            bad += 1
    for i in range(d):
        for y in range(d):
            if bra(pmap[i], 1 << y) != bra(1 << i, br[i][y]):
                bad += 1
    return bad


# ------------------------------------------------------------ the examples
def build_oo(a, b):
    from oo import oo
    from wk import GF
    S, alg, odd = oo(GF(1), a, b, "Pi", True)
    n = S.n
    par = [0] * S.nev + [1] * S.nod
    def tomask(v):
        return sum(1 << k for k, c in enumerate(v) if c)
    br = [[tomask(S.struct[i][j]) for j in range(n)] for i in range(n)]
    sq = [None] * S.nev + [tomask(v) for v in S.sq]
    names = ["e%d" % i for i in range(S.nev)] + ["f%d" % i for i in range(S.nod)]
    # matrices of the basis, reordered like S
    order = [i for i in range(alg.n) if not odd[i]] + [i for i in range(alg.n) if odd[i]]
    mats = [alg.basis[i] for i in order]
    return SA(par, br, sq, names), mats


def vname(g, v):
    return "+".join(g.names[k] for k in bits(v)) if v else "0"


def describe_grading(g, r):
    return ("(g+)ev=<%s> (g-)ev=<%s> (g+)od=<%s> (g-)od=<%s>" % (
        ", ".join(vname(g, v) for v in r["pe"]), ", ".join(vname(g, v) for v in r["me"]),
        ", ".join(vname(g, v) for v in r["po"]), ", ".join(vname(g, v) for v in r["mo"])))


def study(label, g, mats, maxdeg=2, do_ext=True, ext_len=4):
    out("=" * 78)
    out(label)
    out("=" * 78)
    out("   sdim = (%d|%d); axioms: %s; center = %s" % (
        g.par.count(0), g.par.count(1), "OK" if not g.check() else g.check()[:3],
        [vname(g, z) for z in g.center()]))
    for i, M in enumerate(mats):
        out("   %s = %s" % (g.names[i], M))
    for l in g.ops():
        out("   " + l)
    Ds, grs = gradings(g)
    out("   dim Der_ev(g) (respecting s) = %d; Z/2-gradings over F_2 (D^2=D): %d"
        % (len(Ds), len(grs)))
    # classify
    types = {}
    genuine = []
    for D in grs:
        r = analyse_grading(g, D)
        t = (len(r["pe"]), len(r["me"]), len(r["po"]), len(r["mo"]))
        ok24 = all(v is not None for v in r["two"].values()) and \
            all(v is not None for v in r["four"].values())
        ok2 = all(v is not None for v in r["full2"].values())
        key = (t, ok24, len(r["mixed"]) == 0)
        types.setdefault(key, []).append(r)
        if ok24 and t[1] > 0 and len(r["mixed"]) == 0:
            genuine.append(r)
    ok2_any = None
    for key, lst in sorted(types.items()):
        (t, ok24, nomix) = key
        out("   type dims((g+)ev,(g-)ev,(g+)od,(g-)od)=%s  (2,4) on g_ev exists: %s;"
            "  [(g+)od,(g-)od]=0: %s;  #gradings %d" % (t, ok24, nomix, len(lst)))
        for r in lst[:4]:
            out("      " + describe_grading(g, r))
            if r["mixed"]:
                out("        mixed odd brackets land in: %s" % sorted(set(vname(g, m) for m in r["mixed"]))[:4])
            for y, s in r["sqminus"].items():
                out("        (ad_%s)^2 inner (in g_ev)? %s" % (vname(g, y), "yes: ad_" + vname(g, s) if s is not None else "NO"))
        ok2_any = all(v is not None for v in lst[0]["full2"].values())
    out("   2-structure on g_ev compatible with all of g (=> 2|4-structure): %s"
        % ok2_any)
    if ok2_any:
        r0 = analyse_grading(g, [0] * g.n)
        out("      x^[2]: " + "; ".join("%s->%s" % (vname(g, x), vname(g, z)) for x, z in r0["full2"].items()))
    else:
        r0 = analyse_grading(g, [0] * g.n)
        out("      no [2] for: " + ", ".join(vname(g, x) for x, z in r0["full2"].items() if z is None))
    return genuine, types


def cohom_block(label, g, gr, maxdeg, do_ext, ext_len):
    """rebase g along the grading gr, build u, L, compute everything"""
    pe, me, po, mo = gr["pe"], gr["me"], gr["po"], gr["mo"]
    newb = pe + me + po + mo
    par = [0] * (len(pe) + len(me)) + [1] * (len(po) + len(mo))
    names = (["a%d" % i for i in range(len(pe))] + ["y%d" % i for i in range(len(me))]
             + ["f%d" % i for i in range(len(po))] + ["F%d" % i for i in range(len(mo))])
    h = g.rebase(newb, par, names)
    out("-" * 78)
    out(label)
    out("   adapted basis: " + "; ".join("%s=%s" % (names[i], vname(g, newb[i])) for i in range(h.n)))
    out("   (a: (g+)ev, y: (g-)ev, f: (g+)od, F: (g-)od)")
    for l in h.ops():
        out("   " + l)
    npe, nme = len(pe), len(me)
    p2 = [inner(h, matmul(h.ad(1 << i), h.ad(1 << i)), [1 << k for k in range(npe)])
          for i in range(npe)]
    p4 = []
    for j in range(nme):
        A = h.ad(1 << (npe + j))
        A2 = matmul(A, A)
        p4.append(inner(h, matmul(A2, A2), [1 << k for k in range(npe)]))
    assert None not in p2 and None not in p4
    out("   [2]: " + "; ".join("%s^[2]=%s" % (names[i], vname(h, p2[i])) for i in range(npe)))
    out("   [4]: " + "; ".join("%s^[4]=%s" % (names[npe + j], vname(h, p4[j])) for j in range(nme)))
    for i in range(npe + nme, h.n):
        s = h.sq[i]
        # x^[4] := (s(x))^[2]
        x4 = 0
        for k in bits(s):
            assert k < npe, "s(odd) not in (g+)ev"
            x4 ^= p2[k]
        # additivity for [2]: (sum)^[2] = sum + brackets
        sb = s
        x4 = 0
        bs = bits(s)
        for k in bs:
            x4 ^= p2[k]
        for a_ in range(len(bs)):
            for c_ in range(a_ + 1, len(bs)):
                x4 ^= h.br[bs[a_]][bs[c_]]
        out("   %s^[4] := (s(%s))^[2] = %s;  ad check (ad)^4: %s" % (
            names[i], names[i], vname(h, x4),
            "OK" if flat(h.ad(x4)) == flat(matmul(matmul(h.ad(1 << i), h.ad(1 << i)),
                                                    matmul(h.ad(1 << i), h.ad(1 << i)))) else "FAIL"))
    # additivity of [4] (sigma) on all pairs of (g-)ev, and (2,4) axioms on basis
    bad = 0
    def p2v(x):
        b = bits(x)
        o = 0
        for k in b:
            o ^= p2[k]
        for a_ in range(len(b)):
            for c_ in range(a_ + 1, len(b)):
                o ^= h.br[b[a_]][b[c_]]
        return o
    def br(x, y):
        return h.bracket(x, y)
    p4map = {1 << (npe + j): p4[j] for j in range(nme)}
    def p4v(x):
        # via additivity on bits, recursively
        b = bits(x)
        if len(b) == 1:
            return p4map[x]
        x1 = 1 << b[0]
        y1 = x ^ x1
        s = (br(x1, br(x1, br(x1, y1))) ^ br(y1, br(y1, br(y1, x1)))
             ^ br(x1, br(y1, br(x1, y1))) ^ p2v(br(x1, y1)))
        return p4v(x1) ^ p4v(y1) ^ s
    for c in range(1, 1 << nme):
        y = c << npe
        A = h.ad(y)
        A2 = matmul(A, A)
        if flat(h.ad(p4v(y))) != flat(matmul(A2, A2)):
            bad += 1
        if p4v(y) & ~((1 << npe) - 1):
            bad += 1
    out("   [4] by additivity (sigma) on all %d nonzero y in (g-)ev: ad_{y^[4]}=(ad_y)^4 %s"
        % ((1 << nme) - 1, "OK" if bad == 0 else "FAIL %d" % bad))
    # (ad_y)^2 inner?
    for j in range(nme):
        A = h.ad(1 << (npe + j))
        z = inner(h, matmul(A, A), [1 << k for k in range(h.n) if h.par[k] == 0])
        out("   (ad_%s)^2 inner? %s" % (names[npe + j], "ad_" + vname(h, z) if z is not None else "NO"))
    # the closure
    t0 = time.time()
    cl = Closure244(h, npe, nme, p2, p4)
    nod = h.n - npe - nme
    exp = 2 ** npe * 4 ** nme * 2 ** nod
    out("   dim u_{2,4|4} = %d (expected 2^%d 4^%d 2^%d = %d)  [%.1fs]"
        % (len(cl.u.basis), npe, nme, nod, exp, time.time() - t0))
    nb = len(cl.u.basis)
    if nb <= 64:
        out("   associativity of u (all triples): %s" % ("OK" if cl.u.check_associativity() == 0 else "FAIL"))
    else:
        import random
        random.seed(1)
        short = [i for i, w in enumerate(cl.u.basis) if len(w) <= 3]
        samp = random.sample(short, 10)
        out("   associativity of u (10^3 random triples of words of length <= 3): %s"
            % ("OK" if cl.u.check_associativity(samp) == 0 else "FAIL"))
    # centrality of the relations in U is equivalent to [x^[2],.]=ad_x^2 etc. -- checked above.
    # odd x^4 = x^[4] in u, for ALL odd x (homogeneous or not)
    oddidx = list(range(npe + nme, h.n))
    badodd = []
    inhom = []
    homog_ok = 0
    for c in range(1, 1 << nod):
        x = 0
        for a_ in bits(c):
            x ^= cl.Lb[oddidx[a_]]
        x2 = cl.u.mul(x, x)
        x4 = cl.u.mul(x2, x2)
        xv = 0
        for a_ in bits(c):
            xv |= 1 << oddidx[a_]
        # x^[4] := (s(x))^[2] -- defined only if s(x) in (g+)ev
        s = h.square(xv)
        if s & ~((1 << npe) - 1):
            # x is inhomogeneous: x^[4] := (s(x))^[2] is undefined; x^4 = s(x)^2 lies in L
            inL = solve(cl.Lb, x4)
            inhom.append((vname(h, xv), vname(h, s), inL is not None,
                          inL is not None and (inL >> h.n) != 0))
            continue
        homog_ok += 1
        tgt = cl.Lb and 0
        for k in bits(p2v(s)):
            tgt ^= cl.Lb[k]
        if x4 != tgt:
            badodd.append((vname(h, xv), "x^4 != x^[4]"))
    out("   P244 1): odd x with s(x) in (g+)ev: %d, x^4 = x^[4] := (s(x))^[2] for %d of them; failures %s"
        % (homog_ok + len(badodd), homog_ok, badodd[:4]))
    out("   odd x with s(x) NOT in (g+)ev (x^[4] undefined): %d; x^4 in L for all: %s; x^4 outside g for all: %s"
        % (len(inhom), all(r[2] for r in inhom), all(r[3] for r in inhom)))
    for r in inhom[:3]:
        out("      e.g. x=%s, s(x)=%s" % (r[0], r[1]))
    out("   L: dim = %d (dim g + dim (g-)ev = %d), basis independent in u: %s; closed under bracket & square: %s"
        % (cl.dimL, h.n + nme, cl.lin_indep, "OK" if not cl.problems else cl.problems[:3]))
    out("   L squared Lie algebra with L'=L (Jacobi, [x^[2],y]=[x,[x,y]] on basis): %s"
        % ("OK" if L_axioms(cl.br, cl.pmap, cl.dimL) == 0 else "FAIL"))
    out("   dim u_2(L) = 2^%d = %d vs dim u_{2,4|4} = %d: %s"
        % (cl.dimL, 2 ** cl.dimL, nb, "EQUAL" if 2 ** cl.dimL == nb else "DIFFER"))
    # the brackets and squares of the new elements of L
    Lnames = names + ["%s^2" % names[npe + j] for j in range(nme)]
    def Lv(v):
        return "+".join(Lnames[k] for k in bits(v)) if v else "0"
    for j in range(nme):
        a = h.n + j
        brs = ["[%s,%s]=%s" % (Lnames[a], Lnames[b], Lv(cl.br[a][b])) for b in range(cl.dimL) if b != a and cl.br[a][b]]
        out("   L: (%s)^[2]=%s; %s" % (Lnames[a], Lv(cl.pmap[a]), "; ".join(brs)))
    # independent check of degree 1: (restricted) derivations modulo inner ones
    for (mn, Mdim, act_g) in [("K", 1, lambda k, v: 0),
                              ("g", h.n, lambda k, v: h.bracket(1 << k, v))]:
        r1 = h1_direct(h, npe, nme, p2, p4, Mdim, act_g, True)
        o1 = h1_direct(h, npe, nme, p2, p4, Mdim, act_g, False)
        out("   direct H^1 (derivations mod inner), M=%s: restricted %d, ordinary %d" % (mn, r1, o1))
    # cohomology
    L = cl.rla()
    ext_ok = True
    res = {}
    for (mn, Mdim, act_g) in [("K", 1, lambda k, v: 0),
                              ("g", h.n, lambda k, v: h.bracket(1 << k, v))]:
        ML = cl.module_L(Mdim, act_g)
        t0 = time.time()
        hh = L.cohom(ML, maxdeg)
        line = "   M=%s  H^n_res via Gamma(L^*)(x)M, n=0..%d: %s  [%.0fs]" % (mn, maxdeg, hh, time.time() - t0)
        if do_ext:
            t0 = time.time()
            maps = resco.resolution(cl.u, ext_len)
            Mact = resco.module_action(cl.u, act_g, Mdim)
            e = resco.ext_dims(cl.u, maps, Mact, Mdim, ext_len)
            agree = list(hh[:len(e)]) == list(e[:len(hh)])
            ext_ok &= agree
            line += "\n        Ext^n_u(K;M) from a free resolution over u, n=0..%d: %s  ==> %s [%.0fs]" % (
                ext_len - 1, e, "AGREE" if agree else "MISMATCH", time.time() - t0)
        out(line)
        # ordinary cohomology of the Lie superalgebra
        ev_n = npe + nme
        G = RLA(h.n, h.br, ev_n, [h.sq[i] for i in range(ev_n, h.n)], "g")
        M0 = dpco.Module(Mdim, [[act_g(k, 1 << i) for k in range(h.n)] for i in range(Mdim)],
                         [0] * Mdim, "M")
        t0 = time.time()
        ho = G.cohom(M0, maxdeg)
        out("        ordinary H^n(g;M) (Lambda(g_ev^*)(x)Gamma(g_od^*)), n=0..%d: %s  [%.0fs]" % (maxdeg, ho, time.time() - t0))
        res[mn] = (hh, ho)
    return res


def h1_direct(h, npe, nme, p2, p4, Mdim, act, restricted):
    n = h.n
    N = n * Mdim                     # unknown c(e_i)_t  -> index i*Mdim+t
    def actv(k, form_vec):
        """apply e_k to a vector of linear forms (list over t)"""
        res = [0] * Mdim
        for t in range(Mdim):
            if form_vec[t]:
                for s in bits(act(k, 1 << t)):
                    res[s] ^= form_vec[t]
        return res
    def cvec(i):
        return [1 << (i * Mdim + t) for t in range(Mdim)]
    def cof(v):
        r = [0] * Mdim
        for i in bits(v):
            r = [a ^ b for a, b in zip(r, cvec(i))]
        return r
    rows = []
    def emit(vec):
        rows.extend(f for f in vec if f)
    for i in range(n):
        for j in range(i + 1, n):
            l = cof(h.br[i][j])
            r = [a ^ b for a, b in zip(actv(i, cvec(j)), actv(j, cvec(i)))]
            emit([a ^ b for a, b in zip(l, r)])
    for i in range(npe + nme, n):
        emit([a ^ b for a, b in zip(cof(h.sq[i]), actv(i, cvec(i)))])
    if restricted:
        for i in range(npe):
            emit([a ^ b for a, b in zip(cof(p2[i]), actv(i, cvec(i)))])
        for j in range(nme):
            k = npe + j
            emit([a ^ b for a, b in zip(cof(p4[j]), actv(k, actv(k, actv(k, cvec(k)))))])
    der = len(nullspace(rows, N))
    # inner: c(x) = x.m ; dim = Mdim - dim M^g
    inv_rows = []
    for k in range(n):
        for t in range(Mdim):
            r = 0
            for s in range(Mdim):
                if (act(k, 1 << s) >> t) & 1:
                    r |= 1 << s
            if r:
                inv_rows.append(r)
    inv = len(nullspace(inv_rows, Mdim))
    return der - (Mdim - inv)


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    out("ex244.py -- (2,4|4)-structures, exact over F_2;  %s" % time.strftime("%Y-%m-%d %H:%M"))
    g12, m12 = build_oo(1, 1)
    gen12, types12 = study("oo^{(1)}_{I Pi}(1|2) at p = 2", g12, m12)
    # cohomology for every type of grading with a (2,4) structure and no mixed bracket
    for key, lst in sorted(types12.items()):
        (t, ok24, nomix) = key
        if ok24 and nomix:
            cohom_block("oo^{(1)}_{I Pi}(1|2), grading of type %s" % (t,), g12, lst[0], 3, True, 5)
    g32, m32 = build_oo(3, 1)
    gen32, types32 = study("oo^{(1)}_{I Pi}(3|2) at p = 2", g32, m32)
    for key, lst in sorted(types32.items()):
        (t, ok24, nomix) = key
        if ok24 and t[1] > 0:
            cohom_block("oo^{(1)}_{I Pi}(3|2), grading of type %s" % (t,), g32, lst[0],
                        2, False, 3)
            if quick:
                break
    with open(os.path.join(HERE, "ex244.log"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
