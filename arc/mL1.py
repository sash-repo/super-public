"""
The Lie superalgebra lf(S^{1|2}_odd) = Ker s_M explicitly: eqs. (116), (117) of
the text (Subsec. 4.6.6) and the m^L(1) row of Table (61).

Theorem 4.5.1 of the text says that, for m^L(1), the annihilator of the
Schwarzian s_M is isomorphic to sl(1|2) = osp(2|2).  Here we compute its basis
as the radical of the cocycle and compare it with eq. (116).

Realisation (Subsec. 1.2.1, item (iv), and eq. (19)):
    alpha_0 = d tau + q d xi + xi d q,   tau, xi odd,  q even,
    M_f = (2-E)(f) d_tau - ( d_q(f) d_xi + (-1)^p d_xi(f) d_q )
                         - (-1)^p d_tau(f) E,      E = xi d_xi + q d_q.
Functions: Laurent polynomials in q tensor Lambda(tau, xi).
In stringy.py's data structure the even variable is called 't' -- here it is q;
odd labels: 0 = tau, 1 = xi.

Cocycle (eq. (118) of the text, the m^L(1) row of Table (61)):
    c(M_f,M_g) = (-1)^{p(f)} Res( f (D_xi)^2 (g) ),   by eq. (55)
    D_xi := xi d_tau - d_q      (the invariant, "Y-type", operator)
    M_xi := xi d_tau + d_q      (the "X-type" one).
Both candidates, with and without the sign, are tested; the residue is the
coefficient of q^{-1} tau xi.  The signs in the antisymmetry and cocycle
identities are those of the Lie superalgebra m^L(1): p(M_f) = p(f) + 1.
"""
from fractions import Fraction as F
from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show, VF

TAU, XI = 0, 1


# ------------------------------------------------------------------ the fields

def E_op(f):
    """E = xi d_xi + q d_q  (degree in q and xi)."""
    return {(a, S): (a + (1 if XI in S else 0)) * c for (a, S), c in f.items()
            if (a + (1 if XI in S else 0))}


def M_field(f):
    p = parity(f)
    sgn = (-1) ** p
    nab = add(smul(2, f), smul(-1, E_op(f)))          # (2-E)(f)
    dq, dxi, dtau = d_t(f), d_odd(XI, f), d_odd(TAU, f)
    comps = {'t': smul(-sgn, dxi), XI: smul(-1, dq), TAU: nab}
    # - (-1)^p d_tau(f) * E  :  E = xi d_xi + q d_q
    comps['t'] = add(comps['t'], smul(-sgn, mul(dtau, mono(1))))
    comps[XI] = add(comps[XI], smul(-sgn, mul(dtau, mono(0, (XI,)))))
    return VF({k: v for k, v in comps.items() if v}, (TAU, XI))


def monomials(degs):
    out = []
    for a in degs:
        for S in ((), (TAU,), (XI,), (TAU, XI)):
            out.append((a, S))
    return out


# ------------------------------------------------------------------ operators

def D_xi(f):     # xi d_tau - d_q
    return add(mul(mono(0, (XI,)), d_odd(TAU, f)), smul(-1, d_t(f)))


def M_xi(f):     # xi d_tau + d_q
    return add(mul(mono(0, (XI,)), d_odd(TAU, f)), d_t(f))


def sq(op):
    return lambda f: op(op(f))


def Res(f):
    """coefficient of q^{-1} tau xi."""
    return f.get((-1, (TAU, XI)), 0)


# ------------------------------------------------------------------ the bracket

class Inverter:
    """Left inverse of  f -> M_f  on the span of the monomials `keys`:
    one Gaussian elimination, then every bracket costs a back-substitution."""

    def __init__(self, keys):
        self.keys = list(keys)
        cols = [M_field(mono(*k)) for k in self.keys]
        rows = set()
        for V in cols:
            for co, fn in V.c.items():
                for m in fn:
                    rows.add((co, m))
        self.rows = sorted(rows, key=str)
        n, m = len(self.keys), len(self.rows)
        # augmented [A | I]
        A = [[F(cols[j].c.get(co, {}).get(mo, 0)) for j in range(n)]
             + [F(1) if i == k else F(0) for k in range(m)]
             for i, (co, mo) in enumerate(self.rows)]
        piv, r = [], 0
        for j in range(n):
            p = next((i for i in range(r, m) if A[i][j] != 0), None)
            if p is None:
                continue
            A[r], A[p] = A[p], A[r]
            inv = F(1) / A[r][j]
            A[r] = [v * inv for v in A[r]]
            for i in range(m):
                if i != r and A[i][j] != 0:
                    fct = A[i][j]
                    A[i] = [a - fct * c for a, c in zip(A[i], A[r])]
            piv.append(j)
            r += 1
        self.A, self.piv, self.rank, self.n, self.m = A, piv, r, n, m

    def coords(self, X):
        """X (a VF) -> its coefficient vector in the M_f basis, or None."""
        b = [F(X.c.get(co, {}).get(mo, 0)) for (co, mo) in self.rows]
        Tb = [sum(self.A[i][self.n + k] * b[k] for k in range(self.m))
              for i in range(self.m)]
        for i in range(self.rank, self.m):
            if Tb[i] != 0:
                return None
        x = [F(0)] * self.n
        for i, j in enumerate(self.piv):
            x[j] = Tb[i]
        return x

    def bracket(self, f, g):
        return self.coords(M_field(f).bracket(M_field(g)))


def bracket_gf(keys, f, g):
    """[M_f, M_g] = M_h ; return h as a vector in the monomial basis `keys`."""
    X = M_field(f).bracket(M_field(g))
    # solve  sum_k c_k M_{e_k} = X
    cols = [M_field(mono(*k)) for k in keys]
    rows = set()
    for V in cols + [X]:
        for co, fn in V.c.items():
            for m in fn:
                rows.add((co, m))
    rows = sorted(rows, key=str)
    A = [[F(col.c.get(co, {}).get(m, 0)) for col in cols] for (co, m) in rows]
    b = [F(X.c.get(co, {}).get(m, 0)) for (co, m) in rows]
    return solve(A, b)


def solve(A, b):
    m, n = len(A), len(A[0])
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    piv = []
    r = 0
    for j in range(n):
        p = next((i for i in range(r, m) if M[i][j] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = F(1) / M[r][j]
        M[r] = [v * inv for v in M[r]]
        for i in range(m):
            if i != r and M[i][j] != 0:
                f = M[i][j]
                M[i] = [a - f * c for a, c in zip(M[i], M[r])]
        piv.append(j)
        r += 1
    for i in range(r, m):
        if M[i][n] != 0:
            return None                      # inconsistent
    x = [F(0)] * n
    for i, j in enumerate(piv):
        x[j] = M[i][n]
    return x


# ------------------------------------------------------------------ radical

def radical(keys, cfun):
    """{ y : c(x,y) = 0 for all x in keys } as a list of coefficient vectors."""
    A = [[F(cfun(mono(*kx), mono(*ky))) for ky in keys] for kx in keys]
    m, n = len(A), len(A[0])
    piv, r = [], 0
    for j in range(n):
        p = next((i for i in range(r, m) if A[i][j] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = F(1) / A[r][j]
        A[r] = [v * inv for v in A[r]]
        for i in range(m):
            if i != r and A[i][j] != 0:
                f = A[i][j]
                A[i] = [a - f * c for a, c in zip(A[i], A[r])]
        piv.append(j)
        r += 1
    free = [j for j in range(n) if j not in piv]
    basis = []
    for j0 in free:
        v = [F(0)] * n
        v[j0] = F(1)
        for i, j in enumerate(piv):
            v[j] = -A[i][j0]
        basis.append(v)
    return basis


def vec_to_fun(v, keys):
    out = {}
    for c, k in zip(v, keys):
        if c:
            out = add(out, mono(k[0], k[1], c))
    return out


def name(k):
    a, S = k
    s = "".join({TAU: "tau", XI: "xi"}[j] for j in S)
    q = "" if a == 0 else ("q" if a == 1 else "q^%d" % a)
    return (q + s) or "1"


def vec_str(v, keys):
    parts = []
    for c, k in zip(v, keys):
        if c:
            parts.append(("%s*%s" % (c, name(k))) if c != 1 else name(k))
    return " + ".join(parts) or "0"


# ------------------------------------------------------------------ main

def pM(f):
    """parity of the field M_f: p(M_f) = p(f) + 1 (see eq. (116) of the text)."""
    return (parity(f) + 1) % 2


def radical2(xkeys, ykeys, cfun):
    """{ y in span(ykeys) : c(x,y) = 0 for all x in xkeys }."""
    A = [[F(cfun(mono(*kx), mono(*ky))) for ky in ykeys] for kx in xkeys]
    A = [row for row in A if any(row)] or [[F(0)] * len(ykeys)]
    return radical_matrix(A)


def radical_matrix(A):
    m, n = len(A), len(A[0])
    A = [row[:] for row in A]
    piv, r = [], 0
    for j in range(n):
        p = next((i for i in range(r, m) if A[i][j] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = F(1) / A[r][j]
        A[r] = [v * inv for v in A[r]]
        for i in range(m):
            if i != r and A[i][j] != 0:
                f = A[i][j]
                A[i] = [a - f * c for a, c in zip(A[i], A[r])]
        piv.append(j)
        r += 1
    basis = []
    for j0 in [j for j in range(n) if j not in piv]:
        v = [F(0)] * n
        v[j0] = F(1)
        for i, j in enumerate(piv):
            v[j] = -A[i][j0]
        basis.append(v)
    return basis


def rank_of(vectors):
    if not vectors:
        return 0
    return len(vectors[0]) - len(radical_matrix(vectors))


# eq. (116) of the text:  f in Span(tau, xi, q xi, q tau + q^2 xi | 1, q, tau xi, q tau xi)
ANNIH_M = [
    ("tau", [((0, (TAU,)), 1)]),
    ("xi", [((0, (XI,)), 1)]),
    ("q xi", [((1, (XI,)), 1)]),
    ("q tau + q^2 xi", [((1, (TAU,)), 1), ((2, (XI,)), 1)]),
    ("1", [((0, ()), 1)]),
    ("q", [((1, ()), 1)]),
    ("tau xi", [((0, (TAU, XI)), 1)]),
    ("q tau xi", [((1, (TAU, XI)), 1)]),
]


def fun(terms):
    out = {}
    for (a, S), c in terms:
        out = add(out, mono(a, S, c))
    return out


if __name__ == "__main__":
    print("=" * 74)
    print("m^L(1) : the fields M_f, the cocycle and its radical")
    print("=" * 74)

    # the bracket [M_f, M_g] = M_h is solved for h inside a window of q-degrees
    # wide enough to contain every bracket of two monomials of q-degree -2..2
    keys = monomials(range(-8, 9))
    inv = Inverter(keys)

    # 1) the span of the M_f is closed under the bracket
    print("\n1) closure of the bracket on the span of the M_f")
    small = monomials(range(-2, 3))
    bad = 0
    for kf in small:
        for kg in small:
            if inv.bracket(mono(*kf), mono(*kg)) is None:
                bad += 1
                print("   NOT in the span: [M_%s, M_%s]" % (name(kf), name(kg)))
    print("   closed (%d pairs)" % (len(small) ** 2) if not bad else "   %d failures" % bad)

    # 2) which square is a cocycle?  Signs are those of the Lie superalgebra
    #    m^L(1), i.e. the parities p(M_f) = p(f) + 1, not p(f).
    print("\n2) the cocycle identity for c(M_f,M_g) = Res(f s(g))")
    brk = {}
    for kf in small:
        for kg in small:
            brk[kf, kg] = vec_to_fun(inv.bracket(mono(*kf), mono(*kg)), keys)
    results = {}
    for label, op in (("(M_xi)^2", sq(M_xi)), ("(D_xi)^2", sq(D_xi))):
        for sgnname, sgn in (("plain", lambda f: 1),
                             ("(-1)^p(f)", lambda f: (-1) ** parity(f))):
            def c(f, g, op=op, sgn=sgn):
                return sgn(f) * Res(mul(f, op(g)))
            # antisymmetry (super): c(M_f,M_g) = -(-1)^{p(M_f)p(M_g)} c(M_g,M_f)
            anti = all(
                c(mono(*kf), mono(*kg))
                == -(-1) ** (pM(mono(*kf)) * pM(mono(*kg))) * c(mono(*kg), mono(*kf))
                for kf in monomials(range(-3, 4)) for kg in monomials(range(-3, 4)))
            # d c (f,g,h) = c([f,g],h) - (-1)^{p(g)p(h)} c([f,h],g)
            #               + (-1)^{p(f)(p(g)+p(h))} c([g,h],f),   p = p(M_.)
            nbad = ntot = 0
            for kf in small:
                for kg in small:
                    for kh in small:
                        f, g, h = mono(*kf), mono(*kg), mono(*kh)
                        pf, pg, ph = pM(f), pM(g), pM(h)
                        val = (c(brk[kf, kg], h)
                               - (-1) ** (pg * ph) * c(brk[kf, kh], g)
                               + (-1) ** (pf * (pg + ph)) * c(brk[kg, kh], f))
                        ntot += 1
                        nbad += (val != 0)
            results[label, sgnname] = (anti, nbad == 0)
            print("   %-9s %-10s : antisym %s , 2-cocycle %s  (%d triples, %d failures)"
                  % (label, sgnname, "OK" if anti else "NO",
                     "OK" if nbad == 0 else "NO", ntot, nbad))

    # 3) the radical of the cocycle of eq. (118) of the text,
    #    c(M_f,M_g) = (-1)^{p(f)} Res(f (D_xi)^2 (g)),
    #    for g of q-degree -3..3, tested against all f of q-degree -8..8
    #    (c(f,g) != 0 forces deg f + deg g = 4 for deg q = deg xi = 1, deg tau = 2,
    #    so the f-window contains every partner of every g in the y-window)
    print("\n3) the radical of c(M_f,M_g) = (-1)^{p(f)} Res(f (D_xi)^2 (g))")
    opD = sq(D_xi)
    cD = lambda f, g: (-1) ** parity(f) * Res(mul(f, opD(g)))
    ykeys = monomials(range(-3, 4))
    rad = radical2(keys, ykeys, cD)
    ev = [v for v in rad if pM(vec_to_fun(v, ykeys)) == 0]
    od = [v for v in rad if pM(vec_to_fun(v, ykeys)) == 1]
    print("   sdim of the radical = (%d|%d)" % (len(ev), len(od)))
    for v in rad:
        print("     M_f, f = %s" % vec_str(v, ykeys))

    # 4) compare with eq. (116) of the text
    print("\n4) the radical vs eq. (116) of the text")
    idx = {k: i for i, k in enumerate(ykeys)}
    printed = []
    for nm, terms in ANNIH_M:
        v = [F(0)] * len(ykeys)
        for k, c0 in terms:
            v[idx[k]] += c0
        printed.append(v)
    r_rad, r_pr = rank_of(rad), rank_of(printed)
    r_both = rank_of(rad + printed)
    print("   rank(radical) = %d, rank(eq. (116)) = %d, rank(both) = %d  -> %s"
          % (r_rad, r_pr, r_both,
             "SAME SPACE" if r_rad == r_pr == r_both else "DIFFERENT"))
    for nm, terms in ANNIH_M:
        f = fun(terms)
        ok = all(cD(mono(*kx), f) == 0 for kx in keys)
        print("   f = %-16s : p(M_f) = %d, c(., M_f) = 0: %s" % (nm, pM(f), ok))

    # 5) the span of eq. (116) is a subalgebra
    print("\n5) eq. (116) closes under the bracket")
    pk = [k for k in keys]
    pidx = {k: i for i, k in enumerate(pk)}
    pv = []
    for nm, terms in ANNIH_M:
        v = [F(0)] * len(pk)
        for k, c0 in terms:
            v[pidx[k]] += c0
        pv.append(v)
    r0 = rank_of(pv)
    closed = True
    for (n1, t1) in ANNIH_M:
        for (n2, t2) in ANNIH_M:
            X = M_field(fun(t1)).bracket(M_field(fun(t2)))
            h = inv.coords(X)
            if h is None or rank_of(pv + [h]) != r0:
                closed = False
                print("   [M_{%s}, M_{%s}] leaves the span" % (n1, n2))
    print("   closed" if closed else "   NOT closed")
