"""lebedev.py -- A. Lebedev's small-degree cochains for p = 2, implemented
from [BGL, Subsec. 4.1] verbatim, and compared with the dual of the resolution
V_* = U(g) (x) C_* built in dpco.py.

Lebedev's spaces (loc. cit.):

  XC^0 = M
  XC^1 = {c : g -> M linear}
  XC^2 = {(c,q) | c anti-symmetric bilinear, q quadratic on g_1, B_q = c|_{g_1}}
  XC^3 = {(c,p) | c anti-symmetric trilinear, p(.,z) quadratic on g_1,
                  p(x,.) linear, p(x+y,z)+p(x,z)+p(y,z) = c(x,y,z)}

  d^0(m)(x)     = x.m
  d^1(c)        = (dc, q),  dc(x,y) = c([x,y]) + x.c(y) + y.c(x),
                            q(x)    = c(s(x)) + x.c(x)
  d^2(c,q)      = (dc, p),  dc(x,y,z) = x.c(y,z) + c([x,y],z) + cyclic,
                            p(x,z)    = x.c(x,z) + z.q(x) + c(s(x),z)
                                        + c([x,z],x)

Here "anti-symmetric" means: the value is 0 whenever two arguments are equal,
whatever their parities.

The dictionary with C_* = Lambda(g_0) (x) Gamma(g_1) is the obvious one:

  C_1: e_i <-> c(e_i),  gamma_1(f_j) <-> c(f_j)
  C_2: e_i e_i' <-> c(e_i,e_i'),  e_i gamma_1(f_j) <-> c(e_i,f_j),
       gamma_1(f_j) gamma_1(f_l) <-> c(f_j,f_l),  gamma_2(f_j) <-> q(f_j)
  C_3: three distinct letters <-> c(.,.,.),
       gamma_2(f_j) . g_k <-> p(f_j, g_k),   gamma_3(f_j) <-> p(f_j, f_j)

The check below is an equality of MAPS, entry by entry, not of dimensions.
"""

from dpco import SLA2, Env, CE, Resolution, bits, Module


# --------------------------------------------------- Lebedev's cochain maps

def vec(g, k):
    return 1 << k


def lb_d0(g, M, m_elt):
    """d^0(m) as the list [ (x, value) ] over basis x"""
    return {k: M.apply(k, m_elt) for k in range(g.dim)}


def lb_d1(g, M, c):
    """c: dict k -> element of M.  Returns (cc, q) with
    cc: dict (i,j), i<j -> M ; q: dict j (odd index 0..n-1) -> M."""
    cc, q = {}, {}
    for i in range(g.dim):
        for j in range(i + 1, g.dim):
            val = 0
            for k in bits(g.br[i][j]):
                val ^= c[k]
            val ^= M.apply(i, c[j])
            val ^= M.apply(j, c[i])
            cc[(i, j)] = val
    for j in range(g.n):
        k = g.m + j
        val = 0
        for t in bits(g.sq[j]):
            val ^= c[t]
        val ^= M.apply(k, c[k])
        q[j] = val
    return cc, q


def c_of(g, cc, q, x, y):
    """the value c(x,y) of the anti-symmetric form given by (cc,q) on two
    basis vectors given as bitmasks (linearity + polarization)."""
    val = 0
    bx, by = bits(x), bits(y)
    for i in bx:
        for j in by:
            if i == j:
                continue
            val ^= cc[(i, j)] if i < j else cc[(j, i)]
    return val


def lb_d2(g, M, cc, q):
    """returns (ccc, p): ccc dict (i,j,k) i<j<k -> M;
    p dict (j, k) with j an odd index (0..n-1) and k a basis index -> M."""
    ccc = {}
    for i in range(g.dim):
        for j in range(i + 1, g.dim):
            for k in range(j + 1, g.dim):
                val = 0
                for (a, b, cix) in [(i, j, k), (j, k, i), (k, i, j)]:
                    # x.c(y,z) with x = a, (y,z) = the other two
                    pass
                # x.c(y,z) + c([x,y],z) summed cyclically over (i,j,k)
                for (x, y, z) in [(i, j, k), (j, k, i), (k, i, j)]:
                    val ^= M.apply(x, cc[(y, z)] if y < z else cc[(z, y)])
                    val ^= c_of(g, cc, q, g.br[x][y], 1 << z)
                ccc[(i, j, k)] = val
    p = {}
    for j in range(g.n):
        xk = g.m + j
        for k in range(g.dim):
            val = 0
            # x.c(x,z)
            val ^= M.apply(xk, c_of(g, cc, q, 1 << xk, 1 << k))
            # z.q(x)
            val ^= M.apply(k, q[j])
            # c(s(x), z)
            val ^= c_of(g, cc, q, g.sq[j], 1 << k)
            # c([x,z], x)
            val ^= c_of(g, cc, q, g.br[xk][k], 1 << xk)
            p[(j, k)] = val
    return ccc, p


# ------------------------------------------- the same maps from dpco.py

def our_cochain_value(res, phi, c):
    """phi: dict from C-monomials to M-elements; value at monomial c"""
    return phi.get(c, 0)


def our_d(res, M, phi, deg):
    """the codifferential (d phi)(c) = sum_k g_k . phi(partial_k c)+phi(delta c)"""
    out = {}
    for c in res.ce.basis(deg + 1):
        val = 0
        for k in range(res.g.dim):
            pk = res.ce.partial(k, c)
            if pk is not None:
                val ^= M.apply(k, phi.get(pk, 0))
        for c2 in res.ce.delta(c):
            val ^= phi.get(c2, 0)
        out[c] = val
    return out


def mono1(g, k):
    if k < g.m:
        return (frozenset([k]), (0,) * g.n)
    j = k - g.m
    al = [0] * g.n
    al[j] = 1
    return (frozenset(), tuple(al))


def mono2(g, i, j):
    """the C_2 monomial for the pair of distinct basis letters i<j"""
    S = set()
    al = [0] * g.n
    for k in (i, j):
        if k < g.m:
            S.add(k)
        else:
            al[k - g.m] += 1
    return (frozenset(S), tuple(al))


def mono_gamma2(g, j):
    al = [0] * g.n
    al[j] = 2
    return (frozenset(), tuple(al))


def mono3(g, i, j, k):
    S = set()
    al = [0] * g.n
    for t in (i, j, k):
        if t < g.m:
            S.add(t)
        else:
            al[t - g.m] += 1
    return (frozenset(S), tuple(al))


def mono_p(g, j, k):
    """gamma_2(f_j) * g_k  (and gamma_3(f_j) when k = f_j)"""
    S = set()
    al = [0] * g.n
    al[j] = 2
    if k < g.m:
        S.add(k)
    else:
        al[k - g.m] += 1
    return (frozenset(S), tuple(al))


# ------------------------------------------------------------- the compare

def compare(g, M, trials, rng):
    """compare d^1 and d^2 on random cochains; returns (bad1, bad2, n)"""
    res = Resolution(g)
    bad1 = bad2 = 0
    for _ in range(trials):
        # ---- degree 1
        c = {k: rng.randrange(1 << M.dim) for k in range(g.dim)}
        cc, q = lb_d1(g, M, c)
        phi = {mono1(g, k): c[k] for k in range(g.dim)}
        ours = our_d(res, M, phi, 1)
        for i in range(g.dim):
            for j in range(i + 1, g.dim):
                if ours.get(mono2(g, i, j), 0) != cc[(i, j)]:
                    bad1 += 1
        for j in range(g.n):
            if ours.get(mono_gamma2(g, j), 0) != q[j]:
                bad1 += 1
        # ---- degree 2: a random (c,q) in XC^2
        cc = {}
        for i in range(g.dim):
            for j in range(i + 1, g.dim):
                cc[(i, j)] = rng.randrange(1 << M.dim)
        q = {j: rng.randrange(1 << M.dim) for j in range(g.n)}
        # the polar form of q must be c on g_1 x g_1: q is free on the basis,
        # c on pairs of odd letters is then whatever it is -- both are free
        # parameters of XC^2, exactly as in dpco's C_2.
        phi = {}
        for i in range(g.dim):
            for j in range(i + 1, g.dim):
                phi[mono2(g, i, j)] = cc[(i, j)]
        for j in range(g.n):
            phi[mono_gamma2(g, j)] = q[j]
        ccc, p = lb_d2(g, M, cc, q)
        ours = our_d(res, M, phi, 2)
        for i in range(g.dim):
            for j in range(i + 1, g.dim):
                for k in range(j + 1, g.dim):
                    if ours.get(mono3(g, i, j, k), 0) != ccc[(i, j, k)]:
                        bad2 += 1
        for j in range(g.n):
            for k in range(g.dim):
                if ours.get(mono_p(g, j, k), 0) != p[(j, k)]:
                    bad2 += 1
    return bad1, bad2
