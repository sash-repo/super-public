"""
mab.py -- weights and scans for M_{a,b}(n).

Kochetkov's weight basis for M_{a,b}(n) is {'(x_i xi_i - x_{i+1} xi_{i+1}), 1<=i<=n-1;
a z + b h}, so a weight is the n-tuple

    mu(rho) = (rho_1 - rho_2, ..., rho_{n-1} - rho_n),   nu = a . (z-weight) + b . |rho|,

where rho is the pe(n)-weight and |rho| = sum rho_i (the h-eigenvalue).  A module V is
given by its M_{a,b}(n)_0-highest weight (mu; nu); we realise it as the pe(n)-module
V(lambda) with lambda_n = 0 (any lift gives the same spe(n)-module) and z acting by the
scalar c fixed by  a c + b |lambda| = nu   (for a = 0 we shift lambda instead).
"""

from fractions import Fraction as F

from mn import layer, mbr, mdeg
from mimod import Mn, Mab, MInduced
from mreport import render
from mone import weights_of


def mu_of(rho):
    return tuple(rho[i] - rho[i + 1] for i in range(len(rho) - 1))


def nu_of(rho, zw, a, b):
    return a * zw + b * sum(rho)


def lift(n, mu, nu, a, b):
    """(lambda, c) realising the M_{a,b}(n)_0-highest weight (mu; nu)"""
    a, b, nu = F(a), F(b), F(nu)
    lam = [F(0)] * n
    for i in range(n - 2, -1, -1):
        lam[i] = lam[i + 1] + F(mu[i])
    if a != 0:
        c = (nu - b * sum(lam)) / a
    else:
        # z is not in M_{0,b}(n); shift lambda so that b |lambda| = nu
        t = (nu / b - sum(lam)) / n
        lam = [x + t for x in lam]
        c = F(0)
    return tuple(lam), c


def ops_for(mn, a, b, verbose=False):
    """M_{a,b}(n)_1, plus M_{a,b}(n)_2 when L_1 does not generate L_{>=1}"""
    n = mn.n
    A = Mab(mn, a, b)
    ops = list(A.layer(1))
    from mgencheck import span_dim
    amb = [next(iter(f)) for f in layer(n, 2)]
    gen = []
    for f in A.layer(1):
        for g in A.layer(1):
            x = mbr(f, g, n)
            if x and mdeg(x, n) == 2:
                gen.append(x)
    if span_dim(gen, amb) < len(A.layer(2)):
        ops += A.layer(2)
        if verbose:
            print("    (L_1 does not generate L_{>=1}: L_2 imposed as well)")
    return ops


def scan_mab(n, mu, nu, m, a, b, want=None, quiet=False):
    """all singular vectors of degree m in I(V(mu;nu)) over M_{a,b}(n)"""
    mn = Mn(n)
    lam, c = lift(n, mu, nu, a, b)
    ops = ops_for(mn, a, b)
    I = MInduced(mn, lam, c, ops)
    ws = weights_of(I.V, mn.pe)
    dimV = sum(len(I.V.wtbasis(q)) for q in ws)
    print("  M_{%s,%s}(%d):  V(mu=%s; nu=%s) = V_pe(%s), z=%s, dim V = %d;  m = %d"
          % (a, b, n, list(mu), nu, [str(x) for x in lam], c, dimV, m))
    rhos = set()
    for u in I.umonos(m):
        w = I.uwt(u)
        for q in ws:
            rhos.add(tuple(x + y for x, y in zip(q, w)))
    hits = []
    for rho in sorted(rhos):
        mw = (mu_of(rho), nu_of(rho, c - m, a, b))
        if want is not None and mw != want:
            continue
        sv = I.singular(m, rho)
        for s in sv:
            hits.append((rho, mw, s))
            print("    wt(y) = (%s; %s)  [pe-wt %s]  :  %s"
                  % (",".join(str(x) for x in mw[0]), mw[1],
                     ",".join(str(x) for x in rho), render(I, s)))
    if not hits:
        print("    (no singular vector%s)" % ("" if want is None else " at the claimed weight"))
    return hits
