"""Which outer derivations of H(2:n;omega_0) at p=2 fail to be alternating?

Remark 4.5 of arXiv:math/0510431 lists the outer derivations of the graded
Hamiltonian algebra at p=2 (six of them for n=(2,2), by Remark 4.4) and says
that all are alternating with respect to the associative form except
ad x^(p^n1) and ad y^(p^n2), whence dim H^2 = n1+n2.  The cochain computation
of nongraded.py gives one less.  Here the six derivations of the list are written down
explicitly and each is tested for skewness and for being alternating.

The list (Section 4 of arXiv:math/0510431, item (6) as amended by Remark 4.4):
    (1) (ad x)^{p^r},  0 < r < n2
    (2) (ad y)^{p^s},  0 < s < n1
    (3) ad x^(p^n1) : g |-> -x^(p^n1-1) g_x'   ... acting by the Poisson bracket
    (4) ad y^(p^n2)
    (5) ad(e), e = x^(p^n1-1) y^(p^n2-1)
    (6) at p=2: D(x^(i)y^(j)) = (i-1) x^(i) y^(j)
"""
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank                                     # noqa: E402
from nongraded import Ham, binom                              # noqa: E402
from altder import lam_matrix, der_basis                      # noqa: E402


def as_matrix(L, f):
    """the linear map e_m |-> f(m) as a flat d x d list of rows."""
    return [f(m) for m in range(L.dim)]


def poisson_with(L, i0, j0):
    """the derivation g |-> {x^(i0) y^(j0), g}, for x^(i0)y^(j0) possibly
    outside F[2:n] (the cases i0 = p^n1 or j0 = p^n2)."""
    def act(m):
        k, l = L.basis[m]
        out = [0] * L.dim
        c = (binom(i0 + k - 1, i0) * binom(j0 + l - 1, j0 - 1)
             - binom(i0 + k - 1, i0 - 1) * binom(j0 + l - 1, j0)) % L.p
        if c:
            u, v = i0 + k - 1, j0 + l - 1
            if 0 <= u < L.X and 0 <= v < L.Y and (u, v) in L.index:
                out[L.index[(u, v)]] = c
        return out
    return as_matrix(L, act)


def ad_power(L, a, e):
    """(ad e_a)^e as a matrix."""
    cur = [[1 if m == t else 0 for t in range(L.dim)] for m in range(L.dim)]
    for _ in range(e):
        cur = [L.mul([1 if t == a else 0 for t in range(L.dim)], row) for row in cur]
    return cur


def degree_der_p2(L):
    def act(m):
        i, j = L.basis[m]
        out = [0] * L.dim
        out[m] = (i - 1) % L.p
        return out
    return as_matrix(L, act)


def is_derivation(L, D):
    p, d = L.p, L.dim
    for a in range(d):
        for b in range(a + 1, d):
            ea = [1 if t == a else 0 for t in range(d)]
            eb = [1 if t == b else 0 for t in range(d)]
            lhs = [sum(v * D[k][t] for k, v in enumerate(L.mul(ea, eb))) % p
                   for t in range(d)]
            rhs = [(x + y) % p for x, y in zip(L.mul(D[a], eb), L.mul(ea, D[b]))]
            if lhs != rhs:
                return False
    return True


def skew_alt(L, M, D):
    p, d = L.p, L.dim
    skew = all(sum(D[x][t] * M[t][y] + D[y][t] * M[t][x] for t in range(d)) % p == 0
               for x in range(d) for y in range(x, d))
    alt = all(sum(D[x][t] * M[t][x] for t in range(d)) % p == 0 for x in range(d))
    return skew, alt


def run(p, n1, n2):
    L = Ham(p, n1, n2, 0)
    L.table()
    M = lam_matrix(L)
    ads = [ad_power(L, a, 1) for a in range(L.dim)]
    inner = [sum(D, []) for D in ads]
    dim_ad = rank(inner, p=p)

    cand = []
    for r in range(1, n2):
        cand.append(("(ad x)^{%d^%d}" % (p, r), ad_power(L, L.index[(1, 0)], p ** r)))
    for s in range(1, n1):
        cand.append(("(ad y)^{%d^%d}" % (p, s), ad_power(L, L.index[(0, 1)], p ** s)))
    cand.append(("ad x^(%d)" % L.X, poisson_with(L, L.X, 0)))
    cand.append(("ad y^(%d)" % L.Y, poisson_with(L, 0, L.Y)))
    cand.append(("ad e", ad_power(L, L.index[L.top], 1) if L.top in L.index
                 else poisson_with(L, L.X - 1, L.Y - 1)))
    cand.append(("degree der (Rmk 4.4)", degree_der_p2(L)))

    print("p=%d  H(2:(%d,%d);om_0)  dim=%d   dim ad=%d" % (p, n1, n2, L.dim, dim_ad))
    span = list(inner)
    for name, D in cand:
        d_ok = is_derivation(L, D)
        sk, al = skew_alt(L, M, D)
        outer = rank(span + [sum(D, [])], p=p) > rank(span, p=p)
        print("  %-22s derivation=%-5s outer=%-5s skew=%-5s alternating=%s"
              % (name, d_ok, outer, sk, al))
        span = span + [sum(D, [])]
    print("  span(ad + the listed six) has dim %d, Der has dim %d"
          % (rank(span, p=p), len(der_basis(L))))
    print()


if __name__ == "__main__":
    run(2, 2, 2)
    run(2, 2, 3)
