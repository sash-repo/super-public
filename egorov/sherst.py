#!/usr/bin/env python3
"""Exact checks for the Appendix (Lie isomorphisms of gl_{p,*}).

Three independent checks, all in exact arithmetic:

1) supercom(N, p): the identity

       [a_0,b_0]_s + [a_1,b_1]_s = (-1)^{p(0)} e_{00}

   for  a = sum_{k>=0} (-1)^{p(k)} e_{k,k+1},  b = sum_{k>=0} e_{k+1,k},
   split into their homogeneous parts.  Truncated at k<N the right-hand
   side acquires exactly one boundary term at (N,N), which is what makes
   the identity exact in the infinite algebra.

2) stcheck(m,n): the supertranspose st of gl(m|n) is a superanti-
   automorphism, -st is an automorphism of the Lie superalgebra, and
   -st(e_ii) = -e_ii is not an idempotent -- so -st is not of the form
   Ad_g o phi_sigma.

3) der(m,n): the superspace of derivations of the Lie superalgebra
   gl(m|n).  Expected: dim = dim gl(m|n) = (inner, i.e. gl/center) +
   (the one central map lambda*str*1).  The exception of Bahturin-Bresar-
   Spenko at dim 4 must show up as a jump at gl(1|1).

Usage:  python3 sherst.py [--log FILE]
"""

import sys, os, random, argparse
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank, backend  # noqa: E402


# ---------------------------------------------------------------- 1) e_00

def supercom(N, p):
    """p: dict k -> 0/1 on the window.  Returns the diagonal of
    [a_0,b_0]_s + [a_1,b_1]_s as a dict (i -> coefficient)."""
    t = {k: (-1) ** p[k] for k in range(N + 1)}
    # a_eps = sum_{k in K_eps} t_k e_{k,k+1};  b_eps = sum_{k in K_eps} e_{k+1,k}
    K = {0: [], 1: []}
    for k in range(N):
        K[(p[k] + p[k + 1]) % 2].append(k)

    diag = {}

    def add(i, c):
        diag[i] = diag.get(i, 0) + c
        if diag[i] == 0:
            del diag[i]

    for eps in (0, 1):
        sign = -1 if eps == 0 else +1          # -(-1)^{eps*eps}
        for k in K[eps]:
            add(k, t[k])                        # a_eps b_eps
            add(k + 1, sign * t[k])             # -(-1)^{eps} b_eps a_eps
    return diag


def brute(N, p):
    """The same identity computed with honest matrices on 0..N, no algebra
    done by hand.  Returns the full matrix as a dict (i,j) -> coefficient."""
    n = N + 1
    t = {k: (-1) ** p[k] for k in range(n)}

    def zero():
        return [[0] * n for _ in range(n)]

    a = {0: zero(), 1: zero()}
    b = {0: zero(), 1: zero()}
    for k in range(N):
        eps = (p[k] + p[k + 1]) % 2
        a[eps][k][k + 1] = t[k]
        b[eps][k + 1][k] = 1

    def mul(x, y):
        z = zero()
        for i in range(n):
            for j in range(n):
                s = sum(x[i][l] * y[l][j] for l in range(n))
                z[i][j] = s
        return z

    out = zero()
    for eps in (0, 1):
        xy, yx = mul(a[eps], b[eps]), mul(b[eps], a[eps])
        sgn = (-1) ** (eps * eps)
        for i in range(n):
            for j in range(n):
                out[i][j] += xy[i][j] - sgn * yx[i][j]
    return {(i, j): out[i][j] for i in range(n) for j in range(n) if out[i][j]}


def check_supercom(log):
    ok = True
    for N in (3, 5, 8, 12, 20, 40):
        for trial in range(6):
            if trial == 0:
                p = {k: k % 2 for k in range(N + 1)}          # standard
                name = "p = i mod 2"
            elif trial == 1:
                p = {k: 0 for k in range(N + 1)}              # all even
                name = "p = 0"
            elif trial == 2:
                p = {k: 1 if k % 3 == 0 else 0 for k in range(N + 1)}
                name = "p = [3 | i]"
            else:
                p = {k: random.randint(0, 1) for k in range(N + 1)}
                name = "random"
            d = supercom(N, p)
            want = {0: (-1) ** p[0]}
            # the truncation leaves the single boundary term at (N,N)
            bnd = {i: c for i, c in d.items() if i == N}
            body = {i: c for i, c in d.items() if i != N}
            good = (body == want) and set(bnd) <= {N} and len(bnd) <= 1
            # the same thing with honest matrices
            bm = brute(N, p)
            good &= all(i == j for (i, j) in bm)
            good &= ({i: c for (i, j), c in bm.items() if i != N} == want)
            ok &= good
            log(f"  N={N:3d}  {name:12s}  body={body}  boundary={bnd}  "
                f"{'OK' if good else 'FAIL'}")
    return ok


# ------------------------------------------------------- gl(m|n) machinery

class GL:
    def __init__(self, m, n):
        self.m, self.n = m, n
        self.N = m + n
        self.idx = [(i, j) for i in range(self.N) for j in range(self.N)]
        self.pos = {x: k for k, x in enumerate(self.idx)}
        self.dim = len(self.idx)

    def par(self, i):
        return 0 if i < self.m else 1

    def pe(self, a):                      # parity of the basis vector e_{ij}
        i, j = self.idx[a]
        return (self.par(i) + self.par(j)) % 2

    def br(self, a, b):
        """[e_a, e_b]_s as a dict c -> coefficient."""
        i, j = self.idx[a]
        k, l = self.idx[b]
        out = {}
        if j == k:
            out[self.pos[(i, l)]] = out.get(self.pos[(i, l)], 0) + 1
        if l == i:
            s = -(-1) ** (self.pe(a) * self.pe(b))
            c = self.pos[(k, j)]
            out[c] = out.get(c, 0) + s
        return {c: v for c, v in out.items() if v}

    def st(self, a):
        """supertranspose of e_{ij}: returns (coefficient, index)."""
        i, j = self.idx[a]
        s = (-1) ** (self.par(j) * (self.par(i) + self.par(j)))
        return s, self.pos[(j, i)]


def check_st(m, n, log):
    g = GL(m, n)
    ok = True
    # st is a superanti-automorphism of the associative superalgebra:
    #    (xy)^st = (-1)^{|x||y|} y^st x^st   on the basis
    for a in range(g.dim):
        for b in range(g.dim):
            i, j = g.idx[a]
            k, l = g.idx[b]
            lhs = {}
            if j == k:                                  # e_a e_b = e_{il}
                s, c = g.st(g.pos[(i, l)])
                lhs[c] = s
            sa, ca = g.st(a)
            sb, cb = g.st(b)
            ja, jb = g.idx[cb], g.idx[ca]               # e_b^st e_a^st
            rhs = {}
            if jb[0] == ja[1]:
                rhs[g.pos[(ja[0], jb[1])]] = ((-1) ** (g.pe(a) * g.pe(b))
                                              * sa * sb)
            if lhs != rhs:
                ok = False
    log(f"  gl({m}|{n}): st is a superanti-automorphism: {ok}")
    # -st is an automorphism of the Lie superalgebra
    ok2 = True
    for a in range(g.dim):
        for b in range(g.dim):
            lhs = {}
            for c, v in g.br(a, b).items():
                s, d = g.st(c)
                lhs[d] = lhs.get(d, 0) - v * s          # -st applied
            sa, ca = g.st(a)
            sb, cb = g.st(b)
            rhs = {}
            for c, v in g.br(ca, cb).items():
                rhs[c] = rhs.get(c, 0) + v * sa * sb    # (-1)(-1) = 1
            lhs = {c: v for c, v in lhs.items() if v}
            rhs = {c: v for c, v in rhs.items() if v}
            if lhs != rhs:
                ok2 = False
    log(f"  gl({m}|{n}): -st is a Lie superalgebra automorphism: {ok2}")
    # -st is even
    ok3 = all(g.pe(g.st(a)[1]) == g.pe(a) for a in range(g.dim))
    log(f"  gl({m}|{n}): -st preserves the parity: {ok3}")
    # -st(e_ii) = -e_ii is not an idempotent -> not Ad_g o phi_sigma
    ok4 = all(g.st(g.pos[(i, i)]) == (1, g.pos[(i, i)]) for i in range(g.N))
    log(f"  gl({m}|{n}): -st(e_ii) = -e_ii for every i: {ok4}")
    return ok and ok2 and ok3 and ok4


def der_dim(m, n, parity):
    """dim of the space of derivations of gl(m|n) of the given parity."""
    g = GL(m, n)
    d = g.dim
    # unknowns: D(e_a) = sum_b c[a][b] e_b with pe(b) = pe(a)+parity
    cols = []
    colof = {}
    for a in range(d):
        for b in range(d):
            if (g.pe(b) - g.pe(a) - parity) % 2 == 0:
                colof[(a, b)] = len(cols)
                cols.append((a, b))
    #  the equations are  D[e_a,e_b] - [De_a,e_b] - (-1)^{s p(a)}[e_a,De_b] = 0,
    #  one for every basis vector e_c of the value
    eqs = {}
    for a in range(d):
        for b in range(d):
            for c in range(d):
                eqs[(a, b, c)] = {}
    for a in range(d):
        for b in range(d):
            for cc, v in g.br(a, b).items():
                for e in range(d):
                    if (cc, e) in colof:
                        k = (a, b, e)
                        eqs[k][colof[(cc, e)]] = eqs[k].get(colof[(cc, e)], 0) + v
            for e in range(d):
                if (a, e) in colof:
                    for cc, v in g.br(e, b).items():
                        k = (a, b, cc)
                        eqs[k][colof[(a, e)]] = eqs[k].get(colof[(a, e)], 0) - v
                if (b, e) in colof:
                    sgn = (-1) ** (parity * g.pe(a))
                    for cc, v in g.br(a, e).items():
                        k = (a, b, cc)
                        eqs[k][colof[(b, e)]] = (eqs[k].get(colof[(b, e)], 0)
                                                 - sgn * v)
    mat = []
    for k, row in eqs.items():
        row = {c: v for c, v in row.items() if v}
        if row:
            mat.append([row.get(j, 0) for j in range(len(cols))])
    r = rank(mat) if mat else 0
    return len(cols) - r


def check_der(log):
    log(f"  linear algebra backend: {backend()}")
    res = {}
    for (m, n) in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2),
                   (2, 0), (3, 0)]:
        g = GL(m, n)
        de = der_dim(m, n, 0)
        do = der_dim(m, n, 1)
        # sdim gl(m|n), counted off the model, not asserted from m^2+n^2 | 2mn
        ge = sum(1 for a in range(g.dim) if g.pe(a) == 0)
        go = g.dim - ge
        assert (ge, go) == (m * m + n * n, 2 * m * n)
        res[(m, n)] = (de, do, ge, go)
        log(f"  gl({m}|{n}): sdim gl = ({ge}|{go}),  sdim Der = ({de}|{do}), "
            f"  Der - gl = ({de - ge:+d}|{do - go:+d})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    random.seed(20260827)
    fh = open(args.log, "w") if args.log else None

    def log(s):
        print(s)
        if fh:
            fh.write(s + "\n")
            fh.flush()

    log("1) e_00 as a sum of two supercommutators of banded matrices")
    ok1 = check_supercom(log)
    log(f"   => {'all OK' if ok1 else 'FAILED'}\n")

    log("2) the supertranspose")
    ok2 = True
    for (m, n) in [(1, 1), (2, 1), (2, 2), (3, 2)]:
        ok2 &= check_st(m, n, log)
    log(f"   => {'all OK' if ok2 else 'FAILED'}\n")

    log("3) derivations of gl(m|n)")
    check_der(log)
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
