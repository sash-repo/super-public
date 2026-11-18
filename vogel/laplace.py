#!/usr/bin/env python3
"""The derivations Skryabin's method cannot see: differential operators of
order > 1.

For the odd Hamiltonian (Buttin) series le(r|r; N) the odd Laplacian
    Delta = sum_i d_{x_i} d_{xi_i}       (order 2, the BV operator)
generates the antibracket, hence  Delta_f |-> Delta_{Delta f}  is a derivation of
le; it is odd and of degree -2, and it is induced by NO element of Der O, because
Delta is not a derivation of O.

For k(1|3) over F_3 the same happens with the order-3 operator
    d_{xi_1} d_{xi_2} d_{xi_3}:   K_f |-> K_{d d d f}.

This script builds both maps, checks that they are well defined and are
derivations, and checks that they are not in ad(L) + N_{Der O}(L)|_L.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dervect import (Super, Echelon, echelon, poisson, contact,   # noqa: E402
                     normalizer, der_dim)

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laplace.log")


class Coords:
    def __init__(self, S, B):
        self.S, self.B, self.n = S, B, len(B)
        M = [S.vec(X) for X in B]
        aug = [M[i][:] + [1 if j == i else 0 for j in range(self.n)]
               for i in range(self.n)]
        self.A, self.piv = echelon(aug, S.p)

    def __call__(self, v):
        S, p, n = self.S, self.S.p, self.n
        w = [x % p for x in v] + [0] * n
        for row, c in zip(self.A, self.piv):
            if c < S.dimD and w[c]:
                f = w[c]
                w = [(a - f * b) % p for a, b in zip(w, row)]
        return None if any(w[:S.dimD]) else [(-x) % p for x in w[S.dimD:]]


def op_on_mon(S, ops, u):
    """apply a composition of first-order generators to a monomial."""
    cur = {u: 1}
    for g in ops:
        nxt = {}
        for v, c in cur.items():
            w, c1 = S.gen_act(g, v)
            if c1:
                nxt[w] = (nxt.get(w, 0) + c * c1) % S.p
        cur = {k: v for k, v in nxt.items() if v}
    return cur


def main():
    with open(LOG, "w") as fh:
        for p in (3, 5):
            for r in (1, 2):
                S = Super(p, r, r)
                L, fields = poisson(S, 'le', r)
                if Echelon([S.vec(X) for X in L], p).dim > 90:
                    continue
                # Delta = sum_i d_{x_i} d_{xi_i}
                def delta(u, S=S, r=r):
                    out = {}
                    for i in range(r):
                        for w, c in op_on_mon(S, [('x', i, 0), ('xi', i)], u).items():
                            out[w] = (out.get(w, 0) + c) % S.p
                    return {k: v for k, v in out.items() if v}
                test_sum(f"p={p} le({r}|{r}), Delta", S, fields, delta, fh)
        # k(1|3) at p = 3: the order-3 operator
        S = Super(3, 1, 3, wt=[2, 1, 1, 1])
        L, fields = contact(S, 0)
        test_sum("p=3 k(1|3), d_xi1 d_xi2 d_xi3", S, fields,
                 lambda u, S=S: op_on_mon(S, [('xi', 0), ('xi', 1), ('xi', 2)], u),
                 fh)
        S = Super(5, 1, 3, wt=[2, 1, 1, 1])
        L, fields = contact(S, 0)
        test_sum("p=5 k(1|3), d_xi1 d_xi2 d_xi3", S, fields,
                 lambda u, S=S: op_on_mon(S, [('xi', 0), ('xi', 1), ('xi', 2)], u),
                 fh)


def test_sum(tag, S, fields, op, fh):
    """same as test(), but the operator is given as a function on monomials."""
    src = list(fields.values())
    E = Echelon([S.vec(X) for X in src], S.p)
    B = [S.unvec(r) for r in E.rows]
    n = len(B)
    co = Coords(S, B)
    pairs = []
    for u, X in fields.items():
        Y = {}
        for w, c in op(u).items():
            if w in fields:
                Y = S.add(Y, fields[w], c)
        cx, cy = co(S.vec(X)), co(S.vec(Y))
        assert cx is not None and cy is not None, "the operator leaves L"
        pairs.append((cx, cy))
    eqs = []
    for cx, cy in pairs:
        for a in range(n):
            row = [0] * (n * n)
            for b in range(n):
                row[a * n + b] = cx[b] % S.p
            eqs.append(row + [cy[a] % S.p])
    R, P = echelon(eqs, S.p)
    welldef = all(c < n * n for c in P)
    D = [[0] * n for _ in range(n)]
    for row, c in zip(R, P):
        if c < n * n:
            D[c // n][c % n] = row[-1] % S.p

    def apply(k):
        Y = {}
        for a in range(n):
            if D[a][k] % S.p:
                Y = S.add(Y, B[a], D[a][k])
        return Y

    isder = True
    for i in range(n):
        for j in range(n):
            c = co(S.vec(S.bracket(B[i], B[j])))
            lhs = {}
            for k in range(n):
                if c[k] % S.p:
                    lhs = S.add(lhs, apply(k), c[k])
            pi = S.parity(B[i])
            rhs = S.add(S.bracket(apply(i), B[j]),
                        S.bracket(B[i], apply(j)), (-1) ** pi)
            if S.vec(lhs) != S.vec(rhs):
                isder = False
                break
        if not isder:
            break
    ind = []
    NN = normalizer(S, B)
    for Dd in list(NN) + list(B):
        v = [0] * (n * n)
        for i in range(n):
            c = co(S.vec(S.bracket(Dd, B[i])))
            for a in range(n):
                v[a * n + i] = c[a] % S.p
        ind.append(v)
    EI = Echelon(ind, S.p)
    flat = [D[a][b] for a in range(n) for b in range(n)]
    novel = not EI.contains(flat)
    dD = der_dim(S, B)
    dN = Echelon([S.vec(X) for X in NN], S.p).dim
    s = (f"{tag}: dim L={n} well defined={welldef} is a derivation={isder} "
         f"not induced by Der O={novel} | dim Der L={dD} dim N={dN} "
         f"gap={dD - dN}")
    print(s)
    fh.write(s + "\n")
    fh.flush()


if __name__ == "__main__":
    main()
