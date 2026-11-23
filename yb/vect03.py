"""Claim 2.3 and Remark 2.4: the even 2-cocycle omega(x,y) = ell([x,y]) on g = vect(0|3),
where ell is the even linear form equal to 1 on xi_1 d_2, xi_2 d_3, xi_1 xi_2 xi_3 d_1 and to 0
on the other monomial vector fields.

Printed:
  * the four ranks of the proof of Claim 2.3.  With g_k the fields with coefficients of degree
    k+1, the Gram matrix of omega on g_ev = g_0 + g_2 is [[P, Q], [-Q^t, 0]] and on
    g_od = g_{-1} + g_1 it is [[0, R], [+-R^t, S]]; omega is non-degenerate iff rank Q = 3,
    rank R = 3, P is non-degenerate on the 6-dimensional annihilator of g_2 in g_0, and S is
    non-degenerate on the 6-dimensional annihilator of g_{-1} in g_1;
  * the rank of the whole 24 x 24 Gram matrix (a cross-check: 24);
  * the super Jacobi identity of the bracket used (all 24^3 triples of basis fields);
  * Remark 2.4: X in gl(3) acts on the ideal spanned by e_i = xi_1 xi_2 xi_3 d_i by the
    matrix -X^t + tr(X) 1_3.
Exact arithmetic over Q.  Run:  python3 vect03.py
"""
import itertools
import os
import sys
from fractions import Fraction as Fr

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")]
from exactlin import rank, nullspace  # noqa: E402

n = 3


def gmul(S, T):  # product of monomials xi_S xi_T (sorted tuples) -> (sign, U) or None
    if set(S) & set(T):
        return None
    L = list(S) + list(T)
    sign = 1
    for i in range(len(L)):          # bubble sort, counting the transpositions
        for j in range(len(L) - 1 - i):
            if L[j] > L[j + 1]:
                L[j], L[j + 1] = L[j + 1], L[j]
                sign = -sign
    return sign, tuple(L)


def gder(i, S):  # left derivative d/dxi_i of xi_S
    if i not in S:
        return None
    k = S.index(i)
    return (-1) ** k, tuple(x for x in S if x != i)


# basis: (S, i) is the field xi_S d_i
B = [(S, i) for r in range(n + 1) for S in itertools.combinations(range(n), r) for i in range(n)]


def par(b):
    return (len(b[0]) + 1) % 2


def deg(b):
    return len(b[0]) - 1


def apply(b, f):  # vector field b = (S, i) applied to a function f = {S: c}
    S, i = b
    out = {}
    for T, c in f.items():
        d = gder(i, T)
        if d is None:
            continue
        s1, T1 = d
        m = gmul(S, T1)
        if m is None:
            continue
        s2, U = m
        out[U] = out.get(U, 0) + c * s1 * s2
    return out


def bracket(a, b):
    # [X,Y] = sum_j (X(Y^j) - (-1)^{p(X)p(Y)} Y(X^j)) d_j
    out = {}
    for j in range(n):
        Yj = {b[0]: 1} if b[1] == j else {}
        Xj = {a[0]: 1} if a[1] == j else {}
        for U, c in apply(a, Yj).items():
            out[(U, j)] = out.get((U, j), 0) + c
        s = (-1) ** (par(a) * par(b))
        for U, c in apply(b, Xj).items():
            out[(U, j)] = out.get((U, j), 0) - s * c
    return {k: v for k, v in out.items() if v}


ell = {((0,), 1): 1, ((1,), 2): 1, ((0, 1, 2), 0): 1}


def omega(a, b):
    return sum(c * ell.get(k, 0) for k, c in bracket(a, b).items())


M = [[Fr(omega(a, b)) for b in B] for a in B]

# ------------------------------------------------ the four ranks of the proof of Claim 2.3
g = {k: [i for i, b in enumerate(B) if deg(b) == k] for k in (-1, 0, 1, 2)}
print("dim g_k for k = -1, 0, 1, 2:", [len(g[k]) for k in (-1, 0, 1, 2)])
# the blocks that must vanish by the grading argument
zero_blocks = [(a, b) for a in (-1, 0, 1, 2) for b in (-1, 0, 1, 2) if a + b not in (0, 2)]
nonzero = sum(1 for a, b in zero_blocks for i in g[a] for j in g[b] if M[i][j])
print("entries of omega pairing g_a with g_b, a+b not in {0,2}:", nonzero)


def block(I, J):
    return [[M[i][j] for j in J] for i in I]


def restricted_rank(Blk, K):
    """rank of the form Blk restricted to the span of the rows of K"""
    if not K:
        return 0
    return rank([[sum(K[r][i] * Blk[i][j] * K[s][j] for i in range(len(Blk)) for j in range(len(Blk)))
                  for s in range(len(K))] for r in range(len(K))])


Qm = block(g[0], g[2])                  # 9 x 3
Rm = block(g[-1], g[1])                 # 3 x 9
Pm = block(g[0], g[0])
Sm = block(g[1], g[1])
annQ = nullspace([list(col) for col in zip(*Qm)])   # x in g_0 with omega(x, g_2) = 0
annR = nullspace(Rm)                                # y in g_1 with omega(g_{-1}, y) = 0
print("rank Q (g_0 x g_2) =", rank(Qm))
print("rank R (g_-1 x g_1) =", rank(Rm))
print("dim of the annihilator of g_2 in g_0 =", len(annQ),
      "; rank of P on it =", restricted_rank(Pm, annQ))
print("dim of the annihilator of g_-1 in g_1 =", len(annR),
      "; rank of S on it =", restricted_rank(Sm, annR))
print("dim g =", len(B), "; rank of omega =", rank(M))

# ------------------------------------------------ the super Jacobi identity of the bracket
ok = True


def lin_br(x, y):
    out = {}
    for a, ca in x.items():
        for b, cb in y.items():
            for k, v in bracket(a, b).items():
                out[k] = out.get(k, 0) + ca * cb * v
    return {k: v for k, v in out.items() if v}


triples = 0
for a, b, c in itertools.product(B, repeat=3):
    triples += 1
    pa, pb, pc = par(a), par(b), par(c)
    lhs = lin_br({a: 1}, lin_br({b: 1}, {c: 1}))
    r1 = lin_br(lin_br({a: 1}, {b: 1}), {c: 1})
    r2 = lin_br({b: 1}, lin_br({a: 1}, {c: 1}))
    s = (-1) ** (pa * pb)
    tot = {}
    for d, sg in ((lhs, 1), (r1, -1), (r2, -s)):
        for k, v in d.items():
            tot[k] = tot.get(k, 0) + sg * v
    if any(tot.values()):
        ok = False
        break
print("super Jacobi identity on", triples, "triples:", ok)

# ------------------------------------------------ Remark 2.4: X acts on I by -X^t + tr(X) 1_3
cases = 0
for a in range(3):
    for b in range(3):
        X = ((a,), b)
        A = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for (U, j), c in bracket(X, ((0, 1, 2), i)).items():
                A[j][i] += c
        # xi_a d_b maps xi_b to xi_a: the matrix unit E_{ab}
        Xm = [[1 if (r == a and s == b) else 0 for s in range(3)] for r in range(3)]
        tr = 1 if a == b else 0
        exp = [[-Xm[s][r] + (tr if r == s else 0) for s in range(3)] for r in range(3)]
        assert A == exp, (a, b, A, exp)
        cases += 1
print("action of gl(3) on I is X -> -X^t + tr(X) 1_3:", cases, "of 9 matrix units checked")
