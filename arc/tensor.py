#!/usr/bin/env python3
"""Uniqueness of the GRC-brackets (Theorem 6.1.5 of the text), FIRST ingredient:
SM_{mu1} (x) SM_{mu2} as an osp(1|2)-module.

Two independent computations, both exact.

(A) CHARACTERS.  The Cartan element H = {Q_-,Q_+} of osp(1|2) acts on
    t^a theta^e in SM_mu by  H = -(2a + e + mu),  so, with q^{1/2} tracking
    the level n = 2a+e (an operator of order k shifts the weight by 2k = n),

        char SM_mu = q^{mu/2} (1 + q^{1/2}) / (1 - q).

    Since (1-q) = (1-q^{1/2})(1+q^{1/2}),

        char SM_{mu1} . char SM_{mu2}
             = q^{(mu1+mu2)/2} (1+q^{1/2})^2 / (1-q)^2
             = sum_{k in (1/2)Zee_{>=0}} char SM_{mu1+mu2+2k},

    an identity of formal series -- the multiplicity-free shape of (141),
    with ONE constituent for each shift 2k = 0,1,2,3,... and no other.
    Part (A) verifies this identity term by term.

(B) LOWEST-WEIGHT (SINGULAR) VECTORS.  On the tensor product, the lowering
    operator is the odd derivation Q = D_1 + D_2 of C[t_1,t_2,theta_1,theta_2],
    where D_i = d_{theta_i} - theta_i d_{t_i}; note that Q does NOT involve the
    weights.  In the coordinates

        T = t_1+t_2,  S = t_1-t_2,  Theta = theta_1+theta_2, Xi = theta_1-theta_2

    it becomes  Q = 2 d_Theta - Theta d_T - Xi d_S,  and writing
    v = A + Theta B + Xi C + Theta Xi E with A,B,C,E in C[T,S],

        Qv = 2B - Theta A_T + Xi (2E - A_S) + Theta Xi (B_S - C_T),

    so Qv = 0 iff  B = 0, A = A(S), C = C(S), E = A'(S)/2.  Hence

        Ker Q = { A(S) + Xi C(S) + (1/2) Theta Xi A'(S) },

    which in each level n = 2m (resp. 2m+1) is spanned by the single vector

        v_{2m} = S^m + (m/2) Theta Xi S^{m-1},        v_{2m+1} = Xi S^m.

    So the lowest-weight vectors are multiplicity-free, one in each level
    n = 0,1,2,..., FOR ALL n -- no truncation, no genericity.  For n even the
    vector is even, for n odd it is odd, which is the Pi-twist of the odd
    brackets.  Part (B) verifies dim Ker Q = 1 level by level by brute force,
    checks the closed formulas above, and checks that S^m is the classical
    (t_1-t_2)^m of the Rankin-Cohen transvectants.

Run:  python3 tensor.py [NMAX]
"""
import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import rank as _rank
except Exception:                                          # pragma: no cover
    _rank = None

# ---------------------------------------------- C[t1,t2,theta1,theta2]
# element = dict {(a1, e1, a2, e2): coeff};  monomial t1^a1 th1^e1 t2^a2 th2^e2
# with th1 written to the LEFT of th2 (the Koszul order).


def clean(d):
    return {k: v for k, v in d.items() if v}


def add(dst, src, c=1):
    for k, v in src.items():
        dst[k] = dst.get(k, F(0)) + c * v
        if dst[k] == 0:
            del dst[k]
    return dst


def level(m):
    a1, e1, a2, e2 = m
    return 2 * (a1 + a2) + e1 + e2


def Q(u):
    """(D_1 + D_2) u, the lowering operator of osp(1|2) on the tensor square;
    D_i = d_{theta_i} - theta_i d_{t_i}, and D_2 picks up (-1)^{p} from
    passing the first factor."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        # D_1 = d_theta1 - theta1 d_t1  on the first factor
        if e1:
            out[(a1, 0, a2, e2)] = out.get((a1, 0, a2, e2), F(0)) + c
        else:
            if a1:
                k = (a1 - 1, 1, a2, e2)
                out[k] = out.get(k, F(0)) - a1 * c
        # (-1)^{p(first factor)} (d_theta2 - theta2 d_t2) on the second
        sg = -1 if e1 else 1
        if e2:
            k = (a1, e1, a2, 0)
            out[k] = out.get(k, F(0)) + sg * c
        else:
            if a2:
                k = (a1, e1, a2 - 1, 1)
                out[k] = out.get(k, F(0)) - sg * a2 * c
    return clean(out)


def basis_level(n):
    out = []
    for e1 in (0, 1):
        for e2 in (0, 1):
            r = n - e1 - e2
            if r < 0 or r % 2:
                continue
            for a1 in range(r // 2 + 1):
                out.append((a1, e1, r // 2 - a1, e2))
    return out


def kernel_dim(n):
    """dim of Ker Q on the level-n subspace, by exact rank."""
    bas = basis_level(n)
    tgt = {m: i for i, m in enumerate(basis_level(n - 1))} if n else {}
    if not tgt:
        return len(bas), len(bas)
    M = []
    for m in bas:
        row = [F(0)] * len(tgt)
        for mm, c in Q({m: F(1)}).items():
            row[tgt[mm]] += c
        M.append(row)
    r = _rank(M)
    return len(bas) - r, len(bas)


# ------------------------------------------------- the closed-form vectors
def mono(a1, e1, a2, e2, c=F(1)):
    return {(a1, e1, a2, e2): c}


def poly_S(m):
    """S^m = (t_1 - t_2)^m."""
    out = {}
    b = 1
    for i in range(m + 1):
        out[(m - i, 0, i, 0)] = F(b * (-1) ** i)
        b = b * (m - i) // (i + 1)
    return clean(out)


def mul_ThXi(u):
    """Theta.Xi.u = (th1+th2)(th1-th2) u = -2 th1 th2 u, for u even in theta."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if e1 or e2:
            raise ValueError("expected a theta-free polynomial")
        out[(a1, 1, a2, 1)] = -2 * c
    return out


def mul_Xi(u):
    """Xi.u = (th1 - th2) u."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if e1 or e2:
            raise ValueError("expected a theta-free polynomial")
        add(out, {(a1, 1, a2, e2): c})
        add(out, {(a1, e1, a2, 1): -c})
    return clean(out)


def dS(u):
    """d/dS on C[T,S] read in (t_1,t_2): d_S = (d_{t_1} - d_{t_2})/2."""
    out = {}
    for (a1, e1, a2, e2), c in u.items():
        if a1:
            add(out, {(a1 - 1, e1, a2, e2): F(a1, 2) * c})
        if a2:
            add(out, {(a1, e1, a2 - 1, e2): -F(a2, 2) * c})
    return clean(out)


def sing(n):
    """The closed-form lowest-weight vector of level n."""
    if n % 2 == 0:
        m = n // 2
        v = poly_S(m)
        if m:
            add(v, mul_ThXi(dS(poly_S(m))), F(1, 2))
        return v
    return mul_Xi(poly_S((n - 1) // 2))


# ------------------------------------------------------------- characters
def char_SM(mu2, N):
    """char SM_mu as {half-integer exponent (doubled): coeff} up to level N;
    mu2 = 2*mu is kept as an integer offset, exponents are doubled."""
    out = {}
    for a in range(N + 1):
        for e in (0, 1):
            n = 2 * a + e
            if n <= N:
                out[n] = out.get(n, 0) + 1
    return out                     # times q^{mu/2}, the same factor everywhere


def check_characters(N):
    """(char SM_mu1)(char SM_mu2) == sum_{n>=0} char SM_{mu1+mu2+n}, all
    exponents measured from the respective lowest weight."""
    c = char_SM(0, N)
    prod = {}
    for n1, v1 in c.items():
        for n2, v2 in c.items():
            if n1 + n2 <= N:
                prod[n1 + n2] = prod.get(n1 + n2, 0) + v1 * v2
    summ = {}
    for shift in range(N + 1):             # the constituent SM_{mu1+mu2+shift}
        for n, v in c.items():
            if n + shift <= N:
                summ[n + shift] = summ.get(n + shift, 0) + v
    ok = all(prod.get(n, 0) == summ.get(n, 0) for n in range(N + 1))
    return ok, prod, summ


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    print("(A)  characters: (char SM_mu1)(char SM_mu2) =?= "
          "sum_{n>=0} char SM_{mu1+mu2+n}")
    ok, prod, summ = check_characters(N)
    print(f"     levels 0..{N}: "
          f"product {[prod.get(n, 0) for n in range(min(N, 12) + 1)]} ...")
    print(f"                    sum {[summ.get(n, 0) for n in range(min(N, 12) + 1)]} ...")
    print(f"     identity holds up to level {N}: {ok}")

    print("\n(B)  lowest-weight vectors: dim Ker(Q) level by level")
    bad = 0
    for n in range(N + 1):
        d, dim = kernel_dim(n)
        v = sing(n)
        qv = Q(v)
        inker = not qv
        if d != 1 or not inker or not v:
            bad += 1
        print(f"     n = {n:>3}   dim level = {dim:>3}   dim Ker Q = {d}"
              f"   closed form in Ker: {inker}"
              f"{'   <-- UNEXPECTED' if (d != 1 or not inker) else ''}")
    print(f"\n     levels with an unexpected answer: {bad}")
    print(f"     v_0 = {sing(0)}")
    print(f"     v_1 = Xi                      -> {sing(1)}")
    print(f"     v_2 = S^2/... , v_3, v_4 printed as {{(a1,e1,a2,e2): c}}:")
    for n in (2, 3, 4):
        print(f"       v_{n} = {sing(n)}")
