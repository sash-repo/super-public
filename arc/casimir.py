#!/usr/bin/env python3
"""
casimir.py -- the Casimir C = S^2 on SM_{mu_1} (x) SM_{mu_2}, exactly.

This is the lemma behind the proof of the upper bound in Subsection 7.2.8 of
the text: on the level n of the tensor product, the generalized eigenvalues of C
are c(nu_0), ..., c(nu_n), where nu_j = mu_1+mu_2+j and c(lambda) =
(1/4 - lambda/2)^2 -- for ALL weights, degenerate ones included.

Checked here:
  (a) C acts on SM_lambda by the scalar c(lambda);
  (b) c(lambda) = c(lambda') iff lambda' = lambda or lambda' = 1-lambda;
  (c) on the level n of the tensor product, dim ker (C-c)^{n+1} equals
      #{j <= n | c(nu_j) = c}, for every eigenvalue c -- degenerate weights
      included, where the decomposition of Theorem 7.2.5 is unavailable;
  (d) hence dim T^{[c(nu)]}_d <= 2, which is what bounds the multiplicity.

The engine is built on multdeg.py (the modules and the two odd generators).
"""

import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402

import multdeg as M  # noqa: E402


def cval(lam):
    return (F(1, 4) - lam / 2) ** 2


# ------------------------------------------------------------------ matrices

def sm_matrix(op, n, mu):
    """Matrix of op on SM_mu from level n to level n+delta; here 1x1 blocks."""
    raise NotImplementedError


def t_op_matrix(op, n, mu1, mu2):
    """Matrix of Q_op: T_n -> T_{n+d}, d = -1 for '-', +1 for '+'."""
    src, dst = M.t_basis(n), M.t_basis(n - 1 if op == '-' else n + 1)
    pos = {v: i for i, v in enumerate(dst)}
    mat = [[F(0)] * len(src) for _ in dst]
    for j, v in enumerate(src):
        for c, w in M.t_apply(op, v, mu1, mu2):
            mat[pos[w]][j] += c
    return mat


def mul(a, b):
    if not a or not b:
        return []
    n, k, m = len(a), len(b), len(b[0])
    out = [[F(0)] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        for t in range(k):
            if ai[t]:
                bt, c = b[t], ai[t]
                for j in range(m):
                    if bt[j]:
                        out[i][j] += c * bt[j]
    return out


def add(a, b, s=F(1)):
    return [[x + s * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def eye(n):
    return [[F(1) if i == j else F(0) for j in range(n)] for i in range(n)]


def casimir_on_level(n, mu1, mu2):
    """C = S^2 with S = (Q_-Q_+ - Q_+Q_-)/2 + 1/4, as a matrix on T_n."""
    qm_n = t_op_matrix('-', n, mu1, mu2)          # T_n -> T_{n-1}
    qp_n = t_op_matrix('+', n, mu1, mu2)          # T_n -> T_{n+1}
    qm_np = t_op_matrix('-', n + 1, mu1, mu2)     # T_{n+1} -> T_n
    qp_nm = t_op_matrix('+', n - 1, mu1, mu2) if n >= 1 else []   # T_{n-1}->T_n
    a = mul(qm_np, qp_n)                          # Q_-Q_+ on T_n
    b = mul(qp_nm, qm_n) if n >= 1 else [[F(0)] * (n + 1) for _ in range(n + 1)]
    s = add([[x / 2 for x in row] for row in a],
            [[x / 2 for x in row] for row in b], s=F(-1))
    s = add(s, [[F(1, 4) if i == j else F(0) for j in range(n + 1)]
                for i in range(n + 1)])
    return mul(s, s)


def gen_eigdim(mat, c, power):
    """dim ker (mat - c)^power."""
    n = len(mat)
    a = add(mat, eye(n), s=-c)
    p = a
    for _ in range(power - 1):
        p = mul(p, a)
    return len(nullspace(p)) if any(any(r) for r in p) else n


# ---------------------------------------------------------------------- runs

def main():
    with open("runs-casimir.txt", "w") as out:
        print("casimir.py -- exact check of the Casimir lemma of Subsec. 7.2.8\n",
              file=out)

        print("(a) C on SM_lambda is the scalar c(lambda) = (1/4-lambda/2)^2:",
              file=out)
        ok_a = True
        for lam in (F(0), F(-1), F(-2), F(1, 2), F(3), F(-5, 2), F(1, 3)):
            # SM_lambda is the tensor product with a trivial second factor? no:
            # compute directly on the basis t^a theta^e, levels 0..6
            for n in range(7):
                a, e = n // 2, n % 2
                # S = (Q_-Q_+ - Q_+Q_-)/2 + 1/4 on the 1-dimensional level
                qp = M.q_plus(a, e, lam)
                v_qp = qp[0] if qp else None
                val = F(0)
                if v_qp:
                    c1, (b1, f1) = v_qp
                    qm = M.q_minus(b1, f1)
                    if qm:
                        val += c1 * qm[0][0]
                qm = M.q_minus(a, e)
                if qm:
                    c1, (b1, f1) = qm[0]
                    qp2 = M.q_plus(b1, f1, lam)
                    if qp2:
                        val -= c1 * qp2[0][0]
                s = val / 2 + F(1, 4)
                if s * s != cval(lam):
                    ok_a = False
                    print(f"    *** lambda={lam}, level {n}: {s*s} != {cval(lam)}",
                          file=out)
        print(f"    -> {'holds for every level checked' if ok_a else 'FAILS'}",
              file=out)

        print("\n(b) c(lambda)=c(lambda') iff lambda'=lambda or lambda'=1-lambda:",
              file=out)
        ok_b = all(cval(lam) == cval(1 - lam) for lam in
                   (F(0), F(-3), F(5, 2), F(1, 3), F(-7, 4)))
        print(f"    -> {'holds' if ok_b else 'FAILS'}", file=out)

        print("\n(c) generalized eigenvalues of C on the level n of the tensor",
              file=out)
        print("    product are c(nu_0),...,c(nu_n)  [nu_j = mu1+mu2+j]:", file=out)
        ok_c = True
        weights = [(F(0), F(0)), (F(-1), F(-1)), (F(0), F(-2)), (F(-1), F(-2)),
                   (F(-2), F(-2)), (F(1, 2), F(-1, 2)), (F(1, 2), F(1, 2)),
                   (F(1, 3), F(-4, 3)), (F(-3), F(1)), (F(2), F(-5))]
        for mu1, mu2 in weights:
            for n in range(9):
                mat = casimir_on_level(n, mu1, mu2)
                want = {}
                for j in range(n + 1):
                    want[cval(mu1 + mu2 + j)] = want.get(cval(mu1 + mu2 + j), 0) + 1
                got = {c: gen_eigdim(mat, c, n + 1) for c in want}
                if got != want:
                    ok_c = False
                    print(f"    *** ({mu1},{mu2}), n={n}: got {got}, want {want}",
                          file=out)
            print(f"    ({mu1},{mu2}): levels 0..8 checked", file=out)
        print(f"    -> {'the multiset is as claimed in every case' if ok_c else 'FAILS'}",
              file=out)

        print("\n(d) the bound: dim of the generalized eigenspace of c(nu_d) on the",
              file=out)
        print("    level d is 1 outside, and 2 inside, the range M+1 <= d <= 2M+1",
              file=out)
        print("    for mu1+mu2 = -M in Z_{<=0}:", file=out)
        ok_d = True
        for mu1, mu2 in [(F(0), F(0)), (F(-1), F(-1)), (F(-2), F(-3)),
                         (F(1, 2), F(-1, 2)), (F(-1), F(2))]:
            s = mu1 + mu2
            if s != int(s) or s > 0:
                continue
            Mm = -int(s)
            for d in range(2 * Mm + 4):
                mat = casimir_on_level(d, mu1, mu2)
                c = cval(mu1 + mu2 + d)
                got = gen_eigdim(mat, c, d + 1)
                want = 2 if (Mm + 1 <= d <= 2 * Mm + 1) else 1
                if got != want:
                    ok_d = False
                    print(f"    *** ({mu1},{mu2}), d={d}: {got} != {want}", file=out)
            print(f"    ({mu1},{mu2}), M={Mm}: d = 0..{2*Mm+3} checked", file=out)
        print(f"    -> {'as claimed' if ok_d else 'FAILS'}", file=out)

        print(f"\nSUMMARY: (a) {ok_a}  (b) {ok_b}  (c) {ok_c}  (d) {ok_d}", file=out)


if __name__ == "__main__":
    main()
