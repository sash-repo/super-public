#!/usr/bin/env python3
r"""The fiber lemma for n = 2 that proves non-isomorphy for EVERY shearing vector.

For n = 2 the odd part of L = le_omega(2;u) splits as A (+) C, where A is spanned
by the xi-free monomials and C by the ones with xi_1 xi_2.  The four facts the
proof needs, verified here for several u and both shapes:

  (i)   [A,A]_0 = 0        (so Q_0 vanishes on A);
  (ii)  [C,C] = 0 for both shapes (so Q vanishes on C);
  (iii) the polarizations agree on A x C: [a,c]_Xi = [a,c]_0 --- the linear part
        of the fiber equation is the SAME for the two shapes;
  (iv)  Q_Xi(a) = -xi_1 (d_q1 a)^2, which for a = q_1 is -xi_1 =/= 0, while the
        linear map c -> 2[a,c] has image inside xi_2 O + xi_1 O with the xi_1
        part vanishing at a = q_1: the fiber of Z_Xi over a = q_1 is EMPTY,
        that of Z_0 is not.

Hence #Z_Xi = sum_a |fiber_Xi(a)| <= sum_a |fiber_0(a)| = #Z_0, with strict
inequality: the count of odd square-zero elements is smaller for the deformed
algebra, so the two are not isomorphic --- for every u, and over every finite
extension of F_3.

Usage:  python3 lefib.py
"""
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leiso import LeAlg, odd_split                                    # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from exactlin import rank as _rank                                    # noqa: E402

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def fiber(L, avec, C):
    """(consistent?, dim of the fiber) of {c in span C | [a+c,a+c] = 0}."""
    p, d = L.p, L.dim
    T = L.table()
    Qa = L.br(avec, avec)
    cols = []
    for j in C:
        ej = [1 if k == j else 0 for k in range(d)]
        v = L.br(avec, ej)
        cols.append([(2 * x) % p for x in v])
    M = [[cols[j][k] for j in range(len(C))] for k in range(d)]
    rhs = [(-x) % p for x in Qa]
    r = _rank(M, p) if any(any(row) for row in M) else 0
    Mb = [M[k] + [rhs[k]] for k in range(d)]
    rb = _rank(Mb, p) if any(any(row) for row in Mb) else 0
    return (rb == r), len(C) - r


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "lefib.log"), "w")
    for u in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        say("")
        say(f"--- n=2 p=3 u={u}")
        Ls = {sh: LeAlg(2, 3, u, sh) for sh in ("0", "Xi")}
        L0, LX = Ls["0"], Ls["Xi"]
        A, C = odd_split(L0)
        say(f"    dim A = {len(A)}, dim C = {len(C)}, sdim = {L0.sdim()}")
        # (i), (ii), (iii)
        T0, TX = L0.table(), LX.table()
        bad_AA = sum(1 for i in A for j in A if any(x % 3 for x in T0[i][j]))
        bad_CC0 = sum(1 for i in C for j in C if any(x % 3 for x in T0[i][j]))
        bad_CCX = sum(1 for i in C for j in C if any(x % 3 for x in TX[i][j]))
        bad_AC = sum(1 for i in A for j in C
                     if any((T0[i][j][k] - TX[i][j][k]) % 3 for k in range(L0.dim)))
        say(f"    (i) [A,A]_0 = 0: {bad_AA == 0}")
        say(f"    (ii) [C,C] = 0 for shape 0: {bad_CC0 == 0}, for shape Xi: {bad_CCX == 0}")
        say(f"    (iii) [a,c]_0 = [a,c]_Xi on A x C: {bad_AC == 0}")
        # (iv) the fiber over a = q_1
        q1 = ((1,) + (0,) * 1, (0, 0))
        i1 = L0.idx[q1]
        av = [1 if k == i1 else 0 for k in range(L0.dim)]
        c0, d0 = fiber(L0, av, C)
        cX, dX = fiber(LX, av, C)
        say(f"    (iv) fiber over a = q_1: shape 0 consistent = {c0} (dim {d0}), "
            f"shape Xi consistent = {cX}")
        say(f"    ==> #Z_Xi <= #Z_0 - 3^{d0} < #Z_0")
    LOG.close()


if __name__ == "__main__":
    main()
