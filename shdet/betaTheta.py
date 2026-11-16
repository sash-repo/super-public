"""
Read off the third cohomology class (the "beta" one) of k^L'(1|4) in the
theta-realisation: solve for Z^2, quotient by B^2 and by the two classes the
printed support already produces (alpha and gamma), and print the surviving
cocycle on the named basis vectors of (monBas).
"""

from fractions import Fraction as Q
import sys

from stringy import mono
from nsTheta import monomials, weight, name_of, ODD
from solveTheta import (Unknowns, by_weight, eqs, coboundaries, reduce_rows,
                        nullspace, rank, S0KEY)


def run(B=2, kappa=Q(1)):
    A = 2 * B + 1
    keysA = [k for k in monomials(range(-A, A + 1)) if k != S0KEY]
    keysB = [k for k in monomials(range(-B, B + 1)) if k != S0KEY]
    U = Unknowns()
    WA = by_weight(keysA)
    for k1 in keysB:
        for k2 in WA.get(-weight(k1), ()):
            U.get(k1, k2)
    nB = len(U.keys)
    rows = list(eqs(U, keysB, kappa))
    piv = reduce_rows(rows)
    Zb = [v[:nB] for v in nullspace(piv, len(U.keys))]
    cob = coboundaries(U, keysB, nB, kappa)
    return U, nB, Zb, cob


def named(U, vec, nB):
    out = {}
    for i in range(nB):
        if vec[i]:
            k1, k2 = U.keys[i]
            n1, n2 = name_of(k1), name_of(k2)
            out[((n1[0], n1[1]), (n2[0], n2[1]))] = vec[i] * n1[2] * n2[2]
    return out


def reduce_against(vecs, target, n):
    """subtract from `target` its component along span(vecs) as far as
    elimination allows; returns the residue."""
    piv = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in vecs])
    row = {i: x for i, x in enumerate(target) if x}
    while row:
        p = min(row)
        if p not in piv:
            break
        c = row[p]
        for i, v in piv[p].items():
            row[i] = row.get(i, 0) - c * v
        row = {i: v for i, v in row.items() if v}
    out = [Q(0)] * n
    for i, v in row.items():
        out[i] = v
    return out


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    U, nB, Zb, cob = run(B)
    print("dim Z (window) = %d, dim B = %d, dim H = %d"
          % (rank(Zb, nB), rank(cob, nB), rank(Zb, nB) - rank(cob, nB)))

    # the two classes the printed support produces, as vectors on the window
    from fitTheta2 import c_mono as fit_c_mono, IDX, N as NFIT

    def build(coeffs):
        """coeffs: dict name -> value, in fitTheta2's layout."""
        vec = [Q(0)] * nB
        param = [Q(0)] * NFIT
        for k, v in coeffs.items():
            param[IDX[k]] = Q(v)
        for i in range(nB):
            k1, k2 = U.keys[i]
            cv = fit_c_mono(k1, k2)
            vec[i] = sum(cv[j] * param[j] for j in range(NFIT))
        return vec

    alpha = build(dict([("A", 1), ("s", 1), ("f1", 1), ("f2", 1), ("f3", 1),
                        ("f4", 1)] +
                       [("a%d" % i, -1) for i in ODD] +
                       [("b%d%d" % (i, j), -1)
                        for i in ODD for j in ODD if i < j]))
    gamma = build({"p": 2, "u1": 1, "u2": 1, "u3": 1, "u4": 1})

    known = [v for v in cob] + [alpha, gamma]
    print("rank of coboundaries + alpha + gamma = %d" % rank(known, nB))

    # find a cocycle outside that span
    for v in Zb:
        r = reduce_against(known, v, nB)
        if any(r):
            print("\nthe third class, on the named basis "
                  "(entries with |index| <= %d):" % B)
            nm = named(U, r, nB)
            for (x, y), val in sorted(nm.items(),
                                      key=lambda kv: (kv[0][0][0], kv[0][1][0],
                                                      kv[0][0][1])):
                print("   c(%s_%s, %s_%s) = %s" % (x[0], x[1], y[0], y[1], val))
            break
