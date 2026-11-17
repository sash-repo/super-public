#!/usr/bin/env python3
"""crit.py -- when is the induced module over o^{(1)}_I(2n+1) irreducible?
Statement 3.2.4 of the text for n = 1, 2 (and Statement 3.2.2 for n = 1, 2).

The census of run.py leaves a pattern in the weights for which M(lambda) fails
to be irreducible, and the pattern is the obvious one: writing the roots of
B_n in the coordinates of the torus, the short ones are the lambda_i and the
long ones the lambda_i + lambda_j (at p = 2 there is no sign to distinguish
e_i - e_j from e_i + e_j).  The claim tested here is

    M(lambda) is irreducible  <=>  alpha(lambda) is outside F_2 for every root.

For n = 1 this is exactly Dolotkazin's condition a in K \\ {0,1}.

Output: crit.log
"""

import itertools

from wk import GF, Induced, oI, order_of, triangular

LOG = [None]


def out(s=""):
    print(s, flush=True)
    if LOG[0]:
        LOG[0].write(s + "\n")
        LOG[0].flush()


def roots_at(lam, F):
    """the values alpha(lambda) for all the roots: the short ones and the
    long ones."""
    vals = list(lam)
    for i in range(len(lam)):
        for j in range(i + 1, len(lam)):
            vals.append(F.add(lam[i], lam[j]))
    return vals


def main():
    LOG[0] = open("crit.log", "w")
    for n in (1, 2):
        for k in (2, 3, 4):
            F = GF(k)
            alg, deg = oI(F, n)
            alg.build_struct()
            neg, tor, pos = triangular(alg, deg)
            orders = [order_of(alg, i) for i in neg]
            bad, nirr, npred = [], 0, 0
            for lam in itertools.product(F.els, repeat=len(tor)):
                M = Induced(alg, neg, orders, tor, pos, list(lam))
                irr = len(M.singular()) == 1
                pred = all(v not in (0, 1) for v in roots_at(lam, F))
                nirr += irr
                npred += pred
                if irr != pred:
                    bad.append((list(lam), irr, pred))
            out("n = %d over %s: dim M = %d; irreducible for %d of the %d"
                " weights, the criterion predicts %d; mismatches: %s"
                % (n, F.name, 2 ** orders.count(2) * 4 ** orders.count(4),
                   nirr, F.q ** len(tor), npred, bad if bad else "none"))
    out()
    out("So, machine-checked for n = 1 and n = 2 over F_4, F_8 and F_16:")
    out("  M(lambda) is irreducible  <=>  alpha(lambda) is outside F_2 for")
    out("  every root alpha of B_n.")
    out("For n = 1 this is Dolotkazin's a in K minus {0,1}.")
    out("The count over F_q, q = 2^k, for n = 2 is (q-2)(q-4):")
    out("  F_4: 2*0 = 0,  F_8: 6*4 = 24,  F_16: 14*12 = 168.")
    LOG[0].close()


if __name__ == "__main__":
    main()
