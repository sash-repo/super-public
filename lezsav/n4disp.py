#!/usr/bin/env python3
"""n4disp.py -- check the displays printed in the text against the vacuum.

The text displays the level-0 and level-2 parts of the vacuum.  This builds
those two matrices from the printed formulas, in the ring itself, and
compares them with n4rec11-vacuum.pkl entry by entry, so that no display
goes out on a transcription.

Odd labels: 1 = eta^1_-, 2 = rho_{1-}, 3 = eta^2_-, 4 = rho_{2-},
            5 = eta^1_+, 6 = rho_{1+}, 7 = eta^2_+, 8 = rho_{2+}.

Usage: tools/venv/bin/python n4disp.py   ->  n4disp-out.txt
"""
import pickle
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from n4noeth import C, mono, mul, add, smul, sub, eq         # noqa: E402

OUT = []


def log(*a):
    s = " ".join(str(x) for x in a)
    OUT.append(s)
    print(s, flush=True)
    with open("n4disp-out.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")


E1M, R1M, E2M, R2M, E1P, R1P, E2P, R2P = 1, 2, 3, 4, 5, 6, 7, 8
I = C(0, 1)


def th(*S, c=1):
    return mono(S=S, c=c)


def level(f, L):
    return {k: v for k, v in f.items() if len(k[3]) == L}


def main():
    with open("n4rec11-vacuum.pkl", "rb") as fh:
        q = pickle.load(fh)

    # ---- the display of q^{(0)} ----------------------------------------
    pref0 = mono(nw=-1, nm=-1)
    claim0 = [[smul(2, pref0), {}], [{}, smul(2, pref0)]]

    # ---- the display of q^{(2)} ----------------------------------------
    p = mono(nw=-2, nm=-1)

    def E(a, s):          # eta^a_-, eta^a_+
        return {(1, -1): th(E1M), (1, 1): th(E1P),
                (2, -1): th(E2M), (2, 1): th(E2P)}[(a, s)]

    def R(a, s):
        return {(1, -1): th(R1M), (1, 1): th(R1P),
                (2, -1): th(R2M), (2, 1): th(R2P)}[(a, s)]

    def m2(*terms):
        return add(*terms)

    a11 = m2(
        smul(-1, mul(E(1, -1), R(1, -1))),
        mul(E(2, -1), R(2, -1)),
        mul(E(1, 1), R(1, 1)),
        smul(-1, mul(E(2, 1), R(2, 1))),
        smul(2 * I, add(mul(R(1, -1), E(1, 1)), mul(E(2, -1), R(2, 1)))),
    )
    a22 = m2(
        mul(E(1, -1), R(1, -1)),
        smul(-1, mul(E(2, -1), R(2, -1))),
        smul(-1, mul(E(1, 1), R(1, 1))),
        mul(E(2, 1), R(2, 1)),
        smul(2 * I, add(mul(E(1, -1), R(1, 1)), mul(R(2, -1), E(2, 1)))),
    )
    a12 = smul(-2, mul(add(E(1, -1), smul(I, E(1, 1))),
                       add(R(2, -1), smul(I, R(2, 1)))))
    a21 = smul(2, mul(add(R(1, -1), smul(I, R(1, 1))),
                      add(E(2, -1), smul(I, E(2, 1)))))
    claim2 = [[mul(p, a11), mul(p, a12)], [mul(p, a21), mul(p, a22)]]

    ok = True
    for i in (0, 1):
        for j in (0, 1):
            for L, claim in ((0, claim0), (2, claim2)):
                got = level(q[i][j], L)
                want = claim[i][j]
                same = eq(got, want)
                ok = ok and same
                log("q[%d][%d] level %d: %s (%d terms printed, %d in the "
                    "vacuum)" % (i + 1, j + 1, L,
                                 "OK" if same else "MISMATCH",
                                 len(want), len(got)))
                if not same:
                    d = sub(got, want)
                    for k, v in sorted(d.items(), key=lambda t: str(t[0])):
                        log("     difference: %s  %s" % (k, v))
    log("")
    log("all displays reproduce the vacuum: %s" % ok)


if __name__ == "__main__":
    main()
