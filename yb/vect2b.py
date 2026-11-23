"""
Remark 5.22.7: for g = vect^L(1|2), over the windows |a| <= 1 and |a| <= 2 of Laurent exponents,
the recursion for H^1(g, t; g (x) g) has no non-trivial solution: Z^1 = B^1 over the window.
This is part 4 of vect2.py with fewer equations kept in memory.

Why it stays rigorous: dropping cocycle equations can only ENLARGE the solution space, so a
subsystem gives an upper bound for dim Z^1.  The coboundaries are computed exactly: every d r
with r and d r inside the window is one, and the coboundaries satisfy every cocycle equation.
Hence

        dim B^1  <=  dim Z^1  <=  dim Z^1(subsystem),

and if the two ends agree, Z^1 = B^1 over the window, whatever subsystem was used.  So the
number of equations kept is capped (CAP) and the two ends are compared.

Run:  python3 vect2b.py
"""

import os
import sys
from fractions import Fraction as Q

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), HERE]

from vect2 import G, weight, tensor_weight, act, basis, mono
from exactlin import rank, backend

CAP = 9000            # how many cocycle equations to keep


def run(N, say):
    B = basis(N)
    roots = [k for k in B if weight(k) != (0, 0, 0)]
    bywt = {}
    for u in B:
        for v in B:
            bywt.setdefault(tensor_weight((u, v)), []).append((u, v))

    cols, colof = [], {}
    for x in roots:
        for p in bywt.get(weight(x), []):
            colof[(x, p)] = len(cols)
            cols.append((x, p))
    say("  window |a| <= %d: dim g = %d, root vectors = %d, unknowns = %d"
        % (N, len(B), len(roots), len(cols)))

    def phi_of(x):
        if weight(x) == (0, 0, 0):
            return {}
        return {p: colof[(x, p)] for p in bywt.get(weight(x), [])}

    inside = set(B)
    eqs = []
    total = 0
    for i, x in enumerate(B):
        for y in B[i:]:
            br = G.brk(mono(x), mono(y))
            if any(k not in inside for k in br):
                continue
            acc = {}
            for p, col in phi_of(y).items():
                for k, c in act(x, p).items():
                    acc[(k, col)] = acc.get((k, col), 0) + c
            sgn = -((-1) ** (G.par(x) * G.par(y)))
            for p, col in phi_of(x).items():
                for k, c in act(y, p).items():
                    acc[(k, col)] = acc.get((k, col), 0) + sgn * c
            for k, c in br.items():
                for p, col in phi_of(k).items():
                    acc[(p, col)] = acc.get((p, col), 0) - c
            byt = {}
            for (tgt, col), c in acc.items():
                if c:
                    byt.setdefault(tgt, {})[col] = c
            for tgt, row in byt.items():
                if row:
                    total += 1
                    if len(eqs) < CAP:
                        eqs.append(row)
    say("  cocycle equations available: %d, kept for the rank: %d"
        % (total, len(eqs)))

    mat = [[Q(r.get(j, 0)) for j in range(len(cols))] for r in eqs]
    rk = rank(mat) if mat else 0
    z1 = len(cols) - rk
    say("  rank of the kept subsystem = %d  =>  dim Z^1 <= %d" % (rk, z1))
    del mat

    M0 = bywt.get((0, 0, 0), [])
    brows = []
    for p0 in M0:
        vec = [Q(0)] * len(cols)
        skip = False
        for x in roots:
            for k, c in act(x, p0).items():
                if (x, k) not in colof:
                    skip = True
                    break
                vec[colof[(x, k)]] += c
            if skip:
                break
        if not skip:
            brows.append(vec)
    b1 = rank(brows) if brows else 0
    say("  dim M_0 in the window = %d, of which %d give a d r inside it"
        % (len(M0), len(brows)))
    say("  dim B^1 = %d" % b1)
    say("  ==> dim B^1 = %d <= dim Z^1 <= %d %s"
        % (b1, z1, "  ***  EQUAL: H^1 = 0 over this window" if b1 == z1 else ""))
    return b1, z1


def main():
    def say(s=""):
        print(s, flush=True)

    say("backend: %s" % backend())
    for N in (1, 2):
        say("")
        say("=" * 70)
        run(N, say)


if __name__ == "__main__":
    main()
