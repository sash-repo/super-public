"""A lucid shape for the missing degree-5 identity.

The monomials of the free nonassociative algebra are an ugly basis; the natural
one is built from the two products a Volichenko algebra really has,

        [a,b] = ab - ba          and          a o b = ab + ba,

so a basis monomial is a binary tree with UNORDERED children, each internal
node carrying one of the two symbols.  This sweeps those, antisymmetrized over
three of the variables and over the remaining two -- the symmetry type of the
one identity Volichenko's scheme cannot reach.
"""

import itertools
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref, rank as ex_rank  # noqa

from videal import MSpace, identities, mmul, mcomm, mcirc, var, mevaluate
from vcand import orbit, IRRDIM
from vchar import module_of
from vfind import build_C, oracle, test, alt, P
from veventest import even_identities


def cb_monomials(S):
    """All [,]/o-monomials on the index set S: (expression, printable name)."""
    S = tuple(S)
    if len(S) == 1:
        return [({S[0]: Fr(1)}, "x%d" % S[0])]
    out = []
    first = S[0]
    rest = S[1:]
    for k in range(len(rest) + 1):
        for pick in itertools.combinations(rest, k):
            A = (first,) + pick                    # the block holding S[0]
            B = tuple(x for x in S if x not in A)
            if not B:
                continue
            for fa, na in cb_monomials(A):
                for fb, nb in cb_monomials(B):
                    out.append((mcomm(fa, fb), "[%s,%s]" % (na, nb)))
                    out.append((mcirc(fa, fb), "(%s o %s)" % (na, nb)))
    return out


if __name__ == "__main__":
    n = 5
    t0 = time.time()
    ms, C = build_C(n)
    I = identities(n, ms)
    I_rows = [[int(x) % P for x in ms.vec(f)] for f in I]

    # everything Volichenko's scheme can reach: C_5 + all even-element ones
    rows = list(C) + [[int(x) % P for x in ms.vec(f)] for f in even_identities(n)]
    R, _ = ex_rref(rows, p=P)
    R = [r for r in R if any(r)]
    print("known (C_5 + every even-element identity): dim %d of I_5 = %d"
          % (len(R), len(I)), flush=True)
    phis = oracle(R, I_rows)

    mons = cb_monomials(tuple(range(1, n + 1)))
    print("[,]/o-monomials of degree 5: %d" % len(mons), flush=True)

    schemes = [("Alt(a,b,c) Alt(d,e)", lambda f: alt(alt(f, [1, 2, 3], n), [4, 5], n)),
               ("Alt(a,b,c)",          lambda f: alt(f, [1, 2, 3], n)),
               ("Alt(d,e)",            lambda f: alt(f, [4, 5], n)),
               ("Alt(a,b,c,d)",        lambda f: alt(f, [1, 2, 3, 4], n))]

    for label, op in schemes:
        hits = []
        for f, name in mons:
            g = op(f)
            if not g or mevaluate(g):
                continue                      # not an identity
            if test(ms.vec(g), phis):
                hits.append((name, g, len(g)))
        print("\n%s :  %d monomials give a NEW identity" % (label, len(hits)),
              flush=True)
        hits.sort(key=lambda h: h[2])
        for name, g, k in hits[:10]:
            print("      %-40s (%d monomials after expansion)" % (name, k))
        if hits:
            rr = list(R) + [[int(x) % P for x in ms.vec(hits[0][1])]]
            dec, _ = module_of(rr, ms, n, P)
            dec0, _ = module_of(R, ms, n, P)
            dd = {k: dec.get(k, 0) - dec0.get(k, 0)
                  for k in set(dec) | set(dec0)}
            dd = {k: m for k, m in dd.items() if m}
            print("      the first generates %s (dim %d); all together: %d"
                  % (" + ".join(sorted(dd)),
                     sum(m * IRRDIM[n][k] for k, m in dd.items()),
                     ex_rank(list(R) + [[int(x) % P for x in ms.vec(g)]
                                        for _, g, _ in hits], p=P)))
    print("\n[%.0fs]" % (time.time() - t0))
