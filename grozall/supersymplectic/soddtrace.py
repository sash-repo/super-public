"""The trace of the composites: is the picture closed?

Everything of order 2 over a periplectic form factors through the periplectic
trace ptr (Corollary "everything factors through the trace").  The natural next
question is what ptr does to the answers -- if ptr(C_i(A,B)) is again an expression
in pi = ptr A and pi' = ptr B alone, then the whole order-2 theory closes on the
pair of odd functions (pi,pi') and nothing else of A and B ever enters.

The candidates are the only expressions of the right parity.  ptr is odd and a
metric field is even, so ptr(C_i(A,B)) is ODD; the odd expressions available in
two odd functions are {pi,pi'} (Buttin's antibracket, which is odd, so the bracket
of two odd functions is odd) and the two products X_pi(pi'), X_pi'(pi) -- which
are the same bracket up to a sign -- and 0.

Nothing is guessed: each candidate is compared exactly, on random fields, at
k = 2, 3, 4.

Usage:  python3 soddtrace.py
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]

from ssuper import SSetup                                  # noqa: E402
from sodd0 import ptr                                      # noqa: E402


def ratio(A, x, y):
    """The constant c with x = c*y, or None."""
    if not x and not y:
        return Fraction(0)
    if not y or not x:
        return None
    c = None
    keys = set(x) | set(y)
    for kk in keys:
        a, b = x.get(kk, 0), y.get(kk, 0)
        if b == 0:
            if a != 0:
                return None
            continue
        r = Fraction(a, 1) / Fraction(b, 1)
        if c is None:
            c = r
        elif c != r:
            return None
    return c


def main():
    for k in (2, 3, 4):
        S = SSetup(k, k, form="odd")
        A = S.A
        rng = random.Random(60900 + k)
        print(f"--- k = {k}, sdim ({k}|{k}) ---")
        rows = []
        for t in range(3):
            P = S.random_field(3, rng, nterms=3)
            Q = S.random_field(3, rng, nterms=3)
            pi, pip = ptr(S, P), ptr(S, Q)
            C1 = S.lie(pi, Q)
            C2 = S.lie(pip, P)
            C3 = A.mul(S.dform(pi), S.dform(pip))
            br = S.bracket(pi, pip)
            vals = {"ptr C1": ptr(S, C1), "ptr C2": ptr(S, C2),
                    "ptr C3": ptr(S, C3)}
            rows.append({name: ratio(A, v, br) for name, v in vals.items()})
            if t == 0:
                pb = A.parity(br)
                print(f"   {{pi,pi'}} is {'odd' if pb else 'even'}, "
                      f"non-zero: {'YES' if br else 'NO'}")
                for name, v in vals.items():
                    print(f"   {name}: non-zero {'YES' if v else 'NO '}, "
                          f"parity {A.parity(v) if v else '-'}")
        for name in ("ptr C1", "ptr C2", "ptr C3"):
            cs = [r[name] for r in rows]
            same = all(c == cs[0] for c in cs)
            print(f"   {name} = c * {{pi,pi'}} with c = {cs[0]}"
                  f"   (same on all {len(cs)} probes: {same})")

        # the kernel of ptr on the 3-space, in a symmetry-adapted basis:
        # C1+C2 is the symmetric operator, C1-C2+2C3 the antisymmetric one
        rng2 = random.Random(90000 + k)
        for label, coef in (("C1+C2", (1, 1, 0)), ("C1-C2+2C3", (1, -1, 2)),
                            ("C1 (control)", (1, 0, 0))):
            allzero = True
            for _ in range(3):
                P = S.random_field(3, rng2, nterms=3)
                Q = S.random_field(3, rng2, nterms=3)
                pi, pip = ptr(S, P), ptr(S, Q)
                ops = [S.lie(pi, Q), S.lie(pip, P),
                       A.mul(S.dform(pi), S.dform(pip))]
                comb = A.summ([A.scale(o, Fraction(c))
                               for c, o in zip(coef, ops) if c])
                if ptr(S, comb):
                    allzero = False
            print(f"   ptr({label}) = 0 : {allzero}")


if __name__ == "__main__":
    main()
