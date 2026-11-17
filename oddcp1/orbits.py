"""The orbits of g = v_0 on P(H^1(T_1)): how many families with one odd parameter.

Two groups are of interest:
  aut(E) alone -- the convention of [LT], where the underlying CP^1 is held
                  pointwise fixed, so that the answer there is a projective space;
  g = aut(E) + sl(2) -- the whole automorphism group of the retract, which is
                  what an isomorphism of supermanifolds induces.

The orbit of a class v is computed as the span of the X.v for X in a basis of
the group's Lie algebra; adding v itself gives the orbit in the projectivization.
A rank computed at one class is a LOWER bound for the generic orbit dimension,
hence an UPPER bound for the number of moduli; where the bound is the whole of
P(H^1), the orbit is open and the classification is finite.
"""

from fractions import Fraction
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from exactlin import rank                                         # noqa: E402
import oddpar as O


def _primes(n):
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out):
            out.append(k)
        k += 1
    return out


def _samples(n):
    """A handful of classes; the largest orbit found among them is a lower
    bound for the generic one, hence an upper bound for the moduli."""
    pr = _primes(n)
    fib = [1, -1]
    while len(fib) < n:
        fib.append((-1) ** len(fib) * (abs(fib[-1]) + abs(fib[-2])))
    return [pr, fib, [1] * n, [(i * i + 1) % 11 + 1 for i in range(n)],
            [1 if i % 3 else 3 for i in range(n)]]


def report(ks, q=1, verbose=True):
    L = O.Layer(ks, q)
    if L.dim == 0:
        if verbose:
            print(f"{str(tuple(ks)):>10} q={q}  H^1 = 0: no obstruction")
        return None
    mats = {nm: L.mat(nm) for nm, _ in L.gens}
    names_aut = [nm for nm, _ in L.gens if nm not in ("e", "h", "f")]

    def odim(v, names):
        rows = []
        for nm in names:
            M = mats[nm]
            rows.append([sum(M[i][j] * v[j] for j in range(L.dim))
                         for i in range(L.dim)])
        return rank(rows + [list(v)]) - 1

    best_aut = best_all = -1
    for s in _samples(L.dim):
        v = [Fraction(c) for c in s[:L.dim]]
        if not any(v):
            continue
        best_aut = max(best_aut, odim(v, names_aut))
        best_all = max(best_all, odim(v, list(mats)))
    # the basis classes too -- the degenerate strata
    per_basis = []
    for i in range(L.dim):
        v = [Fraction(1 if j == i else 0) for j in range(L.dim)]
        per_basis.append((odim(v, names_aut), odim(v, list(mats))))
        best_aut = max(best_aut, per_basis[-1][0])
        best_all = max(best_all, per_basis[-1][1])
    P = L.dim - 1
    if verbose:
        print(f"{str(tuple(ks)):>10} q={q}  dim H^1 = {L.dim:>3}  dim P = {P:>3}"
              f"   max orbit: aut(E) {best_aut:>3}, g {best_all:>3}"
              f"   moduli <= {P - best_all}"
              f"   {'(g transitive)' if best_all == P else ''}")
    return dict(dim=L.dim, P=P, aut=best_aut, all=best_all,
                moduli=P - best_all, per_basis=per_basis, layer=L)
