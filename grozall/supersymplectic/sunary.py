"""UNARY invariant differential operators on metrics over an ODD symplectic form.

Same ansatz as sgen.py, one factor only.  A unary scheme is

  FA     -- how many derivative indices are left free (U(T) = sum_i u^i d_{z^i}),
  a      -- how many derivative slots are contracted,
  alpha  -- how many fiber slots are contracted (the operators d_{u^k}),

with a + alpha even (they pair up, one omega^{kl} per pair) and
alpha <= 2 + FA.  The target fiber is the space of polynomials of degree
2 + FA - alpha in u, so

        2 + FA - alpha = 2   ->  metrics again   (alpha = FA)
        2 + FA - alpha = 0   ->  functions       (alpha = FA + 2)
        2 + FA - alpha = 1   ->  1-forms         (alpha = FA + 1)

and the order of the operator is a + FA.  Parity = (a+alpha)/2 mod 2 for the odd
form (each omega-contraction is odd), 0 for the even one.

These are what a COMPOSITION is built of: the bilinear operators of order 2 that
are compositions are the mu(P(A), Q(B)) with mu an order-0 invariant pairing and
P, Q unary invariants whose orders add up to 2.
"""

import os
import sys
from fractions import Fraction

from sgen import _matchings


class UScheme:
    def __init__(self, FA, a, alpha, kinds):
        self.FA, self.a, self.alpha, self.kinds = FA, a, alpha, kinds

    def order(self):
        return self.a + self.FA

    def degree(self):
        """degree in u of the result: 2 = metric, 1 = 1-form, 0 = function."""
        return 2 + self.FA - self.alpha

    def parity(self, form):
        return (((self.a + self.alpha) // 2) % 2) if form == "odd" else 0

    def __repr__(self):
        tgt = {2: "metric", 1: "1-form", 0: "function"}.get(self.degree(), "?")
        return (f"U(F={self.FA},d={self.a},u={self.alpha},"
                f"{'|'.join(x + '-' + y for x, y in self.kinds)})->{tgt}")


def enumerate_unary(order, degree=2, maxF=3):
    """All unary schemes of the given order landing in the given u-degree."""
    out, seen = [], set()
    for FA in range(min(order, maxF) + 1):
        a = order - FA
        alpha = 2 + FA - degree
        if alpha < 0 or alpha > 2 + FA:
            continue
        if (a + alpha) % 2:
            continue
        slots = ["Az"] * a + ["Au"] * alpha
        for m in _matchings(slots):
            key = (FA, a, alpha, tuple(sorted(tuple(sorted(p)) for p in m)))
            if key in seen:
                continue
            seen.add(key)
            out.append(UScheme(FA, a, alpha, m))
    return out


def evaluate_unary(S, sch, Atil):
    """Evaluate a unary scheme on one tensor field, in the u-encoding."""
    A = S.A
    P = Atil
    for _ in range(sch.FA):
        P = S.free_deriv(P)
    if not P:
        return {}
    state = [(Fraction(1), P)]
    for (k1, k2) in sch.kinds:
        new = []
        for (c, P) in state:
            for (i, j, w) in S.Omnz:
                cc = c * w
                PP = P
                for (kind, idx) in ((k1, i), (k2, j)):
                    PP = S.dz(PP, idx) if kind[1] == "z" else S.du(PP, idx)
                    if not PP:
                        break
                else:
                    new.append((cc, PP))
        state = new
        if not state:
            return {}
    return A.summ([A.scale(P, c) for (c, P) in state])


def defect_unary(S, sch_or_vec, schemes, f, Atil, pT):
    """L_f(D A) - (-1)^{pT p(X_f)} D(L_f A)."""
    A = S.A
    pX = (A.parity(f) + (1 if S.form == "odd" else 0)) % 2
    e = -1 if (pT * pX) % 2 else 1

    def D(X):
        if schemes is None:
            return evaluate_unary(S, sch_or_vec, X)
        return A.summ([A.scale(evaluate_unary(S, s, X), c)
                       for c, s in zip(sch_or_vec, schemes) if c])

    return A.add(S.lie(f, D(Atil)), A.scale(D(S.lie(f, Atil)), -e))


def invariant_unary(S, order, degree, rng, nsamp=2, fdeg=None, nterms=3,
                    verbose=True):
    """The invariant unary operators of the given order and target degree.

    Returns (schemes, list of coefficient vectors spanning the invariant ones
    that are non-zero as operators).
    """
    _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
    from exactlin import nullspace, rank

    A = S.A
    schemes = enumerate_unary(order, degree)
    if not schemes:
        return [], []
    pars = {s.parity(S.form) for s in schemes}
    assert len(pars) == 1, f"schemes of mixed parity: {pars}"
    pT = pars.pop()
    if fdeg is None:
        fdeg = order + 2
    hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
    rows = []
    for _ in range(nsamp):
        Atil = S.random_field(fdeg, rng, nterms=nterms)
        for f in hams:
            ds = [defect_unary(S, s, None, f, Atil, pT) for s in schemes]
            keys = set()
            for d in ds:
                keys |= set(d)
            for k in sorted(keys):
                rows.append([d.get(k, Fraction(0)) for d in ds])
    ns = (nullspace(rows) if rows else
          [[Fraction(1) if i == j else Fraction(0) for j in range(len(schemes))]
           for i in range(len(schemes))])
    # drop the combinations that are the zero operator
    probes = [S.random_field(max(3, order + 1), rng, nterms=3) for _ in range(2)]
    fps = []
    for v in ns:
        row = []
        for X in probes:
            row.append(A.summ([A.scale(evaluate_unary(S, s, X), c)
                               for c, s in zip(v, schemes) if c]))
        fps.append(row)
    keys = [sorted(set().union(*[set(fp[i]) for fp in fps])) if fps else []
            for i in range(len(probes))]
    M = [[fp[i].get(k, Fraction(0)) for i in range(len(probes)) for k in keys[i]]
         for fp in fps]
    dim = rank(M) if M and M[0] else 0
    if verbose:
        tgt = {2: "metrics", 1: "1-forms", 0: "functions"}[degree]
        print(f"  unary, order {order} -> {tgt:9s}: {len(schemes):2d} schemes, "
              f"kernel {len(ns)}, DIMENSION {dim}   "
              f"(parity {'odd' if pT else 'even'})")
    return schemes, ns, dim
