"""The ansatz of contraction schemes on a symplectic supermanifold.

A bilinear differential operator of order N on tensor fields with fiber S^2(W^*)
is, by the first fundamental theorem for osp(n|2m) (Sergeev; Berele--Regev), a
linear combination of complete contractions, by copies of omega^{ij}, of

        (d...d A~)  x  (d...d B~)

with two indices left free.  In the u-encoding of ssuper.py the free indices are
just the u's that survive, so a scheme is described by

  FA, FB   -- how many derivative indices are left FREE (the operator
              U(T) = sum_i u^i d_{z^i}T applied to each factor), FA+FB <= 2;
  a, b     -- how many derivative slots are contracted on each factor,
              a + b = N - FA - FB;
  alpha,beta - how many fiber slots are contracted on each factor (the
              operators d_{u^k}), alpha + beta = 2 + FA + FB;
  a pairing of those N+2 contracted slots, each pair carrying one omega^{kl}.

Slots of the same kind on the same factor are interchangeable, so only the
multiset of pair-kinds matters; that is what is enumerated.

Every operator acting on the second factor is moved past the first one with the
Koszul sign; that is the ONLY sign the scheme evaluation adds by hand, and the
whole construction is calibrated in test_ssuper.py and verified end to end in
sverify.py.

Note that no scheme is ODD: the two derivatives of an omega-contraction have
equal parity (omega^{kl} = 0 unless p(k) = p(l)) and U is even.  By the first
fundamental theorem there are therefore no odd invariant operators here at all.
"""

from fractions import Fraction
from itertools import product


class Scheme:
    def __init__(self, FA, FB, a, b, alpha, beta, kinds):
        self.FA, self.FB = FA, FB
        self.a, self.b = a, b
        self.alpha, self.beta = alpha, beta
        self.kinds = kinds          # tuple of pairs of slot kinds, e.g. ("Az","Bu")

    def __repr__(self):
        return (f"Sch(F={self.FA}{self.FB},d={self.a}{self.b},"
                f"u={self.alpha}{self.beta},{'|'.join(x+'-'+y for x, y in self.kinds)})")

    def order(self):
        return self.a + self.b + self.FA + self.FB


def _matchings(slots):
    """All perfect matchings of a list of slot labels, as multisets of pairs."""
    if not slots:
        yield ()
        return
    a = slots[0]
    for idx in range(1, len(slots)):
        b = slots[idx]
        rest = slots[1:idx] + slots[idx + 1:]
        for m in _matchings(rest):
            yield ((a, b),) + m


def enumerate_schemes(N):
    out = []
    seen = set()
    for FA in range(3):
        for FB in range(3 - FA):
            F = FA + FB
            if F > N:
                continue
            for a in range(N - F + 1):
                b = N - F - a
                for alpha in range(2 + F + 1):
                    beta = 2 + F - alpha
                    # a factor with more d_u than u's is killed
                    if alpha > 2 + FA or beta > 2 + FB:
                        continue
                    slots = (["Az"] * a + ["Au"] * alpha
                             + ["Bz"] * b + ["Bu"] * beta)
                    for m in _matchings(slots):
                        key = (FA, FB, a, b, alpha, beta,
                               tuple(sorted(tuple(sorted(p)) for p in m)))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(Scheme(FA, FB, a, b, alpha, beta, m))
    return out


def evaluate(S, sch, Atil, Btil):
    """Evaluate the scheme on the two tensor fields, in the u-encoding."""
    A = S.A
    P = Atil
    for _ in range(sch.FA):
        P = S.free_deriv(P)
    Q = Btil
    for _ in range(sch.FB):
        Q = S.free_deriv(Q)
    if not P or not Q:
        return {}
    # state: list of (coeff, P, Q, parity of P)
    state = [(Fraction(1), P, Q, A.parity(P))]
    for (k1, k2) in sch.kinds:
        new = []
        for (c, P, Q, pP) in state:
            for (i, j, w) in S.Omnz:
                cc = c * w
                PP, QQ, ppP = P, Q, pP
                for (kind, idx) in ((k1, i), (k2, j)):
                    pop = S.par[idx] if kind[1] == "z" else S.upar[idx]
                    if kind[0] == "A":
                        PP = S.dz(PP, idx) if kind[1] == "z" else S.du(PP, idx)
                        if not PP:
                            break
                        ppP = (ppP + pop) % 2
                    else:
                        # move the operator past the first factor
                        if pop and ppP:
                            cc = -cc
                        QQ = S.dz(QQ, idx) if kind[1] == "z" else S.du(QQ, idx)
                        if not QQ:
                            break
                else:
                    if PP and QQ:
                        new.append((cc, PP, QQ, ppP))
        state = new
        if not state:
            return {}
    return A.summ([A.scale(A.mul(P, Q), c) for (c, P, Q, _) in state])
