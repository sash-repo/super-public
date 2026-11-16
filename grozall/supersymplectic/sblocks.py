"""Does the printed formula (9) survive superization?

The note gives Gz for every m as one expression in seven blocks,

    Gz = 2P + G1 + G2 + 2G3 + 2G4 - 2G5 - M ,

and, for a symplectic SUPERmanifold, only a combination of 13 contraction
schemes read off by suniv.py -- no formula.  This program closes that gap.

The bridge is that every block of (9) is ONE contraction scheme, or a symmetric
pair of them.  Writing the operator that pairs a slot of A with a slot of A' by
omega as a letter,

    a = (dA , dA')      b = (dA , fiber A')    c = (fiber A , dA')
    e = (fiber A, fiber A')                    f = (dA , fiber A)
    g = (dA', fiber A')                        A,B = a derivative index left free

a scheme of order 2 with values in the metrics is a word in those letters, and

    P  ~ ae        G1 ~ cc, bb        G2 ~ cf, bg      G3 ~ ce|A, be|B
    G4 ~ bc        G5 ~ ef|A, eg|B    M  ~ ee|AB .

The constants are NOT assumed: each block is coded here in index form -- which
at n = 0 carries no signs and so cannot be got wrong -- and fitted against its
scheme(s) exactly.  The resulting vector is then tested, unchanged, on the
supermanifolds.

Usage:  python3 sblocks.py
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace                             # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sgen import enumerate_schemes, evaluate               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

KIND = {("Az", "Bz"): "a", ("Az", "Bu"): "b", ("Au", "Bz"): "c",
        ("Au", "Bu"): "e", ("Az", "Au"): "f", ("Bz", "Bu"): "g",
        # the four same-kind pairs; each of them is the zero operator for an
        # even omega (antisymmetry) but the enumeration still lists them
        ("Az", "Az"): "w", ("Bz", "Bz"): "x",
        ("Au", "Au"): "y", ("Bu", "Bu"): "z"}


def word(sch):
    letters = "".join(sorted(KIND[p] for p in sch.kinds))
    free = "A" * sch.FA + "B" * sch.FB
    return letters + ("|" + free if free else "")


def scheme_index(N=2):
    """word -> index into enumerate_schemes(N); the words are distinct."""
    schemes = enumerate_schemes(N)
    idx = {}
    for i, s in enumerate(schemes):
        w = word(s)
        assert w not in idx, f"two schemes with the word {w}"
        idx[w] = i
    return schemes, idx


# --------------------------------------------------- the blocks, as operators
#
# A block of (9) differs from the contraction scheme of the same word in ONE
# respect, and it is not a formality: in a scheme the free derivative index is
# produced by U(T) = sum u^i d_i T BEFORE the contractions, so a contraction may
# land on that very index; in a block the free index is the index of the answer
# and the contractions are the ones written in the index expression.  So here
# the contractions are performed first, on the fibers of A and A' alone, and U
# is applied at the end.  Everything else -- every Koszul sign -- is carried by
# the supercommutative arithmetic, exactly as in evaluate().

def evaluate_block(S, kinds, FA, FB, Atil, Btil):
    A = S.A
    state = [(Fraction(1), Atil, Btil, A.parity(Atil))]
    for (k1, k2) in kinds:
        new = []
        for (c, P, Q, pP) in state:
            for (i, j, w) in S.Omnz:
                cc = c * w
                PP, QQ, ppP = P, Q, pP
                for (kind, ix) in ((k1, i), (k2, j)):
                    pop = S.par[ix] if kind[1] == "z" else S.upar[ix]
                    if kind[0] == "A":
                        PP = S.dz(PP, ix) if kind[1] == "z" else S.du(PP, ix)
                        if not PP:
                            break
                        ppP = (ppP + pop) % 2
                    else:
                        if pop and ppP:
                            cc = -cc
                        QQ = S.dz(QQ, ix) if kind[1] == "z" else S.du(QQ, ix)
                        if not QQ:
                            break
                else:
                    if PP and QQ:
                        new.append((cc, PP, QQ, ppP))
        state = new
        if not state:
            return {}
    out = []
    for (c, P, Q, _) in state:
        for _ in range(FA):
            P = S.free_deriv(P)
        for _ in range(FB):
            Q = S.free_deriv(Q)
        if P and Q:
            out.append(A.scale(A.mul(P, Q), c))
    return A.summ(out)


# ------------------------------------------------------- the blocks, in indices
#
# Coded for n = 0, where every coordinate is even and no Koszul sign can arise;
# that is the whole point of fixing the constants there.

def comps(S, T):
    """A_{ij} from the u-encoded T, with T = sum_{ij} A_{ij} u^i u^j."""
    n = S.nc
    return [[S.A.scale(S.du(S.du(T, i), j), Fraction(1, 2)) for j in range(n)]
            for i in range(n)]


def rebuild(S, M):
    n = S.nc
    return S.A.summ([S.A.mul(S.A.mul(S.u(i), S.u(j)), M[i][j])
                     for i in range(n) for j in range(n)])


def _sym(S, raw):
    n = S.nc
    out = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = S.A.scale(S.A.add(raw[i][j], raw[j][i]), Fraction(1, 2))
            out[i][j] = s
            out[j][i] = s
    return out


def Dv(S, A):
    n = S.nc
    return [S.A.summ([S.A.scale(S.dz(A[i][l], p), w) for p, l, w in S.Omnz])
            for i in range(n)]


def zeta(S, A):
    d = Dv(S, A)
    return [S.A.summ([S.A.scale(d[r], w) for p2, r, w in S.Omnz if p2 == p])
            for p in range(S.nc)]


def Delta(S, A, f):
    return S.A.summ([S.A.scale(S.A.mul(A[k][l], S.dz(S.dz(f, p), q)), w1 * w2)
                     for k, p, w1 in S.Omnz for l, q, w2 in S.Omnz])


def piece_P(S, A, B):
    n = S.nc
    raw = [[S.A.summ([S.A.scale(S.bracket(A[i][l], B[j][s]), w)
                      for l, s, w in S.Omnz]) for j in range(n)] for i in range(n)]
    return _sym(S, raw)


def piece_G1a(S, A, B):
    """Delta_A(A'_ij) -- the cc half."""
    n = S.nc
    out = [[Delta(S, A, B[i][j]) for j in range(n)] for i in range(n)]
    return _sym(S, out)


def piece_G1b(S, A, B):
    return piece_G1a(S, B, A)


def piece_G2a(S, A, B):
    """zeta_A(A'_ij) -- the cf half."""
    n = S.nc
    zA = zeta(S, A)
    out = [[S.A.summ([S.A.mul(zA[p], S.dz(B[i][j], p)) for p in range(n)])
            for j in range(n)] for i in range(n)]
    return _sym(S, out)


def piece_G2b(S, A, B):
    return piece_G2a(S, B, A)


def piece_G3a(S, A, B):
    """sym  om^{kq} om^{ls} d_i A_kl d_q A'_js  -- the ce|A half."""
    n = S.nc
    raw = [[S.A.summ([S.A.scale(S.A.mul(S.dz(A[k][l], i), S.dz(B[j][s], q)),
                                w1 * w2)
                      for k, q, w1 in S.Omnz for l, s, w2 in S.Omnz])
            for j in range(n)] for i in range(n)]
    return _sym(S, raw)


def piece_G3b(S, A, B):
    return piece_G3a(S, B, A)


def piece_G4(S, A, B):
    n = S.nc
    raw = [[S.A.summ([S.A.scale(S.A.mul(S.dz(A[i][l], p), S.dz(B[j][s], q)),
                                w1 * w2)
                      for p, s, w1 in S.Omnz for l, q, w2 in S.Omnz])
            for j in range(n)] for i in range(n)]
    return _sym(S, raw)


def piece_G5a(S, A, B):
    """sym om^{ls} d_i (Dv A)_l A'_js  -- the ef|A half."""
    n = S.nc
    dA = Dv(S, A)
    raw = [[S.A.summ([S.A.scale(S.A.mul(S.dz(dA[l], i), B[j][s]), w)
                      for l, s, w in S.Omnz])
            for j in range(n)] for i in range(n)]
    return _sym(S, raw)


def piece_G5b(S, A, B):
    return piece_G5a(S, B, A)


def piece_M(S, A, B):
    n = S.nc
    raw = [[S.A.summ([S.A.scale(S.A.mul(S.dz(A[k][r], i), S.dz(B[l][s], j)),
                                w1 * w2)
                      for k, l, w1 in S.Omnz for r, s, w2 in S.Omnz])
            for j in range(n)] for i in range(n)]
    return _sym(S, raw)


# block name -> (index-form function, contraction kinds, FA, FB, its
#                coefficient in the printed formula (9))
BLOCKS = [
    ("P",   piece_P,   [("Az", "Bz"), ("Au", "Bu")], 0, 0, Fraction(2)),
    ("G1a", piece_G1a, [("Au", "Bz"), ("Au", "Bz")], 0, 0, Fraction(1)),
    ("G1b", piece_G1b, [("Az", "Bu"), ("Az", "Bu")], 0, 0, Fraction(1)),
    ("G2a", piece_G2a, [("Au", "Bz"), ("Az", "Au")], 0, 0, Fraction(1)),
    ("G2b", piece_G2b, [("Az", "Bu"), ("Bz", "Bu")], 0, 0, Fraction(1)),
    ("G3a", piece_G3a, [("Au", "Bz"), ("Au", "Bu")], 1, 0, Fraction(2)),
    ("G3b", piece_G3b, [("Az", "Bu"), ("Au", "Bu")], 0, 1, Fraction(2)),
    ("G4",  piece_G4,  [("Az", "Bu"), ("Au", "Bz")], 0, 0, Fraction(2)),
    ("G5a", piece_G5a, [("Au", "Bu"), ("Az", "Au")], 1, 0, Fraction(-2)),
    ("G5b", piece_G5b, [("Au", "Bu"), ("Bz", "Bu")], 0, 1, Fraction(-2)),
    ("M",   piece_M,   [("Au", "Bu"), ("Au", "Bu")], 1, 1, Fraction(-1)),
]


def ratio(lhs, rhs):
    """The rational k with lhs = k * rhs, or None."""
    k = None
    for key in set(lhs) | set(rhs):
        a = lhs.get(key, Fraction(0))
        b = rhs.get(key, Fraction(0))
        if b == 0:
            if a != 0:
                return None
            continue
        r = a / b
        if k is None:
            k = r
        elif r != k:
            return None
    return k


def fit_constant(S, kinds, FA, FB, blockfun, pairs):
    """The rational kappa with (index-form block) = kappa * (operator)."""
    k = None
    for (P, Q) in pairs:
        lhs = rebuild(S, blockfun(S, comps(S, P), comps(S, Q)))
        rhs = evaluate_block(S, kinds, FA, FB, P, Q)
        r = ratio(lhs, rhs)
        if r is None:
            return None
        if k is None:
            k = r
        elif k != r:
            return None
    return k


def gz9(S, kappa):
    """Formula (9), as an operator, with the constants of the encoding."""
    def op(P, Q):
        return S.A.summ([
            S.A.scale(evaluate_block(S, kinds, FA, FB, P, Q), c * kappa[name])
            for (name, fun, kinds, FA, FB, c) in BLOCKS])
    return op


def gz9_index(S, A, B):
    """Formula (9) itself, in indices; only meaningful for n = 0."""
    n = S.nc
    acc = [[{} for _ in range(n)] for _ in range(n)]
    for (name, fun, kinds, FA, FB, c) in BLOCKS:
        T = fun(S, A, B)
        for i in range(n):
            for j in range(n):
                acc[i][j] = S.A.add(acc[i][j], S.A.scale(T[i][j], c))
    return acc


def main():
    print("the constant of the encoding for each block of (9), at n = 0:")
    kappa = {}
    for m in (1, 2, 3):
        S = SSetup(m, 0)
        rng = random.Random(4242 + m)
        pairs = [(S.random_field(3, rng, nterms=3), S.random_field(3, rng, nterms=3))
                 for _ in range(2)]
        row = []
        for (name, fun, kinds, FA, FB, c) in BLOCKS:
            k = fit_constant(S, kinds, FA, FB, fun, pairs)
            row.append((name, k))
            if m == 1:
                kappa[name] = k
            elif k is not None and kappa[name] != k:
                print(f"  !! {name}: {kappa[name]} at m=1 but {k} at m={m}")
        print(f"  m = {m}: " + ", ".join(
            f"{n}={'X' if k is None else str(k)}" for n, k in row))
    print()
    if any(v is None for v in kappa.values()):
        print("A BLOCK IS NOT ITS OPERATOR -- stop")
        return

    # --- the check at n = 0: the operator form IS formula (9)
    print("at n = 0, the operator form equals formula (9) in indices:")
    for m in (1, 2, 3):
        S = SSetup(m, 0)
        rng = random.Random(9 + m)
        P = S.random_field(4, rng, nterms=3)
        Q = S.random_field(4, rng, nterms=3)
        lhs = gz9(S, kappa)(P, Q)
        rhs = rebuild(S, gz9_index(S, comps(S, P), comps(S, Q)))
        d = S.A.add(lhs, S.A.scale(rhs, -1))
        print(f"  m = {m}: {'SAME' if not d else 'DIFFERENT'}   "
              f"(non-zero: {'yes' if lhs else 'NO'})")
    print()

    # --- the test: is formula (9) invariant on a supermanifold?
    print("formula (9), unchanged, on symplectic supermanifolds:")
    for (m, n) in [(1, 0), (2, 0), (3, 0), (1, 1), (1, 2), (1, 3), (1, 4),
                   (2, 1), (2, 2), (2, 3)]:
        S = SSetup(m, n)
        rng = random.Random(77 + 5 * m + n)
        T = gz9(S, kappa)
        hams = [p for d in (1, 2, 3) for p in S.monomials(d)]
        ok, nonzero, nham = True, False, 0
        for _ in range(2):
            P = S.random_field(4, rng, nterms=3)
            Q = S.random_field(4, rng, nterms=3)
            val = T(P, Q)
            if val:
                nonzero = True
            for f in hams:
                lhs = S.lie(f, val)
                rhs = S.A.add(T(S.lie(f, P), Q), T(P, S.lie(f, Q)))
                nham += 1
                if lhs != rhs:
                    ok = False
                    break
            if not ok:
                break
        print(f"  (2m|n) = ({2*m}|{n}):  invariant: {'YES' if ok else 'NO '}"
              f"   non-zero: {'YES' if nonzero else 'NO '}"
              f"   ({nham} Hamiltonians)")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
