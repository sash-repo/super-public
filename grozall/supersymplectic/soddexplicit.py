"""The EXPLICIT SHAPES of the three order-2 operators over a periplectic form.

sodddecomp.py proved that the 3-dimensional space of invariant bilinear operators
of order 2 on metrics over an odd (periplectic) form is spanned by

    C1(A,B) = L_{X_{ptr A}} B,   C2(A,B) = L_{X_{ptr B}} A,   C3 = d(ptr A).d(ptr B).

That says what they ARE; it does not say what they LOOK LIKE in coordinates, which
is what was asked.  This script produces the index formulas and proves them.

The method is the one this folder uses for every sign: NOTHING is guessed.  For each
operator we write down the shape of the formula -- which products of which
derivatives of which arguments can occur -- leave the coefficient of every
PARITY CLASS undetermined, and solve the exact linear system that says the formula
equals the operator on random tensor fields.  The system is solved at k = 2 and the
solution is then verified, unchanged, at k = 3 on fields it was not fitted to.  A
formula is reported only if the solution is UNIQUE (nullity 0 of the homogeneous
system); a non-unique fit would mean the ansatz has redundant terms and the printed
formula would be one of many, which is not a formula.

Coordinates: x^a = q_1..q_k (even), xi_1..xi_k (odd);  omega^{ab} the inverse of
omega = sum dq_i dxi_i;  A_{ij} = (1/2) d_{u^i}d_{u^j}A~ the components of a metric.

Usage:  python3 soddexplicit.py [k]           # fit at k, verify at k and k+1
"""

import os
import random
import sys
from fractions import Fraction

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, solve                      # noqa: E402

from ssuper import SSetup                                  # noqa: E402
from sodd0 import ptr                                      # noqa: E402
from sunary import invariant_unary, evaluate_unary         # noqa: E402


# ---------------------------------------------------------------- components

def comps(S, T):
    """T_{ij} = (1/2) d_{u^i} d_{u^j} T~ -- the components of a metric field."""
    out = {}
    for i in range(S.nc):
        Ti = S.du(T, i)
        if not Ti:
            continue
        for j in range(S.nc):
            v = S.du(Ti, j)
            if v:
                out[(i, j)] = S.A.scale(v, Fraction(1, 2))
    return out


def rebuild(S, C):
    """sum_{i,j} u^i u^j T_{ij} -- the inverse of comps(), up to the check below."""
    A = S.A
    terms = []
    for (i, j), v in C.items():
        terms.append(A.mul(A.mul(S.u(i), S.u(j)), v))
    return A.summ(terms)


def check_roundtrip(S, rng):
    """comps/rebuild must be mutually inverse up to one global factor."""
    for _ in range(3):
        T = S.random_field(2, rng, nterms=3)
        back = rebuild(S, comps(S, T))
        if back == T:
            continue
        if back == S.A.scale(T, 2):
            return Fraction(1, 2)
        raise AssertionError("comps/rebuild are not inverse to each other")
    return Fraction(1)


def ptr_comp(S, C):
    """ptr A = sum omega^{ab} A_{ab}, from the components."""
    A = S.A
    return A.summ([A.scale(C[(i, j)], w) for (i, j, w) in S.Omnz if (i, j) in C])


# ---------------------------------------------------------------- the fit

def classes(S, slots):
    """All parity tuples of the given index slots (each slot is 0 or 1)."""
    out = []
    for mask in range(2 ** slots):
        out.append(tuple((mask >> t) & 1 for t in range(slots)))
    return out


def fit(name, S, targets, termgens, slots, rng, verbose=True, with_dead=False):
    """Solve  sum_c  coeff[c] * term_c  =  target  over all components.

    termgens: list of (label, fn) where fn(i, j) -> list of (parity-class, value)
    contributions.  The unknowns are one coefficient per (label, parity class).
    """
    A = S.A
    unknowns = [(lab, c) for lab, _ in termgens for c in classes(S, slots)]
    idx = {u: t for t, u in enumerate(unknowns)}
    rows, rhs = [], []
    for target, args in targets:
        tc = comps(S, target)
        keys = set(tc) | {(i, j) for i in range(S.nc) for j in range(S.nc)}
        for (i, j) in sorted(keys):
            # every monomial of every contribution gives one scalar equation
            contrib = {}
            for lab, fn in termgens:
                for cls, val in fn(i, j, *args):
                    if not val:
                        continue
                    t = idx[(lab, cls)]
                    contrib[t] = A.add(contrib.get(t, A.zero()), val)
            tgt = tc.get((i, j), A.zero())
            mons = set()
            for v in contrib.values():
                mons |= set(v)
            mons |= set(tgt)
            for mon in mons:
                row = [Fraction(0)] * len(unknowns)
                for t, v in contrib.items():
                    row[t] = Fraction(v.get(mon, 0))
                if any(row) or tgt.get(mon, 0):
                    rows.append(row)
                    rhs.append(Fraction(tgt.get(mon, 0)))
    # An unknown whose column is identically zero is a parity class that never
    # occurs -- a vacuous slot of the ansatz, not a freedom of the formula.  Drop
    # those first; the uniqueness test is about the rest.
    live = [t for t in range(len(unknowns))
            if any(row[t] for row in rows)]
    dead = [unknowns[t] for t in range(len(unknowns)) if t not in live]
    rows = [[row[t] for t in live] for row in rows]
    sol = solve(rows, rhs)
    if sol is None:
        if verbose:
            print(f"   {name}: NO solution -- the ansatz is wrong")
        return (None, dead) if with_dead else None
    ns = nullspace(rows)
    if verbose:
        print(f"   {name}: {len(rows)} equations, {len(live)} live unknowns "
              f"({len(dead)} vacuous classes), nullity {len(ns)}")
    if ns:
        if verbose:
            print(f"   {name}: solution NOT unique -- the ansatz is redundant")
        return (None, dead) if with_dead else None
    out = {unknowns[t]: c for t, c in zip(live, sol) if c}
    return (out, dead) if with_dead else out


def apply_formula(S, coeffs, termgens, i, j, *args):
    A = S.A
    out = []
    for lab, fn in termgens:
        for cls, val in fn(i, j, *args):
            c = coeffs.get((lab, cls))
            if c and val:
                out.append(A.scale(val, c))
    return A.summ(out)


def verify(name, S, coeffs, termgens, targets):
    A = S.A
    bad = 0
    n = 0
    for target, args in targets:
        tc = comps(S, target)
        for i in range(S.nc):
            for j in range(S.nc):
                got = apply_formula(S, coeffs, termgens, i, j, *args)
                want = tc.get((i, j), A.zero())
                n += 1
                if got != want:
                    bad += 1
    print(f"   {name}: {n} components checked, {bad} mismatch"
          f"{'es' if bad != 1 else ''}  {'PASS' if bad == 0 else 'FAIL'}")
    return bad == 0


# ---------------------------------------------------------------- the shapes

def make_terms_C1(S):
    """L_{X_pi} B in components:  X^a d_a B_{ij} + (d_i X^a) B_{aj}
                                                 + (d_j X^a) B_{ia}."""
    A = S.A

    def t_transport(i, j, X, BC, pi):
        out = []
        for a in range(S.nc):
            if not X[a]:
                continue
            v = BC.get((i, j))
            if v is None:
                continue
            dv = S.dz(v, a)
            if dv:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(X[a], dv)))
        return out

    def t_first(i, j, X, BC, pi):
        out = []
        for a in range(S.nc):
            dX = S.dz(X[a], i)
            if not dX:
                continue
            v = BC.get((a, j))
            if v:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(dX, v)))
        return out

    def t_second(i, j, X, BC, pi):
        out = []
        for a in range(S.nc):
            dX = S.dz(X[a], j)
            if not dX:
                continue
            v = BC.get((i, a))
            if v:
                out.append(((S.par[i], S.par[j], S.par[a]), A.mul(dX, v)))
        return out

    return [("transport", t_transport), ("first", t_first), ("second", t_second)]


def make_terms_C3(S):
    """d(pi).d(pi') in components: (d_i pi)(d_j pi') and (d_j pi)(d_i pi')."""
    A = S.A

    def t_ij(i, j, pi, pip):
        v = A.mul(S.dz(pi, i), S.dz(pip, j))
        return [((S.par[i], S.par[j]), v)] if v else []

    def t_ji(i, j, pi, pip):
        v = A.mul(S.dz(pi, j), S.dz(pip, i))
        return [((S.par[i], S.par[j]), v)] if v else []

    return [("ij", t_ij), ("ji", t_ji)]


def hamiltonian_shape(S, f):
    """Is X_f^a = sigma(p(a)) sum_b omega^{ab} d_b f?  Return sigma or None."""
    A = S.A
    X = S.ham(f)
    sig = {}
    for a in range(S.nc):
        cand = A.summ([A.scale(S.dz(f, b), S.Om[a][b]) for b in range(S.nc)
                       if S.Om[a][b]])
        if X[a] == cand:
            s = 1
        elif X[a] == A.scale(cand, -1):
            s = -1
        elif not X[a] and not cand:
            continue
        else:
            return None
        p = S.par[a]
        if p in sig and sig[p] != s:
            return None
        sig[p] = s
    return sig


def targets_C1(S, rng, npair, fdeg=3):
    out = []
    for _ in range(npair):
        P = S.random_field(fdeg, rng, nterms=3)
        Q = S.random_field(fdeg, rng, nterms=3)
        pi = ptr(S, P)
        out.append((S.lie(pi, Q), (S.ham(pi), comps(S, Q), pi)))
    return out


def targets_C3(S, rng, npair, fdeg=3):
    """C3 = d(ptr A) . d(ptr B), with d the honest differential -- the scale of
    the nullspace vector that sunary returns is arbitrary and k-dependent, so it
    must not be what the formula is fitted to."""
    A = S.A
    out = []
    for _ in range(npair):
        P = S.random_field(fdeg, rng, nterms=3)
        Q = S.random_field(fdeg, rng, nterms=3)
        pi, pip = ptr(S, P), ptr(S, Q)
        out.append((A.mul(S.dform(pi), S.dform(pip)), (pi, pip)))
    return out


def get_dv(S, k):
    rng = random.Random(9110 + k)
    schemes, ns, dim = invariant_unary(S, 1, 1, rng, verbose=False)
    assert dim == 1
    v = next(w for w in ns if any(w))
    A = S.A

    def dv(X):
        return A.summ([A.scale(evaluate_unary(S, s, X), c)
                       for c, s in zip(v, schemes) if c])
    return dv


def report(name, coeffs):
    print(f"   {name}:")
    for (lab, cls), c in sorted(coeffs.items()):
        print(f"        {lab:10s} p{cls} : {c}")


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"=== fitting at k = {k} ===")
    S = SSetup(k, k, form="odd")
    rng = random.Random(70901 + k)
    fac = check_roundtrip(S, rng)
    print(f"   comps/rebuild inverse (factor {fac})")

    # 1) the Hamiltonian field of an odd function
    pi = ptr(S, S.random_field(3, rng, nterms=3))
    sig = hamiltonian_shape(S, pi)
    print(f"   X_pi^a = sigma(p(a)) omega^{{ab}} d_b pi   with sigma = {sig}")

    # 2) ptr from the components
    for _ in range(3):
        P = S.random_field(3, rng, nterms=3)
        assert ptr_comp(S, comps(S, P)) == ptr(S, P), "ptr from components"
    print("   ptr A = sum omega^{ab} A_{ab} from the components   PASS")

    # 3) C1
    tg1 = make_terms_C1(S)
    tar1 = targets_C1(S, rng, 3)
    c1 = fit("C1", S, tar1, tg1, 3, rng)
    if c1:
        report("C1", c1)
        verify("C1 (fit data)", S, c1, tg1, tar1)

    # 4) C3
    tg3 = make_terms_C3(S)
    tar3 = targets_C3(S, rng, 3)
    c3 = fit("C3", S, tar3, tg3, 2, rng)
    if c3:
        report("C3", c3)
        verify("C3 (fit data)", S, c3, tg3, tar3)

    # 5) the independent check at k+1, on fields the fit never saw
    kk = k + 1
    print(f"=== verifying the SAME coefficients at k = {kk} ===")
    S2 = SSetup(kk, kk, form="odd")
    rng2 = random.Random(424242)
    check_roundtrip(S2, rng2)
    if c1:
        verify("C1", S2, c1, make_terms_C1(S2), targets_C1(S2, rng2, 2))
    if c3:
        verify("C3", S2, c3, make_terms_C3(S2), targets_C3(S2, rng2, 2))

    sig2 = hamiltonian_shape(S2, ptr(S2, S2.random_field(3, rng2, nterms=3)))
    print(f"   sigma at k = {kk}: {sig2}")


if __name__ == "__main__":
    main()
