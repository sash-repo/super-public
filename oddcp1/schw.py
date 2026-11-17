"""
The Schwarzian layer of a contact superstring, for every odd dimension n.

Two independent halves, both exact.

(A) LOCAL, on the flat model  S^{1|n} = the supercircle with alpha = dt + sum
    theta_i d theta_i.  The Lie superalgebra k^L(1|n) is realised by generating
    functions f(t, theta), Laurent in t, with the contact bracket

      {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
              - (-1)^{p(f)} kappa sum_i d_{theta_i}f d_{theta_i}g,     E = sum theta_i d_{theta_i},

    the normalisation kappa being fixed by the Jacobi identity rather than
    guessed.  Grading: deg K_{t^a theta_S} = a - 1 + |S|/2, stored doubled,
    w(a,S) = 2a - 2 + |S|, the usual doubled grading of k^L(1|n).

    Computed here:
      * the space of weight-0 2-cocycles on a window, modulo the weight-0
        coboundaries -- i.e. the candidates for a central extension;
      * the same space described by an operator: c(f,g) = Res(f P(g)) with
        Res = the coefficient of t^{-1} theta_1...theta_n (residue times the
        Berezin integral) and P a monomial in D_i = d_{theta_i} + theta_i d_t
        and d_t.  The order of P is forced to be 6-n, and the O(n)-invariant
        operators are d_t^k (order 2k) and D_1...D_n d_t^j (order n+2j), so
        that the order 6-n is available exactly for n <= 4 (for n = 6 it is
        available and the resulting form is symmetric, hence not a cocycle).

(B) GLOBAL, on a split superstring X = (X, Lambda(E)) over a curve of genus g
    with a diagonal rigging, E = sum L nu_i.  The sheaf of contact fields is
    K = N (x) Lambda(E) with N = Theta (Lemma 20.4), the
    Berezinian is Ber = O_X (x) K_X (x) (det E)^{-1}, and the SCHWARZIAN SHEAF
    is

          S := Hom(K, Ber) = K^vee (x) Ber.

    Layer by layer this is L^{4-n+q} (x) nu_{S} (x) delta, delta = nu_1...nu_n,
    i.e. the weight-(2 - n/2) densities twisted by delta; and the layer-wise
    Serre duality on the curve gives

          sdim H^0(X; S) = Pi^n sdim H^1(X; K),

    the complementary-subset bookkeeping being the whole proof.  Nothing is
    sampled: h^0 and h^1 are those of srs.py, symbolic in g.
"""

from fractions import Fraction as Q
from itertools import combinations, product

from stringy import mono, add, mul, smul, d_t, d_odd, parity

import srs


# ============================================================ (A) the model

def odd_labels(n):
    return tuple(range(1, n + 1))


def monomials(n, arange):
    out = []
    for a in arange:
        for r in range(n + 1):
            for S in combinations(odd_labels(n), r):
                out.append((a, S))
    return out


def weight(key):
    """2 * deg K_{t^a theta_S}; the grading element is K_t."""
    a, S = key
    return 2 * a - 2 + len(S)


def by_weight(keys):
    d = {}
    for k in keys:
        d.setdefault(weight(k), []).append(k)
    return d


def nabla(h):
    """(2 - E)(h)."""
    out = {}
    for (a, S), c in h.items():
        v = (2 - len(S)) * c
        if v:
            out[(a, S)] = v
    return out


def bracket(f, g, n, kappa=Q(1)):
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in odd_labels(n):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(res, smul(-((-1) ** parity(f)) * kappa, s))


def jacobi_fails(n, keys, kappa=Q(1)):
    bad = 0
    for kf, kg, kh in product(keys, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)
        lhs = bracket(f, bracket(g, h, n, kappa), n, kappa)
        rhs = add(bracket(bracket(f, g, n, kappa), h, n, kappa),
                  smul((-1) ** (parity(f) * parity(g)),
                       bracket(g, bracket(f, h, n, kappa), n, kappa)))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


# ------------------------------------------------------- D_i and the residue

def D(i, f):
    """D_i = d_{theta_i} + theta_i d_t, the odd derivative with D_i^2 = d_t."""
    return add(d_odd(i, f), mul(mono(0, (i,)), d_t(f)))


def apply_op(op, f, n):
    """op is a tuple of items: ('D', i) or ('t',), applied right to left."""
    out = f
    for item in reversed(op):
        if item[0] == 'D':
            out = D(item[1], out)
        else:
            out = d_t(out)
    return out


def res(f, n):
    """coefficient of t^{-1} theta_1...theta_n."""
    return f.get((-1, odd_labels(n)), 0)


def op_order(op):
    return sum(1 if item[0] == 'D' else 2 for item in op)


def invariant_ops(n, order):
    """the O(n)-invariant monomials in D_1..D_n, d_t of the given order:
    d_t^k (order 2k) and D_1...D_n d_t^j (order n+2j)."""
    out = []
    if order >= 0 and order % 2 == 0:
        out.append(('dt^%d' % (order // 2), tuple([('t',)] * (order // 2))))
    if n >= 1 and order >= n and (order - n) % 2 == 0:
        j = (order - n) // 2
        op = tuple([('D', i) for i in odd_labels(n)]) + tuple([('t',)] * j)
        out.append(('D_1..D_%d dt^%d' % (n, j), op))
    return out


# ------------------------------------- the cocycle equations on a window

class Unknowns:
    """c(k1,k2) for pairs of monomials of total weight 0."""

    def __init__(self):
        self.idx = {}
        self.keys = []

    def get(self, k1, k2):
        if weight(k1) + weight(k2) != 0:
            return None
        p1, p2 = len(k1[1]) % 2, len(k2[1]) % 2
        if k1 <= k2:
            key, sgn = (k1, k2), 1
        else:
            key, sgn = (k2, k1), -((-1) ** (p1 * p2))
        if k1 == k2 and p1 == 0:
            return None                       # c(x,x) = -c(x,x)
        if key not in self.idx:
            self.idx[key] = len(self.keys)
            self.keys.append(key)
        return self.idx[key], sgn


def lin_c(U, f, g):
    out = {}
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            r = U.get(k1, k2)
            if r is None:
                continue
            i, s = r
            out[i] = out.get(i, 0) + s * c1 * c2
    return {i: v for i, v in out.items() if v}


def cocycle_rows(U, n, keysB, kappa=Q(1)):
    W = by_weight(keysB)
    seen = set()
    for kf, kg in product(keysB, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            f, g, h = mono(*kf), mono(*kg), mono(*kh)
            row = lin_c(U, f, bracket(g, h, n, kappa))
            for i, v in lin_c(U, bracket(f, g, n, kappa), h).items():
                row[i] = row.get(i, 0) - v
            sgn = (-1) ** (parity(f) * parity(g))
            for i, v in lin_c(U, g, bracket(f, h, n, kappa)).items():
                row[i] = row.get(i, 0) - sgn * v
            row = {i: v for i, v in row.items() if v}
            if not row:
                continue
            k = tuple(sorted(row.items()))
            if k in seen:
                continue
            seen.add(k)
            yield row


def reduce_rows(rows):
    piv = {}
    for row in rows:
        row = dict(row)
        while row:
            p = min(row)
            if p in piv:
                c = row[p]
                for i, v in piv[p].items():
                    row[i] = row.get(i, 0) - c * v
                row = {i: v for i, v in row.items() if v}
            else:
                c = row[p]
                piv[p] = {i: Q(v) / c for i, v in row.items()}
                break
    return piv


def nullspace(piv, nvars):
    free = [j for j in range(nvars) if j not in piv]
    basis = []
    for j in free:
        v = [Q(0)] * nvars
        v[j] = Q(1)
        for p in sorted(piv, reverse=True):
            row = piv[p]
            s = sum(row[i] * v[i] for i in row if i != p)
            v[p] = -s
        basis.append(v)
    return basis


def coboundaries(U, n, keysB, nB, kappa=Q(1)):
    """the weight-0 coboundaries c(f,g) = phi([f,g]), phi supported on the
    weight-0 monomials, projected on the B-window unknowns."""
    zero = [k for k in keysB if weight(k) == 0]
    out = []
    for k0 in zero:
        v = [Q(0)] * nB
        for i in range(nB):
            k1, k2 = U.keys[i]
            b = bracket(mono(*k1), mono(*k2), n, kappa)
            v[i] = Q(b.get(k0, 0))
        if any(v):
            out.append(v)
    return out


def cocycle_space(n, B=2, kappa=Q(1), drop=None, verbose=True):
    """dimension of {weight-0 2-cocycles} and of its quotient by the
    coboundaries, both restricted to the window |a| <= B."""
    A = 2 * B + 1
    keysA = monomials(n, range(-A, A + 1))
    keysB = monomials(n, range(-B, B + 1))
    if drop:
        keysA = [k for k in keysA if k not in drop]
        keysB = [k for k in keysB if k not in drop]
    U = Unknowns()
    WA = by_weight(keysA)
    for k1 in keysB:
        for k2 in WA.get(-weight(k1), ()):
            U.get(k1, k2)
    nB = len(U.keys)
    rows = list(cocycle_rows(U, n, keysB, kappa))
    piv = reduce_rows(rows)
    basis = nullspace(piv, len(U.keys))
    proj = [{i: x for i, x in enumerate(v[:nB]) if x} for v in basis]
    dim = len(reduce_rows(proj))
    cob = coboundaries(U, n, keysB, nB, kappa)
    both = reduce_rows(proj + [{i: x for i, x in enumerate(v) if x} for v in cob])
    dimcob = len(reduce_rows([{i: x for i, x in enumerate(v) if x} for v in cob]))
    if verbose:
        print("    n=%d  B=%d  unknowns=%d(%d in window)  eqs=%d  "
              "cocycles=%d  coboundaries=%d  H^2 >= %d"
              % (n, B, len(U.keys), nB, len(rows), dim, dimcob,
                 dim - dimcob))
    return dim, dimcob, U, nB, basis


def cocycle_of_op(U, n, op, nB):
    """the vector of c(f,g) = Res(f P(g)) on the window unknowns."""
    v = [Q(0)] * nB
    for i in range(nB):
        k1, k2 = U.keys[i]
        v[i] = Q(res(mul(mono(*k1), apply_op(op, mono(*k2), n)), n))
    return v


def is_skew(n, op, keys):
    """c(f,g) = Res(f P g) super-antisymmetric?  returns the list of failures."""
    bad = []
    for k1, k2 in product(keys, repeat=2):
        f, g = mono(*k1), mono(*k2)
        c12 = res(mul(f, apply_op(op, g, n)), n)
        c21 = res(mul(g, apply_op(op, f, n)), n)
        s = (-1) ** (parity(f) * parity(g))
        if c12 + s * c21 != 0:
            bad.append((k1, k2, c12, c21))
    return bad


def is_cocycle(n, op, keys, kappa=Q(1)):
    bad = []
    for kf, kg, kh in product(keys, repeat=3):
        f, g, h = mono(*kf), mono(*kg), mono(*kh)

        def c(x, y):
            return res(mul(x, apply_op(op, y, n)), n)

        lhs = c(f, bracket(g, h, n, kappa))
        rhs = (c(bracket(f, g, n, kappa), h)
               + (-1) ** (parity(f) * parity(g)) * c(g, bracket(f, h, n, kappa)))
        if lhs != rhs:
            bad.append((kf, kg, kh, lhs, rhs))
    return bad


# ======================================================== (B) the sheaves
#
# srs.py bookkeeping: a line bundle is L^k twisted by a 2-torsion class; the
# rigging is (nu_1, ..., nu_n) and delta = nu_1 ... nu_n = det V.

LB = srs.LB
emul = srs.emul
TRIV = srs.TRIV


def contact_layers(n, riggings=None):
    """K = Theta (x) Lambda(E), E = sum L nu_i: the list (bundle, parity)."""
    rig = list(riggings) if riggings else [TRIV] * n
    out = []
    for q in range(n + 1):
        for S in combinations(range(n), q):
            out.append((LB(q - 2, emul(*[rig[i] for i in S])), q % 2))
    return out


def schwarzian_layers(n, riggings=None):
    """S = K^vee (x) Ber = Hom(K, Ber); Ber = O (x) K_X (x) (det E)^{-1}.

    Layer q of S is Lambda^q E (x) L^{4-n} (x) delta, delta = prod nu_i."""
    rig = list(riggings) if riggings else [TRIV] * n
    delta = emul(*rig)
    out = []
    for q in range(n + 1):
        for S in combinations(range(n), q):
            out.append((LB(4 - n + q, emul(*[rig[i] for i in S], delta)),
                        q % 2))
    return out


def ber_layers(n, riggings=None):
    rig = list(riggings) if riggings else [TRIV] * n
    delta = emul(*rig)
    out = []
    for q in range(n + 1):
        for S in combinations(range(n), q):
            out.append((LB(q + 2 - n, emul(*[rig[i] for i in S], delta)),
                        q % 2))
    return out


def sdim_h0(layers, g=srs.G):
    ev = od = 0
    for b, p in layers:
        h = srs.h0(b, g)
        if p:
            od += h
        else:
            ev += h
    return (srs.sp.expand(ev), srs.sp.expand(od))


def sdim_h1(layers, g=srs.G):
    ev = od = 0
    for b, p in layers:
        h = srs.h1(b, g)
        if p:
            od += h
        else:
            ev += h
    return (srs.sp.expand(ev), srs.sp.expand(od))
