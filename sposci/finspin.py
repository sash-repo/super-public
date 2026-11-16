#!/usr/bin/env python3
r"""
finspin.py -- the two finite-dimensional checks of the text, section
"Examples of spinor/oscillator representations":

PART A (the question under eq. (53): "where does the factor of 1/2 come from?")
  o(2k+1) c po(0|2k+1) in the basis (52), quantised as in the text.  The point:
  the quantisation of po(0|n) must send

      K_theta  |-->  (hbar/sqrt 2)(theta + d/d theta),      NOT  hbar(theta + d/d theta),

  because {theta, theta}_{k.b.} = 1 (see (130)) forces Q(theta)^2 = 1/2
  whereas (theta + d/d theta)^2 = 1.  With the 1/sqrt2 in place, the two
  sqrt2's of the basis (52) cancel it and eq. (53) reads, with no stray factor,

      h_k One = [x^+_k, x^-_k] One = (d_k xi_k - xi_k d_k) One = One.

  The text's prefactor 2/2 is exactly this missing (1/sqrt2)^2 = 1/2 (its
  numerator 2 being (sqrt2)^2 from (52)).  Checked below for k = 1, 2, 3, 4:
  the quantisation is a homomorphism of Lie (super)algebras on the nose, the
  Chevalley relations hold, and One is the highest weight vector of the k-th
  fundamental representation.

PART B (the remark "the following is wrong; besides, one should verify that the
answer satisfies the relation sl(2) = o(3)")
  the highest weight of Spin(L^k) resp. Osc(L^k) for the principal embeddings
  sl(2) --> o(k+1) resp. sp(k+1).  Computed from scratch for k <= 12.

Both parts use the canonical quadratic lift

      hat X = (1/2) sum_alpha X(e_alpha) e^alpha,     B(e^alpha, e_beta) = delta,

the unique element of the Clifford/Weyl algebra with [hat X, w] = X(w) for all
w in W; for finite-dimensional W it is an honest homomorphism (verified).
"""

from fractions import Fraction as F
from itertools import product


# =====================================================================  Q[sqrt2]

class Q2:
    """a + b sqrt2, exact."""
    __slots__ = ('a', 'b')

    def __init__(self, a=0, b=0):
        self.a, self.b = F(a), F(b)

    def __add__(s, o):
        o = mk(o)
        return Q2(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __neg__(s):
        return Q2(-s.a, -s.b)

    def __sub__(s, o):
        return s + (-mk(o))

    def __rsub__(s, o):
        return mk(o) + (-s)

    def __mul__(s, o):
        o = mk(o)
        return Q2(s.a * o.a + 2 * s.b * o.b, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def inv(s):
        d = s.a * s.a - 2 * s.b * s.b
        return Q2(s.a / d, -s.b / d)

    def __truediv__(s, o):
        return s * mk(o).inv()

    def __eq__(s, o):
        o = mk(o)
        return s.a == o.a and s.b == o.b

    def __bool__(s):
        return bool(s.a) or bool(s.b)

    def __hash__(s):
        return hash((s.a, s.b))

    def __repr__(s):
        if not s.b:
            return str(s.a)
        if not s.a:
            return '%s*sqrt2' % s.b
        return '(%s + %s sqrt2)' % (s.a, s.b)


def mk(x):
    return x if isinstance(x, Q2) else Q2(x, 0)


SQRT2 = Q2(0, 1)
INV_SQRT2 = Q2(0, F(1, 2))          # 1/sqrt2 = sqrt2 / 2
ZERO, ONEQ = Q2(0), Q2(1)


# ==========================================  PART A: po(0|2k+1) and its spinors
#
# Odd generators: labels 'x1'..'xk' (= xi_i), 'y1'..'yk' (= eta_i), 'th'.
# A superfunction is a dict {monomial (tuple of labels, sorted) : Q2}.
# The k.b. of po(0|n), eq. (130), in the xi-eta-theta basis:
#
#   {f, g} = -(-1)^p(f) [ sum_i (d_xi_i f  d_eta_i g + d_eta_i f  d_xi_i g)
#                         + d_th f d_th g ],
#
# so that {xi_i, eta_i} = 1, {theta, theta} = 1, all other pairs of generators
# bracketing to 0 -- the transport of the orthonormal form of (130).

def gens(k):
    return ['x%d' % i for i in range(1, k + 1)] + \
           ['y%d' % i for i in range(1, k + 1)] + ['th']


def order(k):
    return {g: n for n, g in enumerate(gens(k))}


def pmono(mono, ORD):
    """sort a monomial, return (sorted tuple, sign) or None if repeated."""
    m, sgn = list(mono), 1
    for i in range(len(m)):
        for j in range(len(m) - 1 - i):
            if ORD[m[j]] > ORD[m[j + 1]]:
                m[j], m[j + 1] = m[j + 1], m[j]
                sgn = -sgn
    for i in range(len(m) - 1):
        if m[i] == m[i + 1]:
            return None, 0
    return tuple(m), sgn


def fadd(a, b, s=ONEQ):
    r = dict(a)
    for m, v in b.items():
        r[m] = r.get(m, ZERO) + mk(s) * v
        if not r[m]:
            del r[m]
    return r


def fmul(a, b, ORD):
    out = {}
    for m1, v1 in a.items():
        for m2, v2 in b.items():
            m, sg = pmono(m1 + m2, ORD)
            if sg:
                out[m] = out.get(m, ZERO) + mk(sg) * v1 * v2
    return {m: v for m, v in out.items() if v}


def der(g, f, ORD):
    """left derivative d/d g."""
    out = {}
    for m, v in f.items():
        if g not in m:
            continue
        i = m.index(g)
        new = m[:i] + m[i + 1:]
        out[new] = out.get(new, ZERO) + mk(-1 if i % 2 else 1) * v
    return {m: v for m, v in out.items() if v}


def parity(f):
    return {len(m) % 2 for m in f} .pop() if f else 0


def kb(f, g, k, ORD):
    """the k.b. of po(0|2k+1), eq. (130), in the xi-eta-theta basis."""
    tot = {}
    for i in range(1, k + 1):
        xi, et = 'x%d' % i, 'y%d' % i
        tot = fadd(tot, fmul(der(xi, f, ORD), der(et, g, ORD), ORD))
        tot = fadd(tot, fmul(der(et, f, ORD), der(xi, g, ORD), ORD))
    tot = fadd(tot, fmul(der('th', f, ORD), der('th', g, ORD), ORD))
    sgn = -1 if parity(f) == 0 else 1          # -(-1)^p(f)
    return {m: mk(sgn) * v for m, v in tot.items()}


# ---- the Clifford module Lambda(xi_1..xi_k, theta) and the quantisation Q

def basis_module(k):
    """monomials in xi_1..xi_k, theta -- the space K[xi, theta]."""
    labs = ['x%d' % i for i in range(1, k + 1)] + ['th']
    out = []
    for r in range(len(labs) + 1):
        for combo in product(*[[0, 1]] * len(labs)):
            if sum(combo) == r:
                out.append(tuple(l for l, c in zip(labs, combo) if c))
    return sorted(set(out), key=lambda m: (len(m), m))


class Op:
    """a linear operator on K[xi, theta], stored as a dict of columns."""

    def __init__(self, k, action):
        self.k, self.B = k, basis_module(k)
        self.idx = {m: n for n, m in enumerate(self.B)}
        self.col = [action(m) for m in self.B]

    @staticmethod
    def _apply_col(col, vec, idx):
        out = {}
        for m, c in vec.items():
            for m2, c2 in col[idx[m]].items():
                out[m2] = out.get(m2, ZERO) + c * c2
        return {m: v for m, v in out.items() if v}

    def __call__(self, vec):
        return Op._apply_col(self.col, vec, self.idx)

    def mat(self):
        return [dict(c) for c in self.col]


def op_mul(A, B):
    """A after B."""
    o = Op.__new__(Op)
    o.k, o.B, o.idx = A.k, A.B, A.idx
    o.col = [A(B.col[n] if isinstance(B.col[n], dict) else B.col[n])
             for n in range(len(A.B))]
    return o


def op_lin(*terms):
    """sum of s * A."""
    A0 = terms[0][1]
    o = Op.__new__(Op)
    o.k, o.B, o.idx = A0.k, A0.B, A0.idx
    o.col = []
    for n in range(len(A0.B)):
        acc = {}
        for s, A in terms:
            for m, v in A.col[n].items():
                acc[m] = acc.get(m, ZERO) + mk(s) * v
        o.col.append({m: v for m, v in acc.items() if v})
    return o


def op_zero(A):
    return not any(c for c in A.col)


def make_ops(k):
    """xi_i, d/d xi_i, theta, d/d theta on K[xi, theta]."""
    MORD = {g: n for n, g in enumerate(['x%d' % i for i in range(1, k + 1)] + ['th'])}
    ops = {}
    for g in list(MORD):
        def mult(m, g=g):
            new, sg = pmono((g,) + m, MORD)
            return {new: mk(sg)} if sg else {}

        def deriv(m, g=g):
            if g not in m:
                return {}
            i = m.index(g)
            return {m[:i] + m[i + 1:]: mk(-1 if i % 2 else 1)}
        ops[g] = Op(k, mult)
        ops['d' + g] = Op(k, deriv)
    ops['id'] = Op(k, lambda m: {m: ONEQ})
    return ops


def quantise_linear(k, ops):
    """Q on W = span(xi, eta, theta):  Q(xi_i) = xi_i, Q(eta_i) = d_xi_i,
       Q(theta) = (1/sqrt2)(theta + d_theta)."""
    Q = {}
    for i in range(1, k + 1):
        Q['x%d' % i] = ops['x%d' % i]
        Q['y%d' % i] = ops['dx%d' % i]
    Q['th'] = op_lin((INV_SQRT2, ops['th']), (INV_SQRT2, ops['dth']))
    return Q


def quantise(f, k, ops, Q, ORD):
    """the canonical lift of a quadratic f in po(0|2k+1):
       hat f = (1/2) sum_alpha {f, e_alpha} e^alpha  (e^alpha the B-dual basis)."""
    dual = {}
    for i in range(1, k + 1):
        dual['x%d' % i] = 'y%d' % i
        dual['y%d' % i] = 'x%d' % i
    dual['th'] = 'th'
    terms = []
    for g in gens(k):
        img = kb(f, {(g,): ONEQ}, k, ORD)      # {f, e_alpha} in W
        for m, c in img.items():
            assert len(m) == 1, 'f is not quadratic'
            terms.append((F(1, 2) * 1, c, m[0], dual[g]))
    if not terms:
        return op_lin((0, ops['id']))
    return op_lin(*[(mk(F(1, 2)) * c, op_mul(Q[a], Q[b]))
                    for _, c, a, b in terms])


def sl2_basis_o_odd(k):
    """the basis (52) of o(2k+1) inside po(0|2k+1) -- x^+_i, x^-_i, h_i."""
    xp, xm, h = {}, {}, {}
    for i in range(1, k):
        xp[i] = {('x%d' % (i + 1), 'y%d' % i): ONEQ}
        xm[i] = {('x%d' % i, 'y%d' % (i + 1)): ONEQ}
        h[i] = fadd({('x%d' % (i + 1), 'y%d' % (i + 1)): ONEQ},
                    {('x%d' % i, 'y%d' % i): ONEQ}, -1)
    xp[k] = {('y%d' % k, 'th'): SQRT2}
    xm[k] = {('th', 'x%d' % k): SQRT2}
    h[k] = {('y%d' % k, 'x%d' % k): mk(2)}
    ORD = order(k)
    for d in (xp, xm, h):                      # write every monomial in order
        for i, f in list(d.items()):
            new = {}
            for m, c in f.items():
                sm, sg = pmono(m, ORD)
                if sg:
                    new[sm] = new.get(sm, ZERO) + mk(sg) * c
            d[i] = {m: c for m, c in new.items() if c}
    return xp, xm, h


def part_A():
    print('=' * 78)
    print('PART A -- o(2k+1) c po(0|2k+1), basis (52), and eq. (53)')
    print('=' * 78)
    for k in (1, 2, 3, 4):
        ORD, ops = order(k), make_ops(k)
        Q = quantise_linear(k, ops)
        xp, xm, h = sl2_basis_o_odd(k)
        elts = {}
        for i in range(1, k + 1):
            elts['x+%d' % i], elts['x-%d' % i], elts['h%d' % i] = xp[i], xm[i], h[i]
        # (i) Chevalley relations in po(0|2k+1) itself
        bad_po = 0
        for i in range(1, k + 1):
            if fadd(kb(xp[i], xm[i], k, ORD), h[i], -1):
                bad_po += 1
            if fadd(kb(h[i], xp[i], k, ORD), xp[i], -2):
                bad_po += 1
            if fadd(kb(h[i], xm[i], k, ORD), xm[i], +2):
                bad_po += 1
        # (ii) the quantisation is a homomorphism
        hatted = {n: quantise(f, k, ops, Q, ORD) for n, f in elts.items()}
        bad_hom = 0
        for n1, f1 in elts.items():
            for n2, f2 in elts.items():
                lhs = op_lin((1, op_mul(hatted[n1], hatted[n2])),
                             (-1, op_mul(hatted[n2], hatted[n1])))
                rhs = quantise(kb(f1, f2, k, ORD), k, ops, Q, ORD)
                if not op_zero(op_lin((1, lhs), (-1, rhs))):
                    bad_hom += 1
        # (iii) One is the h.w. vector of the k-th fundamental representation
        one = {(): ONEQ}
        vals, killed = {}, 0
        for i in range(1, k + 1):
            vals[i] = hatted['h%d' % i](one).get((), ZERO)
            if not hatted['x+%d' % i](one):
                killed += 1
        # (iv) the explicit form of eq. (53)
        T = op_lin((1, ops['th']), (1, ops['dth']))
        lhs53 = op_lin((1, op_mul(op_mul(ops['dx%d' % k], T), op_mul(T, ops['x%d' % k]))),
                       (-1, op_mul(op_mul(T, ops['x%d' % k]), op_mul(ops['dx%d' % k], T))))
        rhs53 = op_lin((1, op_mul(ops['dx%d' % k], ops['x%d' % k])),
                       (-1, op_mul(ops['x%d' % k], ops['dx%d' % k])))
        eq53 = op_zero(op_lin((1, lhs53), (-1, rhs53))) and \
            op_zero(op_lin((1, lhs53), (-1, hatted['h%d' % k])))
        print('  k = %d:  Chevalley relations in po: %d failures;'
              '  Q a homomorphism: %d failures' % (k, bad_po, bad_hom))
        print('          x^+_i One = 0 for all i: %s;  (h_1,...,h_k) One = %s'
              % (killed == k, [str(vals[i]) for i in range(1, k + 1)]))
        print('          [d_k T, T xi_k] = d_k xi_k - xi_k d_k = hat h_k : %s'
              '   (so eq. (53) = One, no stray 2/2)' % eq53)
        assert bad_po == 0 and bad_hom == 0 and killed == k and eq53
        assert vals[k] == 1 and all(vals[i] == 0 for i in range(1, k))
    # and what the text's Q(theta) = theta + d_theta would give instead
    k = 3
    ORD, ops = order(k), make_ops(k)
    Q = quantise_linear(k, ops)
    Qbad = dict(Q)
    Qbad['th'] = op_lin((1, ops['th']), (1, ops['dth']))
    xp, xm, h = sl2_basis_o_odd(k)
    hb = op_lin((1, op_mul(quantise(xp[k], k, ops, Qbad, ORD),
                           quantise(xm[k], k, ops, Qbad, ORD))),
                (-1, op_mul(quantise(xm[k], k, ops, Qbad, ORD),
                            quantise(xp[k], k, ops, Qbad, ORD))))
    good = quantise(h[k], k, ops, Q, ORD)
    print()
    print('  with the text\'s Q(theta) = theta + d_theta (k = 3):'
          ' [x^+_k, x^-_k] = %s * hat h_k'
          % ('2' if op_zero(op_lin((1, hb), (-2, good))) else '?'))
    print('  ==> the factor 1/2 of eq. (53) is exactly (1/sqrt2)^2, the square of'
          ' the constant missing in K_theta |--> hbar (theta + d_theta).')


# =============================  PART B: sl(2) on L^k, Spin(L^k) and Osc(L^k)

def sl2_module(k):
    """L^k: H v_i = (k-2i) v_i, X^- v_i = v_{i+1}, X^+ v_i = i(k-i+1) v_{i-1}."""
    n = k + 1

    def H(i):
        return [(i, F(k - 2 * i))]

    def Xm(i):
        return [(i + 1, F(1))] if i + 1 < n else []

    def Xp(i):
        return [(i - 1, F(i * (k - i + 1)))] if i - 1 >= 0 else []
    return n, {'H': H, 'X+': Xp, 'X-': Xm}


def invariant_form(k):
    """the (unique up to scalar) invariant form on L^k, computed from scratch."""
    n, act = sl2_module(k)
    # unknowns B[i][j]; equations B(Xu, v) + B(u, Xv) = 0
    rows = []
    for X in ('H', 'X+', 'X-'):
        for i in range(n):
            for j in range(n):
                row = {}
                for (i2, c) in act[X](i):
                    row[(i2, j)] = row.get((i2, j), F(0)) + c
                for (j2, c) in act[X](j):
                    row[(i, j2)] = row.get((i, j2), F(0)) + c
                if row:
                    rows.append(row)
    # solve by elimination over the n^2 unknowns
    unk = [(i, j) for i in range(n) for j in range(n)]
    pos = {u: m for m, u in enumerate(unk)}
    M = [[r.get(u, F(0)) for u in unk] for r in rows]
    piv = []
    r = 0
    for c in range(len(unk)):
        p = next((rr for rr in range(r, len(M)) if M[rr][c]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        f = M[r][c]
        M[r] = [x / f for x in M[r]]
        for rr in range(len(M)):
            if rr != r and M[rr][c]:
                g = M[rr][c]
                M[rr] = [a - g * b for a, b in zip(M[rr], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(len(unk)) if c not in piv]
    assert len(free) == 1, 'form is not unique: %d parameters' % len(free)
    sol = {u: F(0) for u in unk}
    sol[unk[free[0]]] = F(1)
    for rr, c in enumerate(piv):
        sol[unk[c]] = -M[rr][free[0]]
    return n, act, sol


def part_B():
    print()
    print('=' * 78)
    print('PART B -- the highest weight of Spin(L^k) / Osc(L^k) for sl(2)')
    print('=' * 78)
    print('   k   form    hat H One      hw of Spin(L^k)    formula')
    print('   ' + '-' * 68)
    for k in range(1, 13):
        n, act, B = invariant_form(k)
        sym = all(B[(i, j)] == B[(j, i)] for i in range(n) for j in range(n))
        asym = all(B[(i, j)] == -B[(j, i)] for i in range(n) for j in range(n))
        assert sym != asym
        # sigma and beta
        sig = {}
        for i in range(n):
            js = [j for j in range(n) if B[(i, j)]]
            assert len(js) == 1
            sig[i] = js[0]
        beta = {i: B[(sig[i], i)] for i in range(n)}
        wt = {i: F(k - 2 * i) for i in range(n)}
        # polarisation: creation = positive weight, self-dual = weight 0
        C = [i for i in range(n) if wt[i] > 0]
        S = [i for i in range(n) if sig[i] == i]
        # hat H One = (1/2) sum_i (wt_i/beta_i) v_i v_{sigma(i)} applied to One:
        # only the terms with v_i annihilating and v_{sigma(i)} creating survive,
        # each contributing (1/2)(wt_i/beta_i) B(v_i, v_{sigma(i)}) .
        val = F(0)
        for i in range(n):
            if wt[i] < 0:                    # v_i is an annihilation operator
                val += F(1, 2) * wt[i] / beta[i] * B[(i, sig[i])]
        if sym:                              # Clifford: hw of the whole Lambda(V)
            hw = val + sum(wt[i] for i in C)
            f = 'n(n+1)/2 with k=2n' if k % 2 == 0 else '-'
            pred = F(k // 2 * (k // 2 + 1), 2) if k % 2 == 0 else None
        else:                                # Weyl: S(V), lowest weight vector
            hw = None
            n0 = (k - 1) // 2
            pred = F((n0 + 1) ** 2, 2)
            f = '(n+1)^2/2 with k=2n+1'
        print('  %3d  %-6s  %-13s  %-18s %s'
              % (k, 'sym' if sym else 'anti-sym', str(val),
                 str(hw) if hw is not None else '(inf.dim., l.w. vector)', f))
        if k % 2 == 0:
            assert hw == pred, (k, hw, pred)
            assert val == -pred
        else:
            assert val == pred, (k, val, pred)
    print()
    print('  so:  weight of One  =  -n(n+1)/2  for k = 2n  (Spin, finite-dim.,')
    print('       highest weight  = +n(n+1)/2),  and  = +(n+1)^2/2 for k = 2n+1')
    print('       (Osc, infinite-dim., One is its lowest weight vector).')
    print('  check sl(2) = o(3):  k = 2, i.e. n = 1, gives hw = 1 -- the spinor')
    print('  representation of o(3), of dimension 2.  The text\'s n(n+1) for')
    print('  k = 2n+1 and -n^2/2 for k = 2n fail this test (they give -1/2).')


if __name__ == '__main__':
    part_A()
    part_B()
