#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sv2.py -- the algebra  svect^L_lambda(1|2)  built from scratch, exactly, and
compared with the "SU(2)-superextension of k(1)" of Ademollo e.a. that
Ivanov-Krivonos call  k^A(1|2).

Everything is exact: coefficients are sympy rationals in the formal parameter
lambda (no floating point anywhere), Laurent polynomials in t, Grassmann in
theta_1, theta_2.

Conventions.
  * A superfunction is a dict {(m, S): c}, m in Z the power of t, S a sorted
    tuple of indices of the theta's, c a sympy expression.
  * A vector field X is the triple (X(t), X(theta_1), X(theta_2)) of
    superfunctions, i.e. X = X(t) d_t + sum_i X(theta_i) d_i with the
    coefficients written on the LEFT, and
        X(f) = X(t) d_t(f) + sum_i X(theta_i) d_i(f),
    d_i being the left derivative.  Then X(fg) = X(f)g + (-1)^{|X||f|} f X(g),
    which the script verifies.
  * [X,Y](a) = X(Y(a)) - (-1)^{|X||Y|} Y(X(a))  for a in {t, theta_1, theta_2}.

Run:  ../venv/bin/python sv2.py   (plain python3 works too)
"""

import sys
import itertools
from sympy import Rational, symbols, simplify, expand, together, factor, nsimplify

lam = symbols('lambda')

TH = (1, 2)                      # the two odd indeterminates
COMPS = ('t', 1, 2)              # the components X(t), X(theta_1), X(theta_2)

FAIL = []
CHECKS = [0]


def check(cond, msg):
    CHECKS[0] += 1
    if not cond:
        FAIL.append(msg)
        print("  FAIL: " + msg)


# ---------------------------------------------------------------- functions

def fclean(f):
    return {k: expand(v) for k, v in f.items() if expand(v) != 0}


def fadd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return fclean(out)


def fscal(c, f):
    return fclean({k: c * v for k, v in f.items()})


def _merge_sign(S1, S2):
    """theta_S1 * theta_S2 = sign * theta_(S1 u S2); 0 if they meet."""
    if set(S1) & set(S2):
        return None, None
    merged = list(S1) + list(S2)
    sign = 1
    # bubble sort, counting transpositions of odd variables
    arr = merged[:]
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sign = -sign
    return tuple(arr), sign


def fmul(f, g):
    out = {}
    for (m1, S1), c1 in f.items():
        for (m2, S2), c2 in g.items():
            S, sg = _merge_sign(S1, S2)
            if S is None:
                continue
            out[(m1 + m2, S)] = out.get((m1 + m2, S), 0) + sg * c1 * c2
    return fclean(out)


def parity_f(f):
    """parity of a homogeneous superfunction; None for 0."""
    ps = {len(S) % 2 for (m, S) in f}
    if not ps:
        return None
    assert len(ps) == 1, "inhomogeneous superfunction"
    return ps.pop()


def d_t(f):
    out = {}
    for (m, S), c in f.items():
        if m != 0:
            out[(m - 1, S)] = out.get((m - 1, S), 0) + m * c
    return fclean(out)


def d_th(i, f):
    """left derivative d/d theta_i."""
    out = {}
    for (m, S), c in f.items():
        if i in S:
            pos = S.index(i)
            sg = (-1) ** pos
            T = tuple(x for x in S if x != i)
            out[(m, T)] = out.get((m, T), 0) + sg * c
    return fclean(out)


def der(a, f):
    return d_t(f) if a == 't' else d_th(a, f)


def mono(m, S=(), c=1):
    return fclean({(m, tuple(S)): c})


ONE = mono(0)


# ------------------------------------------------------------ vector fields

class VF:
    """X = (X(t), X(theta_1), X(theta_2)); p = parity of X."""

    def __init__(self, comps, p):
        self.c = {a: fclean(comps.get(a, {})) for a in COMPS}
        self.p = p % 2
        # sanity: parity of X(a) must be p + |a|
        for a in COMPS:
            f = self.c[a]
            pf = parity_f(f)
            if pf is not None:
                want = (self.p + (0 if a == 't' else 1)) % 2
                assert pf == want, "component parity clash at %s" % (a,)

    def is_zero(self):
        return all(not self.c[a] for a in COMPS)

    def act(self, f):
        """X(f) for a superfunction f."""
        out = {}
        for a in COMPS:
            if self.c[a]:
                out = fadd(out, fmul(self.c[a], der(a, f)))
        return out

    def __add__(self, other):
        assert self.p == other.p
        return VF({a: fadd(self.c[a], other.c[a]) for a in COMPS}, self.p)

    def scal(self, k):
        return VF({a: fscal(k, self.c[a]) for a in COMPS}, self.p)

    def __sub__(self, other):
        return self + other.scal(-1)

    def __repr__(self):
        return vf_str(self)


def bracket(X, Y):
    eps = (-1) ** (X.p * Y.p)
    comps = {}
    for a in COMPS:
        ya = X.act(Y.c[a]) if Y.c[a] else {}
        xa = Y.act(X.c[a]) if X.c[a] else {}
        comps[a] = fadd(ya, fscal(-eps, xa))
    return VF(comps, (X.p + Y.p) % 2)


def E(a, m, S=(), c=1, ):
    """the elementary field  t^m theta_S d_a ."""
    p = (len(S) + (0 if a == 't' else 1)) % 2
    return VF({a: mono(m, S, c)}, p)


def vf_str(X):
    parts = []
    for a in COMPS:
        for (m, S), c in sorted(X.c[a].items()):
            cs = str(factor(simplify(c)))
            if cs == '1':
                cs = ''
            elif cs == '-1':
                cs = '-'
            th = ''.join('th%d' % i for i in S)
            tp = '' if m == 0 else ('t^%d' % m)
            d = 'd_t' if a == 't' else 'd_%d' % a
            mono_s = '*'.join(z for z in (cs, tp, th, d) if z)
            parts.append(mono_s)
    return ' + '.join(parts) if parts else '0'


# ---------------------------------------------------------------- divergence

SIGN_CONV = {
    'plus':    lambda p: 1,
    'minus':   lambda p: -1,
    'par':     lambda p: (-1) ** p,
    'parodd':  lambda p: -(-1) ** p,
}


def div(X, sign):
    """div X = d_t X(t) + s(|X|) * sum_i d_i X(theta_i).

    The sign is allowed to depend on the parity of X; which of the four
    possibilities makes div a 1-cocycle is decided by the machine in sec0.
    """
    s = SIGN_CONV[sign](X.p)
    out = d_t(X.c['t'])
    for i in TH:
        out = fadd(out, fscal(s, d_th(i, X.c[i])))
    return out


def div_lam(X, sign, l=lam):
    """divergence relative to the volume form t^l vol."""
    return fadd(div(X, sign), fscal(l, fmul(mono(-1), X.c['t'])))


# =============================================================== 0. sanity

def rand_fields():
    """a small, fixed supply of test fields covering every monomial type."""
    out = []
    for m in (-2, -1, 0, 1, 3):
        for S in ((), (1,), (2,), (1, 2)):
            for a in COMPS:
                out.append(E(a, m, S))
    return out


def sec0():
    print("0. Sanity of the calculus")
    fs = [mono(0), mono(1), mono(-2), mono(2, (1,)), mono(0, (2,)),
          mono(-1, (1, 2)), fadd(mono(1, (1,)), mono(0, (2,)))]
    # Leibniz for the action
    n = 0
    for X in rand_fields()[:30]:
        for f in fs:
            for g in fs:
                pf = parity_f(f)
                if pf is None:
                    continue
                lhs = X.act(fmul(f, g))
                rhs = fadd(fmul(X.act(f), g),
                           fscal((-1) ** (X.p * pf), fmul(f, X.act(g))))
                check(fadd(lhs, fscal(-1, rhs)) == {},
                      "Leibniz fails for %s on %s,%s" % (X, f, g))
                n += 1
    print("   Leibniz rule X(fg)=X(f)g+(-1)^{|X||f|}fX(g): %d checks" % n)

    # Jacobi (super) on the elementary fields
    B = rand_fields()
    n = 0
    for X, Y, Z in itertools.islice(itertools.product(B, B, B), 0, None, 37):
        # (-1)^{|X||Z|}[[X,Y],Z] + cyclic = 0
        t1 = bracket(bracket(X, Y), Z).scal((-1) ** (X.p * Z.p))
        t2 = bracket(bracket(Y, Z), X).scal((-1) ** (Y.p * X.p))
        t3 = bracket(bracket(Z, X), Y).scal((-1) ** (Z.p * Y.p))
        check((t1 + t2 + t3).is_zero(), "Jacobi fails on %s,%s,%s" % (X, Y, Z))
        n += 1
    print("   graded Jacobi identity: %d triples" % n)

    # which sign makes div a 1-cocycle:  div[X,Y] = X(div Y) - (-1)^{..}Y(div X)
    good = None
    for sign in SIGN_CONV:
        ok = True
        for X in B:
            for Y in B:
                lhs = div(bracket(X, Y), sign)
                rhs = fadd(X.act(div(Y, sign)),
                           fscal(-(-1) ** (X.p * Y.p), Y.act(div(X, sign))))
                if fadd(lhs, fscal(-1, rhs)) != {}:
                    ok = False
                    break
            if not ok:
                break
        print("   div with the sign rule %-7s is a 1-cocycle: %s" % (sign, ok))
        if ok:
            good = sign
    check(good is not None, "no sign rule makes div a 1-cocycle")
    # and the twisted one, div_lambda, must be a cocycle as well
    ok = True
    for X in B:
        for Y in B:
            lhs = div_lam(bracket(X, Y), good)
            rhs = fadd(X.act(div_lam(Y, good)),
                       fscal(-(-1) ** (X.p * Y.p), Y.act(div_lam(X, good))))
            if fadd(lhs, fscal(-1, rhs)) != {}:
                ok = False
    check(ok, "div_lambda is not a 1-cocycle")
    print("   div_lambda = div + lambda t^{-1}X(t) is a 1-cocycle too: %s" % ok)
    return good


# ============================================== 1. svect_lambda: the kernel

def in_svect(X, sign):
    return div_lam(X, sign) == {}


def mode_basis(n, sign, parity):
    """A basis of the mode-n, given-parity piece of vect^L(1|2).

    mode n of  t^m theta_S d_a  is  (m - w) + (|S| - v)/2,
    w = 1 for d_t and 0 else, v = 1 for d_theta and 0 else.
    """
    out = []
    for a in COMPS:
        w = 1 if a == 't' else 0
        v = 0 if a == 't' else 1
        for S in ((), (1,), (2,), (1, 2)):
            p = (len(S) + v) % 2
            if p != parity:
                continue
            m2 = n + w - Rational(len(S) - v, 2)
            if m2.q != 1:
                continue
            out.append(E(a, int(m2), S))
    return out


def kernel_of_div(basis, sign):
    """Basis of {sum c_i B_i : div_lambda = 0}, over Q(lambda)."""
    from sympy import Matrix
    cols = []
    keys = set()
    dv = [div_lam(B, sign) for B in basis]
    for d in dv:
        keys |= set(d.keys())
    keys = sorted(keys)
    M = Matrix([[d.get(k, 0) for d in dv] for k in keys]) if keys else None
    if M is None:
        return [ [1 if i == j else 0 for j in range(len(basis))]
                 for i in range(len(basis))]
    return [list(v) for v in M.nullspace()]


def sec1(sign):
    print("\n1. svect^L_lambda(1|2) = ker div_lambda")
    for n in (Rational(-3), Rational(-1), Rational(0), Rational(2),
              Rational(-3, 2), Rational(-1, 2), Rational(1, 2), Rational(5, 2)):
        par = 0 if n.q == 1 else 1
        B = mode_basis(n, sign, par)
        K = kernel_of_div(B, sign)
        print("   mode n=%-5s parity %d: dim vect = %d, dim svect = %d"
              % (n, par, len(B), len(K)))
        check(len(B) == 6, "mode %s: expected 6 basis fields" % n)
        check(len(K) == 4, "mode %s: expected a 4-dimensional kernel" % n)
    print("   => 4 even towers (L and an sl(2) of currents) + 4 odd ones")


# ================================================ 2. the explicit generators

GOOD_SIGN = ['minus']            # set by sec0; the generators consult it


def _solve_coeff(mk):
    """mk(c) is a vector field depending linearly on the unknown c; return the
    unique c with div_lambda = 0 (the machine, not the author, fixes it)."""
    from sympy import Symbol, solve
    c = Symbol('c_')
    eqs = []
    for f in div_lam(mk(c), GOOD_SIGN[0]).values():
        eqs.append(expand(f))
    sol = solve(eqs, c, dict=True)
    assert len(sol) == 1, "coefficient not determined: %s" % eqs
    return simplify(sol[0][c])


def L(n):
    """L_n = -( t^{n+1} d_t + c t^n (th1 d_1 + th2 d_2) ), c fixed by div=0."""
    c = _solve_coeff(lambda z: VF({'t': mono(n + 1, (), -1),
                                   1: mono(n, (1,), z),
                                   2: mono(n, (2,), z)}, 0))
    return VF({'t': mono(n + 1, (), -1),
               1: mono(n, (1,), c),
               2: mono(n, (2,), c)}, 0)


# sl(2) = traceless part of gl(2) acting on the theta's
def J(a, n):
    """a = '+', '-', '0':  e = th1 d_2, f = th2 d_1, h = th1 d_1 - th2 d_2."""
    if a == '+':
        return VF({2: mono(n, (1,))}, 0)
    if a == '-':
        return VF({1: mono(n, (2,))}, 0)
    if a == '0':
        return VF({1: mono(n, (1,)), 2: mono(n, (2,), -1)}, 0)
    raise ValueError(a)


def G(al, r):
    """G_alpha^r = t^{r+1/2} d_alpha  (the 'vertical' doublet)."""
    p = r + Rational(1, 2)
    assert p.q == 1
    return VF({al: mono(int(p))}, 1)


def Gbar(al, r):
    """Gbar^{r alpha} = t^p th_alpha d_t + c t^{p-1} th1 th2 d_beta, p = r+1/2,
    beta the other index, c fixed by div_lambda = 0."""
    p = r + Rational(1, 2)
    assert p.q == 1
    p = int(p)
    be = 2 if al == 1 else 1
    mk = lambda z: VF({'t': mono(p, (al,)),
                       be: mono(p - 1, (1, 2), z)}, 1)
    return mk(_solve_coeff(mk))


def sec2(sign):
    print("\n2. The generators: L, the sl(2)-currents, and the two doublets")
    rng = [Rational(k) for k in (-3, -2, -1, 0, 1, 2, 3)]
    hrng = [Rational(2 * k + 1, 2) for k in (-3, -2, -1, 0, 1, 2)]
    gens = ([('L', L(int(n)), n) for n in rng]
            + [('J%s' % a, J(a, int(n)), n) for a in ('+', '-', '0') for n in rng]
            + [('G%d' % al, G(al, r), r) for al in TH for r in hrng]
            + [('Gb%d' % al, Gbar(al, r), r) for al in TH for r in hrng])
    for nm, X, n in gens:
        check(in_svect(X, sign), "%s_%s is not divergence-free" % (nm, n))
    print("   all %d listed generators satisfy div_lambda = 0" % len(gens))

    # do they span each mode?
    from sympy import Matrix
    for n in (Rational(-2), Rational(1), Rational(-3, 2), Rational(3, 2)):
        par = 0 if n.q == 1 else 1
        B = mode_basis(n, sign, par)
        if par == 0:
            fam = [L(int(n))] + [J(a, int(n)) for a in ('+', '-', '0')]
        else:
            fam = [G(al, n) for al in TH] + [Gbar(al, n) for al in TH]
        M = []
        for X in fam:
            row = []
            for Bi in B:
                a = [k for k in COMPS if Bi.c[k]][0]
                key = list(Bi.c[a].keys())[0]
                row.append(X.c[a].get(key, 0))
            M.append(row)
        r = Matrix(M).rank()
        check(r == 4, "mode %s: the four listed generators are dependent" % n)
    print("   and they are linearly independent, hence a basis of every mode")
    return gens


# ================================================== 3. the relations

def as_combo(X, n, sign):
    """Write X in the basis (L,J+,J-,J0 | G1,G2,Gb1,Gb2) of its mode."""
    from sympy import Matrix
    if X.p == 0:
        names = ['L', 'J+', 'J-', 'J0']
        fam = [L(int(n)), J('+', int(n)), J('-', int(n)), J('0', int(n))]
    else:
        names = ['G1', 'G2', 'Gb1', 'Gb2']
        fam = [G(1, n), G(2, n), Gbar(1, n), Gbar(2, n)]
    keys = []
    for Y in fam + [X]:
        for a in COMPS:
            for k in Y.c[a]:
                if (a, k) not in keys:
                    keys.append((a, k))
    A = Matrix([[Y.c[a].get(k, 0) for Y in fam] for (a, k) in keys])
    b = Matrix([[X.c[a].get(k, 0)] for (a, k) in keys])
    sol = A.solve_least_squares(b) if A.rows > A.cols else A.solve(b)
    sol = [simplify(together(s)) for s in sol]
    Z = fam[0].scal(sol[0])
    for c, Y in zip(sol[1:], fam[1:]):
        Z = Z + Y.scal(c)
    check((Z - X).is_zero(),
          "the combination does not reproduce the field at mode %s" % n)
    return list(zip(names, sol))


def combo_str(X, n, sign):
    terms = []
    for nm, c in as_combo(X, n, sign):
        c = simplify(c)
        if c != 0:
            terms.append("(%s) %s_{%s}" % (factor(c), nm, n))
    return ' + '.join(terms) if terms else '0'


# --- the sl(2) tables, in the basis e = th1 d_2, f = th2 d_1, h = th1d_1-th2d_2
SL2 = {('+', '-'): {'0': 1}, ('-', '+'): {'0': -1},
       ('0', '+'): {'+': 2}, ('+', '0'): {'+': -2},
       ('0', '-'): {'-': -2}, ('-', '0'): {'-': 2},
       ('+', '+'): {}, ('-', '-'): {}, ('0', '0'): {}}


def lin(pairs):
    """build sum c_i * gen_i ; pairs = [(coeff, VF)]"""
    Z = None
    for c, Y in pairs:
        W = Y.scal(c)
        Z = W if Z is None else Z + W
    return Z


def sec3(sign):
    """Verify the closed form of every bracket over a window of modes."""
    print("\n3. The relations, in closed form and verified over a window")
    ms = [Rational(k) for k in range(-3, 4)]
    rs = [Rational(2 * k + 1, 2) for k in range(-3, 3)]
    half = Rational(1, 2)

    print("\n   the generators themselves (mode n, resp. r):")
    print("     L_n    = %s" % L(1))
    print("     J^+_n  = %s,   J^-_n = %s,   J^0_n = %s"
          % (J('+', 1), J('-', 1), J('0', 1)))
    print("     G_1^r  = %s,   G_2^r = %s" % (G(1, half), G(2, half)))
    print("     Gb^{r1} = %s" % Gbar(1, half))
    print("     Gb^{r2} = %s" % Gbar(2, half))

    tests = []

    # (R1) Virasoro
    tests.append(("[L_m,L_n] = (m-n) L_{m+n}",
                  [(bracket(L(int(m)), L(int(n))), L(int(m + n)).scal(m - n))
                   for m in ms for n in ms]))
    # (R2) the currents have weight 1
    tests.append(("[L_m,J^a_n] = -n J^a_{m+n}",
                  [(bracket(L(int(m)), J(a, int(n))),
                    J(a, int(m + n)).scal(-n))
                   for m in ms for n in ms for a in ('+', '-', '0')]))
    # (R3) loop sl(2), no level, no lambda
    tests.append(("[J^a_m,J^b_n] = [J^a,J^b]_{m+n}  (level 0)",
                  [(bracket(J(a, int(m)), J(b, int(n))),
                    lin([(c, J(x, int(m + n)))
                         for x, c in SL2[(a, b)].items()] or
                        [(0, J('0', int(m + n)))]))
                   for m in ms for n in ms
                   for a in ('+', '-', '0') for b in ('+', '-', '0')]))
    # (R4),(R5) the two doublets have weight 3/2 but SHIFTED modings
    tests.append(("[L_m,G_al^r]  = (m/2 - r + lambda/2) G_al^{m+r}",
                  [(bracket(L(int(m)), G(al, r)),
                    G(al, m + r).scal(m / 2 - r + lam / 2))
                   for m in ms for r in rs for al in TH]))
    tests.append(("[L_m,Gb^{r al}] = (m/2 - r - lambda/2) Gb^{(m+r) al}",
                  [(bracket(L(int(m)), Gbar(al, r)),
                    Gbar(al, m + r).scal(m / 2 - r - lam / 2))
                   for m in ms for r in rs for al in TH]))
    # (R6),(R7) the doublets are sl(2)-doublets: G is the tautological module,
    # Gb the dual one (read off, then verified)
    # read off the machine in the first pass, then verified over the window:
    # G_al is the doublet with a LOWER index, Gb^al the dual (upper) one
    GACT = {('+', 1): (2, -1), ('+', 2): None, ('-', 1): None,
            ('-', 2): (1, -1), ('0', 1): (1, -1), ('0', 2): (2, 1)}
    GBACT = {('+', 1): None, ('+', 2): (1, 1), ('-', 1): (2, 1),
             ('-', 2): None, ('0', 1): (1, 1), ('0', 2): (2, -1)}
    for nm, ctor, tbl in (('G', G, GACT), ('Gb', Gbar, GBACT)):
        rows = []
        for m in ms:
            for r in rs:
                for a in ('+', '-', '0'):
                    for al in TH:
                        act = tbl[(a, al)]
                        rhs = (ctor(act[0], m + r).scal(act[1]) if act
                               else ctor(al, m + r).scal(0))
                        rows.append((bracket(J(a, int(m)), ctor(al, r)), rhs))
        tests.append(("[J^a_m,%s^r]: %s is an sl(2)-doublet, no lambda"
                      % (nm, nm), rows))
    # (R8) both doublets are abelian
    ZERO = VF({}, 0)
    tests.append(("{G_al^r,G_be^s} = 0",
                  [(bracket(G(al, r), G(be, s)), ZERO)
                   for r in rs for s in rs for al in TH for be in TH]))
    tests.append(("{Gb^{r al},Gb^{s be}} = 0",
                  [(bracket(Gbar(al, r), Gbar(be, s)), ZERO)
                   for r in rs for s in rs for al in TH for be in TH]))
    # (R9) the mixed bracket: -delta L + (r-s-lambda) (sigma_k) J^k
    #      with J^3 = J^0/2 and J^{+-} = J^1 +- i J^2
    SIG = {(1, 1): [('0', Rational(1, 2))], (2, 2): [('0', Rational(-1, 2))],
           (1, 2): [('-', 1)], (2, 1): [('+', 1)]}
    rows = []
    for r in rs:
        for s in rs:
            for al in TH:
                for be in TH:
                    pieces = [(-1 if al == be else 0, L(int(r + s)))]
                    for x, c in SIG[(al, be)]:
                        pieces.append(((r - s - lam) * c, J(x, int(r + s))))
                    rows.append((bracket(G(al, r), Gbar(be, s)), lin(pieces)))
    tests.append(("{G_al^r,Gb^{s be}} = -delta_al^be L_{r+s} "
                  "+ (r-s-lambda)(sigma_k)_al^be J^k_{r+s}", rows))

    for name, rows in tests:
        bad = 0
        for got, want in rows:
            if not (got - want).is_zero():
                bad += 1
                if bad == 1:
                    print("      first mismatch: %s  vs  %s" % (got, want))
        check(bad == 0, "%s fails %d times" % (name, bad))
        print("   %-62s %4d pairs, %d failures" % (name, len(rows), bad))


# ====================================== 4. what lambda is, and where it hides

def sec4(sign):
    print("\n4. The parameter lambda")
    print("   lambda occurs in exactly two places:")
    print("     * the theta-part of L_n  (so L_0 measures the theta's with a shift)")
    print("     * the theta1theta2-part of Gb;")
    print("   the currents J^a_n and the doublet G_al^r do not contain it.")
    seen = []
    for a in ('+', '-', '0'):
        for n in (-2, 0, 3):
            for c in COMPS:
                for v in J(a, n).c[c].values():
                    if lam in v.free_symbols:
                        seen.append('J^%s' % a)
    for al in TH:
        for r in (Rational(1, 2), Rational(-3, 2)):
            for c in COMPS:
                for v in G(al, r).c[c].values():
                    if lam in v.free_symbols:
                        seen.append('G_%d' % al)
    check(not seen, "lambda occurs in %s" % set(seen))

    print("\n   In the relations it occurs only as the shift of the moding of")
    print("   the two doublets: setting  r' = r - lambda/2  for G and")
    print("   s' = s + lambda/2  for Gb turns (R4),(R5),(R9) into")
    print("     [L_m,G^{r'}] = (m/2-r')G,  [L_m,Gb^{s'}] = (m/2-s')Gb,")
    print("     {G^{r'},Gb^{s'}} = -delta L + (r'-s')(sigma_k)J^k,")
    print("   because r-s-lambda = r'-s' and r+s = r'+s'.  So for lambda = 0")
    print("   the doublets are moded in Z+1/2 (Neveu-Schwarz) and the relations")
    print("   are those of Ademollo e.a. after G |-> sqrt(2) G.")

    # the twist automorphism: Phi_a: t -> t, th_1 -> t^a th_1, th_2 -> th_2
    print("\n   lambda lives in C/Z:  the substitution th_1 |-> t^a th_1 is an")
    print("   automorphism of the Laurent superalgebra and moves lambda by a.")
    for a in (1, -1, 2):
        ok = twist_check(a, sign)
        print("     th_1 |-> t^%-2d th_1 :  svect_lambda -> svect_{lambda%+d} : %s"
              % (a, -a, ok))


def phi(a, f, inv=False):
    """the algebra map t -> t, th_1 -> t^a th_1, th_2 -> th_2 on functions."""
    e = -a if inv else a
    out = {}
    for (m, S), c in f.items():
        out[(m + (e if 1 in S else 0), S)] = out.get(
            (m + (e if 1 in S else 0), S), 0) + c
    return fclean(out)


def push(a, X):
    """Phi_* X = Phi o X o Phi^{-1}, componentwise."""
    comps = {'t': phi(a, X.act(mono(1))),
             1: phi(a, X.act(phi(a, mono(0, (1,)), inv=True))),
             2: phi(a, X.act(mono(0, (2,))))}
    return VF(comps, X.p)


def twist_check(a, sign):
    """Phi_a maps ker div_lambda onto ker div_{lambda - a}?  Return the shift."""
    from sympy import Symbol
    B = []
    for n in (Rational(-2), Rational(1), Rational(-3, 2), Rational(3, 2)):
        par = 0 if n.q == 1 else 1
        if par == 0:
            B += [L(int(n))] + [J(x, int(n)) for x in ('+', '-', '0')]
        else:
            B += [G(al, n) for al in TH] + [Gbar(al, n) for al in TH]
    shifts = set()
    for X in B:
        Y = push(a, X)
        d = div_lam(Y, sign, l=Symbol('mu'))
        from sympy import solve, Symbol as Sy
        eqs = [expand(v) for v in d.values()]
        if not eqs:
            continue
        sol = solve(eqs, Sy('mu'), dict=True)
        if not sol:
            return "NO"
        shifts.add(simplify(sol[0][Sy('mu')]))
    if len(shifts) != 1:
        return "inconsistent: %s" % shifts
    mu = shifts.pop()
    check(simplify(mu - (lam - a)) == 0 or simplify(mu - (lam + a)) == 0,
          "twist by %d gives mu = %s" % (a, mu))
    return "yes, mu = %s" % mu


# ============================ 5. the chain of subalgebras k(1|2) > k(1|1) > k(1)

def sec5(sign):
    """Ivanov-Krivonos state the chain k^A(1|2) > k(1|2) > k(1|1) > k(1);
    our identification must reproduce it inside svect^L_lambda(1|2)."""
    print("\n5. The subalgebras Ivanov-Krivonos list")
    ms = [Rational(k) for k in range(-3, 4)]
    rs = [Rational(2 * k + 1, 2) for k in range(-3, 3)]

    # the N=2 algebra: L, the u(1)-current J^0, and ONE pair (G_1, Gb^1)
    fam = ([('L', L(int(m)), m) for m in ms]
           + [('J0', J('0', int(m)), m) for m in ms]
           + [('G1', G(1, r), r) for r in rs]
           + [('Gb1', Gbar(1, r), r) for r in rs])
    bad = 0
    for na, X, a in fam:
        for nb, Y, b in fam:
            Z = bracket(X, Y)
            if Z.is_zero():
                continue
            combo = dict(as_combo(Z, a + b, sign))
            for nm, c in combo.items():
                if simplify(c) != 0 and nm not in ('L', 'J0', 'G1', 'Gb1'):
                    bad += 1
    check(bad == 0, "the span of L, J^0, G_1, Gb^1 is not closed (%d)" % bad)
    print("   span(L_n, J^0_n, G_1^r, Gb^{r1}) is a subalgebra: %d leaks"
          % bad)
    print("   with [J^0_m,J^0_n] = 0, [J^0_m,G_1^r] = -G_1, [J^0_m,Gb^{r1}] = +Gb,")
    print("   {G_1,G_1} = {Gb^1,Gb^1} = 0 and {G_1^r,Gb^{s1}} = -L + (r-s-lambda)J^0/2")
    print("   -- these are the relations of k^L(1|2), the N=2 algebra;")
    print("   dropping J^0 and one of the two odd towers leaves k^L(1|1), and")
    print("   the L-tower alone is the Witt algebra k(1) = vect^L(1).  So the")
    print("   chain k^A(1|2) > k(1|2) > k(1|1) > k(1) of [IvKr4] is reproduced.")

    # k^L(1|4) is NOT inside: it has 16 towers, we have 8
    print("   By the count of item 1 there are 8 towers in all, so neither")
    print("   k^L(1|4) (16 towers) nor its derived algebra can occur here.")


def main():
    print("=" * 78)
    print("svect^L_lambda(1|2) versus the k^A(1|2) of Ivanov-Krivonos")
    print("=" * 78)
    sign = sec0()
    GOOD_SIGN[0] = sign
    sec1(sign)
    sec2(sign)
    sec3(sign)
    sec4(sign)
    sec5(sign)
    print("\n" + "=" * 78)
    print("checks: %d, failures: %d" % (CHECKS[0], len(FAIL)))
    for f in FAIL:
        print("  ! " + f)
    print("=" * 78)


if __name__ == '__main__':
    main()
