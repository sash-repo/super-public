#!/usr/bin/env python3
"""Do Volichenko algebras stand to metabelian algebras of functions as Lie
algebras stand to commutative ones?  Subsecs. 3.7--3.9 of the text, with
reference to U. Iyer, "Volichenko algebras as algebras
of differential operators", J. Nonlinear Math. Phys. 13:1 (2006) 34-49 [Iy2].

The setting is Iyer's Subsec. 4.2.3, the same as in iyer.py:

    S = k[x_1..x_n] (x) Lambda[y_1..y_n],   z_i = x_i + y_i,
    M = k[Z] = k<z_1,...,z_n> subset S,     d = sum_k y_k d/dx_k,  d^2 = 0.

S is the (universal) supercommutative envelope of the metabelian algebra M,
and d is the odd superderivation that Volichenko's Theorem M supplies.

Three things are computed, all in exact rational arithmetic.

(1) NONTRIVIALITY.  Iyer's Prop. 4.2.1(3) says VD^1(M) = VD^0(M) + V with V a
    Volichenko algebra, and proves it by quoting Example 2.5 of [LSe1]: d is
    homologic, so h_{ad_d} = {a + [d,a] | a in g_ev} is closed under the
    superbracket of g = Der(S).  That argument is silent about whether the
    result is a NEW structure.  By Subsec. 2.9 of the paper, h_{ad_x} is
    isomorphic to the Lie algebra g_ev unless x "ensures nontriviality", i.e.
    unless

        S(a,b) := [[a,x],[b,x]] /= 0   for some a, b in g_ev.

    So we compute S(a,b) for x = d over a basis of g_ev.  This decides whether
    the analogy delivers anything beyond a Lie algebra.

(2) THE NAIVE CANDIDATE.  The object one would write down first is

        v(M) := {D in Der(S) | D(M) subset M},

    the vector fields "tangent to the metaspace".  It is always a Lie ALGEBRA
    (a composite of two M-preserving operators preserves M, so the commutator
    does).  Being a VOLICHENKO algebra is a strictly stronger demand: the
    superbracket of two inhomogeneous elements is NOT their commutator,
        [D,D']_s = [D,D'] + 2 D'_od D_od ,
    and the parity components D_ev, D_od need not preserve M separately.  We
    compute v(M) exactly and test superbracket closure.  This is the precise
    form of Iyer's own worry in Remark 4.2.1 ("It is not clear whether
    [phi,a]_beta is in D_beta(M)"), and of her Remark 4.2.2.

(3) GHOSTS.  The map v(M) -> End(M) by restriction has a kernel -- Iyer's
    "ghost operators".  We compute it: a Volichenko algebra of operators ON M
    is faithful only if the kernel vanishes.

Vector fields are represented by their coefficients, D = sum a_i d/dx_i +
sum b_j d/dy_j with a_i, b_j in S; the parity split, which the superbracket
needs, is then explicit.  Brackets are computed by composing the parity
components as operators and reading the coefficients of the result off its
action on the generators -- never by a hand-derived sign formula; the reading
is verified against the operator (assert_is_vf), so a sign error cannot pass
silently.
"""

import sys
import os
from fractions import Fraction
from itertools import combinations

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "."))
from exactlin import nullspace, rank, backend          # noqa: E402

from iyer import (mono, add, scal, sub, is_zero, Op, op_sum, op_scal,
                 mult_x, d_dx, mult_y, d_dy, basis, compositions, mul_S,
                 z_word_n, words, Span, sort_word)      # noqa: E402


def compose(A, B):
    """iyer.compose, but tolerating an inhomogeneous factor (parity None).
    Only the action matters here -- all brackets below are formed from the
    parity COMPONENTS, so no composite ever needs a parity label."""
    if A.parity is None or B.parity is None:
        par = None
    else:
        par = (A.parity + B.parity) % 2
    return Op(lambda alpha, ys: A(B.fn(alpha, ys)), par)


# --------------------------------------------------------------- elements of S

def s_parity_split(a):
    """Split an element of S into its even and odd halves."""
    ev = {k: c for k, c in a.items() if len(k[1]) % 2 == 0}
    od = {k: c for k, c in a.items() if len(k[1]) % 2 == 1}
    return ev, od


def mult(a):
    """Left multiplication by an arbitrary (possibly inhomogeneous) a in S."""
    return Op(lambda alpha, ys: mul_S(a, mono(alpha, ys)), None)


def s_basis(n, deg):
    """Monomial basis of S with x-degree <= deg (a list of keys)."""
    return basis(n, deg)


# ---------------------------------------------------------------- vector fields
# A vector field is a pair of lists (a, b) of elements of S:
#     D = sum_i a_i d/dx_i + sum_j b_j d/dy_j .
# Parity: d/dx_i is even, d/dy_j is odd, so
#     D_ev = sum a_i^ev d/dx_i + sum b_j^od d/dy_j,
#     D_od = sum a_i^od d/dx_i + sum b_j^ev d/dy_j.

class VF:
    def __init__(self, n, a=None, b=None):
        self.n = n
        self.a = list(a) if a else [{} for _ in range(n)]
        self.b = list(b) if b else [{} for _ in range(n)]

    def copy(self):
        return VF(self.n, [dict(c) for c in self.a], [dict(c) for c in self.b])

    def is_zero(self):
        return all(is_zero(c) for c in self.a) and all(is_zero(c) for c in self.b)

    def parts(self):
        """(even part, odd part) as vector fields."""
        ev = VF(self.n)
        od = VF(self.n)
        for i in range(self.n):
            ev.a[i], od.a[i] = s_parity_split(self.a[i])
            od.b[i], ev.b[i] = s_parity_split(self.b[i])
        return ev, od

    def parity(self):
        ev, od = self.parts()
        if od.is_zero():
            return 0
        if ev.is_zero():
            return 1
        return None                      # genuinely inhomogeneous

    def op(self):
        """The linear operator on S that this vector field is."""
        terms = []
        for i in range(self.n):
            if not is_zero(self.a[i]):
                terms.append(compose(mult(self.a[i]), d_dx(i)))
            if not is_zero(self.b[i]):
                terms.append(compose(mult(self.b[i]), d_dy(i)))
        if not terms:
            return Op(lambda alpha, ys: {}, None)
        return op_sum(*terms)

    def __repr__(self):
        return f"VF(a={self.a}, b={self.b})"


def vf_add(D, E):
    out = VF(D.n)
    for i in range(D.n):
        out.a[i] = add(D.a[i], E.a[i])
        out.b[i] = add(D.b[i], E.b[i])
    return out


def vf_scal(c, D):
    out = VF(D.n)
    for i in range(D.n):
        out.a[i] = scal(c, D.a[i])
        out.b[i] = scal(c, D.b[i])
    return out


def gen_x(n, i):
    return mono([1 if k == i else 0 for k in range(n)], ())


def gen_y(n, j):
    return mono([0] * n, (j,))


def read_vf(A, n):
    """Read a derivation off its action on the generators: a_i = A(x_i),
    b_j = A(y_j)."""
    return VF(n, [A(gen_x(n, i)) for i in range(n)],
              [A(gen_y(n, j)) for j in range(n)])


def assert_is_vf(A, D, n, deg, what):
    """Verify that the operator A really is the vector field D on x-degree
    <= deg.  This is the guard that makes the coefficient reading safe."""
    B = D.op()
    for alpha, ys in s_basis(n, deg):
        if not is_zero(sub(A.fn(alpha, ys), B.fn(alpha, ys))):
            raise AssertionError(
                f"{what}: the bracket is not the vector field read off it, "
                f"at x^{alpha} y_{ys}")


def sbracket(D, E, n, deg=2, check=True):
    """The SUPERbracket of two (possibly inhomogeneous) vector fields, computed
    by splitting into parity components:
        [D,E]_s = sum_{p,q} (D_p E_q - (-1)^{pq} E_q D_p).
    """
    Dp = D.parts()
    Ep = E.parts()
    terms = []
    for p in (0, 1):
        for q in (0, 1):
            X, Y = Dp[p], Ep[q]
            if X.is_zero() or Y.is_zero():
                continue
            xo, yo = X.op(), Y.op()
            sign = (-1) ** (p * q)
            terms.append(op_sum(compose(xo, yo), op_scal(-sign, compose(yo, xo))))
    if not terms:
        return VF(n)
    A = op_sum(*terms)
    out = read_vf(A, n)
    if check:
        assert_is_vf(A, out, n, deg, "sbracket")
    return out


def cbracket(D, E, n, deg=2, check=True):
    """The plain commutator DE - ED, as a vector field."""
    do, eo = D.op(), E.op()
    A = op_sum(compose(do, eo), op_scal(-1, compose(eo, do)))
    out = read_vf(A, n)
    if check:
        assert_is_vf(A, out, n, deg, "cbracket")
    return out


# ------------------------------------------------------ bases of Der(S) by degree

def der_basis(n, xdeg, parity=None):
    """Basis of the truncation of Der(S) whose coefficients have x-degree
    <= xdeg.  parity None: all; 0/1: the homogeneous part of that parity."""
    out = []
    keys = s_basis(n, xdeg)
    for i in range(n):
        for (alpha, ys) in keys:
            p = len(ys) % 2                      # parity of the coefficient
            if parity is not None and p != parity:
                continue                          # d/dx_i is even
            D = VF(n)
            D.a[i] = mono(alpha, ys)
            out.append((f"x^{alpha}y{ys} d/dx_{i+1}", D))
    for j in range(n):
        for (alpha, ys) in keys:
            p = (len(ys) + 1) % 2                # d/dy_j is odd
            if parity is not None and p != parity:
                continue
            D = VF(n)
            D.b[j] = mono(alpha, ys)
            out.append((f"x^{alpha}y{ys} d/dy_{j+1}", D))
    return out


# ---------------------------------------------------------- (1) nontriviality

def nontriviality(n, xdeg=2, verbose=True):
    """Does d = sum y_k d/dx_k ensure nontriviality in g = Der(S)?

    Returns (witness or None, number of nonzero S(a,b) found).
    """
    d = VF(n)
    for k in range(n):
        d.a[k] = gen_y(n, k)                     # d = sum y_k d/dx_k
    # d is odd and homologic
    assert d.parity() == 1, "d must be odd"
    dd = sbracket(d, d, n)
    assert dd.is_zero(), "d must be homologic: [d,d] = 0"

    ev = der_basis(n, xdeg, parity=0)
    if verbose:
        print(f"  n={n}: dim g_ev truncated at x-degree <= {xdeg} is {len(ev)}")

    # ad_d on the even basis
    img = []
    for name, a in ev:
        img.append((name, sbracket(a, d, n)))

    found = 0
    witness = None
    for (na, u) in img:
        if u.is_zero():
            continue
        for (nb, v) in img:
            if v.is_zero():
                continue
            s = sbracket(u, v, n)
            if not s.is_zero():
                found += 1
                if witness is None:
                    witness = (na, nb, s)
    return witness, found


# ------------------------------------------------- (2) the naive candidate v(M)

def m_span(n, length):
    """Row-echelon span of M = k[Z] from all words of length <= length."""
    sp = Span.__new__(Span)
    sp.rows = {}
    for L in range(length + 1):
        for w in words(n, L):
            sp.insert(z_word_n(w, n))
    return sp


def reduce_full(sp, v):
    """Complete reduction of v modulo the echelon basis sp -- a genuinely
    LINEAR normal form.

    Span.reduce of iyer.py stops as soon as the largest surviving key is not a
    pivot.  That is a sound membership oracle (an element of the span has its
    largest key among the pivots), but it is not linear, so it must not be used
    to assemble a linear system: doing so silently over-constrains the system
    and shrinks the answer.  Here every key is processed, in decreasing order,
    which terminates because a subtraction only introduces strictly smaller
    keys."""
    v = {k: c for k, c in v.items() if c}
    out = {}
    while v:
        p = max(v)
        c = v.pop(p)
        r = sp.rows.get(p)
        if r is None:
            out[p] = c
            continue
        f = c / r[p]
        for k, rc in r.items():
            if k == p:
                continue
            nc = v.get(k, Fraction(0)) - f * rc
            if nc:
                v[k] = nc
            else:
                v.pop(k, None)
    return out


def headroom(n, wordlen, xdeg):
    """Word length up to which M must be spanned so that membership of D(w) is
    decided correctly.  D(w) has x-degree at most wordlen + xdeg (a coefficient
    of degree xdeg against d/dy, which removes a y and adds no x).  A word of
    length L carries monomials of x-degree L - (number of y's) >= L - n, so to
    hold every element of M of x-degree <= wordlen + xdeg one needs words up to
    length wordlen + xdeg + n.  Getting this bound wrong makes the linear
    system over-constrained and silently shrinks v(M) -- the same trap
    dereven.py fell into."""
    return wordlen + xdeg + n


def v_of_M(n, xdeg, wordlen, verbose=True):
    """v(M) = {D in Der(S) | D(M) subset M}, truncated: coefficients of
    x-degree <= xdeg, the condition imposed on words in the z_i of length
    <= wordlen.  Returns (basis as list of VF, the basis of Der(S) used)."""
    bas = der_basis(n, xdeg)
    sp = m_span(n, headroom(n, wordlen, xdeg))
    testers = [z_word_n(w, n) for L in range(wordlen + 1) for w in words(n, L)]

    # Build the linear system: for each tester w, reduce D(w) mod M and demand 0.
    rows = {}                                     # key -> row of coefficients
    ops = [D.op() for _, D in bas]
    for w in testers:
        cols = []
        for A in ops:
            cols.append(reduce_full(sp, A(w)))
        keys = set()
        for c in cols:
            keys |= set(c)
        for k in sorted(keys):
            rows[(id(w), k)] = [c.get(k, Fraction(0)) for c in cols]

    mat = list(rows.values())
    if verbose:
        print(f"  n={n}: dim Der(S)_trunc = {len(bas)}, "
              f"{len(mat)} linear conditions ({backend()})")
    if not mat:
        ns = [[Fraction(1) if i == j else Fraction(0) for i in range(len(bas))]
              for j in range(len(bas))]
    else:
        ns = nullspace(mat)

    out = []
    for vec in ns:
        D = VF(n)
        for c, (_, E) in zip(vec, bas):
            if c:
                D = vf_add(D, vf_scal(c, E))
        out.append(D)
    return out, bas


def homogeneous_part(vs, n, parity):
    """Dimension of the subspace of span(vs) consisting of vector fields of the
    given parity -- computed as the nullspace of "kill the other parity"."""
    if not vs:
        return 0, []
    # coordinates: collect all monomial slots appearing
    slots = set()
    for D in vs:
        for i in range(n):
            for k in D.a[i]:
                slots.add(("a", i, k))
            for k in D.b[i]:
                slots.add(("b", i, k))
    slots = sorted(slots)

    def coords_of_wrong_parity(D):
        row = []
        for (which, i, k) in slots:
            p = len(k[1]) % 2 if which == "a" else (len(k[1]) + 1) % 2
            if p == parity:
                continue
            c = (D.a[i] if which == "a" else D.b[i]).get(k, Fraction(0))
            row.append(c)
        return row

    mat = [coords_of_wrong_parity(D) for D in vs]
    if not mat or not mat[0]:
        return len(vs), vs
    # we need the nullspace of the TRANSPOSE: combinations of the vs whose
    # wrong-parity coordinates vanish
    tr = [[mat[r][c] for r in range(len(mat))] for c in range(len(mat[0]))]
    ns = nullspace(tr)
    sub_ = []
    for vec in ns:
        D = VF(n)
        for c, E in zip(vec, vs):
            if c:
                D = vf_add(D, vf_scal(c, E))
        sub_.append(D)
    return len(ns), sub_


def sbracket_op(D, E):
    """The superbracket as an OPERATOR (no assumption that it is a vector
    field -- for the superbracket it always is, and sbracket() checks that)."""
    Dp, Ep = D.parts(), E.parts()
    terms = []
    for p in (0, 1):
        for q in (0, 1):
            X, Y = Dp[p], Ep[q]
            if X.is_zero() or Y.is_zero():
                continue
            xo, yo = X.op(), Y.op()
            terms.append(op_sum(compose(xo, yo),
                                op_scal(-((-1) ** (p * q)), compose(yo, xo))))
    return op_sum(*terms) if terms else Op(lambda alpha, ys: {}, None)


def cbracket_op(D, E):
    """The plain commutator DE - ED as an OPERATOR.  For inhomogeneous D, E
    this is in general NOT a derivation -- only a second-order operator -- so
    it must not be read back as a vector field."""
    do, eo = D.op(), E.op()
    return op_sum(compose(do, eo), op_scal(-1, compose(eo, do)))


def closure_test(vs, n, sp, wordlen, which="s"):
    """Does the (super)bracket of two elements of v(M) still preserve M?
    Tested on the operators, so that the commutator -- which need not be a
    derivation -- is not rejected for the wrong reason.
    Returns a list of failures (i, j, witness word)."""
    testers = [(w, z_word_n(w, n)) for L in range(wordlen + 1) for w in words(n, L)]
    bad = []
    br = sbracket_op if which == "s" else cbracket_op
    for i in range(len(vs)):
        for j in range(len(vs)):
            A = br(vs[i], vs[j])
            for w, zm in testers:
                if reduce_full(sp, A(zm)):
                    bad.append((i, j, w))
                    break
    return bad


def stays_derivation(vs, n, deg=2):
    """Is the superbracket of two elements of v(M) again a derivation, and is
    the plain commutator?  Returns (n_super_fail, n_comm_fail)."""
    sfail = cfail = 0
    for i in range(len(vs)):
        for j in range(len(vs)):
            for br, isS in ((sbracket_op, True), (cbracket_op, False)):
                A = br(vs[i], vs[j])
                try:
                    assert_is_vf(A, read_vf(A, n), n, deg, "x")
                except AssertionError:
                    if isS:
                        sfail += 1
                    else:
                        cfail += 1
    return sfail, cfail


# ---------------------------------------------------------------- (3) ghosts

def ghosts(vs, n, wordlen, verbose=True):
    """The kernel of v(M) -> End(M): the operators acting as 0 on M."""
    testers = [z_word_n(w, n) for L in range(wordlen + 1) for w in words(n, L)]
    ops = [D.op() for D in vs]
    rows = {}
    for w in testers:
        cols = [A(w) for A in ops]
        keys = set()
        for c in cols:
            keys |= set(c)
        for k in sorted(keys):
            rows[(id(w), k)] = [c.get(k, Fraction(0)) for c in cols]
    mat = list(rows.values())
    if not mat:
        return list(vs)
    ns = nullspace(mat)
    out = []
    for vec in ns:
        D = VF(n)
        for c, E in zip(vec, vs):
            if c:
                D = vf_add(D, vf_scal(c, E))
        out.append(D)
    return out


# -------------------------------------------------------------------- reporting

def show_vf(D, n):
    parts = []
    for i in range(n):
        if not is_zero(D.a[i]):
            parts.append(f"({fmt(D.a[i], n)}) d/dx_{i+1}")
    for j in range(n):
        if not is_zero(D.b[j]):
            parts.append(f"({fmt(D.b[j], n)}) d/dy_{j+1}")
    return " + ".join(parts) if parts else "0"


def fmt(a, n):
    out = []
    for (alpha, ys), c in sorted(a.items()):
        m = []
        if c != 1 or (sum(alpha) == 0 and not ys):
            m.append(str(c))
        for i, e in enumerate(alpha):
            if e == 1:
                m.append(f"x{i+1}")
            elif e > 1:
                m.append(f"x{i+1}^{e}")
        for j in ys:
            m.append(f"y{j+1}")
        out.append("*".join(m) if m else "1")
    return " + ".join(out)


def iyer_VD(n, xdeg):
    """Iyer's V = {a + [d,a] | a in g_ev} -- the derivation part of VD^1(M).
    ad_d does not raise the x-degree, so the truncation is consistent."""
    d = VF(n)
    for k in range(n):
        d.a[k] = gen_y(n, k)
    out = []
    for _, a in der_basis(n, xdeg, parity=0):
        out.append(vf_add(a, sbracket(d, a, n)))     # a + [d,a], Iyer's order
    return out


def dim_span(vs, n):
    """Dimension of the span of a list of vector fields."""
    if not vs:
        return 0
    slots = set()
    for D in vs:
        for i in range(n):
            slots |= {("a", i, k) for k in D.a[i]}
            slots |= {("b", i, k) for k in D.b[i]}
    slots = sorted(slots)
    mat = [[(D.a[i] if w == "a" else D.b[i]).get(k, Fraction(0))
            for (w, i, k) in slots] for D in vs]
    return rank(mat)


def in_span(D, vs, n):
    """Is D in the span of vs?  rank(vs) == rank(vs + [D])."""
    return dim_span(vs, n) == dim_span(list(vs) + [D], n)


def ghost_model(n, xdeg):
    """The conjectured ghost space: theta * k[x] * (d/dy_j - d/dx_j), where
    theta = y_1...y_n is the top Grassmann monomial."""
    out = []
    theta = tuple(range(n))
    for (alpha, ys) in s_basis(n, xdeg):
        if ys:
            continue                              # coefficient theta * x^alpha
        for j in range(n):
            D = VF(n)
            D.a[j] = scal(-1, mono(alpha, theta))
            D.b[j] = mono(alpha, theta)
            out.append(D)
    return out


def main():
    print(f"exact linear algebra backend: {backend()}")
    print()
    print("=" * 78)
    print("(1) Does d = sum_k y_k d/dx_k ensure nontriviality in g = Der(S)?")
    print("    If not, Iyer's V = h_{ad_d} is isomorphic to the Lie algebra")
    print("    g_ev and the analogy delivers no new structure (Subsec. 2.9).   ")
    print("=" * 78)
    for n in (1, 2, 3):
        xdeg = 2
        w, cnt = nontriviality(n, xdeg=xdeg)
        if w is None:
            print(f"  n={n}: NO -- S(a,b) = [[a,d],[b,d]] = 0 for every pair "
                  f"of the {len(der_basis(n, xdeg, parity=0))} even basis "
                  f"fields.  h_{{ad_d}} = g_ev, a Lie algebra.")
        else:
            na, nb, s = w
            print(f"  n={n}: YES -- {cnt} nonzero values of S(a,b).  Witness:")
            print(f"          a = {na}")
            print(f"          b = {nb}")
            print(f"          S(a,b) = {show_vf(s, n)}")
        print()

    print("=" * 78)
    print("(2) v(M) = {D in Der(S) | D(M) subset M}: is it closed under the")
    print("    SUPERbracket (Volichenko) as well as the commutator (Lie)?")
    print("=" * 78)
    for n in (1, 2, 3):
        xdeg, wordlen = 1, 3
        vs, bas = v_of_M(n, xdeg, wordlen)
        print(f"  n={n}: dim v(M) = {len(vs)}  (coefficients of x-degree "
              f"<= {xdeg}, tested on words of length <= {wordlen})")
        dev, _ = homogeneous_part(vs, n, 0)
        dod, _ = homogeneous_part(vs, n, 1)
        print(f"        of which even {dev}, odd {dod}; "
              f"{dev + dod} vs {len(vs)} -> "
              + ("genuinely INHOMOGENEOUS" if dev + dod < len(vs)
                 else "parity-homogeneous (a subsuperalgebra, not Volichenko)"))
        # a bracket doubles the coefficient degree, so the span needs more room
        sp = m_span(n, headroom(n, wordlen, 2 * xdeg))
        # the closure test is quadratic in dim v(M); above a threshold test a
        # sample (the first SAMPLE basis vectors) and say so, rather than
        # printing an untested "OK"
        SAMPLE = 40
        sample = vs if len(vs) <= SAMPLE else vs[:SAMPLE]
        tested = "in full" if sample is vs else f"on a sample of {len(sample)}"
        badc = closure_test(sample, n, sp, wordlen, which="c")
        bads = closure_test(sample, n, sp, wordlen, which="s")
        sfail, cfail = stays_derivation(sample, n)
        print(f"        commutator closure ({tested}): "
              f"{'OK' if not badc else str(len(badc)) + ' FAILURES'}"
              f"   (but {cfail} of the commutators are not derivations at all)")
        print(f"        superbracket closure ({tested}): "
              f"{'OK' if not bads else str(len(bads)) + ' FAILURES'}"
              f"   (all stay derivations: {sfail == 0})")
        if bads:
            i, j, w = bads[0]
            B = sbracket(vs[i], vs[j], n)
            print(f"        first failure: D = {show_vf(vs[i], n)}")
            print(f"                       E = {show_vf(vs[j], n)}")
            print(f"                [D,E]_s = {show_vf(B, n)}")
            print(f"           leaves M at the word z{tuple(k+1 for k in w)}")
        # Iyer's VD^1 against the whole tangent space v(M)
        V = iyer_VD(n, xdeg)
        dV = dim_span(V, n)
        inside = all(in_span(D, vs, n) for D in V)
        print(f"        Iyer's V = {{a+[d,a]}}: dim {dV}, "
              f"contained in v(M): {inside}; "
              + ("V = v(M)" if dV == len(vs) else
                 f"V is PROPER in v(M) (codim {len(vs) - dV})"))
        # epimorphy: does the projection to g_ev fill g_ev?
        dev_g = len(der_basis(n, xdeg, parity=0))
        proj = [D.parts()[0] for D in vs]
        print(f"        p_0(v(M)) has dim {dim_span(proj, n)} against "
              f"dim g_ev = {dev_g} -> "
              + ("epimorphic" if dim_span(proj, n) == dev_g else "NOT epimorphic"))

        gh = ghosts(vs, n, wordlen)
        print(f"        ghosts (act as 0 on M): dim {len(gh)}")
        for g in gh[:3]:
            print(f"           {show_vf(g, n)}")
        # Is v(M) exactly Iyer's V plus the ghosts?
        dVG = dim_span(list(V) + list(gh), n)
        print(f"        dim(V + ghosts) = {dVG} against dim v(M) = {len(vs)} -> "
              + ("v(M) = V + ghosts" if dVG == len(vs) else "NOT all of v(M)")
              + f"; dim(V n ghosts) = {dV + len(gh) - dVG}")
        gm = ghost_model(n, xdeg)
        same = (dim_span(gm, n) == len(gh) == dim_span(list(gh) + gm, n))
        print(f"        ghost model theta*k[x]*(d/dy_j - d/dx_j), "
              f"theta = y_1...y_{n}: dim {dim_span(gm, n)} -> "
              + ("MATCHES" if same else "does NOT match"))
        print()


if __name__ == "__main__":
    main()
