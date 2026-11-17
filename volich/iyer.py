#!/usr/bin/env python3
"""Exact check of U. Iyer, "Volichenko algebras as algebras of differential
operators", J. Nonlinear Math. Phys. 13:1 (2006) 34-49 -- the worked example of
her Subsec. 4.2.3 (Subsec. 3.6 of the text).

Setting: S = k[x_1..x_n] (x) Lambda[y_1..y_n], the supercommutative envelope of
the metabelian algebra k[Z] = k[z_1..z_n], z_i = x_i + y_i, sitting
diagonally in S.  The natural odd derivation is d = sum_k y_k d/dx_k.

Elements of S are dicts {(alpha, ymask): Fraction}, alpha a tuple of x-degrees,
ymask a frozenset of the y-indices present (with a sign convention: the monomial
is y_{i_1} ... y_{i_r} with i_1 < ... < i_r).

Checked here:
  (0) d is an odd superderivation of S with d^2 = 0 (d homologic);
  (1) the five bracket formulas of Subsec. 4.2.3:
        [d, x_i] = y_i,  [d, dx_i] = 0,  [d, y_i y_j] = 0,
        [d, dy_i dy_j] = dx_i dy_j - dy_i dx_j,  [d, y_i dy_j] = - y_i dx_j;
  (2) the four listed generators of VD(k[Z]) preserve k[Z], as Prop. 4.2.1(1)
      demands;
  (3) the "ghost operator" y_i d/dy_j - y_i d/dx_j is nonzero on S but acts as 0
      on k[Z];
  (4) the operator psi_1 = x d/dy + y d/dx of her Remark 4.2.2 keeps k[Z]
      invariant while being purely odd (so it is not in VD^1).

Brackets are superbrackets: [A,B] = AB - (-1)^{p(A)p(B)} BA.
"""

from fractions import Fraction
from itertools import combinations


# ---------------------------------------------------------------- elements of S

def mono(alpha, ys, c=1):
    """One monomial: x^alpha * y_{ys[0]} ... , ys a sorted tuple."""
    return {(tuple(alpha), tuple(ys)): Fraction(c)}


def add(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + v
        if out[k] == 0:
            del out[k]
    return out


def scal(c, a):
    if c == 0:
        return {}
    return {k: Fraction(c) * v for k, v in a.items()}


def sub(a, b):
    return add(a, scal(-1, b))


def is_zero(a):
    return all(v == 0 for v in a.values())


def y_sign(ys, j):
    """Sign of inserting y_j into the sorted product y_{ys}; None if j is already
    there (then the product is 0, since y_j^2 = 0)."""
    if j in ys:
        return None, None
    pos = sum(1 for i in ys if i < j)
    new = tuple(sorted(ys + (j,)))
    return (-1) ** pos, new


# ------------------------------------------------------------------- operators
# An operator is a function S -> S, together with its parity.

class Op:
    def __init__(self, fn, parity):
        self.fn = fn
        self.parity = parity

    def __call__(self, a):
        out = {}
        for (alpha, ys), c in a.items():
            out = add(out, scal(c, self.fn(alpha, ys)))
        return out


def mult_x(i):
    def fn(alpha, ys):
        na = list(alpha)
        na[i] += 1
        return mono(na, ys)
    return Op(fn, 0)


def d_dx(i):
    def fn(alpha, ys):
        if alpha[i] == 0:
            return {}
        na = list(alpha)
        e = na[i]
        na[i] -= 1
        return mono(na, ys, e)
    return Op(fn, 0)


def mult_y(j):
    """Left multiplication by y_j: y_j * (x^a y_ys)."""
    def fn(alpha, ys):
        s, new = y_sign(ys, j)
        if s is None:
            return {}
        return mono(alpha, new, s)
    return Op(fn, 1)


def d_dy(j):
    """Left derivative d/dy_j: removes y_j, with the sign of moving it to front."""
    def fn(alpha, ys):
        if j not in ys:
            return {}
        pos = ys.index(j)
        new = tuple(x for x in ys if x != j)
        return mono(alpha, new, (-1) ** pos)
    return Op(fn, 1)


def compose(A, B):
    return Op(lambda alpha, ys: A(B.fn(alpha, ys)), (A.parity + B.parity) % 2)


def bracket(A, B):
    sign = (-1) ** (A.parity * B.parity)
    ab = compose(A, B)
    ba = compose(B, A)
    return Op(lambda alpha, ys: sub(ab.fn(alpha, ys), scal(sign, ba.fn(alpha, ys))),
              (A.parity + B.parity) % 2)


def op_sum(*ops):
    """Sum of operators.  The parity is None when the summands disagree -- and
    that is not an error here: x_i + y_i is exactly the kind of inhomogeneous
    operator Volichenko algebras are made of.  An inhomogeneous operator may be
    applied but not bracketed."""
    par = {o.parity for o in ops}
    parity = ops[0].parity if len(par) == 1 else None

    def fn(alpha, ys):
        out = {}
        for o in ops:
            out = add(out, o.fn(alpha, ys))
        return out
    return Op(fn, parity)


def op_scal(c, A):
    return Op(lambda alpha, ys: scal(c, A.fn(alpha, ys)), A.parity)


# ------------------------------------------------------------------ test space

def basis(n, deg):
    """Monomials of S with total x-degree <= deg."""
    out = []
    for total in range(deg + 1):
        for alpha in compositions(n, total):
            for r in range(n + 1):
                for ys in combinations(range(n), r):
                    out.append((alpha, ys))
    return out


def compositions(n, total):
    if n == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(n - 1, total - first):
            yield (first,) + rest


def op_eq(A, B, n, deg):
    """Do A and B agree on every monomial of x-degree <= deg?"""
    for alpha, ys in basis(n, deg):
        if not is_zero(sub(A.fn(alpha, ys), B.fn(alpha, ys))):
            return False, (alpha, ys)
    return True, None


def op_is_zero(A, n, deg):
    return op_eq(A, Op(lambda a, y: {}, A.parity), n, deg)


# --------------------------------------------------- the metabelian algebra k[Z]

def z_power(exps):
    """The element z_1^{e_1} ... z_n^{e_n} of k[Z] <= S, z_i = x_i + y_i."""
    n = len(exps)
    cur = mono([0] * n, ())
    for i, e in enumerate(exps):
        zi = add(mono([1 if k == i else 0 for k in range(n)], ()),
                 mono([0] * n, (i,)))
        for _ in range(e):
            cur = mul_S(cur, zi)
    return cur


def sort_word(word):
    """Sort a word y_{i_1} ... y_{i_r} into increasing order by bubbling, each
    transposition costing -1.  Returns (sign, sorted tuple), or (0, ()) if a
    letter repeats (y_i^2 = 0)."""
    w = list(word)
    if len(set(w)) != len(w):
        return 0, ()
    sign = 1
    for i in range(len(w)):
        for j in range(len(w) - 1 - i):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                sign = -sign
    return sign, tuple(w)


def mul_S(a, b):
    """Product in S (supercommutative): concatenate the y-words and sort."""
    out = {}
    for (al, ys), ca in a.items():
        for (be, zs), cb in b.items():
            s, new = sort_word(tuple(ys) + tuple(zs))
            if s == 0:
                continue
            gamma = tuple(x + y for x, y in zip(al, be))
            out = add(out, mono(gamma, new, s * ca * cb))
    return out


def z_word(word):
    """The element z_{w_1} z_{w_2} ... of k[Z] <= S, in the given order."""
    n = max(word) + 1 if word else 1
    return z_word_n(word, n)


def z_word_n(word, n):
    cur = mono([0] * n, ())
    for i in word:
        zi = add(mono([1 if k == i else 0 for k in range(n)], ()),
                 mono([0] * n, (i,)))
        cur = mul_S(cur, zi)
    return cur


def kZ_basis(n, deg):
    """k[Z] is the subalgebra of S generated by the z_i.  It is METABELIAN, not
    commutative: [z_i, z_j] = 2 y_i y_j /= 0.  So the ordered monomials
    z_1^{e_1}...z_n^{e_n} do NOT span it -- one must take all words."""
    out = []
    for L in range(deg + 1):
        out.extend(z_word_n(w, n) for w in words(n, L))
    return out


def words(n, L):
    if L == 0:
        yield ()
        return
    for w in words(n, L - 1):
        for i in range(n):
            yield w + (i,)


class Span:
    """Row-echelon form of the span of k[Z] inside S, built once and reused.
    Membership is then one reduction instead of one rank computation per test --
    the difference between minutes and milliseconds here."""

    def __init__(self, n, deg):
        self.rows = {}                       # pivot key -> normalized row (dict)
        for v in kZ_basis(n, deg):
            self.insert(v)

    def reduce(self, v):
        v = dict(v)
        while True:
            v = {k: c for k, c in v.items() if c != 0}
            if not v:
                return v
            p = max(v)                       # pivot = largest key present
            if p not in self.rows:
                return v
            r = self.rows[p]
            f = v[p] / r[p]
            v = sub(v, scal(f, r))

    def insert(self, v):
        r = self.reduce(v)
        if r:
            self.rows[max(r)] = r

    def contains(self, v):
        return not self.reduce(v)


_SPAN_CACHE = {}


def in_kZ(elt, n, deg):
    """Is elt in k[Z], truncated at word length <= deg?  Exact.
    Give deg enough headroom above the degree of elt: a degree-raising operator
    otherwise 'fails' merely because the test basis was truncated too low."""
    key = (n, deg)
    if key not in _SPAN_CACHE:
        _SPAN_CACHE[key] = Span(n, deg)
    return _SPAN_CACHE[key].contains(elt)


# ------------------------------------------------------------------------ main

def run(n=3, deg=4):
    print(f"S = k[x_1..x_{n}] (x) Lambda[y_1..y_{n}], testing on x-degree <= {deg}")
    x = [mult_x(i) for i in range(n)]
    dx = [d_dx(i) for i in range(n)]
    y = [mult_y(i) for i in range(n)]
    dy = [d_dy(i) for i in range(n)]

    d = op_sum(*[compose(y[k], dx[k]) for k in range(n)])
    print("d = sum_k y_k d/dx_k, parity", d.parity)

    fails = []

    def check(name, ok, witness=None):
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}"
              + ("" if ok else f"   at {witness}"))
        if not ok:
            fails.append(name)

    # (0) d^2 = 0, i.e. [d,d] = 2d^2 = 0
    ok, w = op_is_zero(compose(d, d), n, deg)
    check("d^2 = 0  (d is homologic)", ok, w)

    # d is a superderivation of S: d(ab) = d(a)b + (-1)^{p(a)} a d(b)
    ok = True
    bas = basis(n, deg // 2)
    for m1 in bas:
        for m2 in bas:
            a, b = mono(*m1), mono(*m2)
            lhs = d(mul_S(a, b))
            pa = len(m1[1]) % 2
            rhs = add(mul_S(d(a), b), scal((-1) ** pa, mul_S(a, d(b))))
            if not is_zero(sub(lhs, rhs)):
                ok = False
                break
        if not ok:
            break
    check("d is an odd superderivation of S", ok)

    # (1) the five formulas of Subsec. 4.2.3
    for i in range(n):
        ok, w = op_eq(bracket(d, x[i]), y[i], n, deg)
        check(f"[d, x_{i+1}] = y_{i+1}", ok, w)
    for i in range(n):
        ok, w = op_is_zero(bracket(d, dx[i]), n, deg)
        check(f"[d, d/dx_{i+1}] = 0", ok, w)
    for i, j in combinations(range(n), 2):
        ok, w = op_is_zero(bracket(d, compose(y[i], y[j])), n, deg)
        check(f"[d, y_{i+1} y_{j+1}] = 0", ok, w)
    for i, j in combinations(range(n), 2):
        lhs = bracket(d, compose(dy[i], dy[j]))
        rhs = op_sum(compose(dx[i], dy[j]), op_scal(-1, compose(dy[i], dx[j])))
        ok, w = op_eq(lhs, rhs, n, deg)
        check(f"[d, dy_{i+1} dy_{j+1}] = dx_{i+1} dy_{j+1} - dy_{i+1} dx_{j+1}", ok, w)
    for i in range(n):
        for j in range(n):
            lhs = bracket(d, compose(y[i], dy[j]))
            rhs = op_scal(-1, compose(y[i], dx[j]))
            ok, w = op_eq(lhs, rhs, n, deg)
            check(f"[d, y_{i+1} d/dy_{j+1}] = - y_{i+1} d/dx_{j+1}", ok, w)

    # (2) the four listed generators of VD(k[Z]) preserve k[Z]
    gens = []
    for i in range(n):
        gens.append((f"x_{i+1} + y_{i+1}", op_sum(x[i], y[i])))
        gens.append((f"d/dx_{i+1}", dx[i]))
    for i, j in combinations(range(n), 2):
        gens.append((f"dy_{i+1}dy_{j+1} + dx_{i+1}dy_{j+1} - dy_{i+1}dx_{j+1}",
                     op_sum(compose(dy[i], dy[j]), compose(dx[i], dy[j]),
                            op_scal(-1, compose(dy[i], dx[j])))))
    for i in range(n):
        for j in range(n):
            gens.append((f"ghost y_{i+1}d/dy_{j+1} - y_{i+1}d/dx_{j+1}",
                         op_sum(compose(y[i], dy[j]),
                                op_scal(-1, compose(y[i], dx[j])))))
    # Test on words in the z's of length <= deg (NOT just the ordered monomials
    # z^e: k[Z] is metabelian, so the words in other orders are extra elements),
    # checking membership against a basis with two degrees of headroom, so that
    # x_i + y_i is not failed spuriously.
    zb = [(w, z_word_n(w, n)) for L in range(deg + 1) for w in words(n, L)]
    head = deg + 2
    for name, A in gens:
        bad = [e for e, zm in zb if not in_kZ(A(zm), n, head)]
        check(f"{name} maps k[Z] into k[Z]", not bad,
              None if not bad else f"z-word {bad[0]}")

    # (3) the ghost operator is nonzero on S but kills k[Z]
    for i in range(n):
        for j in range(n):
            g = op_sum(compose(y[i], dy[j]), op_scal(-1, compose(y[i], dx[j])))
            nonzero = not op_is_zero(g, n, deg)[0]
            bad = [e for e, zm in zb if not is_zero(g(zm))]
            check(f"ghost_{i+1}{j+1}: nonzero on S", nonzero)
            check(f"ghost_{i+1}{j+1}: acts as 0 on k[Z]", not bad,
                  None if not bad else f"z-word {bad[0]} |-> {g(z_word_n(bad[0], n))}")

    # (4) Remark 4.2.2: psi_1 = x d/dy + y d/dx keeps k[Z] invariant, is odd.
    # Her setting is n = 1; run it there as she states it.
    psi = op_sum(compose(x[0], dy[0]), compose(y[0], dx[0]))
    check("psi_1 = x d/dy + y d/dx is odd", psi.parity == 1)
    bad = [e for e, zm in zb if not in_kZ(psi(zm), n, head)]
    check("psi_1 keeps k[Z] invariant", not bad,
          None if not bad else f"z-word {bad[0]} |-> {psi(z_word_n(bad[0], n))}")

    print()
    if fails:
        print(f"{len(fails)} FAILURES: {fails}")
    else:
        print("all checks passed")
    return fails


if __name__ == "__main__":
    run(n=3, deg=4)
