#!/usr/bin/env python3
r"""Exact check of Tables (231) and (232) of the text and of
formula (196).

Set-up.  R(n) = functions on the supercircle: x (even, invertible) and
\xi_1..\xi_n (odd).  An automorphism \phi of R(n) is written

    \phi(x) = y(x) + X,               fil X >= 2  (X is even, so >= 2),
    \phi(\xi_i) = \sum_j a^j_i(x)\xi_j + s_i,   fil s_i >= 3,

and A = (a^j_i) as in the text, so that the column \xi is sent to A^t\xi in the
convention "row index = row of the matrix".  It induces an automorphism of

  * \fvect^L(n)          whenever A is invertible (functoriality);
  * \fsvect^L_\lambda(n) iff \phi preserves x^\lambda vvol(x,\xi) up to a
                         non-zero constant;
  * \fk^L(n)             iff \phi preserves \alpha = dx + \sum_i \xi_i d\xi_i
                         up to a factor;
  * \fk^M(n)             iff \phi preserves
                         \alpha = dx + \sum_{i<n}\xi_i d\xi_i + x\xi_n d\xi_n
                         up to a factor.  R^M(n) = R(n-1) + R(n-1)\xi_n\sqrt x,
                         so the Gram matrix of the odd part of \alpha is
                         G(x) = diag(1_{n-1}, x); see Lemma 10.3.3(2).

Write G(x) = 1_n for \fk^L and diag(1_{n-1},x) for \fk^M.  What the script
verifies, exactly (sympy over the Grassmann algebra, no floating point):

  (A) the equation \phi(\alpha) = h\alpha forces, at the lowest filtration,
          h = (phitilde x)'  and  A G(phitilde x) A^t = (phitilde x)' G(x),
      and forces in addition the \xi-degree 2 part of X to vanish;
  (B) that condition is sufficient: explicit A's satisfying it do preserve
      \alpha, for both contact series and all three phitilde(x);
  (C) formula (196) of the text satisfies it, for a general f -- which is what pins
      the conventions down;
  (D) for \fsvect^L_\lambda(n) the divergence, built from scratch, gives
          det A = const * (phitilde x)' * (phitilde x / x)^\lambda,
      so formula (61), which has det A = (f')^{-1}, wants (f')^{+1};
  (E) the eight cells of Table 13 as printed, against (A).
"""
import itertools
import sys

import sympy as sp

x = sp.Symbol('x', positive=True)
lam = sp.Symbol('lambda')

# ----------------------------------------------------------------- Grassmann
# element = dict  tuple(increasing indices) -> sympy expression


def gadd(*args):
    r = {}
    for a in args:
        for k, v in a.items():
            r[k] = r.get(k, 0) + v
    return {k: sp.simplify(v) for k, v in r.items() if sp.simplify(v) != 0}


def gscale(a, c):
    c = sp.sympify(c)
    return {} if c == 0 else \
        {k: sp.simplify(c * v) for k, v in a.items() if sp.simplify(c * v) != 0}


def gxi(i, c=1):
    return gscale({(i,): sp.Integer(1)}, c)


def gone(c=1):
    return gscale({(): sp.Integer(1)}, c)


def _merge(u, v):
    if set(u) & set(v):
        return None, 0
    arr, sign = list(u) + list(v), 1
    for i in range(len(arr)):
        for j in range(len(arr) - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                sign = -sign
    return tuple(arr), sign


def gmul(a, b):
    r = {}
    for u, cu in a.items():
        for v, cv in b.items():
            k, s = _merge(u, v)
            if k is None:
                continue
            r[k] = r.get(k, 0) + s * cu * cv
    return {k: sp.simplify(v) for k, v in r.items() if sp.simplify(v) != 0}


def gdx(a):
    return {k: sp.diff(v, x) for k, v in a.items() if sp.diff(v, x) != 0}


def gdxi(a, i):
    r"""left derivative \partial/\partial\xi_i."""
    r = {}
    for u, c in a.items():
        if i not in u:
            continue
        s = (-1) ** u.index(i)
        r[tuple(j for j in u if j != i)] = \
            r.get(tuple(j for j in u if j != i), 0) + s * c
    return {k: sp.simplify(v) for k, v in r.items() if sp.simplify(v) != 0}


def gdeg(a, d):
    return {k: v for k, v in a.items() if len(k) == d}


def gsub(a, ximg, xiimg, n):
    r"""substitute x -> ximg (a Grassmann element) and \xi_i -> xiimg[i]."""
    out = {}
    for u, c in a.items():
        # c is a function of x: expand c(ximg) = c(y + X) as a finite Taylor
        # series in the nilpotent part
        y0 = ximg.get((), sp.Integer(0))
        nil = {k: v for k, v in ximg.items() if k != ()}
        term = gone(c.subs(x, y0))
        if nil:
            pw, fac, dc = nil, 1, c
            for m in range(1, n // 2 + 2):
                dc = sp.diff(dc, x)
                fac *= m
                if dc == 0:
                    break
                term = gadd(term, gscale(pw, dc.subs(x, y0) / fac))
                pw = gmul(pw, nil)
                if not pw:
                    break
        for i in u:
            term = gmul(term, xiimg[i])
        out = gadd(out, term)
    return out


# ------------------------------------------------------------------- 1-forms
# form = (coefficient of dx, [coefficients of d\xi_i]), all Grassmann


def Gram(kind, n):
    return sp.diag(*([1] * (n - 1) + [1 if kind == 'L' else x]))


def alpha(kind, n):
    G = Gram(kind, n)
    return (gone(1), [gxi(i, G[i, i]) for i in range(n)])


def pull(form, ximg, xiimg, n):
    c0, cs = form
    # d(ximg) and d(xiimg[i])
    def dof(g):
        return gdx(g), [gdxi(g, k) for k in range(n)]
    ndx, ndxi = {}, [{} for _ in range(n)]
    a, b = dof(ximg)
    ndx = gadd(ndx, gmul(gsub(c0, ximg, xiimg, n), a))
    for k in range(n):
        ndxi[k] = gadd(ndxi[k], gmul(gsub(c0, ximg, xiimg, n), b[k]))
    for i in range(n):
        ci = gsub(cs[i], ximg, xiimg, n)
        if not ci:
            continue
        a, b = dof(xiimg[i])
        ndx = gadd(ndx, gmul(ci, a))
        for k in range(n):
            ndxi[k] = gadd(ndxi[k], gmul(ci, b[k]))
    return (ndx, ndxi)


def contact_defect(kind, n, ximg, xiimg):
    r"""\phi(\alpha) - h\alpha, where h is the degree-0 part of the dx
    coefficient of \phi(\alpha).  Returns (h, list of Grassmann defects)."""
    al = alpha(kind, n)
    pb = pull(al, ximg, xiimg, n)
    h = pb[0]                                   # the full factor
    defects = []
    for k in range(n):
        defects.append(gadd(pb[1][k], gscale(gmul(h, al[1][k]), -1)))
    return h.get((), sp.Integer(0)), defects


# --------------------------------------------------- (A) the derived condition


def derive(kind, n, y):
    r"""General A(x) and a general \xi-degree-2 part of X: what does the
    degree-1 part of the d\xi equations say?"""
    A = sp.Matrix(n, n, lambda r, c: sp.Function('a_%d%d' % (r + 1, c + 1))(x))
    # \phi(\xi_i) = sum_j a^j_i \xi_j, i.e. the i-th column of A
    xiimg = [gadd(*[gxi(j, A[j, i]) for j in range(n)]) for i in range(n)]
    Xs = {}
    Xg = {}
    for p, q in itertools.combinations(range(n), 2):
        Xs[(p, q)] = sp.Function('X_%d%d' % (p + 1, q + 1))(x)
        Xg[(p, q)] = Xs[(p, q)]
    X = gadd(*[gscale(gmul(gxi(p), gxi(q)), Xg[(p, q)])
               for p, q in itertools.combinations(range(n), 2)]) \
        if n > 1 else {}
    ximg = gadd(gone(y), X)
    h, defects = contact_defect(kind, n, ximg, xiimg)
    G = Gram(kind, n)
    yp = sp.diff(y, x)
    # claim: degree-1 defect vanishes  <=>  A G(y) A^t = y' G  and  X_2 = 0
    claim = sp.simplify(A * G.subs(x, y) * A.T - yp * G)
    eqs = []
    for k in range(n):
        d1 = gdeg(defects[k], 1)
        for key, v in d1.items():
            eqs.append(sp.expand(v))
    # substitute the claim: set A G(y) A^t = y' G and X_2 = 0 and see if the
    # degree-1 defect dies.  Do it the other way round: solve the linear system
    # {eqs} for the unknowns  (A G(y) A^t)_{jk}  and  X_{pq}.
    sym = {}
    M = A * G.subs(x, y) * A.T
    for j in range(n):
        for k in range(j, n):
            sym[(j, k)] = sp.Symbol('M_%d%d' % (j + 1, k + 1))
    subsl = []
    for j in range(n):
        for k in range(n):
            subsl.append((M[j, k], sym[(min(j, k), max(j, k))]))
    # verify by brute force instead: plug in a random A satisfying the claim
    return A, X, h, eqs, claim, defects


def derive_report(kind, n, y, yname):
    A, X, h, eqs, claim, defects = derive(kind, n, y)
    G = Gram(kind, n)
    yp = sp.diff(y, x)
    # the degree-1 defect, entry (k, j): coefficient of \xi_j in defect_k
    mat = sp.zeros(n, n)
    for k in range(n):
        for j in range(n):
            mat[k, j] = sp.expand(defects[k].get((j,), 0))
    # claimed: mat = (A G(y) A^t - y' G) + (antisymmetric in X)  -- check that
    # mat + y'G - A G(y) A^t  is antisymmetric and involves only the X_{pq}
    R = sp.simplify(mat - (A * G.subs(x, y) * A.T - yp * G))
    antisym = sp.simplify(R + R.T)
    onlyX = all(not any(sp.Function('a_%d%d' % (i + 1, j + 1))(x) in e.atoms(sp.Function)
                        for i in range(n) for j in range(n))
                for e in R)
    ok = all(sp.simplify(e) == 0 for e in antisym) and onlyX
    print("   %-6s n=%d  phitilde(x)=%-9s h=%-10s  "
          "deg-1 defect = (A G(y) A^t - y'G) + antisym(X): %s"
          % ('k^L' if kind == 'L' else 'k^M', n, yname, sp.simplify(h),
             "yes" if ok else "NO"))
    return ok


# ------------------------------------------------- (B), (C) sufficiency checks


def check_explicit(kind, n, y, A, label, maxdeg=None):
    r"""A is a sympy Matrix; \phi(\xi_i) = i-th column of A.  Check
    \phi(\alpha) = h\alpha exactly, with X = 0, s_i = 0."""
    xiimg = [gadd(*[gxi(j, A[j, i]) for j in range(n)]) for i in range(n)]
    h, defects = contact_defect(kind, n, gone(y), xiimg)
    G = Gram(kind, n)
    ok_form = all(not d for d in defects)
    resid = sp.simplify(A * G.subs(x, y) * A.T - sp.diff(y, x) * G)
    ok_unif = all(sp.simplify(e) == 0 for e in resid)
    print("     %-48s preserves alpha: %-4s  A G(y) A^t = y' G: %s"
          % (label, "yes" if ok_form else "NO", "yes" if ok_unif else "NO"))
    return ok_form and ok_unif


# --------------------------------------------------------- (D) \fsvect_\lambda


def parity(D, n):
    f, fs = D
    p = set()
    for k in f:
        p.add(len(k) % 2)
    for s in fs:
        for k in s:
            p.add((len(k) + 1) % 2)
    if len(p) > 1:
        return None
    return p.pop() if p else 0


def act(D, g, n):
    f, fs = D
    r = gmul(f, gdx(g))
    for i in range(n):
        r = gadd(r, gmul(fs[i], gdxi(g, i)))
    return r


def brk(D1, D2, n):
    p1, p2 = parity(D1, n), parity(D2, n)
    s = (-1) ** (p1 * p2)
    def comp(Da, Db, coord):
        return act(Da, Db_coord(Db, coord, n), n)
    def Db_coord(D, coord, n):
        return D[0] if coord == 'x' else D[1][coord]
    nf = gadd(act(D1, D2[0], n), gscale(act(D2, D1[0], n), -s))
    ns = [gadd(act(D1, D2[1][i], n), gscale(act(D2, D1[1][i], n), -s))
          for i in range(n)]
    return (nf, ns)


def div(D, n, lm, sign=-1):
    r"""divergence with respect to x^lm vvol(x,\xi).  The sign in front of the
    odd part is the Sign Rule's; which of the two conventions makes the
    divergence-free fields a subalgebra is checked in div_closed()."""
    f, fs = D
    p = parity(D, n)
    s = sign if p == 0 else -sign
    r = gadd(gdx(f), gscale(f, lm / x))
    for i in range(n):
        r = gadd(r, gscale(gdxi(fs[i], i), s))
    return r


def div_closed(n, lm, sign=-1):
    basis = []
    mons = [()] + [(i,) for i in range(n)] + \
           list(itertools.combinations(range(n), 2))
    for m in mons:
        for p in (-1, 0, 1):
            basis.append(({m: x ** p}, [{} for _ in range(n)]))
            for i in range(n):
                s = [{} for _ in range(n)]
                s[i] = {m: x ** p}
                basis.append(({}, s))
    basis = [D for D in basis if parity(D, n) is not None]
    free = [D for D in basis if not div(D, n, lm, sign)]
    bad = 0
    for D1, D2 in itertools.combinations(free, 2):
        if div(brk(D1, D2, n), n, lm, sign):
            bad += 1
    return len(free), bad


def svect_condition(n, y, lm):
    r"""\phi: x->y, \xi_i->\xi_i (i<n), \xi_n -> c\xi_n, with y an involution.
    Which c makes \phi carry divergence-free fields to divergence-free ones?"""
    c = sp.Function('c')(x)
    xiimg = [gxi(i) for i in range(n - 1)] + [gxi(n - 1, c)]
    xiinv = [gxi(i) for i in range(n - 1)] + \
            [gxi(n - 1, 1 / c.subs(x, y))]

    def phi(g):
        return gsub(g, gone(y), xiimg, n)

    conds = []
    for p in (-1, 0, 1, 2):
        # divergence-free: x^p d/dx + (p+lm) x^{p-1} \xi_1 d/d\xi_1
        f = {(): x ** p}
        s = [{} for _ in range(n)]
        s[0] = gxi(0, sp.diff(x ** p, x) + lm * x ** (p - 1))
        D = (f, s)
        assert not div(D, n, lm), D
        # \phi_*D = sum_a \phi(D(\phi^{-1}(z_a))) \partial_{z_a}
        Fx = phi(act(D, gone(y), n))
        Fxi = [phi(act(D, xiinv[i], n)) for i in range(n)]
        dv = div((Fx, Fxi), n, lm)
        for k, v in dv.items():
            conds.append(sp.simplify(v))
    conds = [e for e in conds if sp.simplify(e) != 0]
    return c, conds


# ------------------------------------------------------------------- the table


PRINTED = {
    # (kind, column) -> what Table 13 prints, as a matrix in x
    ('L', 'x'):        lambda n: sp.eye(n),
    ('L', '1/x'):      lambda n: -x ** -2 * sp.eye(n),
    ('L', '-1/x'):     lambda n: x ** 2 * sp.eye(n),
    ('M', 'x'):        lambda n: sp.diag(*([1] * (n - 1) + [x])),
    ('M', '1/x'):      lambda n: sp.diag(*([-x ** -2] * (n - 1) + [-x])),
    ('M', '-1/x'):     lambda n: sp.diag(*([x ** -2] * (n - 1) + [-x])),
}
PRINTED_LHS = {'x': x, '1/x': x, '-1/x': x}      # the G inside A(...)A^t


def main():
    ok = True
    YS = [(x, 'x', 'x'), (1 / x, 'x^{-1}', '1/x'), (-1 / x, '-x^{-1}', '-1/x')]

    print("=" * 78)
    print("(A)  What phi(alpha) = h alpha says at the lowest filtration")
    print("=" * 78)
    for kind in ('L', 'M'):
        for n in (2, 3):
            for y, yn, key in YS:
                ok &= derive_report(kind, n, y, yn)

    print("\n" + "=" * 78)
    print("(B)  Sufficiency, on explicit A (a constant rotation times the")
    print("     scalar, conjugated by T(t)=diag(1_{n-1},sqrt t) for k^M)")
    print("=" * 78)
    t = sp.Symbol('t')
    for kind in ('L', 'M'):
        for n in (2, 3):
            for y, yn, key in YS:
                yp = sp.diff(y, x)
                O = sp.eye(n)
                O[0, 0] = sp.cos(t); O[0, n - 1] = -sp.sin(t)
                O[n - 1, 0] = sp.sin(t); O[n - 1, n - 1] = sp.cos(t)
                if kind == 'L':
                    A = sp.sqrt(yp) * O
                else:
                    T = sp.diag(*([1] * (n - 1) + [sp.sqrt(x)]))
                    Ty = sp.diag(*([1] * (n - 1) + [sp.sqrt(y)]))
                    A = sp.sqrt(yp) * T * O * Ty.inv()
                A = sp.simplify(A)
                ok &= check_explicit(kind, n, y, A,
                                     "%s(%d), phitilde(x)=%s"
                                     % ('k^L' if kind == 'L' else 'k^M', n, yn))

    print("\n" + "=" * 78)
    print("(C)  Formula (196), general f = phitilde(x)")
    print("=" * 78)
    f = sp.Function('f')(x)
    fp = sp.diff(f, x)
    for n in (2, 3):
        ok &= check_explicit('L', n, f, sp.sqrt(fp) * sp.eye(n),
                             "k^L(%d):  A = sqrt(f') 1_n" % n)
        A = sp.diag(*([sp.sqrt(fp)] * (n - 1) + [sp.sqrt(x * fp / f)]))
        ok &= check_explicit('M', n, f, A,
                             "k^M(%d):  A = diag(sqrt(f'),...,sqrt(xf'/f))" % n)

    print("\n" + "=" * 78)
    print("(D)  fsvect^L_lambda(n): the divergence, from scratch")
    print("=" * 78)
    for n in (2, 3):
        for lm in (sp.Integer(0), sp.Rational(1, 2), lam):
            for sign, sname in ((-1, "-"), (+1, "+")):
                cnt, bad = div_closed(n, lm, sign)
                print("     n=%d  lambda=%-8s odd sign %s  divergence-free "
                      "spanning fields %3d,  brackets leaving the algebra %d"
                      % (n, lm, sname, cnt, bad))
                if sign == -1:
                    ok &= (bad == 0)
    print()
    for n in (2, 3):
        for y, yn, key in YS:
            for lm in (sp.Integer(0), lam):
                c, conds = svect_condition(n, y, lm)
                want = sp.diff(y, x) * (y / x) ** lm
                sol = sp.solve([sp.Eq(e, 0) for e in conds], c, dict=True) \
                    if conds else []
                # check directly that c = const * want kills every condition
                K = sp.Symbol('K', nonzero=True)
                sub = [sp.simplify(e.subs(c, K * want).doit()) for e in conds]
                sub = [sp.simplify(sp.expand(s)) for s in sub]
                good = all(s == 0 for s in sub)
                print("     n=%d  phitilde(x)=%-9s lambda=%-8s conditions %2d;"
                      "  c = const*%s works: %s"
                      % (n, yn, lm, len(conds), sp.simplify(want),
                         "yes" if good else "NO"))
                ok &= good
    print()
    print("     formula (61) for fsvect^L(n) has det A = (f')^{-1};")
    print("     the divergence wants det A = const * f' * (f/x)^lambda,")
    print("     i.e. const * f' at lambda = 0.  Ratio of the two readings:")
    print("       (f')^{-1} / f' = (f')^{-2}, non-constant unless f' = const.")

    print("\n" + "=" * 78)
    print("(E)  Table 13 as printed, against (A)")
    print("=" * 78)
    n = sp.Symbol('n')
    for kind in ('L', 'M'):
        for y, yn, key in YS:
            yp = sp.diff(y, x)
            for nn in (3,):
                G = Gram(kind, nn)
                correct_lhs = G.subs(x, y)
                correct_rhs = sp.simplify(yp * G)
                printed_lhs = Gram(kind, nn)           # the table always has G(x)
                printed_rhs = sp.simplify(PRINTED[(kind, key)](nn))
                same = (sp.simplify(correct_lhs - printed_lhs).is_zero_matrix and
                        sp.simplify(correct_rhs - printed_rhs).is_zero_matrix)
                print("   %-6s phitilde(x)=%-9s printed: A(%s)A^t = %s"
                      % ('k^L' if kind == 'L' else 'k^M', yn,
                         sp.simplify(printed_lhs.diagonal().T).T if kind == 'M'
                         else '1_n',
                         list(printed_rhs.diagonal())))
                print("          %-31s correct: A(%s)A^t = %s   %s"
                      % ("", list(correct_lhs.diagonal()) if kind == 'M' else '1_n',
                         list(correct_rhs.diagonal()),
                         "AGREES" if same else "<-- DIFFERS"))

    print("\n" + "=" * 78)
    print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
