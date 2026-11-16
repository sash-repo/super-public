"""
Checker for the relations printed in the old case list of Subsec. 11.2 of the paper (Statement 11.2.1e).

For each relation R claimed on a stratum S it answers three questions:

  (a) is R in the radical on S?          (i.e. is the claim true at all)
  (b) is R in the radical generically?   (i.e. is it really NEW on S)
  (c) does R span the whole new kernel?  (i.e. is the claim complete)

and, when the answer to (c) is no, prints the correct generator.
"""

from fractions import Fraction
from collections import defaultdict
import itertools

import sympy as sp

from shapovalov import free_lie_superalgebra, phi, perms_of
from driver import (a, b, c, sbracket, to_sym, ideal_comp, matrix_of,
                    gram_sym, Stratum, clear_denoms)

X1, X2, X3 = 0, 1, 2


ODDPAR = [1, 1, 1]


def T(e, par=None):
    """Evaluate a nested-tuple bracket expression into a word-vector."""
    par = ODDPAR if par is None else par
    if isinstance(e, int):
        return {(e,): sp.Integer(1)}
    return sbracket(T(e[0], par), T(e[1], par), par)


def scale(vec, s):
    return {w: sp.expand(s * v) for w, v in vec.items()}


def add(*vs):
    out = defaultdict(lambda: sp.Integer(0))
    for v in vs:
        for w, cf in v.items():
            out[w] += cf
    return {w: sp.expand(x) for w, x in out.items() if sp.expand(x) != 0}


def mdeg(vec, n=3):
    w = next(iter(vec))
    d = [0] * n
    for i in w:
        d[i] += 1
    return tuple(d)


def in_radical(vec, A, n, alpha, par=None):
    """Is vec killed by the contravariant form (i.e. in r_+)?"""
    par = ODDPAR if par is None else par
    ph = phi(vec, A, n, alpha, par)
    return all(sp.simplify(v) == 0 for v in ph.values())


def spec(A_sym, subs):
    return [[sp.expand(x.subs(subs)) for x in row] for row in A_sym]


def proportional(u, v):
    """Are two word-vectors proportional over the function field?"""
    ws = set(u) | set(v)
    ratios = set()
    for w in ws:
        uu, vv = sp.simplify(u.get(w, 0)), sp.simplify(v.get(w, 0))
        if uu == 0 and vv == 0:
            continue
        if uu == 0 or vv == 0:
            return None
        ratios.add(sp.simplify(sp.cancel(uu / vv)))
    if len(ratios) == 1:
        return ratios.pop()
    return None


# --------------------------------------------- the paper's relations -------

# (rel3b)  b[x3,[x1,x2]] - a[x2,[x1,x3]]
REL3B = add(scale(T((X3, (X1, X2))), b), scale(T((X2, (X1, X3))), -a))

# (rel3d)  2[[x1,x3],[x3,[x1,x2]]] - [[x1,x3],[x2,[x1,x3]]]
REL3D = add(scale(T(((X1, X3), (X3, (X1, X2)))), sp.Integer(2)),
            scale(T(((X1, X3), (X2, (X1, X3)))), sp.Integer(-1)))

# (rel3e)  2a[[x1,x2],[x2,[x1,x3]]] - b[[x1,x2],[x3,[x1,x2]]]
REL3E = add(scale(T(((X1, X2), (X2, (X1, X3)))), 2 * a),
            scale(T(((X1, X2), (X3, (X1, X2)))), -b))

# (rel3f)  b[[x2,x3],[x3,[x1,x2]]] - a[[x2,x3],[x2,[x1,x3]]]
REL3F = add(scale(T(((X2, X3), (X3, (X1, X2)))), b),
            scale(T(((X2, X3), (X2, (X1, X3)))), -a))

CLAIMS = [
    ('(rel3b)', REL3B, 'c = -a-b',      {c: -a - b}),
    ('(rel3d)', REL3D, 'c = -a-2b',     {c: -a - 2 * b}),
    ('(rel3e)', REL3E, 'c = -2a-b',     {c: -2 * a - b}),
    ('(rel3f)', REL3F, 'c = -(a+b)/2',  {c: -(a + b) / 2}),
    # the two lines that the old case list gives as cases but that are NOT walls
    ('(rel3d) @ (-1,1,-1) on c=-b',  REL3D, 'c = -b, a = -1',
     {c: -b, a: sp.Integer(-1), b: sp.Integer(1)}),
    ('(rel3b) @ (-2,1,1) on c=-a/2', REL3B, 'c = -a/2, (a,b,c)=(-2,1,1)',
     {a: sp.Integer(-2), b: sp.Integer(1), c: sp.Integer(1)}),
    ('(rel3e) @ (-2/3,1,1/3)',       REL3E, 'c = -a/2, (a,b,c)=(-2/3,1,1/3)',
     {a: sp.Rational(-2, 3), b: sp.Integer(1), c: sp.Rational(1, 3)}),
    # the alleged clash
    ('(rel3d) @ (1,1,-3)', REL3D, '(a,b,c) = (1,1,-3)',
     {a: sp.Integer(1), b: sp.Integer(1), c: sp.Integer(-3)}),
    ('(rel3e) @ (1,1,-3)', REL3E, '(a,b,c) = (1,1,-3)',
     {a: sp.Integer(1), b: sp.Integer(1), c: sp.Integer(-3)}),
]


def main():
    A_sym = [[sp.Integer(0), a, b], [a, sp.Integer(0), c], [b, c, sp.Integer(0)]]
    Agen = A_sym
    L = free_lie_superalgebra(3, 5)

    print('=' * 78)
    print('CHECK OF THE RELATIONS OF THE OLD CASE LIST (Subsec. 11.2)')
    print('=' * 78)
    for name, rel, cond, subs in CLAIMS:
        alpha = mdeg(rel)
        r = {w: sp.expand(v.subs(subs)) for w, v in rel.items()}
        r = {w: v for w, v in r.items() if v != 0}
        A = spec(A_sym, subs)
        ok = in_radical(r, A, 3, alpha)
        gen = in_radical(rel, Agen, 3, alpha)
        print('\n%-34s degree %s   on  %s' % (name, alpha, cond))
        print('   holds on the stratum : %s' % ('YES' if ok else '*** NO ***'))
        print('   holds generically    : %s  (so it is %s)'
              % ('yes' if gen else 'no',
                 'vacuous' if gen else 'genuinely new'))
        if ok:
            # is it the whole new kernel?
            st = Stratum(subs, cond, [s for s in (a, b, c) if s not in subs])
            ker = new_kernel(A_sym, st, L, alpha)
            if ker is None:
                print('   spans the new kernel : (no new kernel found here)')
            else:
                dim, vecs, lower = ker
                if dim == 1:
                    # r spans the new kernel iff r lies in I^alpha + <v>, I = the ideal
                    # generated by the relations of lower degree (r itself is not in I,
                    # since it does not hold generically)
                    ok_span = spans_mod_ideal(r, vecs[0], lower, alpha, subs)
                    print('   spans the new kernel : %s'
                          % ('YES (modulo the relations of lower degree)' if ok_span
                             else '*** NO ***'))
                    if not ok_span:
                        print('   correct generator    : %s' % vecs[0])
                else:
                    print('   new kernel dimension : %d  (relation gives only 1)' % dim)


def spans_mod_ideal(r, v, lower, alpha, subs):
    ivecs = [{w: sp.expand(x.subs(subs)) for w, x in u.items()}
             for u in ideal_comp(lower, alpha, 3, ODDPAR)]
    words = sorted({w for u in ivecs + [r, v] for w in u}, key=lambda w: (len(w), w))
    base = matrix_of(ivecs + [v], words).applyfunc(sp.simplify)
    full = matrix_of(ivecs + [v, r], words).applyfunc(sp.simplify)
    return base.rank() == full.rank()


def new_kernel(A_sym, stratum, L, alpha):
    """Kernel of the Gram form on L^alpha / (ideal of lower relations)."""
    from driver import analyse
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyse(A_sym, 3, sum(alpha), stratum, L, maxdepth=0,
                log=lambda *x: None, par=ODDPAR)
    hits = [v for al, v in stratum.rels if al == alpha]
    lower = [(al, v) for al, v in stratum.rels if al != alpha]
    if not hits:
        return None
    return len(hits), hits, lower


if __name__ == '__main__':
    main()
