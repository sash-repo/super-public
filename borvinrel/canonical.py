"""
Express the new defining relation of each stratum in the *bracket monomials
the paper itself uses*, by solving for the coefficients directly.

For a chosen list of bracket monomials m_1..m_r of degree alpha, we look for
lambda_1..lambda_r over Q(a,b) with  sum lambda_i m_i  in the radical.  The
solution space is computed from Phi (the iterated-lowering map), so it is the
honest answer, not a guess; and it is compared with the relations displayed in Subsec. 11.2.
"""

import sympy as sp

from shapovalov import free_lie_superalgebra, phi, perms_of
from driver import a, b, c, sbracket, ideal_comp, matrix_of
from check_paper import T, add, scale, mdeg, spec, in_radical

X1, X2, X3 = 0, 1, 2


def solve_relation(monomials, subs, A_sym, par=None, verbose=True):
    """Find all combos of `monomials` lying in the radical after `subs`."""
    par = [1, 1, 1] if par is None else par
    A = spec(A_sym, subs)
    vecs = [{w: sp.expand(v.subs(subs)) for w, v in T(m, par).items()}
            for m in monomials]
    vecs = [{w: v for w, v in u.items() if v != 0} for u in vecs]
    alpha = mdeg(vecs[0])
    words = perms_of(alpha, 3)
    rows = []
    for v in vecs:
        ph = phi(v, A, 3, alpha, par)
        rows.append([sp.simplify(ph.get(w, sp.Integer(0))) for w in words])
    M = sp.Matrix(rows).T          # columns = monomials
    ns = M.nullspace()
    return alpha, ns


def show(name, monomials, subs, A_sym, printed=None, par=None):
    alpha, ns = solve_relation(monomials, subs, A_sym, par)
    print('\n%s   degree %s   %s' % (name, alpha,
                                     ', '.join('%s=%s' % (k, v) for k, v in subs.items())))
    for m in monomials:
        print('      m: %s' % tree_str(m))
    if not ns:
        print('   -> NO combination of these monomials lies in the radical')
        return None
    for v in ns:
        den = sp.lcm([sp.denom(sp.together(x)) for x in v])
        v = [sp.simplify(sp.expand(x * den)) for x in v]
        g = sp.gcd([sp.nsimplify(x) for x in v])
        if g and not g.free_symbols and g != 0:
            v = [sp.simplify(x / g) for x in v]
        terms = ' + '.join('(%s)*%s' % (sp.factor(v[i]), tree_str(monomials[i]))
                           for i in range(len(v)) if v[i] != 0)
        print('   -> %s  =  0' % terms)
    if printed is not None:
        print('   paper prints: %s' % printed)
    return ns


def tree_str(e):
    if isinstance(e, int):
        return 'x_%d' % (e + 1)
    return '[%s,%s]' % (tree_str(e[0]), tree_str(e[1]))


def main():
    A_sym = [[sp.Integer(0), a, b], [a, sp.Integer(0), c], [b, c, sp.Integer(0)]]
    print('=' * 78)
    print('THE NEW RELATION OF EACH STRATUM, IN THE PAPER\'S OWN MONOMIALS')
    print('=' * 78)

    # ---- (1,1,1), wall a+b+c = 0 ------------------------------------ rel3b
    show('stratum  a+b+c = 0        [paper: c = -a-b, rel (3b)]',
         [(X3, (X1, X2)), (X2, (X1, X3))],
         {c: -a - b}, A_sym,
         printed='b[x_3,[x_1,x_2]] - a[x_2,[x_1,x_3]] = 0')

    # ---- (2,1,2), wall a+2b+c = 0 ----------------------------------- rel3d
    show('stratum  a+2b+c = 0       [paper: c = -a-2b, rel (3d)]',
         [((X1, X3), (X3, (X1, X2))), ((X1, X3), (X2, (X1, X3)))],
         {c: -a - 2 * b}, A_sym,
         printed='2[[x_1,x_3],[x_3,[x_1,x_2]]] - [[x_1,x_3],[x_2,[x_1,x_3]]] = 0')

    # ---- (2,2,1), wall 2a+b+c = 0 ----------------------------------- rel3e
    show('stratum  2a+b+c = 0       [paper: c = -2a-b, rel (3e)]',
         [((X1, X2), (X2, (X1, X3))), ((X1, X2), (X3, (X1, X2)))],
         {c: -2 * a - b}, A_sym,
         printed='2a[[x_1,x_2],[x_2,[x_1,x_3]]] - b[[x_1,x_2],[x_3,[x_1,x_2]]] = 0')

    # ---- (1,2,2), wall a+b+2c = 0 ----------------------------------- rel3f
    show('stratum  a+b+2c = 0       [paper: c = -(a+b)/2, rel (3f)]',
         [((X2, X3), (X3, (X1, X2))), ((X2, X3), (X2, (X1, X3)))],
         {c: -(a + b) / 2}, A_sym,
         printed='b[[x_2,x_3],[x_3,[x_1,x_2]]] - a[[x_2,x_3],[x_2,[x_1,x_3]]] = 0')

    print('\n' + '=' * 78)
    print('THE ALLEGED CLASH AT (a,b,c) = (1,1,-3)')
    print('=' * 78)
    pt = {a: sp.Integer(1), b: sp.Integer(1), c: sp.Integer(-3)}
    for nm, mons in [('(3d)-shape, degree (2,1,2)',
                      [((X1, X3), (X3, (X1, X2))), ((X1, X3), (X2, (X1, X3)))]),
                     ('(3e)-shape, degree (2,2,1)',
                      [((X1, X2), (X2, (X1, X3))), ((X1, X2), (X3, (X1, X2)))])]:
        show(nm, mons, pt, A_sym)
    print("""
(1,1,-3) satisfies BOTH  a+2b+c = 0  and  2a+b+c = 0 : it is the intersection
point of the two walls, so BOTH relations hold there simultaneously.  The
old case list says each 'should be replaced by' the other; the correct word is
'together with'.""")


if __name__ == '__main__':
    main()
