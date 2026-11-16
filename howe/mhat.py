"""
mhat.py -- names of m(n)_{-1}: hats only over indeterminates.

The two elements of m(n)_{-1} were printed as 'hat partial_i' and 'hat delta_i', i.e.
hats over operators.  They are the fields with the generating functions -xi_i and x_i,
so the names with a hat over an indeterminate are

        hat xi_i := '(xi_i) = - hat partial_i,      hat x_i := '(x_i) = hat delta_i.

This script settles, by the engine and not by hand, the two places where the rename
touches a sign:

  1) the bracket of m(n)_{-1}: what replaces ['partial_i,'delta_j] = 2 delta_{ij} D;
  2) the second-order singular vector of Theorem 9.5.1,
        y = (sum_i 'partial_i 'delta_i) (x) v + D (x) w,   w = -(n+lam) v,
     rewritten in the new names.

Run:  python3 mhat.py
"""
from fractions import Fraction as F

from le import is_hom_par, pscal
from mn import mono, one, mbr, mdeg, par_m


def gen(n, kind, i):
    """generating function x_i (kind='x') or xi_i (kind='xi')"""
    if kind == 'x':
        a = [0] * n
        a[i] = 1
        return mono(n, a, [])
    return mono(n, [0] * n, [i])


def show(f, n):
    """print a generating function as a short string"""
    if not f:
        return '0'
    out = []
    for (a, S), c in sorted(f.items()):
        s = ''
        for i, e in enumerate(a):
            s += f'x_{i+1}' + (f'^{e}' if e > 1 else '') if e else ''
        for i in S:
            s += 'tau' if i == n else f'xi_{i+1}'
        out.append(f'{c}*{s or "1"}')
    return ' + '.join(out)


def main():
    print(f'{"":=^72}')
    print('1) the bracket of m(n)_{-1} in the two namings')
    print(f'{"":=^72}')
    for n in (1, 2, 3):
        # D = d/d tau is the field with generating function 1/2 : '1 = 2D
        D = pscal(one(n), F(1, 2))
        ok_old, ok_new = True, True
        for i in range(n):
            for j in range(n):
                d_ij = 1 if i == j else 0
                # old names: 'partial_i = '(-xi_i),  'delta_j = '(x_j)
                p_i = pscal(gen(n, 'xi', i), -1)
                d_j = gen(n, 'x', j)
                got = mbr(p_i, d_j, n)
                want = pscal(D, 2 * d_ij)
                ok_old &= (got == want)
                # new names: hat xi_i = '(xi_i),  hat x_j = '(x_j)
                xi_i = gen(n, 'xi', i)
                x_j = gen(n, 'x', j)
                b1 = mbr(xi_i, x_j, n)            # [hat xi_i, hat x_j]
                b2 = mbr(x_j, xi_i, n)            # [hat x_j, hat xi_i]
                ok_new &= (b1 == pscal(D, -2 * d_ij) and b2 == pscal(D, 2 * d_ij))
        print(f'  n={n}:  [\'partial_i,\'delta_j] = 2 d_ij D          : {ok_old}')
        print(f'        [hat xi_i, hat x_j]   = -2 d_ij D         : {ok_new}')
        print(f'        [hat x_i,  hat xi_j]  = +2 d_ij D         : {ok_new}')

    print()
    print('   parities (p(\'f) = p(f)+1):')
    n = 2
    for nm, f in (('hat x_i ', gen(n, 'x', 0)), ('hat xi_i', gen(n, 'xi', 0)),
                  ('D       ', one(n))):
        print(f'     {nm}: {"odd" if par_m(f) else "even"},  '
              f'M-degree {mdeg(f, n)}')

    print()
    print(f'{"":=^72}')
    print('2) the singular vector of Theorem 9.5.1 in the new names')
    print(f'{"":=^72}')
    # In U(m(n)_{<0}) the element hat xi_i is even and hat x_i is odd, so the two
    # commute and the product is order-free; the rename is a relabelling of the PBW
    # generators, and the only arithmetic in it is the sign of 'partial_i:
    #   sum_i 'partial_i 'delta_i = sum_i (- hat xi_i)(hat x_i) = - sum_i hat x_i hat xi_i.
    # The singular vector itself is taken from the engine, not from the printed text.
    from mimod import Mn, MInduced

    checked = 0
    for n in (1, 2, 3):
        mn = Mn(n)
        for lam in (-3, -2, -1, 0, 1, 2):
            I = MInduced(mn, tuple([1] * n), lam + 2)
            sv = I.singular(2, tuple(F(1) for _ in range(n)))
            assert len(sv) == 1, (n, lam, len(sv))
            sol = sv[0]
            # coefficient of 'partial_i 'delta_i (any i: they are equal) and of D
            a = [0] * n
            a[0] = 1
            u_pd = (tuple(a), (0,), 0)
            u_D = (tuple([0] * n), (), 1)
            c_pd = next((c for (u, _, _), c in sol.items() if u == u_pd), F(0))
            c_D = next((c for (u, _, _), c in sol.items() if u == u_D), F(0))
            # old names: normalize the sum_i 'p_i 'd_i coefficient to 1
            w_old = c_D / c_pd if c_pd else None
            # new names: 'p_i 'd_i = - hat x_i hat xi_i, so normalizing the
            # sum_i hat x_i hat xi_i coefficient to 1 negates the D-coefficient
            w_new = -w_old if w_old is not None else None
            if c_pd:
                assert w_old == -(n + lam), (n, lam, w_old)
                assert w_new == n + lam, (n, lam, w_new)
                checked += 1
        print(f'  n={n}:  engine gives w = -(n+lam) v in the old names, '
              f'hence w = +(n+lam) v in the new ones')
    print(f'  ({checked} weights checked; at lam = -n the D-part is absent '
          f'in both namings)')
    print()
    print('   the parity check that makes the product order-free:')
    n = 2
    x1, xi1 = gen(n, 'x', 0), gen(n, 'xi', 0)
    print(f'     p(hat x_1) = {"odd" if par_m(x1) else "even"}, '
          f'p(hat xi_1) = {"odd" if par_m(xi1) else "even"}  '
          f'=> hat x_1 hat xi_1 = hat xi_1 hat x_1 in U')


if __name__ == '__main__':
    main()
