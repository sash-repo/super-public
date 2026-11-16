"""
The radical in degree (1,1,1,1) of the family of Statement 10.1.10, in the
basis that makes it readable.

The six elements

   T1 = [[x1,x2],[x3,x4]]      S1 = [[[x1,x2],x3],x4]
   T2 = [[x1,x3],[x2,x4]]      S2 = [[[x1,x3],x2],x4]
   T3 = [[x1,x4],[x2,x3]]      S3 = [[[x1,x4],x2],x3]

form a BASIS of the multilinear component (rank 6, checked).  So every
linear relation among them in g(A) is a genuine relation, and the radical is
a 2-dimensional subspace of that 6-space; we print its reduced row echelon
form.  Statement 10.1.10 is the unique radical vector with S1=S2=S3 = 0.
"""

import itertools
from fractions import Fraction

import sympy as sp

from shapovalov import bracket
from fourbyfour import (A_SYM, N, a, b, multilinear_basis, gram_of,
                        claimed_relation, kernel_as_vectors)

d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')


def br(i, j, par):
    return bracket({(i,): Fraction(1)}, {(j,): Fraction(1)}, par)


def sym(v):
    return {w: sp.Rational(c.numerator, c.denominator) for w, c in v.items()}


def TS(par):
    T1 = sym(bracket(br(0, 1, par), br(2, 3, par), par))
    T2 = sym(bracket(br(0, 2, par), br(1, 3, par), par))
    T3 = sym(bracket(br(0, 3, par), br(1, 2, par), par))
    S1 = sym(bracket(bracket(br(0, 1, par), {(2,): Fraction(1)}, par),
                     {(3,): Fraction(1)}, par))
    S2 = sym(bracket(bracket(br(0, 2, par), {(1,): Fraction(1)}, par),
                     {(3,): Fraction(1)}, par))
    S3 = sym(bracket(bracket(br(0, 3, par), {(1,): Fraction(1)}, par),
                     {(2,): Fraction(1)}, par))
    return [T1, T2, T3, S1, S2, S3], ['T1', 'T2', 'T3', 'S1', 'S2', 'S3']


def main():
    out = open('log-4x4-second-relation.txt', 'w')

    def P(s=''):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    for par in ([1, 1, 1, 1], [0, 0, 0, 0], [1, 0, 1, 0]):
        P('=' * 70)
        P('parity %s' % par)
        P('=' * 70)
        gens, names = TS(par)
        words = sorted({w for v in gens for w in v})
        Bm = sp.Matrix([[v.get(w, sp.Integer(0)) for w in words] for v in gens])
        P('rank of {T1,T2,T3,S1,S2,S3} = %d  (a basis of the 6-space)'
          % Bm.rank())

        # Gram of the contravariant form in THIS basis
        G = gram_of(gens, A_SYM, par)
        P('rank of the contravariant form = %d -> dim radical = %d'
          % (G.rank(), 6 - G.rank()))
        P('the main diagonal d1..d4 occurs in the Gram matrix: %s'
          % bool(G.free_symbols & {d1, d2, d3, d4}))

        ns = G.nullspace()          # coefficient vectors in the T,S basis
        M = sp.Matrix([[sp.simplify(x) for x in v.T] for v in ns])
        R = M.rref()[0]
        R = R.applyfunc(lambda x: sp.cancel(sp.together(x)))
        P('')
        P('radical, reduced row echelon form in the basis (%s):'
          % ', '.join(names))
        for i in range(R.rows):
            row = [sp.factor(sp.cancel(R[i, j])) for j in range(6)]
            den = sp.lcm([sp.denom(sp.together(x)) for x in row])
            row = [sp.factor(sp.expand(x * den)) for x in row]
            g = sp.gcd(row)
            if g != 0 and not g.free_symbols:
                row = [sp.cancel(x / g) for x in row]
            terms = ['(%s)*%s' % (c, nm) for c, nm in zip(row, names) if c != 0]
            P('   ' + '  +  '.join(terms) + '  = 0')

        # the unique radical vector with zero S-part = Statement 10.1.10
        P('')
        P('the unique radical vector whose S1,S2,S3 coefficients vanish:')
        # solve  sum_i l_i * ns_i  has zero coords 3,4,5
        ls = sp.symbols('l0:%d' % len(ns))
        comb = sum((l * v for l, v in zip(ls, ns)), sp.zeros(6, 1))
        sol = sp.solve([sp.numer(sp.together(comb[3])),
                        sp.numer(sp.together(comb[4])),
                        sp.numer(sp.together(comb[5]))], ls, dict=True)
        if sol:
            c2 = comb.subs(sol[0])
            free = sorted(c2.free_symbols & set(ls), key=str)
            if free:
                c2 = c2.subs({free[0]: 1})
                for f in free[1:]:
                    c2 = c2.subs({f: 0})
            c2 = [sp.cancel(sp.together(x)) for x in c2]
            den = sp.lcm([sp.denom(x) for x in c2])
            c2 = [sp.factor(sp.expand(x * den)) for x in c2]
            g = sp.gcd(c2)
            if g != 0 and not g.free_symbols:
                c2 = [sp.cancel(x / g) for x in c2]
            terms = ['(%s)*%s' % (c, nm) for c, nm in zip(c2, names) if c != 0]
            P('   ' + '  +  '.join(terms) + '  = 0')
            s2 = -1 if (par[1] * par[2]) % 2 else 1
            s3 = -1 if ((par[1] + par[2]) * par[3]) % 2 else 1
            P('   Statement 10.1.10 predicts (ab, -(-1)^{p2p3} b, -(-1)^{(p2+p3)p4})')
            P('                          = (%s, %s, %s)' % (a * b, -s2 * b, -s3))
            ratio = sp.simplify(c2[0] / (a * b)) if c2[0] != 0 else None
            P('   ratio of the computed vector to the predicted one: %s, %s, %s'
              % (ratio,
                 sp.simplify(c2[1] / (-s2 * b)) if c2[1] != 0 else None,
                 sp.simplify(c2[2] / (-s3)) if c2[2] != 0 else None))
        P('')

    out.close()


if __name__ == '__main__':
    main()
