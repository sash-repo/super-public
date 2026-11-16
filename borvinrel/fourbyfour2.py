"""
Statement 10.1.10, multilinear degree: WHAT the radical is, in readable form.

fourbyfour.py established: over Q(a,b), for every one of the 16 parity
vectors and for every main diagonal, the 6-dimensional multilinear component
carries a contravariant form of rank 4, so the radical is 2-dimensional,
while Statement 10.1.10 states ONE relation.  Here we identify the second one
and check where the rank drops further.
"""

import itertools
from fractions import Fraction
from collections import defaultdict

import sympy as sp

from shapovalov import bracket, phi
from fourbyfour import (A_SYM, N, a, b, multilinear_basis, gram_of,
                        claimed_relation, kernel_as_vectors)

PAR = [1, 1, 1, 1]          # all odd: the setting of Section 11 ("Relations")


def br(i, j, par):
    return bracket({(i,): Fraction(1)}, {(j,): Fraction(1)}, par)


def dbl(i, j, k, l, par):
    return bracket(br(i, j, par), br(k, l, par), par)


def to_sym(v):
    return {w: sp.Rational(c.numerator, c.denominator) for w, c in v.items()}


def mat(vecs, words):
    return sp.Matrix([[sp.expand(v.get(w, sp.Integer(0))) for w in words]
                      for v in vecs])


def coords_in(target, spanvecs, words, names):
    """Write target as a combination of spanvecs; None if impossible."""
    M = mat(spanvecs, words).T
    y = sp.Matrix([[sp.expand(target.get(w, sp.Integer(0)))] for w in words])
    sol, params = M.gauss_jordan_solve(M.copy(), y) if False else (None, None)
    try:
        s = sp.linsolve((M, y))
    except Exception:
        return None
    s = list(s)
    if not s:
        return None
    return s[0]


def main():
    out = open('log-4x4-radical.txt', 'w')

    def P(s=''):
        print(s)
        out.write(str(s) + '\n')
        out.flush()

    for PAR in ([1, 1, 1, 1], [0, 0, 0, 0]):
        P('=' * 72)
        P('parity vector %s' % PAR)
        P('=' * 72)
        bas = [v for _, v in multilinear_basis(PAR)]
        labels = ['[[[x1,x%d],x%d],x%d]' % s
                  for s, _ in [(tuple(x + 1 for x in s), None)
                               for s in itertools.permutations((1, 2, 3))]]
        G = gram_of(bas, A_SYM, PAR)
        P('Gram matrix on the 6 left-normed basis elements:')
        for i in range(6):
            P('   ' + '  '.join(str(sp.factor(G[i, j])) for j in range(6)))
        P('rank = %d, so dim radical = %d' % (G.rank(), 6 - G.rank()))
        P('')

        ker = kernel_as_vectors(G, bas)
        words = sorted({w for v in bas for w in v}
                       | {w for v in ker for w in v})

        # the three double brackets of the Statement
        D12_34 = to_sym(dbl(0, 1, 2, 3, PAR))
        D13_24 = to_sym(dbl(0, 2, 1, 3, PAR))
        D14_23 = to_sym(dbl(0, 3, 1, 2, PAR))
        trio = [D12_34, D13_24, D14_23]
        Mtrio = mat(trio, words)
        P('span of the three double brackets [[x1,x2],[x3,x4]], [[x1,x3],[x2,x4]],')
        P('   [[x1,x4],[x2,x3]] has dimension %d (Jacobi already relates them)'
          % Mtrio.rank())

        # intersection of that span with the radical
        Mker = mat(ker, words)
        both = Mtrio.col_join(Mker)
        inter = Mtrio.rank() + Mker.rank() - both.rank()
        P('dim( span(trio) cap radical ) = %d' % inter)
        P('')

        rel = claimed_relation(PAR)
        P('Statement 10.1.10 relation is in the radical: %s'
          % (mat(ker, words).rank()
             == mat(ker + [rel], words).rank()))
        P('')

        # Is the radical spanned by trio-combinations alone?
        if inter < Mker.rank():
            P('*** the radical is NOT contained in span(trio): there is a second,')
            P('    independent relation involving brackets outside the three.')
        P('')

        # exhibit the radical in the left-normed basis, cleanly
        P('radical, in the left-normed basis  u_s = [[[x1,x_s1],x_s2],x_s3]:')
        perms = list(itertools.permutations((1, 2, 3)))
        Bm = mat(bas, words).T                      # words x 6
        for v in ker:
            y = sp.Matrix([[sp.expand(v.get(w, sp.Integer(0)))] for w in words])
            sol = sp.linsolve((Bm, y))
            sol = list(sol)[0]
            sol = [sp.simplify(sp.together(x)) for x in sol]
            den = sp.lcm([sp.denom(x) for x in sol])
            sol = [sp.factor(sp.expand(x * den)) for x in sol]
            g = sp.gcd(sol)
            if g != 0 and not g.free_symbols:
                sol = [sp.simplify(x / g) for x in sol]
            terms = []
            for co, s in zip(sol, perms):
                if co == 0:
                    continue
                terms.append('(%s)*u_%s' % (co, ''.join(str(t + 1) for t in s)))
            P('   ' + '  +  '.join(terms) + '  = 0')
        P('')

        # a nicer generating set: express the radical against the trio + one more
        P('the same radical against the natural spanning set')
        P('   T1=[[x1,x2],[x3,x4]], T2=[[x1,x3],[x2,x4]], T3=[[x1,x4],[x2,x3]],')
        P('   S1=[[[x1,x2],x3],x4], S2=[[[x1,x3],x2],x4], S3=[[[x1,x4],x2],x3]:')
        S1 = to_sym(bracket(bracket(br(0, 1, PAR), {(2,): Fraction(1)}, PAR),
                            {(3,): Fraction(1)}, PAR))
        S2 = to_sym(bracket(bracket(br(0, 2, PAR), {(1,): Fraction(1)}, PAR),
                            {(3,): Fraction(1)}, PAR))
        S3 = to_sym(bracket(bracket(br(0, 3, PAR), {(1,): Fraction(1)}, PAR),
                            {(2,): Fraction(1)}, PAR))
        gens = [D12_34, D13_24, D14_23, S1, S2, S3]
        gnames = ['T1', 'T2', 'T3', 'S1', 'S2', 'S3']
        Gm = mat(gens, words).T
        P('   rank of that spanning set = %d' % mat(gens, words).rank())
        for v in ker:
            y = sp.Matrix([[sp.expand(v.get(w, sp.Integer(0)))] for w in words])
            try:
                sol = list(sp.linsolve((Gm, y)))
            except Exception:
                sol = []
            if not sol:
                P('   (not expressible)')
                continue
            sol = [sp.simplify(x) for x in sol[0]]
            free = sorted({s for x in sol for s in x.free_symbols
                           if str(s).startswith('tau')}, key=str)
            sol0 = [sp.simplify(x.subs({s: 0 for s in free})) for x in sol]
            den = sp.lcm([sp.denom(sp.together(x)) for x in sol0])
            sol0 = [sp.factor(sp.expand(x * den)) for x in sol0]
            terms = ['(%s)*%s' % (c, nm) for c, nm in zip(sol0, gnames) if c != 0]
            P('   ' + '  +  '.join(terms) + '  = 0')
        P('')

        # --- walls: where does the rank drop below 4? --------------------
        P('walls: values of (a,b) where the rank drops below the generic 4')
        minors = []
        idx = list(itertools.combinations(range(6), 4))
        for rs in idx:
            for cs in idx:
                m = G[list(rs), list(cs)].det()
                if m != 0:
                    minors.append(sp.factor(sp.expand(m)))
        g = sp.factor(sp.gcd(minors)) if minors else sp.Integer(0)
        P('   number of nonzero 4x4 minors: %d' % len(minors))
        P('   gcd of all 4x4 minors = %s' % g)
        allfacs = set()
        for m in minors:
            for f, _ in sp.factor_list(m)[1]:
                if f.free_symbols:
                    allfacs.add(sp.factor(f))
        P('   irreducible factors occurring in some minor: %s'
          % sorted(map(str, allfacs)))
        # true walls = common zeros of ALL 4x4 minors
        P('   candidate walls (zeros of the gcd): %s'
          % ([sp.factor(f) for f, _ in sp.factor_list(g)[1] if f.free_symbols]
             if g != 0 else 'gcd is 0'))
        # test the candidates
        cands = set()
        for f in allfacs:
            cands.add(f)
        for f in sorted(cands, key=str):
            sols = sp.solve(sp.Eq(f, 0), b, dict=True) or \
                   sp.solve(sp.Eq(f, 0), a, dict=True)
            for s in sols:
                Gs = G.subs(s).applyfunc(lambda x: sp.simplify(x))
                P('      %-14s -> %-18s rank = %d' %
                  (f, s, Gs.rank()))
        P('')

    out.close()


if __name__ == '__main__':
    main()
