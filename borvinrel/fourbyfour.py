"""
Statement 10.1.10 ("On 4x4 matrices"): are there OTHER relations?

The statement concerns the Cartan matrix (anything on the main diagonal)

        /  *    1    a   -ab \
        |  1    *   ab   -a  |
   A =  |  a   ab    *   -1  |
        \ -ab  -a   -1    *  /

and asserts one relation among [[x1,x2],[x3,x4]], [[x1,x3],[x2,x4]],
[[x1,x4],[x2,x3]] -- i.e. one relation in the MULTILINEAR degree
alpha = (1,1,1,1).

Part 1 (this file) settles the multilinear degree completely and exactly:

  * the multilinear component of the free Lie superalgebra on x_1..x_4 has
    dimension (n-1)! = 6 whatever the parities (Remark 10.1.14 of the paper),
    with basis the left-normed brackets [[[x_1,x_s2],x_s3],x_s4];
  * the diagonal entries of A CANNOT enter the Gram matrix there: the lowering
    operator D_i only ever multiplies by A_{i,j} with j a letter of the word
    other than the deleted one, and in a multilinear word every letter occurs
    once.  So "anything instead of each *" is a theorem, not a hope, and the
    same computation covers all 16 parity vectors at once;
  * therefore rank/kernel of the 6x6 Gram matrix over Q(a,b) is the whole
    truth about relations in that degree, for every diagonal and every parity.

"""

import sys
import itertools
from fractions import Fraction
from collections import defaultdict

import sympy as sp

from shapovalov import bracket, phi, pw

a, b = sp.symbols('a b')
d1, d2, d3, d4 = sp.symbols('d1 d2 d3 d4')

N = 4
A_SYM = [[d1, sp.Integer(1), a, -a * b],
         [sp.Integer(1), d2, a * b, -a],
         [a, a * b, d3, sp.Integer(-1)],
         [-a * b, -a, sp.Integer(-1), d4]]


def check_symmetric():
    for i in range(N):
        for j in range(N):
            assert sp.simplify(A_SYM[i][j] - A_SYM[j][i]) == 0, (i, j)


# ---------------------------------------------------------------- basis ----

def multilinear_basis(par):
    """Left-normed [[[x_0,x_s1],x_s2],x_s3], s a permutation of (1,2,3)."""
    out = []
    for s in itertools.permutations((1, 2, 3)):
        v = {(0,): Fraction(1)}
        for i in s:
            v = bracket(v, {(i,): Fraction(1)}, par)
        out.append((s, v))
    return out


def gram_of(basis, A, par, alpha=(1, 1, 1, 1)):
    d = len(basis)
    phis = [phi(u, A, N, alpha, par) for u in basis]
    G = sp.zeros(d, d)
    for i in range(d):
        for j in range(d):
            s = sp.Integer(0)
            for w, cf in basis[j].items():
                pv = phis[i].get(w)
                if pv is not None and pv != 0:
                    s += sp.Rational(cf.numerator, cf.denominator) * pv
            G[i, j] = sp.expand(s)
    return G


# ------------------------------------------------- the claimed relation ----

def claimed_relation(par):
    """ab[[x1,x2],[x3,x4]] - (-1)^{p2 p3} b [[x1,x3],[x2,x4]]
                           - (-1)^{(p2+p3)p4} [[x1,x4],[x2,x3]]  =  0."""
    def br(i, j):
        return bracket({(i,): Fraction(1)}, {(j,): Fraction(1)}, par)

    t1 = bracket(br(0, 1), br(2, 3), par)
    t2 = bracket(br(0, 2), br(1, 3), par)
    t3 = bracket(br(0, 3), br(1, 2), par)
    s2 = -1 if (par[1] * par[2]) % 2 else 1
    s3 = -1 if ((par[1] + par[2]) * par[3]) % 2 else 1
    out = defaultdict(lambda: sp.Integer(0))
    for w, c in t1.items():
        out[w] += a * b * sp.Rational(c.numerator, c.denominator)
    for w, c in t2.items():
        out[w] += -s2 * b * sp.Rational(c.numerator, c.denominator)
    for w, c in t3.items():
        out[w] += -s3 * sp.Rational(c.numerator, c.denominator)
    return {w: sp.expand(x) for w, x in out.items() if sp.expand(x) != 0}


def variant_with_factor_a(par):
    """The variant with a factor a in front of the third summand (Statement 10.1.11, item 4)."""
    rel = claimed_relation(par)          # rebuild with a in front of the 3rd
    def br(i, j):
        return bracket({(i,): Fraction(1)}, {(j,): Fraction(1)}, par)
    t1 = bracket(br(0, 1), br(2, 3), par)
    t2 = bracket(br(0, 2), br(1, 3), par)
    t3 = bracket(br(0, 3), br(1, 2), par)
    s2 = -1 if (par[1] * par[2]) % 2 else 1
    s3 = -1 if ((par[1] + par[2]) * par[3]) % 2 else 1
    out = defaultdict(lambda: sp.Integer(0))
    for w, c in t1.items():
        out[w] += a * b * sp.Rational(c.numerator, c.denominator)
    for w, c in t2.items():
        out[w] += -s2 * b * sp.Rational(c.numerator, c.denominator)
    for w, c in t3.items():
        out[w] += -s3 * a * sp.Rational(c.numerator, c.denominator)
    return {w: sp.expand(x) for w, x in out.items() if sp.expand(x) != 0}


def in_span(vec, kernel_vecs, words):
    """Is `vec` in the span of kernel_vecs (over Q(a,b))?"""
    if not kernel_vecs:
        return not vec
    M = sp.Matrix([[sp.simplify(v.get(w, sp.Integer(0))) for w in words]
                   for v in kernel_vecs])
    r0 = M.rank()
    M2 = M.col_join(sp.Matrix([[sp.simplify(vec.get(w, sp.Integer(0)))
                                for w in words]]))
    return M2.rank() == r0


def kernel_as_vectors(G, basis):
    out = []
    for v in G.nullspace():
        vec = defaultdict(lambda: sp.Integer(0))
        for i, coeff in enumerate(v):
            if coeff == 0:
                continue
            for w, cf in basis[i].items():
                vec[w] += coeff * sp.Rational(cf.numerator, cf.denominator)
        vec = {w: sp.simplify(sp.together(x)) for w, x in vec.items()
               if sp.simplify(x) != 0}
        out.append(vec)
    return out


# ----------------------------------------------------------------- main ----

def main():
    log = open('log-4x4-multilinear.txt', 'w')

    def P(s=''):
        print(s)
        log.write(str(s) + '\n')
        log.flush()

    check_symmetric()
    P('=== Statement 10.1.10, multilinear degree alpha=(1,1,1,1) ===')
    P('A = %s' % sp.Matrix(A_SYM).tolist())
    P('diagonal kept SYMBOLIC (d1..d4); all 16 parity vectors run.')
    P('')

    summary = []
    for par in itertools.product((0, 1), repeat=4):
        par = list(par)
        bas = multilinear_basis(par)
        vecs = [v for _, v in bas]
        assert len(vecs) == 6
        words = sorted({w for v in vecs for w in v})
        M = sp.Matrix([[sp.Integer(v.get(w, 0)) for w in words] for v in vecs])
        assert M.rank() == 6, (par, M.rank())     # (n-1)! = 6, any parity

        G = gram_of(vecs, A_SYM, par)
        used_diag = G.free_symbols & {d1, d2, d3, d4}
        det = sp.factor(sp.expand(G.det()))
        rk = G.rank()
        ker = kernel_as_vectors(G, vecs)
        allwords = sorted({w for v in vecs for w in v}
                          | {w for v in ker for w in v})
        rel = claimed_relation(par)
        relvar = variant_with_factor_a(par)
        ok = in_span(rel, ker, allwords)
        okvar = in_span(relvar, ker, allwords)
        summary.append((tuple(par), rk, len(ker), det, bool(used_diag),
                        ok, okvar))
        P('par=%s  rank=%d  dim ker=%d  diag used: %s' %
          (par, rk, len(ker), sorted(used_diag, key=str) or 'NONE'))
        P('    det = %s' % det)
        P('    stated relation (10.1.10) lies in the radical : %s' % ok)
        P('    variant with a factor "a" on the 3rd summand: %s' % okvar)
        P('')

    P('=== summary over the 16 parity vectors ===')
    P('%-14s %5s %6s %9s %9s  %s' %
      ('parity', 'rank', 'dimker', 'stated', 'a-variant', 'det'))
    for par, rk, nk, det, ud, ok, okv in summary:
        P('%-14s %5d %6d %9s %9s  %s' % (par, rk, nk, ok, okv, det))

    diagfree = all(not ud for _, _, _, _, ud, _, _ in summary)
    P('')
    P('diagonal entries never enter any Gram matrix: %s' % diagfree)
    P('  => "with anything instead of each *" is exact, not generic.')
    ranks = {rk for _, rk, _, _, _, _, _ in summary}
    kers = {nk for _, _, nk, _, _, _, _ in summary}
    P('ranks over all parities: %s ; kernel dims: %s' % (sorted(ranks), sorted(kers)))
    P('stated relation holds for every parity: %s'
      % all(ok for *_, ok, _ in summary))
    P('a-variant holds for every parity      : %s'
      % all(okv for *_, okv in summary))
    log.close()


if __name__ == '__main__':
    main()
