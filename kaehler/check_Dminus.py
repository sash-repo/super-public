r"""The algebra of Theorem 1.9.4a.  With cD^- := [cX^-, cD^+] as the definition,
print cD^- in closed form, compare it with eq. (Dpm2) of the text, and recompute
the square root relations, [cD^+, cD^-] = -hbar cH, the superdimension (8|8) of
the closure of {cD^+, cD^-, cX^+, cX^-}, and the abelian ideal hbar*g of sdim (5|3).

Run (from this folder):  ../venv/bin/python check_Dminus.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from fractions import Fraction
from flatmodel import Model, Echelon, closure, d_operator

half = Fraction(1, 2)


def span(M, elts):
    ech = {0: Echelon(), 1: Echelon()}
    for e in elts:
        if e:
            ech[M.parity(e)].add(e)
    return ech


def contains(M, ech, e):
    return not ech[M.parity(e)].reduce(e)[0]


def pretty(M, e, n):
    """print an element of the super-Weyl algebra in the paper's notation"""
    def nm(g):
        kind, a = g
        if kind == 'h':
            return 'hbar'
        base = ('q_%d' % (a + 1)) if a < n else ('pi_%d' % (a - n + 1))
        return {'c': 'D%s', 'b': 'd_D%s', 'x': '%s', 't': 'd_%s'}[kind] % base
    out = []
    for m, c in sorted(e.items(), key=lambda kv: (len(kv[0]), kv[0])):
        out.append('%+s*%s' % (c, ' '.join(nm(g) for g in m) or '1'))
    return '  '.join(out)


for n in (1, 2, 3):
    P = Model([0] * n + [1] * n)
    q = list(range(n))
    pi = list(range(n, 2 * n))
    c, b, x, t = (lambda a: P.gen('c', a)), (lambda a: P.gen('b', a)), \
                 (lambda a: P.gen('x', a)), (lambda a: P.gen('t', a))
    hb = P.gen('h', 0)
    d = d_operator(P)

    Xp = P.add(*[P.mul(c(q[i]), c(pi[i])) for i in range(n)])
    Xm = P.add(*[P.mul(b(pi[i]), b(q[i])) for i in range(n)])
    Hc = P.br(Xp, Xm)

    Dp = P.add(d, P.smul(-half, P.mul(hb, P.add(
        *[P.add(P.mul(c(q[i]), x(pi[i])),
                P.smul(-1, P.mul(c(pi[i]), x(q[i])))) for i in range(n)]))))

    # the operator as printed in eq. (Dpm2)
    Dm_old = P.add(
        *[P.add(P.mul(b(pi[i]), t(q[i])),
                P.smul(-1, P.mul(b(q[i]), t(pi[i])))) for i in range(n)])
    Dm_old = P.add(Dm_old, P.smul(-half, P.mul(hb, P.add(
        *[P.add(P.mul(x(pi[i]), b(pi[i])), P.mul(x(q[i]), b(q[i])))
          for i in range(n)]))))

    # the definition of cD^-
    Dm = P.br(Xm, Dp)

    print('=== C^{%d|%d} ===' % (n, n))
    print('  [X^-,D^+]      =', pretty(P, Dm, n))
    print('  eq.(Dpm2) D^-  =', pretty(P, Dm_old, n))
    print('  parity of [X^-,D^+] :', P.parity(Dm))

    # the closed form we intend to print in the paper:
    #   D^- = sum(d_Dpi d_q + d_Dq d_pi) - (hbar/2) sum(q d_Dq - pi d_Dpi)
    guess = P.add(
        *[P.add(P.mul(b(pi[i]), t(q[i])), P.mul(b(q[i]), t(pi[i])))
          for i in range(n)])
    guess = P.add(guess, P.smul(-half, P.mul(hb, P.add(
        *[P.add(P.mul(x(q[i]), b(q[i])), P.smul(-1, P.mul(x(pi[i]), b(pi[i]))))
          for i in range(n)]))))
    print('  closed form matches :', not P.add(Dm, P.smul(-1, guess)))

    # the hbar = 0 part of the new D^- is delta = [X^-, d] (it was not, before)
    delta = P.br(Xm, d)
    Dm0 = {m: cc for m, cc in Dm.items() if not (m and m[0][0] == 'h')}
    Dm_old0 = {m: cc for m, cc in Dm_old.items() if not (m and m[0][0] == 'h')}
    print('  new D^-|_{hbar=0} = delta := [X^-,d] :', not P.add(Dm0, P.smul(-1, delta)),
          '; old one :', not P.add(Dm_old0, P.smul(-1, delta)))

    basis, sd = closure(P, [Dp, Dm, Xp, Xm], limit=300)
    print('  sdim <D^+,D^-,X^+,X^-> :', sd)
    basis_old, sd_old = closure(P, [Dp, Dm_old, Xp, Xm], limit=300)
    ech_new, ech_old = span(P, basis), span(P, basis_old)
    print('  sdim with the old D^-  :', sd_old,
          '; new subset of old :', all(contains(P, ech_old, e) for e in basis),
          '; old D^- in the new one :', contains(P, ech_new, Dm_old))

    # Theorem 1.9.4a, item 1)
    print('  [D^+,D^+] = -2 hbar X^+ :',
          not P.add(P.br(Dp, Dp), P.smul(2, P.mul(hb, Xp))))
    print('  [D^-,D^-] = 0           :', not P.br(Dm, Dm))
    print('  [D^+,X^+] = 0           :', not P.br(Dp, Xp))
    print('  [D^-,X^-] = 0           :', not P.br(Dm, Xm))
    Dt = P.br(Dm, Xp)
    print('  Dtilde^+ := [D^-,X^+]   =', pretty(P, Dt, n))
    Dphm = P.add(d, P.smul(half, P.mul(hb, P.add(
        *[P.add(P.mul(c(q[i]), x(pi[i])),
                P.smul(-1, P.mul(c(pi[i]), x(q[i])))) for i in range(n)]))))
    print('  Dtilde^+ = D^+|_{hbar -> -hbar} :', not P.add(Dt, P.smul(-1, Dphm)))
    print('  [Dtilde^+,Dtilde^+] = 2 hbar X^+ :',
          not P.add(P.br(Dt, Dt), P.smul(-2, P.mul(hb, Xp))))

    # Theorem 1.9.4a, item 2)
    S = P.br(Dp, Dm)
    print('  [D^+,D^-] =', pretty(P, S, n))
    for sgn in (2, -2, 1, -1):
        lap = P.smul(sgn, P.add(*[P.mul(t(q[i]), t(pi[i])) for i in range(n)]))
        if not P.add(S, P.smul(-1, lap)):
            print('    i.e. %+d sum d_q d_pi (no hbar)' % sgn)
    print('  [D^+,D^-] = [d,delta] :', not P.add(S, P.smul(-1, P.br(d, delta))))
    # Theorem 1.9.4a, item 3)
    ker = [e for e in basis if all(m and m[0][0] == 'h' for m in e)]
    kech = span(P, ker)
    print('  sdim of the ideal hbar*g :', (len(kech[0]), len(kech[1])),
          '; abelian :', all(not P.br(a, b_)
                             for a in kech[0].elements + kech[1].elements
                             for b_ in kech[0].elements + kech[1].elements))
    # Theorem 1.9.4a, item 4)
    _, sdh = closure(P, [Xp, Xm, Hc, P.mul(hb, Xp), P.mul(hb, Xm), P.mul(hb, Hc)])
    print('  sdim <X^+,X^-,H, hbar*(same)> :', sdh)

    # the quotient g/hbar*g and whether the extension splits
    _, sd_h = closure(P, [Xp, Xm, d])                 # ba(2) + sl(1|1)
    print('  sdim of <X^+,X^-,d> = h        :', sd_h)
    print('  sdim h + Pi(h)                 :', (sd_h[0] + sd_h[1], sd_h[1] + sd_h[0]))
    print('  [D^+,D^-] = -hbar*H :', not P.add(S, P.mul(hb, Hc)),
          ' ; = +hbar*H :', not P.add(S, P.smul(-1, P.mul(hb, Hc))))

    # the hbar-free part of g: if it is of sdim (3|5) the extension splits
    # (the hbar-free elements of the super-Weyl algebra form a subalgebra)
    def hfree_part(e):
        return {m: cc for m, cc in e.items() if not (m and m[0][0] == 'h')}

    free = []
    Efree = {0: Echelon(), 1: Echelon()}
    for e in basis:
        if e == hfree_part(e):
            if Efree[P.parity(e)].add(e):
                free.append(e)
    # also test all linear combinations: solve for the hbar-free subspace of g
    mons = sorted({m for e in basis for m in e})
    hmons = [m for m in mons if m and m[0][0] == 'h']
    dims = {}
    for par in (0, 1):
        vecs = [e for e in basis if P.parity(e) == par]
        E = Echelon()
        red = [e for e in vecs if E.add(e)]
        # kernel of the map "hbar-part" on span(red)
        rows = [[e.get(m, Fraction(0)) for e in red] for m in hmons]
        ncol, r0 = len(red), 0
        mat = [r[:] for r in rows]
        for col in range(ncol):
            piv = next((r for r in range(r0, len(mat)) if mat[r][col] != 0), None)
            if piv is None:
                continue
            mat[r0], mat[piv] = mat[piv], mat[r0]
            pv = mat[r0][col]
            mat[r0] = [v / pv for v in mat[r0]]
            for r in range(len(mat)):
                if r != r0 and mat[r][col] != 0:
                    f = mat[r][col]
                    mat[r] = [a - f * bb for a, bb in zip(mat[r], mat[r0])]
            r0 += 1
        dims[par] = ncol - r0
    print('  sdim of the hbar-free part of g:', (dims[0], dims[1]),
          '==> the extension splits, g = h (x) C[hbar]/(hbar^2)'
          if (dims[0], dims[1]) == sd_h else '==> NON-split extension')
    print()
