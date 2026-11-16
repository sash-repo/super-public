#!/usr/bin/env python3
"""i(pe(2))-invariant unary and binary differential operators between spaces of
weighted densities F_{a,b} = F alpha_0^a vol^b over m(1)  (Section 7 of the paper).

Invariant derivatives (all supercommute with pe(2)_- = <M_1,M_q,M_xi>):
   D_1  = d_tau            odd,   degree -2, charge  0
   D_xi = xi d_tau - d_q   even,  degree -1, charge -1
   D_q  = q d_tau + d_xi   odd,   degree -1, charge +1
(degree = weight w.r.t. M_tau = 2 tau d_tau + q d_q + xi d_xi;
 charge = weight w.r.t. M_qxi = q d_q - xi d_xi.)

Weights: for an operator of degree s and charge w,
   a = sum a_i + (s+w)/2 ,   b = sum b_i - w/2 .
Invariance under pe(2)_- and under m(1)_0 cap pe(2) = <M_tau, M_qxi, M_q2> is
then automatic except for M_{q^2}, so the equations are taken from
M_{q^2}, M_{q tau}, M_{tau xi} (and everything is re-verified against all 8).
"""
import sys
from itertools import product
from sympy import symbols, S, Matrix, expand, simplify, factor, Rational
from super12 import add, smul, mul, mono, parity
from m1 import M, mbb, D_1, D_xi, D_q, L, PE, TESTM

a1, b1, a2, b2 = symbols('a_1 b_1 a_2 b_2')
GEN = {'1': (D_1, -2, 0, 1), 'x': (D_xi, -1, -1, 0), 'q': (D_q, -1, +1, 1)}
FIELDS = ('q2', 'qtau', 'tauxi')
ALLF = tuple(PE)


def monos(nmax):
    return [(a, e, f) for a in range(nmax + 1) for e in (0, 1) for f in (0, 1)]


def word_op(word):
    ops = [GEN[c][0] for c in word]

    def r(u):
        for o in reversed(ops):
            u = o(u)
        return u
    return r


def words(s, w, maxlen=8):
    """all words in D_1,D_xi,D_q of degree s and charge w, then thinned out to
    a linearly independent set (as operators on polynomials)."""
    out = []
    for n in range(0, maxlen + 1):
        for word in product('1xq', repeat=n):
            deg = sum(GEN[c][1] for c in word)
            ch = sum(GEN[c][2] for c in word)
            if deg == s and ch == w:
                out.append((''.join(word), word_op(''.join(word)),
                            sum(GEN[c][3] for c in word) % 2))
    # thin out: evaluate on monomials, keep those adding rank
    basis, rows, keep = monos(-s + 3), [], []
    cols = {}
    for nm, op, p in out:
        vec = []
        for m in basis:
            r = op(mono(*m))
            vec.append(r)
        rows.append(vec)
    # build a numeric matrix over the monomial basis of the image
    allmon = sorted({k for vec in rows for r in vec for k in r})
    Mx = Matrix([[r.get(k, 0) for r in vec for k in allmon] for vec in rows])
    ind = []
    cur = Matrix(0, Mx.cols, [])
    for i in range(Mx.rows):
        test = cur.col_join(Mx[i, :])
        if test.rank() > cur.rank():
            cur = test
            ind.append(out[i])
    return ind


def eqs_bin(terms, A1, B1, A2, B2, At, Bt, pB, nmax, fields):
    cs = symbols(f'c0:{len(terms)}')
    eqs = set()
    MS = monos(nmax)
    for nm in fields:
        f = PE[nm]
        pf = (parity(f) + 1) % 2
        sB = -1 if (pf and pB) else 1
        Lf1, Lf2, Lft = L(f, A1, B1), L(f, A2, B2), L(f, At, Bt)
        for m1 in MS:
            u = mono(*m1)
            pu = (m1[1] + m1[2]) % 2
            Lu = Lf1(u)
            for m2 in MS:
                v = mono(*m2)
                Lv = Lf2(v)

                def B(x, y, px):
                    return add(*[smul(cs[i] * (-1 if (terms[i][3] and px) else 1),
                                      mul(terms[i][1](x), terms[i][2](y)))
                                 for i in range(len(terms))])
                s2 = -1 if (pf and pu) else 1
                res = add(Lft(B(u, v, pu)),
                          smul(-sB, add(B(Lu, v, (pu + pf) % 2), smul(s2, B(u, Lv, pu)))))
                for val in res.values():
                    e = expand(val)
                    if e:
                        eqs.add(e)
    return cs, sorted(eqs, key=str)


def eqs_un(terms, A, Bb, At, Bt, pB, nmax, fields):
    cs = symbols(f'c0:{len(terms)}')
    eqs = set()
    for nm in fields:
        f = PE[nm]
        pf = (parity(f) + 1) % 2
        s = -1 if (pf and pB) else 1
        Ls, Lt = L(f, A, Bb), L(f, At, Bt)
        for m in monos(nmax):
            u = mono(*m)
            Bu = add(*[smul(cs[i], terms[i][1](u)) for i in range(len(terms))])
            BLu = add(*[smul(cs[i], terms[i][1](Ls(u))) for i in range(len(terms))])
            res = add(Lt(Bu), smul(-s, BLu))
            for v in res.values():
                e = expand(v)
                if e:
                    eqs.add(e)
    return cs, sorted(eqs, key=str)


if __name__ == '__main__':
    SMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    aa, bb = symbols('a b')

    print("=" * 78)
    print("UNARY  (should reproduce [BLS1, Thm 5.1])")
    print("=" * 78)
    for s in range(-1, -SMAX - 1, -1):
        for w in range(-4, 5):
            ws = words(s, w)
            if not ws:
                continue
            pB = ws[0][2]
            if any(x[2] != pB for x in ws):
                # split by parity
                pass
            for par in (0, 1):
                tt = [(x[0] or 'id', x[1], x[2]) for x in ws if x[2] == par]
                if not tt:
                    continue
                terms = [(nm, op, None, p) for nm, op, p in tt]
                At, Bt = aa + S(s + w) / 2, bb - S(w) / 2
                cs, eqs = eqs_un(terms, aa, bb, At, Bt, par, -s + 2, FIELDS)
                Mx = Matrix([[expand(e).coeff(c) for c in cs] for e in eqs])
                # rank-drop locus
                from sympy import solve
                sols = []
                for i in range(len(cs)):
                    sub = {cs[j]: (S(1) if j == i else symbols(f'y{j}'))
                           for j in range(len(cs))}
                    unk = [sub[cs[j]] for j in range(len(cs)) if j != i] + [aa, bb]
                    try:
                        r = solve([expand(e.subs(sub)) for e in eqs], unk, dict=True)
                    except Exception:
                        r = []
                    sols += [(i, d) for d in r]
                if sols:
                    print(f"  s={s} w={w:+d} parity {par}: words {[x[0] for x in tt]}"
                          f"  -> F_{{{At},{Bt}}}")
                    seen = set()
                    for i, d in sols:
                        k = str(sorted(d.items(), key=str))
                        if k in seen:
                            continue
                        seen.add(k)
                        print(f"      c_{i}=1: {d}")
                sys.stdout.flush()

    print("=" * 78)
    print("BINARY")
    print("=" * 78)
    for s in range(-1, -SMAX - 1, -1):
        for w in range(-4, 5):
            terms = []
            for sp in range(0, -s + 1):
                for wp in range(-4, 5):
                    W1 = words(-sp, wp)
                    W2 = words(s + sp, w - wp)
                    for n1, o1, p1 in W1:
                        for n2, o2, p2 in W2:
                            terms.append((f"{n1 or 'id'}|{n2 or 'id'}", o1, o2,
                                          p2, (p1 + p2) % 2))
            for par in (0, 1):
                tt = [(n, o1, o2, pq) for (n, o1, o2, pq, pt) in terms if pt == par]
                if not tt:
                    continue
                At, Bt = a1 + a2 + S(s + w) / 2, b1 + b2 - S(w) / 2
                cs, eqs = eqs_bin(tt, a1, b1, a2, b2, At, Bt, par,
                                  (-s) // 2 + 2, FIELDS)
                Mx = Matrix([[expand(e).coeff(c) for c in cs] for e in eqs])
                ns = Mx.nullspace()
                print(f"  s={s:3d} w={w:+d} parity {par}: {len(tt):2d} terms, "
                      f"rank {Mx.rank()}, generic dim {len(ns)}  -> F_{{{At},{Bt}}}")
                for v in ns:
                    print("     ", {tt[i][0]: factor(simplify(v[i]))
                                    for i in range(len(tt)) if v[i] != 0})
                sys.stdout.flush()
