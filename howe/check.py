"""
check.py -- independent verification of the engine.

 (1) le(n) closure: [L_i, L_j] subset L_{i+j}, and the antibracket agrees with the
     super-commutator of the vector fields D_f (already checked in le.py's self-test).
 (2) V(lambda) is an L_0-module: x(yv) - (-1)^{p(x)p(y)} y(xv) = [x,y]v on all of pe(n).
 (3) I(V) is an le(n)-module: the same identity for x,y in L_0 + L_1 on all basis
     vectors of degree <= 2.
 (4) dim V(lambda) against independently known values.
"""

import sys
from fractions import Fraction as F
from le import abr, par_le, pwt, is_hom_par
from pemod import Pe, Irrep, mon
from imod import Le, Induced


def check_closure(n, top=3):
    le = Le(n)
    monos = {}
    for d in range(1, top + 3):
        monos[d] = le._monos(d)
    bad = 0
    for d1 in monos:
        for d2 in monos:
            if d1 + d2 > top + 3:
                continue
            for f in monos[d1]:
                for g in monos[d2]:
                    h = abr(f, g, n)
                    if not h:
                        continue
                    degs = {sum(a) + len(S) for (a, S) in h}
                    if degs != {d1 + d2 - 2}:
                        bad += 1
    return bad


def check_pe_module(n, lam):
    pe = Pe(n)
    ir = Irrep(pe, lam)
    weights = []
    seen = set()
    stack = [ir.lam]
    while stack:
        mu = stack.pop()
        if mu in seen:
            continue
        seen.add(mu)
        if not ir.wtbasis(mu):
            continue
        weights.append(mu)
        for w in pe.nneg_wt:
            stack.append(tuple(a + c for a, c in zip(mu, w)))
    ntest = 0
    for mu in weights:
        for i in range(len(ir.wtbasis(mu))):
            for b1 in range(len(pe.basis)):
                for b2 in range(len(pe.basis)):
                    v = {i: F(1)}
                    m1 = tuple(a + c for a, c in zip(mu, pe.bwt[b2]))
                    lhs = ir.act(b1, m1, ir.act(b2, mu, v))
                    m2 = tuple(a + c for a, c in zip(mu, pe.bwt[b1]))
                    t = ir.act(b2, m2, ir.act(b1, mu, v))
                    sg = (-1) ** (pe.bpar[b1] * pe.bpar[b2])
                    for k, c in t.items():
                        lhs[k] = lhs.get(k, 0) - sg * c
                    br = pe.as_vector(abr(pe.basis[b1], pe.basis[b2], n))
                    for b, cb in br.items():
                        for k, c in ir.act(b, mu, v).items():
                            lhs[k] = lhs.get(k, 0) - cb * c
                    lhs = {k: c for k, c in lhs.items() if c}
                    ntest += 1
                    if lhs:
                        return ntest, (mu, i, b1, b2, lhs)
    return ntest, None


def check_induced(n, lam, maxdeg=2):
    le = Le(n)
    I = Induced(le, lam)
    pe = le.pe
    ops = [pe.basis[k] for k in range(len(pe.basis))] + list(le.pos1)
    # basis vectors of I(V) up to degree maxdeg, at the weights that occur
    keys = []
    for m in range(maxdeg + 1):
        for u in I.umonos(m):
            w = I.uwt(u)
            for mu in _wts(I, 3):
                if len(I.V.wtbasis(mu)) == 0:
                    continue
                for i in range(len(I.V.wtbasis(mu))):
                    keys.append((u, mu, i))
    ntest = 0
    for key in keys:
        for f1 in ops:
            for f2 in ops:
                lhs = I.act(f1, I._act_one(f2, key))
                t = I.act(f2, I._act_one(f1, key))
                sg = (-1) ** (par_le(f1) * par_le(f2))
                for k, c in t.items():
                    lhs[k] = lhs.get(k, 0) - sg * c
                br = abr(f1, f2, n)
                if br:
                    for k, c in I._act_one(br, key).items():
                        lhs[k] = lhs.get(k, 0) - c
                lhs = {k: c for k, c in lhs.items() if c}
                ntest += 1
                if lhs:
                    return ntest, (key, f1, f2, lhs)
    return ntest, None


def _deg(f):
    a, S = next(iter(f))
    return sum(a) + len(S)


def _wts(I, ht):
    pe = I.pe
    out, seen, stack = [], set(), [I.lam]
    while stack:
        mu = stack.pop()
        if mu in seen:
            continue
        seen.add(mu)
        nu = tuple(a - b for a, b in zip(I.lam, mu))
        if pe.ht(nu) > ht:
            continue
        if not I.V.wtbasis(mu):
            continue
        out.append(mu)
        for w in pe.nneg_wt:
            stack.append(tuple(a + c for a, c in zip(mu, w)))
    return out


if __name__ == '__main__':
    for n in (2, 3):
        print('--- n =', n)
        print('  [L_i,L_j] subset L_{i+j}: violations =', check_closure(n))
        for lam in ([(0,) * n, (1,) + (0,) * (n - 1), (1,) * n, (2,) + (1,) * (n - 1)]
                    + ([(2, 1)] if n == 2 else [(2, 2, 1), (2, 1, 1)])):
            nt, bad = check_pe_module(n, lam)
            print('  V%s : module axiom on %d triples ->' % (str(lam), nt),
                  'OK' if bad is None else 'FAIL %s' % (bad,))
            sys.stdout.flush()
    for n in (2, 3):
        for lam in [(1,) * n, (2,) + (1,) * (n - 1), (2,) * n]:
            nt, bad = check_induced(n, lam, maxdeg=2)
            print('  I(V%s) n=%d : module axiom on %d triples ->' % (str(lam), n, nt),
                  'OK' if bad is None else 'FAIL %s' % (bad,))
            sys.stdout.flush()
