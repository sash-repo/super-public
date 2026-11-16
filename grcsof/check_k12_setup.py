#!/usr/bin/env python3
"""Sanity checks of the k(1|2) framework, and a recomputation of the unary
answer [BLS1, Theorem 1.5] (Bol operators for i(osp(2|2)) subset k(1|2)).
"""
from sympy import symbols, S, Matrix, simplify, factor, expand, Rational
from super12 import (clean, add, smul, mul, mono, parity, d_even)
from k12 import (K, kb, D_xi, D_eta, div2, L, OSP, TEST, d_t)

a_, b_ = symbols('a b')          # weights of the source F_{a,b}
ok = True


def chk(cond, msg):
    global ok
    print(('  OK   ' if cond else '  FAIL ') + msg)
    ok = ok and bool(cond)


def eqop(o1, o2, monos=TEST):
    for m in monos:
        if add(o1(mono(*m)), smul(-1, o2(mono(*m)))):
            return False
    return True


print("1. K is a homomorphism: [K_f,K_g] = K_{{f,g}} on osp(2|2)")
for n1, f in OSP.items():
    for n2, g in OSP.items():
        pf, pg = parity(f), parity(g)
        s = -1 if (pf and pg) else 1
        lhs = lambda u, f=f, g=g, s=s: add(K(f)(K(g)(u)), smul(-s, K(g)(K(f)(u))))
        rhs = K(kb(f, g))
        if not eqop(lhs, rhs):
            chk(False, f"[K_{n1},K_{n2}]")
chk(True, "all 64 pairs")

print("2. osp(2|2) = span{K_f} is closed under the bracket")
import itertools
basis = list(OSP.values())


def coords(h):
    """express h in the basis OSP (as generating functions)"""
    v = {}
    rest = dict(h)
    for n, f in OSP.items():
        # each OSP element is a single monomial
        m = list(f)[0]
        if m in rest:
            v[n] = rest.pop(m) / f[m]
    return v, clean(rest)


for n1, f in OSP.items():
    for n2, g in OSP.items():
        _, rest = coords(kb(f, g))
        if rest:
            chk(False, f"{{{n1},{n2}}} = ... + {rest} leaves osp(2|2)")
chk(True, "all brackets stay in span{1,xi,eta,t,xieta,txi,teta,t2}")

print("3. weights: K_t alpha = 2 alpha, K_t beta = 0, K_xieta beta = beta, "
      "K_t2 beta = 2 xi eta beta   [BLS1 3.2.2]")
chk(d_t(OSP['t']) == mono() and div2(OSP['t']) == {}, "f=t : 2 d_t f = 2, div_2 = 0")
chk(div2(OSP['xieta']) == mono(), "f=xi eta : div_2 = 1")
chk(div2(OSP['t2']) == mono(0, 1, 1, 2), "f=t^2 : div_2 = 2 xi eta")
chk(smul(2, d_t(OSP['t2'])) == mono(1, 0, 0, 4), "f=t^2 : 2 d_t f = 4t")

print("4. F_{a,b} is a module (both sign conventions tried)")
for sgn2 in (1, -1):
    good = True
    for n1, f in OSP.items():
        for n2, g in OSP.items():
            pf, pg = parity(f), parity(g)
            s = -1 if (pf and pg) else 1
            Lf, Lg = L(f, a_, b_, sgn2), L(g, a_, b_, sgn2)
            lhs = lambda u: add(Lf(Lg(u)), smul(-s, Lg(Lf(u))))
            rhs = L(kb(f, g), a_, b_, sgn2)
            if not eqop(lhs, rhs, TEST[:12]):
                good = False
    chk(good, f"sgn2 = {sgn2:+d}")

print("5. D's: D_xi^2 = D_eta^2 = 0, [D_xi,D_eta] = 2 d_t, "
      "[K_xieta,D_xi]=D_xi, [K_xieta,D_eta]=-D_eta")
chk(eqop(lambda u: D_xi(D_xi(u)), lambda u: {}), "D_xi^2 = 0")
chk(eqop(lambda u: D_eta(D_eta(u)), lambda u: {}), "D_eta^2 = 0")
chk(eqop(lambda u: add(D_xi(D_eta(u)), D_eta(D_xi(u))), lambda u: smul(2, d_t(u))),
    "D_xi D_eta + D_eta D_xi = 2 d_t")
Kxe = K(OSP['xieta'])
chk(eqop(lambda u: add(Kxe(D_xi(u)), smul(-1, D_xi(Kxe(u)))), D_xi), "[K_xieta,D_xi]=D_xi")
chk(eqop(lambda u: add(Kxe(D_eta(u)), smul(-1, D_eta(Kxe(u)))),
         lambda u: smul(-1, D_eta(u))), "[K_xieta,D_eta]=-D_eta")

print("6. D's supercommute with g_- = <K_1, K_xi, K_eta>")
for nm in ('1', 'xi', 'eta'):
    Kf = K(OSP[nm])
    pf = parity(OSP[nm])
    for dn, Dop in (('D_xi', D_xi), ('D_eta', D_eta)):
        s = -1 if pf else 1                    # D's are odd
        chk(eqop(lambda u, Kf=Kf, Dop=Dop, s=s: add(Kf(Dop(u)), smul(-s, Dop(Kf(u)))),
                 lambda u: {}), f"[K_{nm}, {dn}] = 0")

# --------------------------------------------------------------------------
# 7.  unary classification: reproduce [BLS1, Theorem 1.5]
# --------------------------------------------------------------------------
print()
print("7. unary i(osp(2|2))-invariant operators F_{a,b} -> F_{c,d}")
print("   ansatz: linear combinations of d_t^n D_I, |I| in {0,1,2}, fixed degree")

DOPS = {(): (lambda u: u, 0, 0),
        ('xi',): (D_xi, 1, 1),
        ('eta',): (D_eta, 1, -1),
        ('xi', 'eta'): (lambda u: D_xi(D_eta(u)), 2, 0)}
# (operator, #D's, weight w.r.t. K_xieta)


def dtn(n):
    def r(u):
        for _ in range(n):
            u = d_t(u)
        return u
    return r


def unary_terms(deg, w):
    """all d_t^n D_I with 2n+|I| = -deg and xi-eta-weight w"""
    out = []
    for I, (op, nD, wt) in DOPS.items():
        r = -deg - nD
        if r >= 0 and r % 2 == 0 and wt == w:
            out.append((I, r // 2, op, nD))
    return out


def solve_unary(deg, w, verbose=True):
    terms = unary_terms(deg, w)
    if not terms:
        return None
    c = symbols(f'c0:{len(terms)}')
    # target weights: a' = a - deg/2 ,  b' = b - w   (see the note in the paper)
    A, B = a_, b_
    A2 = A - S(deg)
    B2 = B - w
    Bop = lambda u: add(*[smul(c[i], compose_dt(terms[i], u)) for i in range(len(terms))]) \
        if terms else {}
    eqs = set()
    for nm in ('t', 'xieta', 'txi', 'teta', 't2', '1', 'xi', 'eta'):
        f = OSP[nm]
        pf = parity(f)
        pB = (-deg - 2 * (0)) % 2   # parity of B = number of D's mod 2 = w mod 2
        pB = w % 2
        Lsrc, Ltgt = L(f, A, B), L(f, A2, B2)
        for m in TEST:
            u = mono(*m)
            pu = par_mon(m) if False else (m[1] + m[2]) % 2
            s = -1 if (pf and pB) else 1
            res = add(Ltgt(Bop(u)), smul(-s, Bop(Lsrc(u))))
            for k, v in res.items():
                e = expand(v)
                if e != 0:
                    eqs.add(e)
    eqs = list(eqs)
    M = Matrix([[expand(e).coeff(ci) for ci in c] for e in eqs])
    return c, M, A2, B2, terms


def compose_dt(term, u):
    I, n, op, nD = term
    return op(dtn(n)(u))


from super12 import par_mon
for deg in range(-1, -7, -1):
    for w in (-2, -1, 0, 1, 2):
        r = solve_unary(deg, w)
        if r is None:
            continue
        c, M, A2, B2, terms = r
        ns = M.nullspace()
        lbl = ' + '.join(f"d_t^{n} D_{''.join(I) if I else '1'}" for I, n, _, _ in terms)
        if ns:
            print(f"  deg={deg} w={w:+d}: generic solutions {len(ns)}  [{lbl}]")
            for v in ns:
                print("      ", [simplify(x) for x in v], f"-> F_{{{A2},{B2}}}")
        else:
            # rank is full generically; look for the locus where it drops
            sq = M.rank()
            print(f"  deg={deg} w={w:+d}: no generic solution (rank {sq}/{len(c)})  [{lbl}]")

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
