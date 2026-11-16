#!/usr/bin/env python3
"""Sanity checks for m(1) and pe(2), and the realisation of pe(2) printed in
Section 7 of the paper."""
from sympy import symbols, S, expand
from super12 import add, smul, mul, mono, parity, clean
from m1 import (M, mbb, D_1, D_xi, D_q, Delta_m, L, PE, TESTM,
                d_q, d_tau, d_xi, E_m)

a_, b_ = symbols('a b')
ok = True


def chk(c, msg):
    global ok
    print(('  OK   ' if c else '  FAIL ') + msg)
    ok = ok and bool(c)


def eqop(o1, o2, monos=TESTM):
    for m in monos:
        if add(o1(mono(*m)), smul(-1, o2(mono(*m)))):
            return False
    return True


def show(op, name):
    """print a vector field as a table of its values on the coordinates."""
    v = {c: op(mono(*m)) for c, m in (('q', (1, 0, 0)), ('tau', (0, 1, 0)),
                                      ('xi', (0, 0, 1)))}
    print(f"      {name}: " + ", ".join(f"{c} -> {val}" for c, val in v.items()))


print("1. M is a homomorphism: [M_f,M_g] = M_{{f,g}_m.b.}  "
      "(p(M_f) = p(f)+1)")
bad = []
for n1, f in PE.items():
    for n2, g in PE.items():
        pf, pg = (parity(f) + 1) % 2, (parity(g) + 1) % 2
        s = -1 if (pf and pg) else 1
        lhs = lambda u, f=f, g=g, s=s: add(M(f)(M(g)(u)), smul(-s, M(g)(M(f)(u))))
        if not eqop(lhs, M(mbb(f, g))):
            bad.append(f"[{n1},{n2}]")
chk(not bad, f"all 64 pairs" + ('' if not bad else f" -- failures {bad}"))

print("2. pe(2) = span{M_f : f in 1,q,xi,q2,qxi,tau,qtau,tauxi} is closed")
bad = []
for n1, f in PE.items():
    for n2, g in PE.items():
        rest = dict(mbb(f, g))
        for n, h in PE.items():
            m = list(h)[0]
            if m in rest:
                del rest[m]
        if clean(rest):
            bad.append(f"{{{n1},{n2}}} -> {clean(rest)}")
chk(not bad, "closure" + ('' if not bad else f" -- {bad}"))

print("3. the realisation of pe(2) printed in Section 7 of the paper")
for nm in PE:
    show(M(PE[nm]), f"M_{nm}")
print("   paper:  M_xi = xi d_tau + d_q;  M_qtau = q tau d_tau - tau d_xi + q E;")
print("           M_{tau+q xi} = 2 tau d_tau + 2 q d_q;  "
      "M_{tau-q xi} = 2 tau d_tau + 2 xi d_xi;")
print("           M_1 = 2 d_tau;  M_q = q d_tau - d_xi;  M_q2 = -2 q d_xi;")
print("           M_tauxi = (tau - q xi) d_q + tau xi d_tau")
tp = add(PE['tau'], PE['qxi'])
tm = add(PE['tau'], smul(-1, PE['qxi']))
show(M(tp), "M_{tau+q xi}")
show(M(tm), "M_{tau-q xi}")

print("4. D's supercommute with pe(2)_- = <M_1, M_q, M_xi>")
for nm in ('1', 'q', 'xi'):
    f = PE[nm]
    pf = (parity(f) + 1) % 2
    for dn, Dop, pD in (('D_1', D_1, 1), ('D_xi', D_xi, 0), ('D_q', D_q, 1)):
        s = -1 if (pf and pD) else 1
        chk(eqop(lambda u, f=f, Dop=Dop, s=s: add(M(f)(Dop(u)), smul(-s, Dop(M(f)(u)))),
                 lambda u: {}), f"[M_{nm}, {dn}] = 0")

print("5. F_{a,b} is a module over pe(2)")
bad = []
for n1, f in PE.items():
    for n2, g in PE.items():
        pf, pg = (parity(f) + 1) % 2, (parity(g) + 1) % 2
        s = -1 if (pf and pg) else 1
        Lf, Lg = L(f, a_, b_), L(g, a_, b_)
        lhs = lambda u: add(Lf(Lg(u)), smul(-s, Lg(Lf(u))))
        if not eqop(lhs, L(mbb(f, g), a_, b_), TESTM[:12]):
            bad.append(f"[{n1},{n2}]")
chk(not bad, "representation property" + ('' if not bad else f" -- {bad}"))

print("6. weights of the two diagonal elements of m(1)_0")
print("   M_tau  =", {c: M(PE['tau'])(mono(*m)) for c, m in
                      (('q', (1, 0, 0)), ('tau', (0, 1, 0)), ('xi', (0, 0, 1)))})
print("   M_qxi  =", {c: M(PE['qxi'])(mono(*m)) for c, m in
                      (('q', (1, 0, 0)), ('tau', (0, 1, 0)), ('xi', (0, 0, 1)))})
print("   d_tau(tau) =", d_tau(PE['tau']), " Delta_m(tau) =", Delta_m(PE['tau']))
print("   d_tau(qxi) =", d_tau(PE['qxi']), " Delta_m(qxi) =", Delta_m(PE['qxi']))
print("   d_tau(q2)  =", d_tau(PE['q2']), " Delta_m(q2)  =", Delta_m(PE['q2']))

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
