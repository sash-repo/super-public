"""Evolutionary fields on supercommutative jet polynomials (spdo.P), 1 space variable x."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sdkp'))
from spdo import P, PAR
from fractions import Fraction
def mono(ev, od, c=1):
    r = P.const(c)
    for v, a in ev:
        for _ in range(a): r = r * P.var(v[0], v[1])
    for v in od: r = r * P.var(v[0], v[1])
    return r
def dxk(p, k):
    for _ in range(k): p = p.dx()
    return p
def apply(phi, sig, p):
    """X_phi(p), X of parity sig, phi: name -> P."""
    r = P()
    for (ev, od), c in p.t.items():
        for idx, (v, a) in enumerate(ev):
            ev2 = list(ev); ev2[idx] = (v, a - 1)
            ev2 = tuple(e for e in ev2 if e[1])
            r = r + dxk(phi.get(v[0], P()), v[1]) * mono(ev2, od, c * a)
        for j, v in enumerate(od):
            s = -1 if (sig and j % 2) else 1
            r = r + mono(ev, od[:j], c * s) * dxk(phi.get(v[0], P()), v[1]) * mono((), od[j+1:])
    return r
def bracket(phi, sp_, chi, sc, names):
    """[X_phi, X_chi] = X_{X_phi(chi) - (-1)^{sp sc} X_chi(phi)}"""
    sg = -1 if (sp_ and sc) else 1
    return {n: apply(phi, sp_, chi.get(n, P())) - apply(chi, sc, phi.get(n, P())) * sg for n in names}
def is_sym(K, phi, sig, names):
    """evolution system u_t = K (even); phi t-independent"""
    b = bracket(K, 0, phi, sig, names)
    return all(v.iszero() for v in b.values())
