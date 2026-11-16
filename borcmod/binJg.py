"""Symbolic proof: J_0, J_1, J_2 are invariant under K_{g theta} for an arbitrary function g (over Z[a,b]),
with generic phi, psi (arbitrary functions u(t), v(t)); J_3 is not (shows the conditions)  (Proposition 7.8).
Usage: python3 binJg.py"""
import sympy as sp
t, a, b = sp.symbols('t a b')
g = sp.Function('g')(t); u = sp.Function('u')(t); v = sp.Function('v')(t)
def D(f): return (f[1], sp.diff(f[0], t))
def Dn(f, n):
    for _ in range(n): f = D(f)
    return f
def L(f, lam): return (sp.expand(-g*f[1]), sp.expand(g*sp.diff(f[0], t) + lam*sp.diff(g, t)*f[0]))
def mul(f, h): return (sp.expand(f[0]*h[0]), sp.expand(f[0]*h[1] + f[1]*h[0]))
def add(*fs): return (sp.expand(sum(f[0] for f in fs)), sp.expand(sum(f[1] for f in fs)))
def sc(s, f): return (s*f[0], s*f[1])
Js = {0: {(0, 0): 1}, 1: {(1, 0): b, (0, 1): -a}, 2: {(0, 2): -a, (1, 1): 1, (2, 0): b},
      3: {(0, 3): a*(a+1), (1, 2): -(a+1)*(b+1), (2, 1): -(a+1)*(b+1), (3, 0): b*(b+1)}}
def apply(cs, phi, pphi, psi):
    out = (0, 0)
    for (i, j), w in cs.items(): out = add(out, sc(w*(-1)**(j*pphi), mul(Dn(phi, i), Dn(psi, j))))
    return out
for k, cs in Js.items():
    c = a+b+k; pi = k % 2; bad = []
    for pphi in (0, 1):
        for ppsi in (0, 1):
            phi = (u, 0) if pphi == 0 else (0, u); psi = (v, 0) if ppsi == 0 else (0, v)
            d = add(L(apply(cs, phi, pphi, psi), c),
                    sc(-(-1)**pi, add(apply(cs, L(phi, a), 1-pphi, psi), sc((-1)**pphi, apply(cs, phi, pphi, L(psi, b))))))
            bad += [sp.factor(x) for x in d if sp.simplify(x) != 0]
    print("J_%d:" % k, "invariant under every K_{g theta}" if not bad else ("defect", bad[:2]))
