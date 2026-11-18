"""the k^M(1|n)-module of volume forms .  Functions in t (Laurent, sympy) and Grassmann
theta_1..theta_{n-1}, zeta.  K^M_f = nabla(f) D + H^M_f + D(f) E  (the text),  D = d_t - zeta/(2t) d_zeta.
Test: Div(K^M_f) = lam*D(f) + mu*nabla(f)/t  for all f  <=>  Vol ~ Pi^n(F^M_{lam;mu})  (F^M_lam = R alpha~^{lam/2}, multiplier 2D(f)).
Div(a d_t + sum b_i d_{theta_i}) = d_t a + sum (-1)^{p(b_i)} d_{theta_i} b_i, the volume being vol(t,theta,zeta)."""
import sympy as sp, itertools, random
t = sp.symbols('t', positive=True)
def mono_mul(a, b):
    if set(a) & set(b): return None, 0
    s = list(a) + list(b); sg = 1
    for i in range(len(s)):
        for j in range(len(s) - 1 - i):
            if s[j] > s[j + 1]: s[j], s[j + 1] = s[j + 1], s[j]; sg = -sg
    return tuple(s), sg
def mul(x, y):
    o = {}
    for a, u in x.items():
        for b, v in y.items():
            k, s = mono_mul(a, b)
            if k is None: continue
            o[k] = sp.expand(o.get(k, 0) + s * u * v)
    return {k: v for k, v in o.items() if v != 0}
def add(*xs):
    o = {}
    for x in xs:
        for k, v in x.items(): o[k] = sp.expand(o.get(k, 0) + v)
    return {k: v for k, v in o.items() if sp.simplify(v) != 0}
def sc(c, x): return {k: sp.expand(c * v) for k, v in x.items()}
def dodd(x, g):  # left derivative d/d(gen g)
    o = {}
    for m, v in x.items():
        if g in m:
            i = m.index(g); o[m[:i] + m[i + 1:]] = o.get(m[:i] + m[i + 1:], 0) + (-1) ** i * v
    return o
def dt(x): return {k: sp.diff(v, t) for k, v in x.items() if sp.diff(v, t) != 0}
def par(x):
    ps = {len(k) % 2 for k in x}; assert len(ps) <= 1; return ps.pop() if ps else 0
def run(n, trials=6):
    Z = n - 1            # index of zeta; thetas 0..n-2
    gens = list(range(n))
    def D(f): return add(dt(f), sc(-1 / (2 * t), mul({(Z,): 1}, dodd(f, Z))))
    def E(f): return add(*[mul({(g,): 1}, dodd(f, g)) for g in gens])
    def nabla(f): return add(sc(2, f), sc(-1, E(f)))
    def KM(f):  # returns (a, {g: b_g}) with K = a d_t + sum b_g d_g
        p = par(f); s = (-1) ** p
        a = nabla(f)
        b = {}
        for g in gens:
            # nabla(f) D contributes -nabla(f) zeta/(2t) on d_zeta; H^M; D(f) E contributes D(f) g on d_g
            bg = mul(D(f), {(g,): 1})
            if g == Z:
                bg = add(bg, sc(-1 / (2 * t), mul(nabla(f), {(Z,): 1})), sc(s / t, dodd(f, Z)))
            else:
                bg = add(bg, sc(s, dodd(f, g)))
            b[g] = bg
        return a, b
    def Div(f):
        a, b = KM(f)
        out = dt(a)
        for g in gens:
            bg = b[g]
            if bg: out = add(out, sc((-1) ** par(bg), dodd(bg, g)))
        return out
    R = random.Random(n)
    rows = []
    lam, mu = sp.symbols('lam mu')
    eqs = []
    for _ in range(trials):
        for d in range(n + 1):
            for I in itertools.combinations(gens, d):
                k = R.randint(-3, 3)
                f = {tuple(I): t ** k}
                lhs = Div(f); rhs = add(sc(lam, D(f)), sc(mu / t, nabla(f)))
                diff = add(lhs, sc(-1, rhs))
                eqs += [sp.expand(v * t ** 10) for v in diff.values()]
    eqs = [sp.Poly(e, t).coeffs() for e in eqs if e != 0]
    sol = sp.solve([c for cs in eqs for c in cs], [lam, mu], dict=True)
    return sol
for n in (1, 2, 3, 4, 7):
    print("k^M(1|%d): Div(K^M_f) = lam D(f) + mu nabla(f)/t  ->" % n, run(n, 2))
