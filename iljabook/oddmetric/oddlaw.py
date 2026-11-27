"""Statement 5.A.8.2: the tractor connection of an odd metric with an UNKNOWN Schouten-type
form beta (entries = unknown Grassmann-valued symbols at a point PT).
1) signs s1, s2 in  nabla_a d_s = s1*beta_as p + Gamma + s2*g_as q,  nabla_a p = d_a,  nabla_a q = r d_a
   (beta(X,Y) = g(rX,Y)) for which the connection preserves the odd form <p,q>=1, <X,Y>=g(X,Y):
   tested at PT for unknown beta, symmetric or not.
2) psi(p)=lam p, psi(d_s)=lam d_s + d_s(lam) p, psi(q)=q/lam + t*G/lam^2 (+ u*p), G = grad lam:
   solve psi D' = D psi for beta' in terms of beta, and compare beta'-beta with -Hess phi + dphi dphi."""
import sys, itertools, sympy as sp
from stractor import Alg
from oddgeom import geometry_odd, pairing
from oddexample import A, N, P, x, y, metric, PT, ev, evM

idx = ['p'] + list(range(N)) + ['q']
bpar = {'p': 0, 'q': 1, **{i: P(i) for i in range(N)}}

def unknown_form(name):
    syms = []; M = [[{} for _ in range(N)] for _ in range(N)]
    for a, s in itertools.product(range(N), repeat=2):
        par = (P(a) + P(s)) % 2
        monos = [(), (0, 1)] if par == 0 else [(0,), (1,)]
        for mo in monos:
            t = sp.Symbol(f"{name}_{a}{s}_{''.join(map(str, mo)) or 'e'}"); syms.append(t)
            M[a][s][mo] = t
    return M, syms

def r_from(beta, Gi):
    out = []
    for a in range(N):
        w = A.matmul([beta[a]], Gi)[0]
        out.append([A.sc((-1)**(P(a)+P(c)+1), w[c]) for c in range(N)])
    return out

def make_D(geo, g, beta, signs, at_point=True):
    s1f, s2f = signs
    Gam = {k: ev(v) for k, v in geo['Gam'].items()}
    gP = evM(g); Gi = evM(geo['Gi'])
    r0 = r_from(beta, Gi)
    def Dbasis(a, s):
        out = {i: {} for i in idx}
        if s == 'p': out[a] = A.const(1)
        elif s == 'q':
            for c in range(N): out[c] = r0[a][c]
        else:
            out['p'] = A.sc(s1f(a, s), beta[a][s]); out['q'] = A.sc(s2f(a, s), gP[a][s])
            for c in range(N): out[c] = Gam[a, s, c]
        return out
    def D(a, sec, dsec):
        """sec: section coefficients AT the point; dsec: d_a of the coefficients at the point."""
        out = {i: dict(dsec.get(i, {})) for i in idx}
        for s, f in sec.items():
            if not f: continue
            fs = A.sgn(P(a), f); Db = Dbasis(a, s)
            for i in idx: out[i] = A.add(out[i], A.mul(fs, Db[i]))
        return out
    return D, gP

metric_g = metric()
def compat(D, gP):
    B = {('p', 'q'): A.const(1), ('q', 'p'): A.const(1)}
    for i, j in itertools.product(range(N), repeat=2): B[i, j] = gP[i][j]
    res = []
    for a in range(N):
        for u, v in itertools.product(idx, repeat=2):
            Du = D(a, {u: A.const(1)}, {}); Dv = D(a, {v: A.const(1)}, {})
            lhs = ev(A.d(a, metric_g[u][v])) if (u in range(N) and v in range(N)) else {}
            rhs = A.add(A.sc((-1)**P(a), pairing(A, 1, B, bpar, Du, {v: A.const(1)})),
                        A.sc((-1)**(P(a)*(1+bpar[u])), pairing(A, 1, B, bpar, {u: A.const(1)}, Dv)))
            r = A.add(lhs, A.neg(rhs))
            if r: res.append((a, u, v, r))
    return res

if __name__ == "__main__" and "psi" not in sys.argv:
    metric_g = metric()
    geo = geometry_odd(A, metric_g)
    print("geometry done", flush=True)
    beta, bs = unknown_form('b')
    variants = {}
    for e1, f1, o1, e2, f2, o2 in itertools.product((0, 1), repeat=6):
        s1 = (lambda e, f, o: (lambda a, s: (-1)**(o + e*P(a) + f*P(s))))(e1, f1, o1)
        s2 = (lambda e, f, o: (lambda a, s: (-1)**(o + e*P(a) + f*P(s))))(e2, f2, o2)
        D, gP = make_D(geo, metric_g, beta, (s1, s2))
        bad = compat(D, gP)
        if not bad: variants[(e1, f1, o1, e2, f2, o2)] = (s1, s2)
        print((e1, f1, o1, e2, f2, o2), "compat failures", len(bad), flush=True)
    print("compatible sign variants:", list(variants), flush=True)

def psi_test(geo, geop, g, lam, beta, betap, signs, xv, yv, t):
    """returns list of residual entries (Grassmann dicts with symbolic coefficients)."""
    D, _ = make_D(geo, g, beta, signs)
    gp = [[A.mul(A.mul(lam, lam), g[i][j]) for j in range(N)] for i in range(N)]
    Dp, _ = make_D(geop, gp, betap, signs)
    il = A.inv(lam); Gi = geo['Gi']
    v = [[A.sc((-1)**(xv*P(b)), A.d(b, lam)) for b in range(N)]]
    w = A.matmul(v, Gi)[0]
    G = [A.sc((-1)**(P(c)+1), w[c]) for c in range(N)]
    def pb(s):   # symbolic coefficients of psi(e_s)
        out = {}
        if s == 'p': out['p'] = lam
        elif s == 'q':
            out['q'] = il
            for c in range(N): out[c] = A.sc(t, A.mul(A.mul(il, il), G[c]))
        else:
            out[s] = lam; out['p'] = A.sc((-1)**(yv*P(s)), A.d(s, lam))
        return out
    res = []
    for a in range(N):
        for s in idx:
            # lhs: psi(D'_a e_s)
            De = Dp(a, {s: A.const(1)}, {})
            lhs = {i: {} for i in idx}
            for j, f in De.items():
                if not f: continue
                for i, h in pb(j).items():
                    lhs[i] = A.add(lhs[i], A.mul(f, ev(h)))
            sec = {i: ev(h) for i, h in pb(s).items()}
            dsec = {i: ev(A.d(a, h)) for i, h in pb(s).items()}
            rhs = D(a, sec, dsec)
            for i in idx:
                r = A.add(lhs[i], A.neg(rhs[i]))
                if r: res.append(r)
    return res

def phi_tensors(geo, lam):
    il = A.inv(lam)
    dphi = [A.mul(il, A.d(b, lam)) for b in range(N)]
    Hess = [[A.add(A.d(b, dphi[d]), A.neg(A.add(*[A.mul(geo['Gam'][b, d, c], dphi[c]) for c in range(N)]))) for d in range(N)] for b in range(N)]
    dd = [[A.mul(dphi[b], dphi[d]) for d in range(N)] for b in range(N)]
    return evM(Hess), evM(dd)

def run_psi():
    g = metric_g
    lam = {(): 1 + x**2 + y, (0, 1): 1 + x}
    gp = [[A.mul(A.mul(lam, lam), g[i][j]) for j in range(N)] for i in range(N)]
    geo = geometry_odd(A, g); geop = geometry_odd(A, gp)
    print("geometries done", flush=True)
    s1 = lambda a, s: (-1)**(1 + P(a)); s2 = lambda a, s: (-1)**(P(a) + P(s))
    beta, bs = unknown_form('b'); betap, bps = unknown_form('c')
    Hess, dd = phi_tensors(geo, lam)
    for xv, yv, t in itertools.product((0, 1), (0, 1), (1, -1)):
        res = psi_test(geo, geop, g, lam, beta, betap, (s1, s2), xv, yv, t)
        eqs = [c for r in res for c in r.values()]
        sol = sp.solve(eqs, bps, dict=True)
        if not sol:
            print((xv, yv, t), "no solution", flush=True); continue
        sol = sol[0]
        if len(sol) < len(bps):
            print((xv, yv, t), "underdetermined", len(sol), flush=True)
        # Q = beta' - beta
        Q = [[A.clean({mo: sp.expand(sp.sympify(betap[a][s][mo]).subs(sol) - beta[a][s][mo]) for mo in betap[a][s]}) for s in range(N)] for a in range(N)]
        free = any(sym in sp.sympify(v).free_symbols for r in Q for e in r for v in e.values() for sym in bs)
        # fit Q = c1 Hess + c2 dd + c3 Hess^T-ish
        c1, c2 = sp.symbols('c1 c2')
        fit = [sp.expand(Q[a][s].get(mo, 0) - c1*Hess[a][s].get(mo, 0) - c2*dd[a][s].get(mo, 0))
               for a in range(N) for s in range(N) for mo in set(Q[a][s]) | set(Hess[a][s]) | set(dd[a][s])]
        fs = sp.solve(fit, [c1, c2], dict=True)
        print((xv, yv, t), "solved; Q depends on beta:", free, " fit Q = c1 Hess + c2 dphi dphi:", fs, flush=True)
        if fs:
            print("  Q symmetric:", all(A.add(Q[b][d], A.sc(-(-1)**(P(b)*P(d)), Q[d][b])) == {} for b in range(N) for d in range(N)), flush=True)

if __name__ == "__main__" and sys.argv[1:] == ["psi"]:
    run_psi()

def run_psi2():
    g = metric_g
    lam = {(): 1 + x**2 + y, (0, 1): 1 + x}
    gp = [[A.mul(A.mul(lam, lam), g[i][j]) for j in range(N)] for i in range(N)]
    geo = geometry_odd(A, g); geop = geometry_odd(A, gp)
    print("geometries done", flush=True)
    s1 = lambda a, s: (-1)**(1 + P(a)); s2 = lambda a, s: (-1)**(P(a) + P(s))
    beta, bs = unknown_form('b'); betap, bps = unknown_form('c')
    Hess, dd = phi_tensors(geo, lam)
    D, _ = make_D(geo, g, beta, (s1, s2)); Dp, _ = make_D(geop, gp, betap, (s1, s2))
    # unknown psi(q) = lam^{-1} q + sum K^c d_c + u p   (at the point)
    ks = []; K = []
    for c in range(N):
        mos = [(), (0, 1)] if (P(c)+1) % 2 == 0 else [(0,), (1,)]
        e = {}
        for mo in mos:
            t = sp.Symbol(f"K{c}_{''.join(map(str, mo)) or 'e'}"); ks.append(t); e[mo] = t
        K.append(e)
    u = {(0,): sp.Symbol('u0'), (1,): sp.Symbol('u1')}; ks += list(u.values())
    il = A.inv(lam)
    def pb(s):
        if s == 'p': return {'p': ev(lam)}
        if s == 'q':
            out = {'q': ev(il), 'p': u}
            for c in range(N): out[c] = K[c]
            return out
        return {s: ev(lam), 'p': ev(A.d(s, lam))}
    eqs = []
    for a in range(N):
        for s in ['p'] + list(range(N)):
            De = Dp(a, {s: A.const(1)}, {})
            lhs = {i: {} for i in idx}
            for j, f in De.items():
                if not f: continue
                for i, h in pb(j).items(): lhs[i] = A.add(lhs[i], A.mul(f, h))
            if s == 'p': sec = {'p': lam}
            else: sec = {s: lam, 'p': A.d(s, lam)}
            rhs = D(a, {i: ev(h) for i, h in sec.items()}, {i: ev(A.d(a, h)) for i, h in sec.items()})
            for i in idx:
                r = A.add(lhs[i], A.neg(rhs[i]))
                eqs += list(r.values())
    sol = sp.solve(eqs, bps + ks, dict=True)
    print("solutions:", len(sol), flush=True)
    if not sol: return
    sol = sol[0]
    print("u =", [sp.simplify(u[m].subs(sol)) for m in u], " K solved:", all(k in sol for k in ks), flush=True)
    Kv = [{mo: sp.sympify(v).subs(sol) for mo, v in K[c].items()} for c in range(N)]
    print("K =", Kv, flush=True)
    Gi = geo['Gi']
    for xv in (0, 1):
        v = [[A.sc((-1)**(xv*P(b)), A.d(b, lam)) for b in range(N)]]
        w = A.matmul(v, Gi)[0]
        for zv in (0, 1):
            G = [ev(A.sc((-1)**(zv*P(c)), A.mul(A.mul(il, il), w[c]))) for c in range(N)]
            for t in (1, -1):
                ok = all(A.add(A.sc(t, G[c]), A.neg(A.clean(Kv[c]))) == {} for c in range(N))
                if ok: print("K = t * lam^-2 * (-1)^{zv p(c)} (v Gi)_c with v_b=(-1)^{xv p(b)} d_b lam:", dict(xv=xv, zv=zv, t=t), flush=True)
    Q = [[A.clean({mo: sp.expand(sp.sympify(betap[a][s][mo]).subs(sol) - beta[a][s][mo]) for mo in betap[a][s]}) for s in range(N)] for a in range(N)]
    free = [sym for sym in bs if any(sym in sp.sympify(v).free_symbols for r in Q for e in r for v in e.values())]
    print("Q depends on beta:", free, flush=True)
    c1, c2 = sp.symbols('c1 c2')
    for nm, H in (('Hess', Hess), ('HessT', [[Hess[d][b] for d in range(N)] for b in range(N)])):
      for nm2, E in (('dd', dd), ('ddT', [[dd[d][b] for d in range(N)] for b in range(N)])):
        fit = [sp.expand(Q[a][s].get(mo, 0) - c1*H[a][s].get(mo, 0) - c2*E[a][s].get(mo, 0))
               for a in range(N) for s in range(N) for mo in set(Q[a][s]) | set(H[a][s]) | set(E[a][s])]
        print(nm, nm2, sp.solve(fit, [c1, c2], dict=True), flush=True)
    print("Q symmetric:", all(A.add(Q[b][d], A.sc(-(-1)**(P(b)*P(d)), Q[d][b])) == {} for b in range(N) for d in range(N)),
          " anti:", all(A.add(Q[b][d], A.sc((-1)**(P(b)*P(d)), Q[d][b])) == {} for b in range(N) for d in range(N)), flush=True)
    import pickle; pickle.dump(dict(Q=Q, Hess=Hess, dd=dd), open('oddlaw.pkl', 'wb'))

if __name__ == "__main__" and "psi2" in sys.argv:
    run_psi2()
