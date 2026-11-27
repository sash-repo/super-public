"""Super check of Miklashevskii's Prop. 4.1: on a supermanifold of sdim (n|k) with an even metric g,
L_g = Rp + TM + Rq with connection (4.1), rho^0 = (rho - S/(2N-2) g)/(N-2), N = n-k = str(id).
Test: psi: L_{lambda^2 g} -> L_g of (4.2) intertwines the connections.  Exact, sympy."""
import sys, itertools, sympy as sp

class Alg:
    def __init__(self, evars, nodd):
        self.ev = evars; self.n = len(evars); self.k = nodd; self.N = self.n + nodd
    def par(self, a): return 0 if a < self.n else 1
    # functions: dict sorted-odd-tuple -> sympy expr
    def mul(self, f, g):
        R = {}
        for a, ca in f.items():
            for b, cb in g.items():
                if set(a) & set(b): continue
                lst = list(a) + list(b); s = 1
                for i in range(len(lst)):
                    for j in range(len(lst)-1-i):
                        if lst[j] > lst[j+1]: lst[j], lst[j+1] = lst[j+1], lst[j]; s = -s
                t = tuple(lst); R[t] = R.get(t, 0) + s*ca*cb
        return self.clean(R)
    def clean(self, f):
        out = {}
        for k, v in f.items():
            v = sp.cancel(sp.together(v))
            if v != 0: out[k] = v
        return out
    def add(self, *fs):
        R = {}
        for f in fs:
            for k, v in f.items(): R[k] = R.get(k, 0) + v
        return self.clean(R)
    def sc(self, c, f): return self.clean({k: c*v for k, v in f.items()})
    def neg(self, f): return self.sc(-1, f)
    def sgn(self, pa, f):   # (-1)^{pa * p(f)} termwise
        return {k: (v if (pa*len(k)) % 2 == 0 else -v) for k, v in f.items()}
    def d(self, a, f):
        R = {}
        for mono, c in f.items():
            if a < self.n:
                R[mono] = R.get(mono, 0) + sp.diff(c, self.ev[a])
            else:
                j = a - self.n
                if j in mono:
                    pos = mono.index(j); m2 = mono[:pos] + mono[pos+1:]
                    R[m2] = R.get(m2, 0) + c*(-1)**pos
        return self.clean(R)
    def const(self, c): return self.clean({(): sp.sympify(c)})
    def body(self, f): return f.get((), 0)
    def inv(self, f):     # even invertible function
        b = self.body(f); nil = self.add(f, self.const(-b)); u = self.sc(1/b, nil)
        res = self.const(1); term = self.const(1)
        for _ in range(self.k + 1):
            term = self.neg(self.mul(term, u)); res = self.add(res, term)
        return self.sc(1/b, res)
    def matmul(self, A, B):
        return [[self.add(*[self.mul(A[i][l], B[l][j]) for l in range(len(B))]) for j in range(len(B[0]))] for i in range(len(A))]
    def matinv(self, G):
        N = len(G); G0 = sp.Matrix(N, N, lambda i, j: self.body(G[i][j])); G0i = G0.inv()
        G0iM = [[self.const(G0i[i, j]) for j in range(N)] for i in range(N)]
        Nil = [[self.add(G[i][j], self.const(-G0[i, j])) for j in range(N)] for i in range(N)]
        U = self.matmul(G0iM, Nil)
        res = G0iM; term = G0iM
        for _ in range(self.k + 1):
            term = [[self.neg(x) for x in r] for r in self.matmul(U, term)]
            res = [[self.add(res[i][j], term[i][j]) for j in range(N)] for i in range(N)]
        return res

def geometry(A, g, ric_sign, S_sign):
    N = A.N; P = A.par
    Gi = A.matinv(g)
    # Christoffel 1st kind via super Koszul formula, then check
    Gam1 = {}
    for a, b, c in itertools.product(range(N), repeat=3):
        t1 = A.d(a, g[b][c])
        t2 = A.sc((-1)**(P(a)*P(b)), A.d(b, g[a][c]))
        t3 = A.sc(-(-1)**(P(c)*(P(a)+P(b))), A.d(c, g[a][b]))
        Gam1[a, b, c] = A.sc(sp.Rational(1, 2), A.add(t1, t2, t3))
    Gam = {}
    for a, b in itertools.product(range(N), repeat=2):
        row = [[Gam1[a, b, c] for c in range(N)]]
        r = A.matmul(row, Gi)[0]
        for e in range(N): Gam[a, b, e] = r[e]
    # checks: torsion-free, metric
    for a, b, e in itertools.product(range(N), repeat=3):
        assert A.add(Gam[a, b, e], A.sc(-(-1)**(P(a)*P(b)), Gam[b, a, e])) == {}, "torsion"
    for a, b, c in itertools.product(range(N), repeat=3):
        rhs = A.add(*[A.mul(Gam[a, b, d], g[d][c]) for d in range(N)],
                    *[A.sc((-1)**(P(a)*P(b)), A.mul(A.sgn(P(b), Gam[a, c, d]), g[b][d])) for d in range(N)])
        assert A.add(A.d(a, g[b][c]), A.neg(rhs)) == {}, "metric"
    def nabla(a, vec):  # vec: list of coefficient functions on d_e
        out = [A.d(a, vec[e]) for e in range(N)]
        for d in range(N):
            f = A.sgn(P(a), vec[d])
            for e in range(N): out[e] = A.add(out[e], A.mul(f, Gam[a, d, e]))
        return out
    R = {}
    for a, b, c in itertools.product(range(N), repeat=3):
        v1 = nabla(a, [Gam[b, c, e] for e in range(N)]); v2 = nabla(b, [Gam[a, c, e] for e in range(N)])
        s = (-1)**(P(a)*P(b))
        for e in range(N): R[a, b, c, e] = A.add(v1[e], A.sc(-s, v2[e]))
    Ric = [[A.add(*[A.sc(ric_sign(P(a), P(b), P(c)), R[a, b, c, a]) for a in range(N)]) for c in range(N)] for b in range(N)]
    r = A.matmul(Ric, Gi)
    S = A.add(*[A.sc(S_sign(P(b)), r[b][b]) for b in range(N)])
    return dict(Gi=Gi, Gam=Gam, Ric=Ric, S=S, nabla=nabla)

def tractor(A, g, geo):
    N = A.N; Ns = sum(1 if A.par(i) == 0 else -1 for i in range(N))
    rho0 = [[A.sc(sp.Rational(1, Ns-2), A.add(geo['Ric'][a][b], A.sc(-sp.Rational(1, 2*Ns-2), A.mul(geo['S'], g[a][b]))))
             for b in range(N)] for a in range(N)]
    r0 = A.matmul(rho0, geo['Gi'])
    idx = ['p'] + list(range(N)) + ['q']
    def Dbasis(a, s):
        out = {i: {} for i in idx}
        if s == 'p': out[a] = A.const(1)
        elif s == 'q':
            for c in range(N): out[c] = r0[a][c]
        else:
            out['p'] = A.neg(rho0[a][s]); out['q'] = A.neg(g[a][s])
            for c in range(N): out[c] = geo['Gam'][a, s, c]
        return out
    def D(a, sec):
        out = {i: {} for i in idx}
        for s, f in sec.items():
            if not f: continue
            df = A.d(a, f)
            if df: out[s] = A.add(out[s], df)
            fs = A.sgn(A.par(a), f); Db = Dbasis(a, s)
            for i in idx: out[i] = A.add(out[i], A.mul(fs, Db[i]))
        return out
    return D, idx, rho0

def test(A, g, lam, ric_sign, S_sign):
    N = A.N
    gp = [[A.mul(A.mul(lam, lam), g[i][j]) for j in range(N)] for i in range(N)]
    geo = geometry(A, g, ric_sign, S_sign); geop = geometry(A, gp, ric_sign, S_sign)
    # Ric symmetry
    sym = all(A.add(geo['Ric'][b][c], A.sc(-(-1)**(A.par(b)*A.par(c)), geo['Ric'][c][b])) == {} for b in range(N) for c in range(N))
    D, idx, _ = tractor(A, g, geo); Dp, _, _ = tractor(A, gp, geop)
    il = A.inv(lam); dl = [[A.d(c, lam) for c in range(N)]]
    G = A.matmul(dl, geo['Gi'])[0]
    norm = A.add(*[A.mul(G[c], A.d(c, lam)) for c in range(N)])
    def psi_basis(s):
        out = {i: {} for i in idx}
        if s == 'p': out['p'] = lam
        elif s == 'q':
            out['q'] = il
            for c in range(N): out[c] = A.neg(A.mul(A.mul(il, il), G[c]))
            out['p'] = A.neg(A.sc(sp.Rational(1, 2), A.mul(A.mul(A.mul(il, il), il), norm)))
        else:
            out[s] = lam; out['p'] = A.d(s, lam)
        return out
    def psi(sec):
        out = {i: {} for i in idx}
        for s, f in sec.items():
            if not f: continue
            pb = psi_basis(s)
            for i in idx: out[i] = A.add(out[i], A.mul(f, pb[i]))
        return out
    bad = 0; cnt = 0
    for a in range(N):
        for s in idx:
            lhs = psi(Dp(a, {s: A.const(1)})); rhs = D(a, psi_basis(s))
            for i in idx:
                cnt += 1
                if A.add(lhs[i], A.neg(rhs[i])) != {}: bad += 1
    return sym, bad, cnt

if __name__ == "__main__":
    x, y, z = sp.symbols('x y z')
    signs = {
      'A': (lambda pa, pb, pc: (-1)**pa, lambda pb: (-1)**pb),
      'B': (lambda pa, pb, pc: (-1)**(pa*(1+pb+pc)), lambda pb: (-1)**pb),
      'C': (lambda pa, pb, pc: (-1)**(pa*(pb+pc)), lambda pb: (-1)**pb),
      'D': (lambda pa, pb, pc: 1, lambda pb: 1),
    }
    cases = []
    A3 = Alg([x, y, z], 0)
    g3 = [[A3.const(1+x**2), A3.const(x), A3.const(0)], [A3.const(x), A3.const(1+y**2), A3.const(0)], [A3.const(0), A3.const(0), A3.const(2+z*x)]]
    cases.append(("(3|0)", A3, g3, A3.const(1+x**2+y*z)))
    A12 = Alg([x], 2)
    g12 = [[{(): 1+x**2, (0, 1): x}, {(0,): x, (1,): sp.Integer(1)}, {(0,): sp.Integer(1)}],
           [{(0,): x, (1,): sp.Integer(1)}, {}, {(): 1+x, (0, 1): x**2}],
           [{(0,): sp.Integer(1)}, {(): -(1+x), (0, 1): -x**2}, {}]]
    cases.append(("(1|2)", A12, g12, {(): 1+x**2, (0, 1): 2+x}))
    which = sys.argv[1:] or list(signs)
    for name, A, g, lam in cases:
        for sname in which:
            rs, Ss = signs[sname]
            if name == "(3|0)": continue
            sym, bad, cnt = test(A, g, lam, rs, Ss)
            print(f"{name} signs {sname}: Ric symmetric={sym}; psi intertwines: {cnt-bad}/{cnt} components agree", flush=True)
