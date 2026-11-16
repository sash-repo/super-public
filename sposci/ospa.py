"""Reducibility of the modules T(K_c) over osp_a(4|2)
in the depth-1 grading with sdim g_{-1}=(2|2), z=(h2+h3)/2.

osp_sigma(4|2) on sl2^3 + V1(x)V2(x)V3, sigma1+sigma2+sigma3=0; copy 1 is the one inside g_0.
T(K_c) is the restricted dual of the generalized Verma module M(-c) induced from the
character of p=g_0+g_1 (no shift: str ad_z on g_{-1} is 0), so T(K_c) is reducible iff
M(-c) has a nonzero g_1-singular vector of positive degree.
"""
import sys, itertools, random
from fractions import Fraction as F
import sympy as sp

def build(sig):
    # basis: even e_i,h_i,f_i (i=0,1,2) ; odd v_s, s in {+1,-1}^3
    names = []
    par = []
    for i in range(3):
        for nm in 'ehf':
            names.append((nm, i)); par.append(0)
    odd = list(itertools.product([1, -1], repeat=3))
    for s in odd:
        names.append(('v', s)); par.append(1)
    idx = {n: k for k, n in enumerate(names)}
    N = len(names)
    # sl2 action on V=span(u+,u-): e u- = u+, f u+ = u-, h u+- = +-u+-
    def act(nm, s):  # returns list of (coef, s')
        if nm == 'e':
            return [(1, 1)] if s == -1 else []
        if nm == 'f':
            return [(1, -1)] if s == 1 else []
        return [(s, s)]
    def om(s, t):  # symplectic form om(u+,u-)=1
        if s == 1 and t == -1: return 1
        if s == -1 and t == 1: return -1
        return 0
    # P(u,w) in sl2: P(u,w)x = om(w,x)u + om(u,x)w ; expressed in e,h,f
    def P(s, t):
        # compute matrix of x -> om(t,x)u_s + om(s,x)u_t on basis u+,u-
        M = {}
        for x in (1, -1):
            for (c, y) in ((om(t, x), s), (om(s, x), t)):
                if c: M[(y, x)] = M.get((y, x), 0) + c
        # e = E_{+-}, f=E_{-+}, h=E_{++}-E_{--}
        out = {}
        if M.get((1, -1)): out['e'] = M[(1, -1)]
        if M.get((-1, 1)): out['f'] = M[(-1, 1)]
        a, d = M.get((1, 1), 0), M.get((-1, -1), 0)
        assert a == -d
        if a: out['h'] = a
        return out
    br = {}
    sl2 = {('e', 'f'): {'h': 1}, ('h', 'e'): {'e': 2}, ('h', 'f'): {'f': -2}}
    def setb(x, y, d):
        br[(x, y)] = d
    for i in range(3):
        for (a, b), d in sl2.items():
            setb(idx[(a, i)], idx[(b, i)], {idx[(k, i)]: F(v) for k, v in d.items()})
            setb(idx[(b, i)], idx[(a, i)], {idx[(k, i)]: F(-v) for k, v in d.items()})
        for nm in 'ehf':
            for s in odd:
                d = {}
                for c, t in act(nm, s[i]):
                    s2 = list(s); s2[i] = t
                    d[idx[('v', tuple(s2))]] = d.get(idx[('v', tuple(s2))], 0) + F(c)
                setb(idx[(nm, i)], idx[('v', s)], d)
                setb(idx[('v', s)], idx[(nm, i)], {k: -v for k, v in d.items()})
    for s in odd:
        for t in odd:
            d = {}
            for i in range(3):
                j, k = [m for m in range(3) if m != i]
                c = sig[i] * om(s[j], t[j]) * om(s[k], t[k])
                if c:
                    for nm, v in P(s[i], t[i]).items():
                        d[idx[(nm, i)]] = d.get(idx[(nm, i)], 0) + F(c) * v
            setb(idx[('v', s)], idx[('v', t)], {k: v for k, v in d.items() if v})
    def bracket(x, y):
        return br.get((x, y), {})
    return names, par, idx, bracket, N

def jacobi(names, par, bracket, N):
    def b(u, v):  # u,v dicts
        out = {}
        for x, a in u.items():
            for y, bb in v.items():
                for z, c in bracket(x, y).items():
                    out[z] = out.get(z, 0) + a * bb * c
        return {k: v for k, v in out.items() if v}
    bad = 0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                X, Y, Z = {x: 1}, {y: 1}, {z: 1}
                lhs = b(X, b(Y, Z))
                r1 = b(b(X, Y), Z)
                r2 = b(Y, b(X, Z))
                sg = (-1) ** (par[x] * par[y])
                tot = dict(lhs)
                for k, v in r1.items(): tot[k] = tot.get(k, 0) - v
                for k, v in r2.items(): tot[k] = tot.get(k, 0) - sg * v
                if any(v for v in tot.values()):
                    bad += 1
    return bad

def setup(sig):
    names, par, idx, bracket, N = build(sig)
    # z = (h2+h3)/2 -> copies 1,2 in 0-based
    def zev(n):
        nm, i = n
        if nm in 'ehf':
            if i == 0: return 0
            return {'e': 1, 'h': 0, 'f': -1}[nm]
        return F(n[1][1] + n[1][2], 2)
    deg = [zev(n) for n in names]
    gm = [k for k in range(N) if deg[k] == -1]
    g0 = [k for k in range(N) if deg[k] == 0]
    g1 = [k for k in range(N) if deg[k] == 1]
    return names, par, idx, bracket, N, deg, gm, g0, g1

def character(names, idx, bracket, g0, c):
    """chi on g_0: zero on [g0,g0], chi(z)=c, z=(h2+h3)/2. Returns dict basis->value."""
    # [g0,g0] span
    rows = []
    for x in g0:
        for y in g0:
            d = bracket(x, y)
            if d:
                rows.append([d.get(k, 0) for k in g0])
    M = sp.Matrix(rows)
    # chi is a row vector killing rows; solve chi . r = 0 for all r, chi(z)=c
    ns = M.nullspace()  # vectors v with M v = 0 -> chi coefficients
    assert len(ns) == 1, len(ns)
    v = ns[0]
    zvec = [0] * len(g0)
    zvec[g0.index(idx[('h', 1)])] = F(1, 2)
    zvec[g0.index(idx[('h', 2)])] = F(1, 2)
    zval = sum(v[i] * zvec[i] for i in range(len(g0)))
    assert zval != 0, "z lies in [g0,g0]"
    return {g0[i]: v[i] * c / zval for i in range(len(g0))}

class Verma:
    def __init__(self, S, chi):
        (self.names, self.par, self.idx, self.bracket, self.N, self.deg,
         self.gm, self.g0, self.g1) = S
        self.chi = chi
        self.order = {k: n for n, k in enumerate(self.gm)}
    def mul_left(self, y, mono):
        """y in g_- times PBW monomial (sorted tuple) -> dict mono->coef"""
        # supercommutative g_- ; insert y with sign
        lst = list(mono)
        sign = 1
        pos = 0
        while pos < len(lst) and self.order[lst[pos]] < self.order[y]:
            if self.par[y] and self.par[lst[pos]]:
                sign = -sign
            pos += 1
        if pos < len(lst) and lst[pos] == y and self.par[y]:
            return {}
        lst.insert(pos, y)
        return {tuple(lst): sign}
    def act(self, x, mono):
        """x in g (basis index) acting on mono.v ; returns dict"""
        if not mono:
            if x in self.gm:
                return {(x,): 1}
            if x in self.g0:
                v = self.chi.get(x, 0)
                return {(): v} if v != 0 else {}
            return {}
        y, rest = mono[0], mono[1:]
        out = {}
        # x y rest = [x,y] rest + (-1)^{p x p y} y x rest
        for z, c in self.bracket(x, y).items():
            for m, a in self.act(z, rest).items():
                out[m] = out.get(m, 0) + c * a
        sg = (-1) ** (self.par[x] * self.par[y])
        for m, a in self.act(x, rest).items():
            for m2, b in self.mul_left(y, m).items():
                out[m2] = out.get(m2, 0) + sg * a * b
        return {k: v for k, v in out.items() if v != 0}
    def basis(self, k):
        ev = [y for y in self.gm if self.par[y] == 0]
        od = [y for y in self.gm if self.par[y] == 1]
        out = []
        for i in range(0, min(k, len(od)) + 1):
            for oc in itertools.combinations(od, i):
                for ec in itertools.combinations_with_replacement(ev, k - i):
                    out.append(tuple(sorted(ec + oc, key=lambda t: self.order[t])))
        return out

def singular_cs(sig, kmax, csym):
    S = setup(sig)
    chi = character(S[0], S[2], S[3], S[7], csym)
    V = Verma(S, chi)
    res = {}
    for k in range(1, kmax + 1):
        B = V.basis(k)
        Bm = V.basis(k - 1)
        pos = {m: i for i, m in enumerate(Bm)}
        rows = []
        for x in V.g1:
            blk = [[0] * len(B) for _ in Bm]
            for j, m in enumerate(B):
                for m2, a in V.act(x, m).items():
                    blk[pos[m2]][j] += a
            rows.extend(blk)
        A = sp.Matrix(rows)
        res[k] = (A, len(B))
    return res, V

if __name__ == '__main__':
    pass
