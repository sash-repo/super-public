#!/usr/bin/env python3
"""Real (almost real-complex) version of the structure functions for the
degenerate Ramond symbol m_0, after BGLS arXiv:1010.4480, Sect. 3.3:
h_0 = g_0 cap aut(J), circumcised Nijenhuis tensor = H^2(g_-^R; g_*^R).

Complexified, g_-^R (x) C = g_- (+) conj(g_-) and the prolong of (g_-^R, h_0)
complexifies to g_* (+) conj(g_*); so H^2(g_-^R; g_*^R) (x) C =
H^2(L; g_*) (+) conj, L = g_- (+) gbar_-, gbar_- acting trivially on g_*.
This file computes H^q(L; M)_d DIRECTLY (generic super CE complex from
structure constants, Q^2 = 0 asserted) and also H^q(g_-; g_*), H^q(g_-; C),
to compare with the Kunneth decomposition."""
import sys, itertools
from fractions import Fraction as Fr
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import rank
from rc import Distr, vzero, fadd, fmul, vpar, vbr, coords

def model(n, kind):
    frame = []
    m = n if kind == 'NS' else n-1
    for i in range(1, m+1):
        L = vzero(n); L[i] = {(0, ()): Fr(1)}; L[0] = {(0, (i,)): Fr(-1)}; frame.append(L)
    if kind == 'R0':
        L = vzero(n); L[n] = {(0, ()): Fr(1)}; frame.append(L)
    def alpha(Y):
        a = dict(Y[0])
        for i in range(1, m+1):
            a = fadd(a, fmul(Y[i], {(0, (i,)): Fr(1)}))
        return a
    return Distr(n, frame, alpha)

def extract(n, kind, K):
    """structure data: L-basis (deg, par), brackets; M-basis (deg, par), action."""
    D = model(n, kind)
    gm = [(k, b) for k in (-2, -1) for b in D.comp(k)]
    xs = [b for _, b in gm]
    Lb = [(k, vpar(b, n)) for k, b in gm]
    br = {}
    for i in range(len(xs)):
        for j in range(len(xs)):
            v = coords(vbr(xs[i], xs[j], n), xs, n)
            br[i, j] = {k: c for k, c in enumerate(v) if c}
    Mb, Mx = [], []
    for k in range(-2, K+1):
        for b in D.comp(k):
            Mb.append((k, vpar(b, n))); Mx.append(b)
    bydeg = {}
    for a, (k, p) in enumerate(Mb): bydeg.setdefault(k, []).append(a)
    act = {}
    for i, x in enumerate(xs):
        for a, m in enumerate(Mx):
            tg = Mb[a][0] + Lb[i][0]
            if tg < -2: act[i, a] = {}; continue
            idx = bydeg[tg]
            v = coords(vbr(x, m, n), [Mx[q] for q in idx], n)
            act[i, a] = {idx[q]: c for q, c in enumerate(v) if c}
    return Lb, br, Mb, act

def double(Lb, br, act, nM):
    """L (+) Lbar, Lbar acting trivially on M."""
    N = len(Lb)
    Lb2 = Lb + Lb
    br2 = dict(br)
    for (i, j), v in br.items():
        br2[i+N, j+N] = {k+N: c for k, c in v.items()}
    for i in range(N):
        for j in range(N):
            br2[i, j+N] = {}; br2[i+N, j] = {}
    act2 = dict(act)
    for i in range(N):
        for a in range(nM): act2[i+N, a] = {}
    return Lb2, br2, act2

class CE:
    """C = Lambda(c) (x) M, c^i of parity p_i+1, deg c^i = -deg x_i.
    Element: dict {(mono, a): coef}, mono = sorted tuple of ghost indices
    (repeats allowed only for even ghosts, i.e. odd x_i)."""
    def __init__(self, Lb, br, Mb, act, s=-1):
        self.Lb, self.br, self.Mb, self.act, self.s = Lb, br, Mb, act, s
        self.gp = [(p+1) % 2 for _, p in Lb]
    def norm(self, mono):  # sort ghosts, return (mono, sign) or (None,0)
        m = list(mono); sg = 1
        for i in range(len(m)):
            for j in range(len(m)-1-i):
                if m[j] > m[j+1]:
                    if self.gp[m[j]] and self.gp[m[j+1]]: sg = -sg
                    m[j], m[j+1] = m[j+1], m[j]
        for j in range(len(m)-1):
            if m[j] == m[j+1] and self.gp[m[j]]: return None, 0
        return tuple(m), sg
    def mpar(self, mono): return sum(self.gp[i] for i in mono) % 2
    def Qghost(self, k):  # Q(c^k) = s/2 sum_ij f^k_ij c^i c^j
        out = {}
        for (i, j), v in self.br.items():
            if k in v:
                mono, sg = self.norm((i, j))
                if sg: out[mono] = out.get(mono, 0) + Fr(self.s, 2)*v[k]*sg
        return {m: c for m, c in out.items() if c}
    def Q(self, el):
        out = {}
        def add(key, c):
            out[key] = out.get(key, 0) + c
        for (mono, a), c in el.items():
            # Q(c^{i1}...c^{ir}) m = sum Leibniz + (-1)^{|mono|} mono Q(m)
            for pos, k in enumerate(mono):
                pre = mono[:pos]; post = mono[pos+1:]
                sg0 = (-1)**self.mpar(pre)
                for qm, qc in self.Qghost(k).items():
                    nm, sg = self.norm(pre + qm + post)
                    if sg: add((nm, a), c*sg0*sg*qc)
            sg0 = (-1)**self.mpar(mono)
            pa = self.Mb[a][1]
            for i in range(len(self.Lb)):
                # Q(m) = sum_i c^i rho_i(m); mono c^i = sign * sorted
                for b, v in self.act[i, a].items():
                    nm, sg = self.norm(mono + (i,))
                    # move c^i to the left of nothing: Q(m)=c^i rho_i m, then mono * c^i
                    if sg: add((nm, b), c*sg0*sg*v)
        return {k: v for k, v in out.items() if v}
    def basis(self, q, d):
        N = len(self.Lb); out = []
        for mono in itertools.combinations_with_replacement(range(N), q):
            if any(mono[j] == mono[j+1] and self.gp[mono[j]] for j in range(q-1)): continue
            cd = -sum(self.Lb[i][0] for i in mono)
            for a, (k, p) in enumerate(self.Mb):
                if cd + k == d: out.append((mono, a))
        return out
    def par(self, key):
        mono, a = key
        return (self.mpar(mono) + self.Mb[a][1]) % 2
    def dmat(self, q, d, parity):
        src = [b for b in self.basis(q, d) if self.par(b) == parity]
        tgt = [b for b in self.basis(q+1, d) if (self.par(b)+1) % 2 == parity]
        ti = {b: k for k, b in enumerate(tgt)}
        rows = []
        for b in src:
            img = self.Q({b: Fr(1)})
            r = [Fr(0)]*len(tgt)
            for key, v in img.items():
                r[ti[key]] += v
            rows.append(r)
        return src, tgt, rows
    def H(self, q, d):
        """dims of H^q_d split by cochain parity (parity of the element of C)."""
        res = {}
        for p in (0, 1):
            src, tgt, A = self.dmat(q, d, p)
            rA = rank(A) if A and tgt else 0
            if q > 0:
                s2, t2, B = self.dmat(q-1, d, (p+1) % 2)
                rB = rank(B) if B and t2 else 0
            else: rB = 0
            res[p] = len(src) - rA - rB
        return res
    def checkQ2(self, q, d):
        for b in self.basis(q, d):
            assert not self.Q(self.Q({b: Fr(1)})), ("Q^2 != 0", b)

def trivial(Lb, br):
    return [(0, 0)], {(i, 0): {} for i in range(len(Lb))}
