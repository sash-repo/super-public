"""Ambient Lie algebras over F_2 as matrix algebras, with random-element samplers.
  sl(n)      in gl(n): sparse sums of E_ab (a != b) and E_aa + E_bb;
  W(m;1)     in gl(2^m), acting on O(m;1) = divided powers, basis x^(a) d_i.
Samplers: 'sparse' (a sum of k random basis elements, k random), 'graded' for W:
x = sum of all d_i (or a random nonzero part of g_{-1}), y = random of degree >= 1."""
import numpy as np
from lie import MatAmb

def sl(n):
    amb = MatAmb(n, 'sl(%d)' % n)
    basis = []
    for a in range(n):
        for b in range(n):
            if a != b:
                m = np.zeros((n, n), np.uint8); m[a, b] = 1; basis.append(m.ravel())
    for a in range(n - 1):
        m = np.zeros((n, n), np.uint8); m[a, a] = m[a + 1, a + 1] = 1; basis.append(m.ravel())
    amb.basis = np.array(basis, np.uint8)
    amb.deg = None
    return amb

def witt(m):
    N = 1 << m
    amb = MatAmb(N, 'W(%d;1)' % m)
    basis, deg = [], []
    for a in range(N):
        for i in range(m):
            M = np.zeros((N, N), np.uint8)
            for b in range(N):
                if b >> i & 1:
                    bp = b ^ (1 << i)
                    if a & bp == 0:
                        M[a | bp, b] = 1
            basis.append(M.ravel()); deg.append(bin(a).count('1') - 1)
    amb.basis = np.array(basis, np.uint8)
    amb.deg = np.array(deg)
    return amb

def sparse(amb, rng, kmax=4):
    k = int(rng.integers(1, kmax + 1))
    idx = rng.choice(len(amb.basis), size=k, replace=False)
    return (amb.basis[idx].sum(axis=0) & 1).astype(np.uint8)

def graded_pair(amb, rng, kmax=4):
    """For W(m;1): g_{-1} part (random nonzero subset of the d_i) + one random element of degree >= 1."""
    neg = np.nonzero(amb.deg == -1)[0]
    pos = np.nonzero(amb.deg >= 1)[0]
    s = rng.integers(0, 2, len(neg))
    if not s.any(): s[0] = 1
    x = (amb.basis[neg[s == 1]].sum(axis=0) & 1).astype(np.uint8)
    k = int(rng.integers(1, kmax + 1))
    y = (amb.basis[rng.choice(pos, size=k, replace=False)].sum(axis=0) & 1).astype(np.uint8)
    return [x, y]

def wittN(Ns):
    """W(m;N) acting on O(m;N) (divided powers, x_i^(a_i) with a_i < 2^{N_i});
    exponent vectors packed, coordinate i in N_i bits; x^(a) x^(b) = x^(a|b) if a & b == 0."""
    m = len(Ns)
    off = [sum(Ns[:i]) for i in range(m)]
    N = 1 << sum(Ns)
    amb = MatAmb(N, 'W(%d;(%s))' % (m, ','.join(map(str, Ns))))
    def field(a, i):
        return (a >> off[i]) & ((1 << Ns[i]) - 1)
    basis, deg = [], []
    for a in range(N):
        for i in range(m):
            M = np.zeros((N, N), np.uint8)
            for b in range(N):
                if field(b, i) > 0:
                    bp = b - (1 << off[i])
                    if a & bp == 0:
                        M[a | bp, b] = 1
            basis.append(M.ravel()); deg.append(sum(field(a, j) for j in range(m)) - 1)
    amb.basis = np.array(basis, np.uint8)
    amb.deg = np.array(deg)
    return amb
