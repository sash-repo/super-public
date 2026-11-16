"""The W-grading defined by a maximal subalgebra L_0 = n_L(s):
L_{-1} = a minimal L_0-invariant subspace containing L_0 properly (all of them are tried,
the dims reported), L_{-i-1} = L_{-i} + [L_{-1}, L_{-i}], L_{i+1} = {x in L_i | [x, L_{-1}] in L_i}.
Output: dims of g_i = L_i / L_{i+1}, i = -d, ..., top."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sand import Alg, Span, reduce, kernel


def close_under(A, M, L0b):
    S = Span(M)
    queue = list(S.basis())
    while queue:
        a = queue.pop()
        for b in L0b:
            w = A.br(a, b)
            if S.add(w):
                queue.append(w)
    return S


def wgrading(A, L0):
    L0b = L0.basis()
    comp = []
    T = Span(L0b)
    for j in range(A.n):
        if T.add(1 << j):
            comp.append(1 << j)
    k = len(comp)
    minimal = None
    for c in range(1, 1 << k):
        u = 0
        for i in range(k):
            if (c >> i) & 1:
                u ^= comp[i]
        M = close_under(A, L0b + [u], L0b)
        if minimal is None or len(M) < len(minimal):
            minimal = M
    Lm1 = minimal
    neg = [len(Lm1)]
    cur = Lm1
    while len(cur) < A.n:
        S = Span(cur.basis())
        for a in Lm1.basis():
            for b in cur.basis():
                S.add(A.br(a, b))
        if len(S) == len(cur):
            break
        cur = S
        neg.append(len(cur))
    pos = [len(L0)]
    Li = L0
    while len(Li):
        # x in Li with [x, Lm1] in Li: linear conditions
        bs = Li.basis()
        imgs = []
        for x in bs:
            v = 0
            sh = 0
            for y in Lm1.basis():
                v |= reduce(A.br(x, y), Li.piv) << sh
                sh += A.n
            imgs.append(v)
        kc = kernel(imgs, len(bs))
        nxt = []
        for c in kc:
            v = 0
            for i in range(len(bs)):
                if (c >> i) & 1:
                    v ^= bs[i]
            nxt.append(v)
        N = Span(nxt)
        if len(N) == len(Li):
            break
        pos.append(len(N))
        Li = N
    dims_neg = [neg[i] - (neg[i - 1] if i else len(L0)) for i in range(len(neg))][::-1]
    dims_pos = [pos[i] - (pos[i + 1] if i + 1 < len(pos) else 0) for i in range(len(pos))]
    return dims_neg, dims_pos
