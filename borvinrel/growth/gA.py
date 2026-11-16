"""Root multiplicities of the contragredient Lie superalgebra g(B), B symmetric.

n_+ is built degree by degree: an element u of g_alpha (alpha>0) is identified
with its lowering vector (ad f_i u)_i in (+)_i g_{alpha-alpha_i}; this map is
injective on g(B) (the radical is exactly its kernel), so g_alpha = span of the
lowering vectors of the candidates [e_j, b], b a basis vector of g_{alpha-alpha_j}.
  ad f_i [e_j,v] = -delta_ij (-1)^{p_i} alpha'(h_i) v + (-1)^{p_i p_j} [e_j, ad f_i v],
alpha' = deg v, [h_i, x] = alpha'(h_i) x, alpha'(h_i) = sum_k m_k B[i][k].
Degree 0 (the Cartan) is treated by hand: ad f_i e_j = -delta_ij (-1)^{p_i} ... see code.
"""
import sys
from fractions import Fraction
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rref

def build(B, par, maxh, verbose=False):
    n = len(B)
    # store: deg tuple -> dict with 'basis': list of lowering vectors (unused beyond rank),
    #        'brk': dict (j, idx) -> coord vector of [e_j, basis_idx] in basis of deg+e_j
    dims = {}
    # coords of elements: vectors in basis of g_alpha.
    # height 1
    unit = lambda i: tuple(1 if t == i else 0 for t in range(n))
    for i in range(n):
        dims[unit(i)] = 1
    # E[(alpha, j)] = matrix: rows indexed by basis of g_alpha, giving coords of [e_j, b] in g_{alpha+e_j}
    E = {}
    # lowering of basis elements: low[alpha] = list over basis b of dict i -> coord vector in g_{alpha-e_i}
    low = {}
    for i in range(n):
        # e_i : ad f_k e_i = [f_k,e_i] = -(-1)^{p_k p_i} [e_i,f_k] = -(-1)^{p_i} delta h_i
        low[unit(i)] = [None]  # handled specially
    def bracket_e_with(j, alpha, vec):
        """coords of [e_j, sum vec_b b] in g_{alpha+e_j}; vec in basis of g_alpha."""
        beta = tuple(alpha[t] + (t == j) for t in range(n))
        if beta not in dims or dims[beta] == 0:
            return None
        M = E[(alpha, j)]
        out = [Fraction(0)] * dims[beta]
        for b, c in enumerate(vec):
            if c:
                for t, x in enumerate(M[b]):
                    if x:
                        out[t] += c * x
        return out
    def ht(a): return sum(a)
    def degpar(a): return sum(a[t] * par[t] for t in range(n)) % 2
    levels = {1: [unit(i) for i in range(n)]}
    # E at height 1 -> height 2 computed in loop
    for h in range(2, maxh + 1):
        # candidate degrees
        cands = {}
        for alpha in levels[h - 1]:
            for j in range(n):
                beta = tuple(alpha[t] + (t == j) for t in range(n))
                cands.setdefault(beta, []).append((alpha, j))
        levels[h] = []
        for beta, srcs in cands.items():
            # candidate list: (alpha, j, b)
            cand = []
            for (alpha, j) in srcs:
                for b in range(dims[alpha]):
                    cand.append((alpha, j, b))
            # lowering vectors: concatenated over i of coords in g_{beta - e_i}
            blocks = []
            for i in range(n):
                g = tuple(beta[t] - (t == i) for t in range(n))
                if min(g) < 0 or dims.get(g, 0) == 0:
                    continue
                blocks.append((i, g))
            rows = []
            for (alpha, j, b) in cand:
                row = []
                for (i, g) in blocks:
                    d = dims[g]
                    v = [Fraction(0)] * d
                    # term1: -delta_ij (-1)^{p_i} alpha(h_i) b   (lands in g_alpha = g_{beta-e_j})
                    if i == j:
                        ah = sum(alpha[k] * B[i][k] for k in range(n))
                        s = -1 if par[i] == 0 else 1
                        v[b] += s * ah
                    # term2: (-1)^{p_i p_j} [e_j, ad f_i b]
                    sign = -1 if (par[i] and par[j]) else 1
                    if ht(alpha) == 1:
                        # ad f_i e_b'(alpha=e_m) = -(-1)^{p_m} delta_im h_m ; [e_j, h_m] = -B[m][j] e_j
                        m = alpha.index(1)
                        if i == m:
                            # [e_j, -(-1)^{p_m} h_m] = (-1)^{p_m} B[m][j] e_j, lands in g_{e_j}
                            if g == unit(j):
                                v[0] += sign * (-1 if par[m] else 1) * B[m][j]
                    else:
                        gam = tuple(alpha[t] - (t == i) for t in range(n))
                        if min(gam) >= 0 and dims.get(gam, 0) > 0:
                            fv = low[alpha][b].get(i)
                            if fv is not None:
                                r = bracket_e_with(j, gam, fv)
                                if r is not None:
                                    for t, x in enumerate(r):
                                        v[t] += sign * x
                    row += v
                rows.append(row)
            # rank and coordinates
            if not rows or not rows[0]:
                dims[beta] = 0
                continue
            R, piv = rref(rows)  # rref of rows
            # choose independent candidates greedily
            basis_idx = []
            basisrows = []
            # greedy via incremental rref on transposed: simpler: pick rows one by one
            from exactlin import rank as rk
            cur = []
            for t, row in enumerate(rows):
                if rk(cur + [row]) > len(cur):
                    cur.append(row); basis_idx.append(t)
            d = len(cur)
            dims[beta] = d
            if d == 0:
                continue
            levels[h].append(beta)
            # coordinates of each candidate in the chosen basis: solve cur^T x = row
            from exactlin import solve
            T = [list(col) for col in zip(*cur)]  # len(row) x d
            low[beta] = []
            for t in basis_idx:
                (alpha, j, b) = cand[t]
                lv = {}
                pos = 0
                for (i, g) in blocks:
                    dd = dims[g]
                    lv[i] = rows[t][pos:pos + dd]; pos += dd
                low[beta].append(lv)
            for t, (alpha, j, b) in enumerate(cand):
                x = solve(T, rows[t])
                E.setdefault((alpha, j), [None] * dims[alpha])
                E[(alpha, j)][b] = x
        # fill E for (alpha,j) with zero target
        for alpha in levels[h - 1]:
            for j in range(n):
                beta = tuple(alpha[t] + (t == j) for t in range(n))
                if dims.get(beta, 0) == 0:
                    E[(alpha, j)] = [[] for _ in range(dims[alpha])]
                    dims.setdefault(beta, 0)
        if verbose:
            print(h, sum(dims[b] for b in levels[h]), flush=True)
    return dims, levels
