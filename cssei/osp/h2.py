"""Even H^2(s; s) of a Lie superalgebra s over F_2 with bracket and squaring (char 2):
infinitesimal deformations (c, q), c super-antisymmetric even bilinear, q: s_1 -> s_0 quadratic with
polar form c|_{s_1 x s_1}; cocycle = first-order Jacobi on distinct basis triples + first-order
[s(x), y] = [x,[x,y]] on odd basis x.  Coboundaries: phi even linear."""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from f2vf import kernel_bits, BitEchelon

def h2(tab, sq, par, verbose=True):
    d = len(par)
    E = [i for i in range(d) if par[i] == 0]; O = [i for i in range(d) if par[i] == 1]
    var = {}   # (a, b, k) with a <= b  -> index
    def nv(key):
        var[key] = len(var)
    for a in range(d):
        for b in range(a, d):
            if a == b and par[a] == 0: continue       # alternating on even
            for k in range(d):
                if (par[a] + par[b] + par[k]) % 2 == 0: nv((a, b, k))
    # c(a,b) as dict k -> bitmask over variables ; (a,a) odd is q(e_a) in the squaring slot, c(a,a) = 0
    def c(a, b):
        if a == b: return {}
        x, y = min(a, b), max(a, b)
        return {k: 1 << var[(x, y, k)] for k in range(d) if (x, y, k) in var}
    def q(a):
        return {k: 1 << var[(a, a, k)] for k in range(d) if (a, a, k) in var}
    def c_vec(a, w):   # c(e_a, sum w_l e_l)
        out = {}
        for l in range(d):
            if w[l]:
                for k, m in c(a, l).items(): out[k] = out.get(k, 0) ^ m
        return out
    def br_varvec(a, V):   # [e_a, sum_k V_k e_k], V_k bitmasks
        out = {}
        for k, m in V.items():
            if not m: continue
            vec = tab[a][k]
            for l in range(d):
                if vec[l]: out[l] = out.get(l, 0) ^ m
        return out
    def addto(acc, V):
        for k, m in V.items(): acc[k] = acc.get(k, 0) ^ m
    eqs = []
    for x, y, z in itertools.combinations(range(d), 3):
        acc = {}
        for (a, b, cc) in ((x, y, z), (y, z, x), (z, x, y)):
            addto(acc, c_vec(a, tab[b][cc]))
            addto(acc, br_varvec(a, c(b, cc)))
        eqs.extend(m for m in acc.values() if m)
    for i in O:
        for y in range(d):
            acc = {}
            # [q(e_i), y] = sum_k q_k [e_k, y]  -> -[y, q] ; char 2 symmetric: [e_k,y] = [y,e_k]
            addto(acc, br_varvec(y, q(i)))
            s = sq[i]
            addto(acc, {k: m for k, m in {}.items()})
            # c(s(e_i), y)
            for l in range(d):
                if s[l]:
                    addto(acc, c(l, y))
            addto(acc, c_vec(i, tab[i][y]))
            addto(acc, br_varvec(i, c(i, y)))
            eqs.extend(m for m in acc.values() if m)
    nvars = len(var)
    # solve: kernel of equations. transpose: image of each variable
    img = [0] * nvars
    for r, m in enumerate(eqs):
        while m:
            v = m.bit_length() - 1
            img[v] |= 1 << r
            m ^= 1 << v
    Z = kernel_bits(img)
    # coboundaries
    B = BitEchelon()
    def cob(l, k):     # phi: e_l -> e_k (same parity)
        vec = 0
        def put(a, b, kk):
            nonlocal vec
            if a == b: key = (a, a, kk)
            else: key = (min(a, b), max(a, b), kk)
            vec ^= 1 << var[key]
        for a in range(d):
            for b in range(a, d):
                if a == b and par[a] == 0: continue
                if a == b:   # q(e_a) = phi(s(e_a)) + [e_a, phi e_a]
                    if sq[a][l]: put(a, a, k)
                    if a == l:
                        w = tab[a][k]
                        for kk in range(d):
                            if w[kk]: put(a, a, kk)
                    continue
                # c(a,b) = phi([a,b]) + [phi a, b] + [a, phi b]
                if tab[a][b][l]: put(a, b, k)
                if a == l:
                    w = tab[k][b]
                    for kk in range(d):
                        if w[kk]: put(a, b, kk)
                if b == l:
                    w = tab[a][k]
                    for kk in range(d):
                        if w[kk]: put(a, b, kk)
        return vec
    zset = BitEchelon()
    for zv in Z: zset.add(zv)
    ok = True
    for l in range(d):
        for k in range(d):
            if par[l] == par[k]:
                v = cob(l, k)
                if zset.reduce(v): ok = False
                B.add(v)
    dimZ, dimB = len(Z), len(B)
    if verbose:
        print('  cochains %d, equations %d, dim Z^2 %d, dim B^2 %d (coboundaries are cocycles: %s), dim H^2_ev %d'
              % (nvars, len(eqs), dimZ, dimB, ok, dimZ - dimB))
    return dimZ - dimB, Z, B, var

if __name__ == '__main__':
    from galg import GFField
    from msa import MSA
    F = GFField(2)
    S = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 's_gs2.json')))
    for t in ('inner', 'outer'):
        A = MSA(F, S[t]['even'], S[t]['odd'])
        print('gs2', t, 'sdim', (A.par.count(0), A.par.count(1)))
        h2([[[x & 1 for x in v] for v in row] for row in A.tab], [None if s is None else [x & 1 for x in s] for s in A.sq], A.par)
