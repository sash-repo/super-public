"""Hamiltonian Lie algebras h(omega) in 4 indeterminates over F_2, divided powers O(4;N).
Forms: omega0, omega1 (two readings), omega_{2,B}. Computes h, h', h'', simplicity (dim of
associative envelope of ad = n^2), dim Der/ad, dim H^2(g;K) with alternating cochains,
ad-rank distribution (dim 14 only).  Single log writer: main()."""
import os, sys, itertools, random
from math import comb

M = 4
def setup(N):
    global DIMS, MONS, IDX, DO
    DIMS = [2**n for n in N]
    MONS = list(itertools.product(*[range(d) for d in DIMS]))
    IDX = {a: i for i, a in enumerate(MONS)}
    DO = len(MONS)

# ---- O over F_2: elements = int bitmask over MONS
def mono_mul(a, b):
    c = []
    for ai, bi, d in zip(a, b, DIMS):
        s = ai + bi
        if s >= d or comb(s, ai) % 2 == 0: return None
        c.append(s)
    return tuple(c)
def mul(f, g):
    r = 0
    fi = [i for i in range(DO) if f >> i & 1]; gi = [i for i in range(DO) if g >> i & 1]
    for i in fi:
        for j in gi:
            c = mono_mul(MONS[i], MONS[j])
            if c is not None: r ^= 1 << IDX[c]
    return r
def der(f, k):
    r = 0
    for i in range(DO):
        if f >> i & 1:
            a = MONS[i]
            if a[k] > 0:
                b = list(a); b[k] -= 1; r ^= 1 << IDX[tuple(b)]
    return r
def mon(a): return 1 << IDX[tuple(a)]
def xbar(i):
    a = [0]*M; a[i] = DIMS[i]-1; return mon(a)
def xvar(i):
    a = [0]*M; a[i] = 1; return mon(a)
ONE = None
def expo(i):
    r = 0
    for m in range(DIMS[i]):
        a = [0]*M; a[i] = m; r ^= mon(a)
    return r

# ---- forms: 1-form = list of M elts; 2-form = dict (i,j) i<j -> elt
def wedge11(a, b):
    w = {}
    for i in range(M):
        for j in range(M):
            if i == j: continue
            t = mul(a[i], b[j])
            key = (min(i, j), max(i, j))
            w[key] = w.get(key, 0) ^ t       # char 2: sign irrelevant
    return w
def d1(a):
    w = {}
    for i in range(M):
        for j in range(M):
            if i != j:
                key = (min(i, j), max(i, j)); w[key] = w.get(key, 0) ^ der(a[j], i)
    return w
def addf(*ws):
    r = {}
    for w in ws:
        for k, v in w.items(): r[k] = r.get(k, 0) ^ v
    return {k: v for k, v in r.items() if v}
def closed(w):
    for i, j, k in itertools.combinations(range(M), 3):
        if der(w.get((j, k), 0), i) ^ der(w.get((i, k), 0), j) ^ der(w.get((i, j), 0), k): return False
    return True
def pf_const(w):
    g = lambda i, j: w.get((i, j), 0)
    pf = mul(g(0,1), g(2,3)) ^ mul(g(0,2), g(1,3)) ^ mul(g(0,3), g(1,2))
    return pf & 1   # constant term (index of (0,0,0,0) is 0)

# ---- vect: element = int bitmask, bit k*DO + i  <-> x^(MONS[i]) d_k
def vcomp(v, k): return (v >> (k*DO)) & ((1 << DO)-1)
def vmake(comps):
    r = 0
    for k, c in enumerate(comps): r |= c << (k*DO)
    return r
def vapply(v, f):
    r = 0
    for k in range(M): r ^= mul(vcomp(v, k), der(f, k))
    return r
def vbr(u, v):
    return vmake([vapply(u, vcomp(v, k)) ^ vapply(v, vcomp(u, k)) for k in range(M)])

# ---- F_2 linear algebra on ints
def echelon(vecs):
    """returns list of (pivot, vec) reduced"""
    piv = {}
    for v in vecs:
        for p, b in piv.items():
            if v >> p & 1: v ^= b
        if v:
            p = v.bit_length()-1
            for q in list(piv):
                if piv[q] >> p & 1: piv[q] ^= v
            piv[p] = v
    return piv
def rank2(vecs): return len(echelon(vecs))
def nullspace2(rows, n):
    """rows: ints over n columns; return basis of {x: row.x=0 all}"""
    piv = echelon(rows)
    pivcols = set(piv)
    basis = []
    for f in range(n):
        if f in pivcols: continue
        x = 1 << f
        for p, r in piv.items():
            if r >> f & 1: x |= 1 << p
        basis.append(x)
    return basis

def hamiltonian(w):
    """{D in vect : iota_D w closed}"""
    n = M*DO
    cols = []   # for each basis vector field, the closedness defect vector
    for b in range(n):
        k, i = divmod(b, DO); v = 1 << b
        alpha = [0]*M
        for (p, q), wpq in w.items():
            # iota_D (wpq dx_p dx_q) = wpq (D_p dx_q - D_q dx_p)
            alpha[q] ^= mul(wpq, vcomp(v, p)); alpha[p] ^= mul(wpq, vcomp(v, q))
        defect = 0; s = 0
        for i1, j1 in itertools.combinations(range(M), 2):
            defect |= (der(alpha[i1], j1) ^ der(alpha[j1], i1)) << s; s += DO
        cols.append(defect)
    # matrix with columns cols: rows = defect bits
    nrows = 6*DO
    rows = [sum(((cols[b] >> r) & 1) << b for b in range(n)) for r in range(nrows)]
    return nullspace2(rows, n)

def span_basis(vecs): return list(echelon(vecs).values())
def derived(basis):
    return span_basis([vbr(u, v) for u, v in itertools.combinations(basis, 2)])

def coords_fn(basis):
    """express vectors in basis (ints over vect) -> int over len(basis)"""
    piv = {}
    for idx, v in enumerate(basis):
        t = 1 << idx
        for p, (b, tb) in piv.items():
            if v >> p & 1: v ^= b; t ^= tb
        assert v
        p = v.bit_length()-1
        for q in list(piv):
            if piv[q][0] >> p & 1: piv[q] = (piv[q][0] ^ v, piv[q][1] ^ t)
        piv[p] = (v, t)
    def co(v):
        t = 0
        for p, (b, tb) in piv.items():
            if v >> p & 1: v ^= b; t ^= tb
        assert v == 0, "not in span"
        return t
    return co

def struct(basis):
    co = coords_fn(basis); n = len(basis)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = co(vbr(basis[i], basis[j])); C[i][j] = C[j][i] = c
    return C

def ad_matrix(C, x):
    """ad_x as list of column ints: column j = [x, e_j]"""
    n = len(C); cols = []
    for j in range(n):
        c = 0
        for i in range(n):
            if x >> i & 1: c ^= C[i][j]
        cols.append(c)
    return cols
def matmul_cols(A, B, n):   # (A B) columns: A applied to columns of B
    out = []
    for bc in B:
        c = 0
        for i in range(n):
            if bc >> i & 1: c ^= A[i]
        out.append(c)
    return out
def flat(cols, n): return sum(c << (j*n) for j, c in enumerate(cols))

def envelope_dim(C):
    n = len(C)
    gens = [ad_matrix(C, 1 << i) for i in range(n)]
    ident = [1 << j for j in range(n)]
    piv = {}; mats = []
    def add(m):
        v = flat(m, n)
        for p, b in piv.items():
            if v >> p & 1: v ^= b
        if v:
            p = v.bit_length()-1
            for q in list(piv):
                if piv[q] >> p & 1: piv[q] ^= v
            piv[p] = v; return True
        return False
    frontier = []
    for m in [ident] + gens:
        if add(m): mats.append(m); frontier.append(m)
    while frontier and len(piv) < n*n:
        new = []
        for m in frontier:
            for g in gens:
                p = matmul_cols(g, m, n)
                if add(p): new.append(p)
        frontier = new
    return len(piv)

def der_dim(C):
    n = len(C)
    # unknown D: D(e_j) = sum_i d_{ij} e_i; var index i + n*j
    rows = []
    for a in range(n):
        for b in range(a+1, n):
            cab = C[a][b]
            # D[e_a,e_b] + [De_a,e_b] + [e_a,De_b] = 0, component t
            for t in range(n):
                r = 0
                for j in range(n):
                    if cab >> j & 1: r ^= 1 << (t + n*j)
                for i in range(n):
                    if C[i][b] >> t & 1: r ^= 1 << (i + n*a)
                    if C[a][i] >> t & 1: r ^= 1 << (i + n*b)
                if r: rows.append(r)
    return n*n - rank2(rows)

def h2_dim(C):
    n = len(C)
    pairs = list(itertools.combinations(range(n), 2)); pidx = {p: i for i, p in enumerate(pairs)}
    # d1: c in g^* -> (dc)(a,b) = c([a,b]); image rank
    d1rows = []   # columns = pairs; rank of d1 = rank of the map g^* -> Lambda^2
    imgs = []
    for k in range(n):
        v = 0
        for (a, b), i in pidx.items():
            if C[a][b] >> k & 1: v |= 1 << i
        imgs.append(v)
    rk1 = rank2(imgs)
    # d2: Lambda^2 -> Lambda^3: dc(a,b,c)=c([a,b],c)+c([b,c],a)+c([c,a],b)
    def term(x, c, row):
        # c(x, e_c) with x vector
        for j in range(n):
            if x >> j & 1 and j != c:
                row ^= 1 << pidx[(min(j, c), max(j, c))]
        return row
    rows = []
    for a, b, c in itertools.combinations(range(n), 3):
        r = term(C[a][b], c, 0); r = term(C[b][c], a, r); r = term(C[c][a], b, r)
        if r: rows.append(r)
    z2 = len(pairs) - rank2(rows)
    return z2 - rk1, z2, rk1

def adrank_dist(C):
    n = len(C)
    adb = [ad_matrix(C, 1 << i) for i in range(n)]
    dist = {}; sq0 = 0
    for x in range(1, 1 << n):
        cols = [0]*n
        for i in range(n):
            if x >> i & 1:
                cols = [c ^ d for c, d in zip(cols, adb[i])]
        r = rank2(cols); dist[r] = dist.get(r, 0) + 1
        if all(c == 0 for c in matmul_cols(cols, cols, n)): sq0 += 1
    return dict(sorted(dist.items())), sq0

def omega0():
    return addf(wedge11(dx(0), dx(2)), wedge11(dx(1), dx(3)))
def dx(i):
    a = [0]*M; a[i] = 1 << IDX[(0,)*M]; return a
def scal1(f, a): return [mul(f, c) for c in a]

def forms():
    F = {}
    F['omega0'] = omega0()
    lam = [0, 0, xvar(0), xvar(1)]          # x1 dx3 + x2 dx4
    e1 = expo(0)
    F['omega1lit'] = wedge11(d1_0form(e1), lam)          # d(exp x1) ^ (x1dx3+x2dx4)
    F['omega1'] = d1(scal1(e1, lam))                     # d(exp(x1) (x1dx3+x2dx4))
    return F
def d1_0form(f): return [der(f, k) for k in range(M)]

def omega2(B):
    w = dict(omega0())
    for (i, j), bit in B.items():
        if bit:
            w[(i, j)] = w.get((i, j), 0) ^ mul(xbar(i), xbar(j))
    return {k: v for k, v in w.items() if v}

def analyse(w, log, full=True, adr=False):
    if not closed(w): log("  NOT closed"); return None
    if not pf_const(w): log("  degenerate (Pf has no constant term)"); return None
    h = hamiltonian(w); h1 = derived(h); h2 = derived(h1); h3 = derived(h2)
    res = dict(h=len(h), h1=len(h1), h2=len(h2), h3=len(h3))
    g = h2 if len(h3) == len(h2) else h3
    C = struct(g); n = len(C)
    res['env'] = envelope_dim(C); res['abs_simple'] = res['env'] == n*n
    if full:
        res['der'] = der_dim(C); res['outer'] = res['der'] - n
        res['H2'] = h2_dim(C)[0]
    if adr: res['adrank'], res['ad2zero'] = adrank_dist(C)
    return res

def main():
    out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ham4.log'), 'w')
    def log(*a):
        s = ' '.join(str(x) for x in a); print(s); out.write(s+'\n'); out.flush()
    which = sys.argv[1] if len(sys.argv) > 1 else 'N1'
    Ns = {'N1': (1,1,1,1), 'N2a': (2,1,1,1), 'N2b': (1,2,1,1)}
    for key in which.split(','):
        N = Ns[key]; setup(N)
        log(f"=== p=2, N={N}, dim O={DO}")
        small = DO == 16
        for name, w in forms().items():
            log(name, sorted((k, bin(v).count('1')) for k, v in w.items()))
            r = analyse(w, log, full=True, adr=small)
            log("  ", r)
        pairs = list(itertools.combinations(range(M), 2))
        seen = {}
        for bits in range(1, 64):
            B = {p: (bits >> t) & 1 for t, p in enumerate(pairs)}
            r = analyse(omega2(B), log, full=True, adr=False)
            sig = tuple(sorted((k, str(v)) for k, v in r.items()))
            seen.setdefault(sig, []).append(bits)
            log(f"omega2 B={[p for p in pairs if B[p]]}: {r}")
        log("omega2 classes by invariants:")
        for sig, lst in seen.items(): log("  ", len(lst), "choices", lst[:8], dict(sig))
    out.close()
if __name__ == '__main__': main()
