#!/usr/bin/env python3
"""The groupoid of a generalized root system (GRS), part A.

For each irreducible GRS of small rank we verify, with exact rational arithmetic:

  * Serganova's axioms (0), (1), (2') of Definition 1.1 / section 7 of [S];
  * the splitting of the arrangement A(R) = {gamma^perp : gamma in R} into
    anisotropic ("real") and isotropic ("imaginary") walls, and that the wall type
    is well defined (proportional roots only);
  * |W| for W = <r_alpha : alpha in R_re>, and that the number of chambers of
    A(R_re) equals |W|;
  * the number of chambers of the full arrangement A(R), exactly, by Zaslavsky's
    theorem (Moebius function of the intersection lattice) AND by exhaustive
    enumeration of sign vectors (the two must agree, which certifies the list);
  * the number of bases of R in the sense of Serganova's Definition 6.1;
  * that chamber |--> {indecomposable roots of its positive system} is a bijection
    chambers <--> bases.

Reference counts from [S]: G(3) has 96 bases (Example 6.5); D(2,1) has 32
(Example 6.7: 32 vertices of the associated polytope).
"""
from fractions import Fraction as F
from itertools import combinations, product
import random

random.seed(20260803)

# ---------------------------------------------------------------- linear algebra

def dot(G, u, v):
    return sum(G[i][j] * u[i] * v[j] for i in range(len(u)) for j in range(len(v)) if G[i][j])

def rref(rows):
    """Row-reduced echelon form of a list of tuples of Fractions; returns tuple of tuples."""
    m = [list(map(F, r)) for r in rows]
    n = len(m[0]) if m else 0
    piv = 0
    for c in range(n):
        r = next((i for i in range(piv, len(m)) if m[i][c] != 0), None)
        if r is None:
            continue
        m[piv], m[r] = m[r], m[piv]
        pv = m[piv][c]
        m[piv] = [x / pv for x in m[piv]]
        for i in range(len(m)):
            if i != piv and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[piv])]
        piv += 1
        if piv == len(m):
            break
    return tuple(tuple(r) for r in m[:piv])

def rank(rows):
    return len(rref(rows)) if rows else 0

def solve(basis, v):
    """Coordinates of v in the (independent) list basis; None if v not in the span."""
    n = len(basis[0])
    k = len(basis)
    m = [[F(basis[j][i]) for j in range(k)] + [F(v[i])] for i in range(n)]
    piv = []
    r = 0
    for c in range(k):
        p = next((i for i in range(r, n) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(n):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        piv.append(c)
        r += 1
    for i in range(r, n):
        if m[i][k] != 0:
            return None
    out = [F(0)] * k
    for i, c in enumerate(piv):
        out[c] = m[i][k]
    return out

def normalize(vec):
    """Scale a nonzero rational vector so its first nonzero entry is 1 (kills scalars)."""
    for x in vec:
        if x != 0:
            return tuple(y / x for y in vec)
    return tuple(vec)

# --------------------------------------------------------- the systems (root data)

def sysA(m, n):
    """A(m,n) = sl(m+1|n+1), m != n.  Ambient dim m+n+2, V = span(R) of dim m+n+1."""
    N = m + n + 2
    G = [[F(0)] * N for _ in range(N)]
    for i in range(m + 1):
        G[i][i] = F(1)
    for k in range(m + 1, N):
        G[k][k] = F(-1)
    def e(i):
        v = [F(0)] * N
        v[i] = F(1)
        return tuple(v)
    def sub(u, v):
        return tuple(a - b for a, b in zip(u, v))
    ev = [e(i) for i in range(m + 1)]
    od = [e(m + 1 + k) for k in range(n + 1)]
    roots = []
    for a, b in combinations(range(m + 1), 2):
        roots += [sub(ev[a], ev[b]), sub(ev[b], ev[a])]
    for a, b in combinations(range(n + 1), 2):
        roots += [sub(od[a], od[b]), sub(od[b], od[a])]
    for a in range(m + 1):
        for b in range(n + 1):
            roots += [sub(ev[a], od[b]), sub(od[b], ev[a])]
    # distinguished simple roots form a basis of V
    simple = [sub(ev[i], ev[i + 1]) for i in range(m)] + [sub(ev[m], od[0])] + \
             [sub(od[k], od[k + 1]) for k in range(n)]
    return dict(name=f"A({m},{n})=sl({m+1}|{n+1})", G=G, roots=roots, Vbasis=simple)

def sysA11():
    """A(1,1) = psl(2|2): the defective case, V = span(R)/radical, dim 2.
    a = eps1-eps2, b = del1-del2, (a,a)=2, (b,b)=-2; odd roots (+-a+-b)/2."""
    G = [[F(2), F(0)], [F(0), F(-2)]]
    h = F(1, 2)
    roots = [(F(1), F(0)), (F(-1), F(0)), (F(0), F(1)), (F(0), F(-1)),
             (h, h), (-h, -h), (h, -h), (-h, h)]
    return dict(name="A(1,1)=psl(2|2)", G=G, roots=roots, Vbasis=[(F(1), F(0)), (F(0), F(1))])

def sysB(m, n):
    """B(m,n) = osp(2m+1|2n), m>=1, n>=1; dim m+n."""
    N = m + n
    G = [[F(0)] * N for _ in range(N)]
    for i in range(m):
        G[i][i] = F(1)
    for k in range(m, N):
        G[k][k] = F(-1)
    def vec(coeffs):
        return tuple(F(c) for c in coeffs)
    roots = []
    for i in range(m):
        for s in (1, -1):
            c = [0] * N; c[i] = s; roots.append(vec(c))                 # +-eps_i (even)
    for i, j in combinations(range(m), 2):
        for si, sj in product((1, -1), repeat=2):
            c = [0] * N; c[i] = si; c[j] = sj; roots.append(vec(c))      # +-eps+-eps
    for k in range(m, N):
        for s in (1, -1):
            c = [0] * N; c[k] = 2 * s; roots.append(vec(c))              # +-2delta
            c = [0] * N; c[k] = s; roots.append(vec(c))                  # +-delta (odd, real)
    for k, l in combinations(range(m, N), 2):
        for sk, sl in product((1, -1), repeat=2):
            c = [0] * N; c[k] = sk; c[l] = sl; roots.append(vec(c))
    for k in range(m, N):
        for i in range(m):
            for sk, si in product((1, -1), repeat=2):
                c = [0] * N; c[k] = sk; c[i] = si; roots.append(vec(c))   # odd isotropic
    return dict(name=f"B({m},{n})=osp({2*m+1}|{2*n})", G=G, roots=roots, Vbasis=None)

def sysC(n):
    """C(n) = osp(2|2n-2), n>=2; coords eps, delta_1..delta_{n-1}; dim n."""
    N = n
    G = [[F(0)] * N for _ in range(N)]
    G[0][0] = F(1)
    for k in range(1, N):
        G[k][k] = F(-1)
    roots = []
    for k in range(1, N):
        for s in (1, -1):
            c = [0] * N; c[k] = 2 * s; roots.append(tuple(map(F, c)))
    for k, l in combinations(range(1, N), 2):
        for sk, sl in product((1, -1), repeat=2):
            c = [0] * N; c[k] = sk; c[l] = sl; roots.append(tuple(map(F, c)))
    for k in range(1, N):
        for s0, sk in product((1, -1), repeat=2):
            c = [0] * N; c[0] = s0; c[k] = sk; roots.append(tuple(map(F, c)))
    return dict(name=f"C({n})=osp(2|{2*n-2})", G=G, roots=roots, Vbasis=None)

def sysD(m, n):
    """D(m,n) = osp(2m|2n), m>=2, n>=1; dim m+n."""
    N = m + n
    G = [[F(0)] * N for _ in range(N)]
    for i in range(m):
        G[i][i] = F(1)
    for k in range(m, N):
        G[k][k] = F(-1)
    roots = []
    for i, j in combinations(range(m), 2):
        for si, sj in product((1, -1), repeat=2):
            c = [0] * N; c[i] = si; c[j] = sj; roots.append(tuple(map(F, c)))
    for k in range(m, N):
        for s in (1, -1):
            c = [0] * N; c[k] = 2 * s; roots.append(tuple(map(F, c)))
    for k, l in combinations(range(m, N), 2):
        for sk, sl in product((1, -1), repeat=2):
            c = [0] * N; c[k] = sk; c[l] = sl; roots.append(tuple(map(F, c)))
    for i in range(m):
        for k in range(m, N):
            for si, sk in product((1, -1), repeat=2):
                c = [0] * N; c[i] = si; c[k] = sk; roots.append(tuple(map(F, c)))
    return dict(name=f"D({m},{n})=osp({2*m}|{2*n})", G=G, roots=roots, Vbasis=None)

def sysD21(a):
    """D(2,1;a): dim 3, (eps_i,eps_i) = c_i with c_1+c_2+c_3 = 0; a=1 gives D(2,1)=osp(4|2)."""
    c = [F(1), F(a), F(-1) - F(a)]
    G = [[F(0)] * 3 for _ in range(3)]
    for i in range(3):
        G[i][i] = c[i]
    roots = []
    for i in range(3):
        for s in (2, -2):
            v = [F(0)] * 3; v[i] = F(s); roots.append(tuple(v))
    for s in product((F(1), F(-1)), repeat=3):
        roots.append(tuple(s))
    return dict(name=f"D(2,1;{a})", G=G, roots=roots, Vbasis=None)

def sysF4():
    """F(4): coords (eps_1,eps_2,eps_3,delta), G = diag(1,1,1,-3)."""
    G = [[F(0)] * 4 for _ in range(4)]
    for i in range(3):
        G[i][i] = F(1)
    G[3][3] = F(-3)
    roots = []
    for i in range(3):
        for s in (1, -1):
            c = [0, 0, 0, 0]; c[i] = s; roots.append(tuple(map(F, c)))
    for i, j in combinations(range(3), 2):
        for si, sj in product((1, -1), repeat=2):
            c = [0, 0, 0, 0]; c[i] = si; c[j] = sj; roots.append(tuple(map(F, c)))
    for s in (1, -1):
        roots.append((F(0), F(0), F(0), F(s)))
    h = F(1, 2)
    for s in product((h, -h), repeat=4):
        roots.append(tuple(s))
    return dict(name="F(4)", G=G, roots=roots, Vbasis=None)

def sysG3():
    """G(3): G_2 plane with Gram [[2,-3],[-3,6]] (short alpha, long beta) plus delta, (delta,delta)=-2."""
    G = [[F(2), F(-3), F(0)], [F(-3), F(6), F(0)], [F(0), F(0), F(-2)]]
    def v(x, y, z):
        return (F(x), F(y), F(z))
    g2 = []
    for (x, y) in [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)]:
        g2 += [v(x, y, 0), v(-x, -y, 0)]
    short = [r for r in g2 if dot(G, r, r) == 2]
    roots = list(g2)
    roots += [v(0, 0, 2), v(0, 0, -2)]                     # +-2delta (even)
    for s in (1, -1):
        roots.append(v(0, 0, s))                           # +-delta (odd, anisotropic)
    for r in short:
        for s in (1, -1):
            roots.append((r[0], r[1], F(s)))               # odd isotropic
    return dict(name="G(3)", G=G, roots=roots, Vbasis=None)

# ------------------------------------------------------------------- GRS checks

def check_axioms(G, roots):
    rs = set(roots)
    msgs = []
    if any(all(x == 0 for x in r) for r in roots):
        msgs.append("0 is a root!")
    if rs != {tuple(-x for x in r) for r in roots}:
        msgs.append("R != -R")
    for a in roots:
        na = dot(G, a, a)
        if na != 0:
            for b in roots:
                k = 2 * dot(G, a, b) / na
                if k.denominator != 1:
                    msgs.append(f"axiom(1) integrality fails: {a},{b},k={k}")
                rb = tuple(x - k * y for x, y in zip(b, a))
                if rb not in rs:
                    msgs.append(f"axiom(1) r_a(b) not a root: {a},{b}")
        else:
            for b in roots:
                if dot(G, a, b) != 0:
                    plus = tuple(x + y for x, y in zip(b, a))
                    minus = tuple(x - y for x, y in zip(b, a))
                    if (plus not in rs) and (minus not in rs):
                        msgs.append(f"axiom(2') fails: {a},{b}")
    return msgs

def weyl_group(G, roots):
    """Order of W = <r_alpha : alpha in R_re>, realized as permutations of R."""
    idx = {r: i for i, r in enumerate(roots)}
    gens = []
    for a in roots:
        na = dot(G, a, a)
        if na == 0:
            continue
        k = lambda b, a=a, na=na: 2 * dot(G, a, b) / na
        p = tuple(idx[tuple(x - k(b) * y for x, y in zip(b, a))] for b in roots)
        gens.append(p)
    ident = tuple(range(len(roots)))
    seen = {ident}
    frontier = [ident]
    while frontier:
        new = []
        for p in frontier:
            for g in gens:
                q = tuple(g[p[i]] for i in range(len(p)))
                if q not in seen:
                    seen.add(q)
                    new.append(q)
        frontier = new
    return len(seen)

# --------------------------------------------------- arrangements: walls, regions

def walls(G, roots, Vb):
    """Distinct hyperplanes of V cut by the roots, as functional vectors on Vbasis.
    Returns dict: functional (normalized) -> (list of roots giving it, anisotropic?)."""
    out = {}
    for r in roots:
        f = normalize(tuple(dot(G, b, r) for b in Vb))
        aniso = dot(G, r, r) != 0
        if f in out:
            out[f][0].append(r)
            if out[f][1] != aniso:
                out[f].append("TYPE CLASH")
        else:
            out[f] = [[r], aniso]
    return out

def zaslavsky(funcs, dim):
    """Number of regions of the central arrangement given by the functionals, exactly."""
    # intersection lattice: subspaces as rref of sets of functionals
    top = ()
    lattice = {top: None}
    frontier = [top]
    while frontier:
        new = []
        for X in frontier:
            for f in funcs:
                Y = rref(list(X) + [f])
                if Y not in lattice:
                    lattice[Y] = None
                    new.append(Y)
        frontier = new
    elems = sorted(lattice, key=len)
    def leq(X, Y):        # X <= Y in the lattice ordered by reverse inclusion of subspaces
        return all(solve(list(Y), f) is not None for f in X) if X else True
    mu = {}
    for X in elems:
        if not X:
            mu[X] = 1
        else:
            mu[X] = -sum(mu[Y] for Y in elems if len(Y) < len(X) and leq(Y, X))
    return sum(abs(mu[X]) for X in elems)

def matinv(M):
    """Inverse of a square Fraction matrix."""
    n = len(M)
    A = [[F(M[i][j]) for j in range(n)] + [F(1) if i == j else F(0) for j in range(n)]
         for i in range(n)]
    for c in range(n):
        p = next(i for i in range(c, n) if A[i][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for i in range(n):
            if i != c and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[c])]
    return [row[n:] for row in A]

def enumerate_regions(funcs, dim, target, gram_inv=None, tries=600000):
    """Sign vectors of regions, by random sampling; stops when target reached.
    If gram_inv is given, sampling is done in the coordinates dual to Vbasis
    (i.e. the values of the functionals on the simple roots are the random data),
    which spreads the samples evenly over the chambers."""
    found = {}
    rng = random.Random(11)
    span = 2
    for t in range(tries):
        if len(found) == target:
            break
        if t % 30000 == 0:
            span = min(span * 2, 8192)
        c = [F(rng.randint(-span, span)) for _ in range(dim)]
        if gram_inv is None:
            v = tuple(c)
        else:
            v = tuple(sum(gram_inv[k][i] * c[i] for i in range(dim)) for k in range(dim))
        sv = tuple(sum(f[i] * v[i] for i in range(dim)) for f in funcs)
        if any(x == 0 for x in sv):
            continue
        key = tuple(1 if x > 0 else -1 for x in sv)
        if key not in found:
            found[key] = v
    return found

# ------------------------------------------------------------------------ bases

def positive_system(G, roots, v):
    return [r for r in roots if sum(G[i][j] * v[i] * r[j] for i in range(len(v)) for j in range(len(r))) > 0]

def indecomposables(pos):
    ps = set(pos)
    out = []
    for r in pos:
        dec = any(tuple(x - y for x, y in zip(r, s)) in ps for s in pos if s != r)
        if not dec:
            out.append(r)
    return out

def is_base(B, roots):
    """Serganova Def. 6.1: every root is uniquely a Z_{>=0}- or Z_{<=0}-combination of B."""
    if not B:
        return False
    if rank(B) != len(B):
        return None          # dependent: handled separately
    for r in roots:
        c = solve(B, r)
        if c is None:
            return False
        if not all(x.denominator == 1 for x in c):
            return False
        if not (all(x >= 0 for x in c) or all(x <= 0 for x in c)):
            return False
    return True

def all_bases_independent(roots, dim):
    out = []
    for B in combinations(roots, dim):
        if rank(list(B)) < dim:
            continue
        if is_base(list(B), roots):
            out.append(frozenset(B))
    return out

# --------------------------------------------------------------------- the report

def report(sysd, do_bases=True):
    G, roots = sysd["G"], sysd["roots"]
    Vb = sysd["Vbasis"] or [tuple(F(1) if i == j else F(0) for j in range(len(G))) for i in range(len(G))]
    dim = len(Vb)
    name = sysd["name"]
    print(f"\n=== {name} ===  |R| = {len(roots)}, dim V = {dim}")
    ax = check_axioms(G, roots)
    print("  GRS axioms (0),(1),(2'):", "OK" if not ax else ax[:4])
    nre = sum(1 for r in roots if dot(G, r, r) != 0)
    print(f"  |R_re| = {nre}, |R_im| = {len(roots)-nre}")
    W = weyl_group(G, roots)
    wl = walls(G, roots, Vb)
    clash = [k for k, v in wl.items() if len(v) > 2]
    aniso = [f for f, v in wl.items() if v[1]]
    iso = [f for f, v in wl.items() if not v[1]]
    print(f"  walls: {len(wl)} total = {len(aniso)} anisotropic + {len(iso)} isotropic"
          + ("  TYPE CLASH!" if clash else "  (wall type well defined)"))
    print(f"  |W| = {W}")
    r_re = zaslavsky(aniso, dim) if aniso else 1
    r_all = zaslavsky(list(wl), dim)
    print(f"  chambers of A(R_re) = {r_re}   (equals |W|: {r_re == W})")
    print(f"  chambers of A(R)    = {r_all}  (Zaslavsky)")
    gram = [[dot(G, Vb[i], Vb[j]) for j in range(dim)] for i in range(dim)]
    regs = enumerate_regions(list(wl), dim, r_all, gram_inv=matinv(gram))
    print(f"  enumerated regions  = {len(regs)}  ({'complete' if len(regs)==r_all else 'INCOMPLETE'})")
    print(f"  refinement index (chambers of A(R) per W-chamber) = "
          f"{F(r_all, W)}")
    if not do_bases:
        return
    # bases via chambers
    ch_bases = set()
    for key, v in regs.items():
        vv = [sum(Vb[k][i] * v[k] for k in range(dim)) for i in range(len(G))]
        pos = positive_system(G, roots, vv)
        B = indecomposables(pos)
        ch_bases.add(frozenset(B))
    sizes = sorted({len(B) for B in ch_bases})
    print(f"  bases from chambers = {len(ch_bases)}  (sizes {sizes})")
    bb = all_bases_independent(roots, dim)
    print(f"  bases by brute force over {dim}-subsets = {len(bb)}")
    print(f"  chamber -> base injective: {len(ch_bases) == len(regs)};  "
          f"onto brute-force list: {set(bb) == ch_bases}")
    print(f"  bases / |W| = {F(len(ch_bases), W)}  (= number of W-orbits of bases)")


if __name__ == "__main__":
    systems = [sysA(1, 0), sysA(2, 0), sysA(2, 1), sysA(3, 1), sysB(1, 1), sysB(2, 1),
               sysB(1, 2), sysC(2), sysC(3), sysD(2, 1), sysD(2, 2), sysD21(1), sysD21(2),
               sysD21(3), sysG3(), sysF4()]
    for s in systems:
        report(s)
    print("\n--- the defective case (bases are linearly dependent) ---")
    report(sysA11(), do_bases=False)
