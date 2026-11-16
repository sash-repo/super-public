#!/usr/bin/env python3
"""Integrability of the degree-3 infinitesimal filtered deformations of
g = svect(3;1) over F_3.

Everything is split by the Z^3-multidegree (a genuine grading of vect(3;1),
hence of svect(3;1)); the Z-degree is its coordinate sum.

Conventions.  For 2-cochains mu, nu put
    J(mu,nu)(x,y,z) = cyc mu(nu(x,y),z).
A bracket mu_0 + mu_3 + mu_6 (mu_k of degree k) satisfies Jacobi iff its
graded pieces J(.,.) vanish degree by degree.  With L(phi) := J(mu_0,phi) +
J(phi,mu_0) one checks L = -d (d the Chevalley-Eilenberg differential), so
    degree 3:  d mu_3 = 0,     degree 6:  J(mu_3,mu_3) = d mu_6,
and nothing else, since 2-cochains live in degrees <= 4+2 = 6 and 3-cochains
in degrees <= 4+3 = 7 (top degree of svect(3;1) is 4), so degree 9, 12 vanish.
"""
import sys, time, random
from itertools import combinations
from collections import defaultdict
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace as xnull, rank as xrank, rref as xrref, backend
from vect3 import build_svect, subalg_from_rows
from coh import Complex

P = 3
t0 = time.time()
DERIVED = "derived" in sys.argv
LOG = open("svint.log", "a" if DERIVED else "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n"); LOG.flush()


log("backend:", backend())
vect, ker = build_svect()
g = subalg_from_rows(vect, ker)
if DERIVED:            # run everything for g^(1) = [g, g] instead
    from k3 import rref as krref
    br = [vect.bracket(a, b) for a in ker for b in ker]
    br = [r for r in br if any(r)]
    red = krref(br, vect.n)[0]
    g = subalg_from_rows(vect, red)
    log("\n" + "=" * 70 + "\nSAME COMPUTATION FOR THE DERIVED ALGEBRA svect^(1)(3;1)\n" + "=" * 70)
n = g.n
WT = [tuple(w) for w in g.wt]
DEG = [sum(w) for w in WT]
log("dim g =", n, " degrees:", {d: DEG.count(d) for d in sorted(set(DEG))})
C = g.c                                   # C[i][j] = {k: coef}
# pairs (a<b) with a component on s
PAIRS_TO = defaultdict(list)
for a in range(n):
    for b in range(a + 1, n):
        for s, co in C[a][b].items():
            PAIRS_TO[s].append((a, b, co))

def wsub(u, v): return tuple(x - y for x, y in zip(u, v))
def wadd(u, v): return tuple(x + y for x, y in zip(u, v))

BYWT = defaultdict(list)
for i, w in enumerate(WT):
    BYWT[w].append(i)


def cochains(k, w):
    """basis (S, m) of C^k of multidegree w."""
    out = []
    for S in combinations(range(n), k):
        ws = (0, 0, 0)
        for i in S:
            ws = wadd(ws, WT[i])
        for m in BYWT.get(wadd(w, ws), []):
            out.append((S, m))
    return out


def sort_sign(t):
    t = list(t)
    s = 1
    for i in range(len(t)):
        for j in range(len(t) - 1 - i):
            if t[j] > t[j + 1]:
                t[j], t[j + 1] = t[j + 1], t[j]; s = -s
            elif t[j] == t[j + 1]:
                return None, 0
    return tuple(t), s


def d_basis(S, m):
    """d(e_S^* (x) m) as dict (T, k) -> coef.
    (d phi)(x_0..x_n) = sum_i (-1)^i [x_i, phi(..^i..)]
                      + sum_{i<j} (-1)^{i+j} phi([x_i,x_j], ..^i..^j..)."""
    out = defaultdict(int)
    for t in range(n):
        if t in S:
            continue
        T = tuple(sorted(S + (t,)))
        i = T.index(t)
        for k, co in C[t][m].items():
            out[(T, k)] += (-1) ** i * co
    Sset = set(S)
    for s in S:
        rest0 = tuple(x for x in S if x != s)
        for a, b, co in PAIRS_TO[s]:
            if a in rest0 or b in rest0:
                continue
            T = tuple(sorted(rest0 + (a, b)))
            i, j = T.index(a), T.index(b)
            rest = tuple(x for x in T if x not in (a, b))
            S2, s2 = sort_sign((s,) + rest)
            assert S2 == S
            out[(T, m)] += (-1) ** (i + j) * s2 * co
    return {k: v % P for k, v in out.items() if v % P}


def dmatrix(src, tgt):
    """rows = targets, cols = sources."""
    pos = {c: i for i, c in enumerate(tgt)}
    M = [[0] * len(src) for _ in tgt]
    for j, s in enumerate(src):
        for c, v in d_basis(*s).items():
            M[pos[c]][j] = v
    return M


def kernel(M, ncols):
    if ncols == 0:
        return []
    if not M:
        return [[1 if i == j else 0 for i in range(ncols)] for j in range(ncols)]
    return [[int(x) % P for x in r] for r in xnull(M, P)]


def colspace(M, ncols):
    """basis of the image of M (as rows = image vectors)."""
    if not M or ncols == 0:
        return []
    Mt = [[M[r][c] for r in range(len(M))] for c in range(ncols)]
    R, piv = xrref(Mt, P)
    return [[int(x) % P for x in r] for r in R[:len(piv)]]


def rk(rows):
    if not rows:
        return 0
    return xrank(rows, P)


# ---------------------------------------------------------------- H^2(g;g)_k
def weights_of_degree(k, deg_range):
    """all multidegrees of C^2 of Z-degree k (collected from the cochains)."""
    ws = set()
    for a in range(n):
        for b in range(a + 1, n):
            for m in range(n):
                w = wsub(WT[m], wadd(WT[a], WT[b]))
                if sum(w) == k:
                    ws.add(w)
    return sorted(ws)


H2 = {}           # w -> (reps (list of dicts pair->vec), src)
SUMMARY = {}
for k in range(1, 7):
    tot = 0
    for w in weights_of_degree(k, None):
        src2 = cochains(2, w)
        if not src2:
            continue
        src1 = cochains(1, w)
        src3 = cochains(3, w)
        D2 = dmatrix(src2, src3)
        Z = kernel(D2, len(src2))
        if not Z:
            continue
        D1 = dmatrix(src1, src2)
        B = colspace(D1, len(src1))
        h = len(Z) - len(B)
        assert h >= 0 and rk(B + Z) == len(Z), "B not inside Z?"
        if h:
            reps, cur, r0 = [], B[:], len(B)
            for z in Z:
                if rk(cur + [z]) > r0:
                    cur.append(z); r0 += 1; reps.append(z)
            H2[w] = (reps, src2, B)
            tot += h
            log(f"   H^2(g;g) degree {k}, multidegree {w}: dim {h}"
                f"   (|C^1|={len(src1)}, |C^2|={len(src2)}, |C^3|={len(src3)})")
    SUMMARY[k] = tot
    log(f"dim H^2(g;g)_{k} = {tot}      [{time.time()-t0:.0f}s]")

# ---------------------------------------------------- restriction to g_-
gm = [i for i in range(n) if DEG[i] < 0]
cxm = Complex(g, gm)
Hm = cxm.cohomology(2)
log("\nH^2(g_-;g) by multidegree:",
    {w: d for w, (d, *_ ) in sorted(Hm.items())})


def to_pairs(vec, src):
    c = {}
    for co, (S, m) in zip(vec, src):
        if co:
            c.setdefault(S, defaultdict(int))
            c[S][m] = (c[S][m] + co) % P
    return c


for w, (reps, src2, B) in sorted(H2.items()):
    if sum(w) not in (3, 6):
        continue
    # restriction: keep only pairs inside g_-
    if w in Hm:
        dH, srcm, repsm, Bm = Hm[w]
        posm = {(tuple(gm[i] for i in S), m): j for j, (S, m) in enumerate(srcm)}
        imgs = []
        for z in reps:
            v = [0] * len(srcm)
            for co, (S, m) in zip(z, src2):
                if co and (S, m) in posm:
                    v[posm[(S, m)]] = co
            imgs.append(v)
        r = rk(Bm + imgs) - rk(Bm)
        log(f"   multidegree {w}: dim H^2(g;g)_w = {len(reps)},"
            f" rank of restriction to H^2(g_-;g)_w (dim {dH}) = {r}")
    else:
        log(f"   multidegree {w}: dim H^2(g;g)_w = {len(reps)},"
            f" H^2(g_-;g)_w = 0 (restriction zero)")

# ---------------------------------------------------- obstruction J(z,z)
def J(mu, nu):
    """cyc mu(nu(x,y),z) as dict (T, k) -> coef.  mu, nu: dict pair -> {k: c}."""
    out = defaultdict(int)
    def muval(v, c):          # mu(v, e_c) for a vector v (dict)
        res = defaultdict(int)
        for a, va in v.items():
            if a == c:
                continue
            key, sg = ((a, c), 1) if a < c else ((c, a), -1)
            for k, co in mu.get(key, {}).items():
                res[k] += sg * va * co
        return res
    for (a, b), v in nu.items():
        for c in range(n):
            if c in (a, b):
                continue
            T, sg = sort_sign((a, b, c))
            for k, co in muval(v, c).items():
                if co % P:
                    out[(T, k)] += sg * co
    return {k: v % P for k, v in out.items() if v % P}


def add(c1, c2, f=1):
    out = {S: defaultdict(int, v) for S, v in c1.items()}
    for S, v in c2.items():
        out.setdefault(S, defaultdict(int))
        for k, co in v.items():
            out[S][k] = (out[S][k] + f * co) % P
    return out


B3cache = {}
def in_B3(vec3):
    """is the dict (T,k)->c in d(C^2)?  split by multidegree."""
    parts = defaultdict(dict)
    for (T, k), c in vec3.items():
        w = WT[k]
        for t in T:
            w = wsub(w, WT[t])
        parts[w][(T, k)] = c
    res = {}
    for w, part in parts.items():
        if w not in B3cache:
            src2 = cochains(2, w)
            src3 = cochains(3, w)
            D2 = dmatrix(src2, src3)
            B3cache[w] = (src3, colspace(D2, len(src2)))
        src3, B = B3cache[w]
        pos = {c: i for i, c in enumerate(src3)}
        v = [0] * len(src3)
        for c, x in part.items():
            v[pos[c]] = x
        res[w] = rk(B + [v]) == rk(B)
    return res


Z3 = []   # (w, dict)
for w, (reps, src2, B) in sorted(H2.items()):
    if sum(w) == 3:
        for z in reps:
            Z3.append((w, to_pairs(z, src2)))
log(f"\n{len(Z3)} representatives of H^2(g;g)_3; multidegrees:",
    [w for w, _ in Z3])


# ================================================= verification when H^2_3 = 0
DEG3W = sorted(w for w in Hm if sum(w) == 3)
log("\nVERIFICATION.  degree-3 multidegrees of H^2(g_-;g):", DEG3W)
# (a) d o d = 0 on the degree-3 blocks C^1 -> C^2 -> C^3
bad = 0
for w in weights_of_degree(3, None):
    s1, s2, s3 = cochains(1, w), cochains(2, w), cochains(3, w)
    if not (s1 and s2 and s3):
        continue
    D1, D2 = dmatrix(s1, s2), dmatrix(s2, s3)
    for r in D2:
        for j in range(len(s1)):
            if sum(r[i] * D1[i][j] for i in range(len(s2))) % P:
                bad += 1
log("(a) d o d = 0 on all degree-3 blocks:", bad == 0)

# (b) my differential against coh.py's (independent code) on the degree-3 blocks
cxf = Complex(g, list(range(n)))
agree = True
for kdeg in (1, 2):
    cols = cxf.diff(kdeg)
    for w in DEG3W:
        for s in cochains(kdeg, w):
            a = d_basis(*s)
            b = {kk: v % P for kk, v in cols.get(s, {}).items() if v % P}
            if a != b:
                agree = False
log("(b) differential agrees with coh.py's on C^1, C^2 of those blocks:", agree,
    f"   [{time.time()-t0:.0f}s]")

# (c) direct extension test: phi in C^2(g;g)_w, psi in C^1(g_-;g)_w with
#     d phi = 0  and  phi|_{g_-} - d_- psi = c   (c the g_- class)
dm1 = cxm.diff(1)
for w in DEG3W:
    dH, srcm, repsm, Bm = Hm[w]
    s2, s3 = cochains(2, w), cochains(3, w)
    s1m = [c for c in cxm.cochains(1) if cxm.weight(*c) == w]
    posm = {c: i for i, c in enumerate(srcm)}
    # unknowns: phi (len s2) then psi (len s1m)
    nu = len(s2) + len(s1m)
    rows, rhs = [], []
    D2 = dmatrix(s2, s3)
    for r in D2:
        rows.append(r + [0] * len(s1m)); rhs.append(0)
    R = [[0] * nu for _ in srcm]
    for j, (S, m) in enumerate(s2):
        if all(DEG[x] < 0 for x in S):
            Sm = tuple(gm.index(x) for x in S)
            R[posm[(Sm, m)]][j] = 1
    for j, c in enumerate(s1m):
        for tgt, v in dm1.get(c, {}).items():
            if tgt in posm:
                R[posm[tgt]][len(s2) + j] = (-v) % P
    for i in range(len(srcm)):
        rows.append(R[i]); rhs.append(repsm[0][i])
    aug = [r + [b] for r, b in zip(rows, rhs)]
    solvable = rk(aug) == rk(rows)
    log(f"(c) multidegree {w}: the g_- class extends to a cocycle on g ? {solvable}"
        f"   (|C^2(g;g)_w| = {len(s2)})")

# (c') where the extension breaks: impose d phi = 0 only on triples whose
#      degrees sum to <= s, and find the least s with no solution
for w in DEG3W:
    dH, srcm, repsm, Bm = Hm[w]
    s2, s3 = cochains(2, w), cochains(3, w)
    s1m = [c for c in cxm.cochains(1) if cxm.weight(*c) == w]
    posm = {c: i for i, c in enumerate(srcm)}
    nu = len(s2) + len(s1m)
    D2 = dmatrix(s2, s3)
    R = [[0] * nu for _ in srcm]
    for j, (S, m) in enumerate(s2):
        if all(DEG[x] < 0 for x in S):
            R[posm[(tuple(gm.index(x) for x in S), m)]][j] = 1
    for j, c in enumerate(s1m):
        for tgt, v in dm1.get(c, {}).items():
            if tgt in posm:
                R[posm[tgt]][len(s2) + j] = (-v) % P
    first = None
    for smax in range(-3, 13):
        rows = [r + [0] * len(s1m) for r, (T, k) in zip(D2, s3)
                if sum(DEG[x] for x in T) <= smax]
        rows += R
        rhs = [0] * (len(rows) - len(R)) + [repsm[0][i] for i in range(len(srcm))]
        aug = [r + [b] for r, b in zip(rows, rhs)]
        if rk(aug) > rk(rows):
            first = smax
            break
    log(f"(c') multidegree {w}: extension impossible once d phi = 0 is imposed on"
        f" triples of total degree <= {first}  (i.e. arguments in g_-, g_-, g_{{{first+2}}})"
        if first is not None else f"(c') {w}: no break found")

# (d) calibration: H^1(g;g) (outer derivations) by degree
H1 = defaultdict(int)
ws1 = set()
for a in range(n):
    for m in range(n):
        ws1.add(wsub(WT[m], WT[a]))
for w in sorted(ws1):
    s0, s1, s2 = cochains(0, w), cochains(1, w), cochains(2, w)
    Z = kernel(dmatrix(s1, s2), len(s1))
    B = colspace(dmatrix(s0, s1), len(s0)) if s0 else []
    if len(Z) - len(B):
        H1[w] = len(Z) - len(B)
log("(d) H^1(g;g) (outer derivations) by multidegree:", dict(H1))

# (e) the degree-6 class: its deform satisfies Jacobi
w6 = [w for w in H2 if sum(w) == 6]
for w in w6:
    reps, src2, B = H2[w]
    z6 = to_pairs(reps[0], src2)
    def brk6(i, j, z6=z6):
        out = defaultdict(int, C[i][j])
        if i < j:
            for k, v in z6.get((i, j), {}).items(): out[k] += v
        elif j < i:
            for k, v in z6.get((j, i), {}).items(): out[k] -= v
        return {k: v % P for k, v in out.items() if v % P}
    from k3 import LieAlg
    A6 = LieAlg(list(range(n)), brk6, g.wt, g.names)
    log(f"(e) g with bracket mu_0 + z_6 (multidegree {w}): Jacobi on all triples:",
        A6.check_jacobi()[0])
    log("    z_6 =", {(g.names[a], g.names[b]): {g.names[k]: v for k, v in val.items() if v}
                     for (a, b), val in z6.items() if any(val.values())})

if not Z3:
    log("\nH^2(g;g)_3 = 0: no degree-3 infinitesimal filtered deformation exists"
        " in the full complex; the obstruction part is void.")
    log(f"done [{time.time()-t0:.0f}s]")
    if DERIVED:
        # the leading terms of Tyurin's subalgebra {D | div D + D(x_1^{(3)}) = 0}
        from svecth import stabiliser
        from vect3 import BASIS as VB
        from k3 import rref as krref
        vdeg = [sum(m) - 1 for (m, i) in VB]
        T = stabiliser({(3, 0, 0): 1})
        # gr: reduce T so that leading (lowest-degree) parts are independent
        order = sorted(range(vect.n), key=lambda i: (vdeg[i], i))
        Tp = [[r[i] for i in order] for r in T]
        red = krref(Tp, vect.n)[0]
        lead = []
        for r in red:
            f = next(c for c, x in enumerate(r) if x)
            dd = vdeg[order[f]]
            v = [0] * vect.n
            for c, x in enumerate(r):
                if x and vdeg[order[c]] == dd:
                    v[order[c]] = x
            lead.append(v)
        grT = krref(lead, vect.n)[0]
        svr = krref(ker, vect.n)[0]
        gr1 = [r for r in g.rows]
        rkk = lambda rows: len(krref(rows, vect.n)[0]) if rows else 0
        log("\nTyurin's algebra for h = exp(x_1^{(3)}): dim", len(T),
            "; dim gr =", len(grT), "; gr inside svect(3;1):",
            rkk(svr + grT) == len(svr), "; gr contains svect^(1)(3;1):",
            rkk(grT + gr1) == len(grT), "; dims gr by degree:",
            {d: sum(1 for r in grT if vdeg[next(i for i, x in enumerate(r) if x)] == d)
             for d in range(-1, 6)})
    if not DERIVED:
        import subprocess
        LOG.close()
        subprocess.run([sys.executable, __file__, "derived"])
    sys.exit(0)

# sanity: d z = 0 via J  (L = -d) -- use J(mu0, z) + J(z, mu0)
MU0 = {}
for a in range(n):
    for b in range(a + 1, n):
        if C[a][b]:
            MU0[(a, b)] = dict(C[a][b])
chk = []
for w, z in Z3:
    t = J(MU0, z)
    t2 = J(z, MU0)
    s = defaultdict(int)
    for kk, v in list(t.items()) + list(t2.items()):
        s[kk] = (s[kk] + v) % P
    chk.append(not any(s.values()))
log("J(mu0,z)+J(z,mu0) = 0 (cocycle, via J):", chk)

# the check L = -d on a random 2-cochain of the degree-3 multidegree
w0 = Z3[0][0]
src2 = cochains(2, w0)
random.seed(1)
phi = {}
rv = {}
for s in random.sample(src2, min(20, len(src2))):
    c = random.randrange(1, P)
    phi.setdefault(s[0], {})[s[1]] = c
    rv[s] = c
L = defaultdict(int)
for kk, v in list(J(MU0, phi).items()) + list(J(phi, MU0).items()):
    L[kk] = (L[kk] + v) % P
dphi = defaultdict(int)
for s, c in rv.items():
    for kk, v in d_basis(*s).items():
        dphi[kk] = (dphi[kk] + c * v) % P
ok = all((L[kk] + dphi[kk]) % P == 0 for kk in set(L) | set(dphi))
log("check L = -d on a random cochain:", ok)

log("\nObstructions J(z_a, z_b) + J(z_b, z_a)  (for a=b: J(z,z)), in B^3 ?")
OBS = {}
for i in range(len(Z3)):
    for j in range(i, len(Z3)):
        wi, zi = Z3[i]; wj, zj = Z3[j]
        if i == j:
            o = J(zi, zi)
        else:
            o = defaultdict(int)
            for kk, v in list(J(zi, zj).items()) + list(J(zj, zi).items()):
                o[kk] = (o[kk] + v) % P
            o = {k: v for k, v in o.items() if v}
        r = in_B3(o) if o else {}
        OBS[(i, j)] = (o, r)
        log(f"   ({i},{j}) multidegrees {wi}+{wj}: |J| = {len(o)},"
            f" coboundary per block: {r if o else 'J = 0'}"
            f"   [{time.time()-t0:.0f}s]")

# invariance of the class of J(z,z) under z -> z + d phi
w0, z0 = Z3[0]
src1 = cochains(1, w0)
random.seed(2)
dphi = {}
for s in random.sample(src1, min(5, len(src1))):
    c = random.randrange(1, P)
    for (T, k), v in d_basis(*s).items():
        dphi.setdefault(T, defaultdict(int))
        dphi[T][k] = (dphi[T][k] + c * v) % P
z1 = add(z0, dphi)
o0 = J(z0, z0); o1 = J(z1, z1)
diff = defaultdict(int)
for kk, v in o1.items(): diff[kk] = (diff[kk] + v) % P
for kk, v in o0.items(): diff[kk] = (diff[kk] - v) % P
diff = {k: v for k, v in diff.items() if v}
log("J(z+d phi, z+d phi) - J(z,z) is a coboundary:", in_B3(diff) if diff else "difference 0",
    " (J(z,z) itself coboundary:", in_B3(o0) if o0 else True, ")")

import pickle
pickle.dump({"Z3": Z3, "H2dims": SUMMARY,
             "OBS": {k: v[1] for k, v in OBS.items()}}, open("svint.pkl", "wb"))
log(f"done [{time.time()-t0:.0f}s]")
