#!/usr/bin/env python3
"""S^2(g) for e(6), e(7), e(8) and ab(3)=F(4) over F_p, from scratch.

e(n): the Chevalley basis by the Frenkel--Kac sign rule, h_1..h_n, e_alpha;
  [e_a,e_b] = eps(a,b) e_{a+b},  [e_a,e_{-a}] = eps(a,-a) h_a,  [h_i,e_a] = a(h_i) e_a,
  structure constants in Z, super-Jacobi checked before use.
ab(3): the rational weight-basis model of exc3.py (sdim (24|16)),
  reduced mod p (its denominators are powers of 2 and 3).

For a prime p (a large prime stands in for characteristic 0):
  * the even supersymmetric invariant forms are SOLVED for (not assumed);
  * the Casimir Omega = sum_k rho(b_k) rho(b^k), B(b^k, b_l) = delta_kl,
    is checked to commute with ad g on g and on S^2(g);
  * the singular vectors of S^2(g) = n_+-invariants, computed weight by weight
    (the root-lattice grading is finer than the h-weights, and a kernel of a
    graded map is graded), n_+ = the root vectors positive for a generic
    functional;
  * Omega on the span of the singular vectors of each weight: its eigenvalues.
Optionally g is replaced by g/z (z = the center), for a degenerate Cartan
matrix (e(6) at p=3, e(7) at p=2).
Output: vogele.log (one writer, in main()).
"""
import sys, itertools, random
from fractions import Fraction as F
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace, rank, backend

BIG = 1000003
CR = '--cr' in sys.argv


E = {
 6: [[2,0,-1,0,0,0],[0,2,0,-1,0,0],[-1,0,2,-1,0,0],[0,-1,-1,2,-1,0],[0,0,0,-1,2,-1],[0,0,0,0,-1,2]],
 7: [[2,0,-1,0,0,0,0],[0,2,0,-1,0,0,0],[-1,0,2,-1,0,0,0],[0,-1,-1,2,-1,0,0],[0,0,0,-1,2,-1,0],[0,0,0,0,-1,2,-1],[0,0,0,0,0,-1,2]],
 8: [[2,0,-1,0,0,0,0,0],[0,2,0,-1,0,0,0,0],[-1,0,2,-1,0,0,0,0],[0,-1,-1,2,-1,0,0,0],[0,0,0,-1,2,-1,0,0],[0,0,0,0,-1,2,-1,0],[0,0,0,0,0,-1,2,-1],[0,0,0,0,0,0,-1,2]],
}   # Bourbaki numbering


class Alg:
    """basis 0..N-1, parity PAR, weight WT (tuple), brackets SC[(i,j)] = {k: c}."""
    def __init__(self, name, PAR, WT, SC):
        self.name, self.PAR, self.WT, self.SC = name, PAR, WT, SC
        self.N = len(PAR)

    def br(self, x, y, p):
        out = {}
        for i, a in x.items():
            for j, b in y.items():
                d = self.SC.get((i, j))
                if d:
                    for k, c in d.items():
                        out[k] = (out.get(k, 0) + a * b * c) % p
        return {k: v for k, v in out.items() if v}


def e_type(n):
    A = E[n]
    def ip(a, b): return sum(a[i] * A[i][j] * b[j] for i in range(n) for j in range(n))
    simple = [tuple(int(i == j) for j in range(n)) for i in range(n)]
    pos = set(simple); frontier = list(simple)
    while frontier:
        new = []
        for a in frontier:
            for i in range(n):
                if ip(a, simple[i]) == -1:
                    b = tuple(x + y for x, y in zip(a, simple[i]))
                    if b not in pos:
                        pos.add(b); new.append(b)
        frontier = new
    pos = sorted(pos, key=lambda r: (sum(r), r))
    roots = pos + [tuple(-x for x in r) for r in pos]
    def eps(a, b):
        s = 0
        for i in range(n):
            for j in range(n):
                if i == j or (i < j and A[i][j] == -1):
                    s += a[i] * b[j]
        return -1 if s % 2 else 1
    N = n + len(roots)
    WT = [tuple([0] * n)] * n + roots
    idx = {r: n + k for k, r in enumerate(roots)}
    SC = {}
    for r in roots:
        k = idx[r]
        for i in range(n):           # [h_i, e_r] = r(h_i) e_r ; r(h_i) = sum_j r_j A_ji
            c = sum(r[j] * A[j][i] for j in range(n))
            if c:
                SC[(i, k)] = {k: c}; SC[(k, i)] = {k: -c}
    for a in roots:
        for b in roots:
            s = tuple(x + y for x, y in zip(a, b))
            if s in idx:
                SC[(idx[a], idx[b])] = {idx[s]: eps(a, b)}
            elif not any(s):
                SC[(idx[a], idx[b])] = {i: eps(a, b) * a[i] for i in range(n) if a[i]}
    return Alg("e(%d)" % n, [0] * N, WT, SC), len(pos)


def with_d(alg, n, p):
    """g(A) for a Cartan matrix degenerate mod p: add d with [d, e_r] = r(d) e_r,
    r(d) = sum_i r_i v_i, v = e_k for a node k on which ker A (mod p) is nonzero."""
    A = E[n]
    K = nullspace([[x % p for x in row] for row in A], p)
    z = [int(x) % p for x in K[0]]
    k = max(i for i in range(n) if z[i])
    v = [int(i == k) for i in range(n)]
    N = alg.N
    SC = {key: dict(d) for key, d in alg.SC.items()}
    for j in range(n, N):
        r = alg.WT[j]
        c = sum(r[i] * v[i] for i in range(n))
        if c:
            SC[(N, j)] = {j: c}; SC[(j, N)] = {j: -c}
    return Alg(alg.name + "+d", alg.PAR + [0], alg.WT + [alg.WT[0]], SC), z, k


def ab3():
    import exc3
    m = exc3.build_F4()
    SC = {}
    for (i, j), d in m.B.items():
        SC[(i, j)] = dict(d)
    WT = [tuple(F(x) for x in w) for w in m.WT]
    return Alg("ab(3)", list(m.PAR), WT, SC)


def reduce(alg, p):
    SC = {}
    for key, d in alg.SC.items():
        dd = {}
        for k, c in d.items():
            c = F(c)
            assert c.denominator % p, ("denominator divisible by p", c)
            v = c.numerator * pow(c.denominator, -1, p) % p
            if v: dd[k] = v
        if dd: SC[key] = dd
    return Alg(alg.name, alg.PAR, alg.WT, SC)


def jacobi_bad(alg, p, sample=None, seed=1):
    N, PAR = alg.N, alg.PAR
    triples = itertools.product(range(N), repeat=3)
    if sample:
        rnd = random.Random(seed)
        triples = ((rnd.randrange(N), rnd.randrange(N), rnd.randrange(N)) for _ in range(sample))
    bad = cnt = 0
    for a, b, c in triples:
        cnt += 1
        x, y, z = {a: 1}, {b: 1}, {c: 1}
        l = alg.br(x, alg.br(y, z, p), p)
        r1 = alg.br(alg.br(x, y, p), z, p)
        r2 = alg.br(y, alg.br(x, z, p), p)
        sg = -1 if PAR[a] * PAR[b] else 1
        for k in set(l) | set(r1) | set(r2):
            if (l.get(k, 0) - r1.get(k, 0) - sg * r2.get(k, 0)) % p:
                bad += 1; break
    return bad, cnt


def quotient_by_center(alg, p):
    """g/z, z = center; basis: drop the pivot coordinates of z (h-part only)."""
    N = alg.N
    rows = []                       # z = {v | [b_i, v] = 0 for all i}
    for i in range(N):
        for k in range(N):
            row = [0] * N
            for j in range(N):
                d = alg.SC.get((i, j))
                if d and k in d: row[j] = d[k]
            if any(row): rows.append(row)
    Z = nullspace(rows, p) if rows else [[int(i == j) for j in range(N)] for i in range(N)]
    Z = [[int(x) % p for x in v] for v in Z]
    if not Z:
        return alg, []
    # eliminate: for each z-vector choose a pivot coordinate, express b_piv = -(rest)/c mod z
    drop = {}
    for v in Z:
        piv = max(i for i in range(N) if v[i])
        inv = pow(v[piv], -1, p)
        drop[piv] = {i: (-v[i] * inv) % p for i in range(N) if v[i] and i != piv}
    for piv in list(drop):                     # resolve chains
        changed = True
        while changed:
            changed = False
            for i in list(drop[piv]):
                if i in drop:
                    c = drop[piv].pop(i)
                    for j, d in drop[i].items():
                        drop[piv][j] = (drop[piv].get(j, 0) + c * d) % p
                    changed = True
    keep = [i for i in range(N) if i not in drop]
    pos = {i: k for k, i in enumerate(keep)}
    SC = {}
    for (i, j), d in alg.SC.items():
        if i in drop or j in drop: continue
        dd = {}
        for k, c in d.items():
            if k in drop:
                for kk, cc in drop[k].items():
                    dd[pos[kk]] = (dd.get(pos[kk], 0) + c * cc) % p
            else:
                dd[pos[k]] = (dd.get(pos[k], 0) + c) % p
        dd = {k: v for k, v in dd.items() if v}
        if dd: SC[(pos[i], pos[j])] = dd
    return Alg(alg.name + "/z", [alg.PAR[i] for i in keep], [alg.WT[i] for i in keep], SC), Z


def invariant_forms(alg, p):
    N, WT, PAR = alg.N, alg.WT, alg.PAR
    zero = tuple(0 * x for x in WT[0])
    def add(a, b): return tuple(x + y for x, y in zip(a, b))
    pairs = [(a, b) for a in range(N) for b in range(a, N)
             if not any(add(WT[a], WT[b])) and PAR[a] == PAR[b]]
    var = {pr: k for k, pr in enumerate(pairs)}
    def Bv(a, b):             # B(a,b) as (var index, sign)
        if (a, b) in var: return var[(a, b)], 1
        if (b, a) in var: return var[(b, a)], (-1 if PAR[a] * PAR[b] else 1)
        return None
    rows = []
    bywt = {}
    for i in range(N): bywt.setdefault(WT[i], []).append(i)
    for x in range(N):
        for y in range(N):
            # B([x,y], z) - B(x, [y,z]) = 0 for z with wt x+y+z = 0
            target = tuple(-t for t in add(WT[x], WT[y]))
            for z in bywt.get(target, []):
                row = {}
                for k, c in alg.SC.get((x, y), {}).items():
                    v = Bv(k, z)
                    if v: row[v[0]] = (row.get(v[0], 0) + c * v[1]) % p
                for k, c in alg.SC.get((y, z), {}).items():
                    v = Bv(x, k)
                    if v: row[v[0]] = (row.get(v[0], 0) - c * v[1]) % p
                row = {k: v for k, v in row.items() if v}
                if row:
                    r = [0] * len(pairs)
                    for k, v in row.items(): r[k] = v
                    rows.append(r)
    sols = nullspace(rows, p) if rows else []
    forms = []
    for s in sols:
        Bm = {}
        for (a, b), k in var.items():
            v = int(s[k]) % p
            if v:
                Bm[(a, b)] = v; Bm[(b, a)] = v * (-1 if PAR[a] * PAR[b] else 1) % p
        forms.append(Bm)
    return forms


def dual_basis(alg, Bm, p):
    """returns list of (k, dual vector of b_k as dict); None if degenerate."""
    N = alg.N
    M = [[Bm.get((i, j), 0) for j in range(N)] for i in range(N)]
    if rank(M, p) < N: return None
    # b^k = sum_j c_kj b_j with B(b^k, b_l) = sum_j c_kj M[j][l] = delta_kl -> C = M^{-1}
    from flint import nmod_mat
    Mi = nmod_mat(M, p).inv()
    return [{j: int(Mi[k, j]) for j in range(N) if int(Mi[k, j])} for k in range(N)]


class S2:
    """the supersymmetric square: monomials (i,j), i<=j, no square of an odd vector
       (char != 2); in char 2 squares of even vectors are kept."""
    def __init__(self, alg, p):
        self.alg, self.p = alg, p
        PAR = alg.PAR
        self.mons = [(i, j) for i in range(alg.N) for j in range(i, alg.N)
                     if not (i == j and PAR[i] == 1)]
        self.bywt = {}
        for m in self.mons:
            w = tuple(x + y for x, y in zip(alg.WT[m[0]], alg.WT[m[1]]))
            self.bywt.setdefault(w, []).append(m)

    def norm(self, i, j, c):
        if i <= j: return (i, j), c
        s = -1 if self.alg.PAR[i] * self.alg.PAR[j] else 1
        return (j, i), c * s

    def act(self, x, vec):
        """x: basis index; vec: {mon: coeff}. returns {mon: coeff}."""
        alg, p, PAR = self.alg, self.p, self.alg.PAR
        out = {}
        for (i, j), c in vec.items():
            for k, a in alg.SC.get((x, i), {}).items():          # [x,b_i] b_j
                if k == j and PAR[k] == 1: continue
                m, cc = self.norm(k, j, c * a)
                out[m] = (out.get(m, 0) + cc) % p
            sg = -1 if PAR[x] * PAR[i] else 1                  # (-1)^{p(x)p(b_i)} b_i [x,b_j]
            for k, a in alg.SC.get((x, j), {}).items():
                if k == i and PAR[k] == 1: continue
                m, cc = self.norm(i, k, sg * c * a)
                out[m] = (out.get(m, 0) + cc) % p
        return {m: v for m, v in out.items() if v}

    def actv(self, xv, vec):
        out = {}
        for x, a in xv.items():
            for m, c in self.act(x, vec).items():
                out[m] = (out.get(m, 0) + a * c) % self.p
        return {m: v for m, v in out.items() if v}


def adj_act(alg, p):
    class Adj:
        def actv(self, xv, vec): return alg.br(xv, vec, p)
    return Adj()


def casimir(mod, dual, vec, sgnmode):
    """Omega vec = sum_k b_k (b^k vec) [mode 0] or sum_k b^k (b_k vec) [mode 1]."""
    out = {}
    p = None
    for k, dk in enumerate(dual):
        if sgnmode == 0:
            w = mod.actv({k: 1}, mod.actv(dk, vec))
        else:
            w = mod.actv(dk, mod.actv({k: 1}, vec))
        for m, c in w.items():
            out[m] = out.get(m, 0) + c
    return out


def run(alg, p, log, functional, label):
    N = alg.N
    ev = sum(1 for x in alg.PAR if x == 0)
    log("== %s over F_%s   sdim (%d|%d)" % (label, p if p != BIG else "BIG(char 0 proxy)", ev, N - ev))
    forms = invariant_forms(alg, p)
    log("   even supersymmetric invariant forms: %d" % len(forms))
    dual = None
    for Bm in forms:
        dual = dual_basis(alg, Bm, p)
        if dual: break
    if not dual:
        log("   no NIS -> no quadratic Casimir from a form; stop"); return None
    Bm = forms[forms.index(Bm)]
    # normalize: B on the highest root pair / e-type: B(h_1,h_1)=2 if possible
    def wf(w): return sum(F(a) * b for a, b in zip(w, functional))
    nplus = [i for i in range(N) if wf(alg.WT[i]) > 0]
    # Casimir on g: pick the sign mode that commutes with ad
    A = adj_act(alg, p)
    mode = None
    for sm in (0, 1):
        ok = True
        rnd = random.Random(7)
        for x in rnd.sample(range(N), min(N, 25)):
            for b in rnd.sample(range(N), min(N, 25)):
                l = casimir(A, dual, alg.br({x: 1}, {b: 1}, p), sm)
                r = alg.br({x: 1}, casimir(A, dual, {b: 1}, sm), p)
                if any((l.get(k, 0) - r.get(k, 0)) % p for k in set(l) | set(r)):
                    ok = False; break
            if not ok: break
        if ok: mode = sm; break
    assert mode is not None, "Casimir not central"
    top = max(range(N), key=lambda i: wf(alg.WT[i]))
    cg = casimir(A, dual, {top: 1}, mode)
    cg = {k: v % p for k, v in cg.items() if v % p}
    assert set(cg) <= {top}
    c_g = cg.get(top, 0)
    log("   Casimir on g: %d   (form normalized by B as solved; order mode %d)" % (c_g, mode))
    S = S2(alg, p)
    log("   dim S^2 = %d, weights %d, |n_+| = %d" % (len(S.mons), len(S.bywt), len(nplus)))
    sing = []
    for w, mons in S.bywt.items():
        col = {m: k for k, m in enumerate(mons)}
        tgt = {}
        rows = []
        imgs = []
        for x in nplus:
            for k, m in enumerate(mons):
                imgs.append((x, k, S.act(x, {m: 1})))
        targets = {}
        for x, k, img in imgs:
            for mm, c in img.items():
                key = (x, mm)
                if key not in targets: targets[key] = len(targets)
        if not targets:
            K = [[int(i == j) for j in range(len(mons))] for i in range(len(mons))]
        else:
            M = [[0] * len(mons) for _ in range(len(targets))]
            for x, k, img in imgs:
                for mm, c in img.items():
                    M[targets[(x, mm)]][k] = c % p
            K = nullspace(M, p)
        if K:
            vecs = [{mons[j]: int(v[j]) % p for j in range(len(mons)) if int(v[j]) % p} for v in K]
            # Casimir on the span
            Cm = []
            for v in vecs:
                cv = casimir(S, dual, v, mode)
                cv = {m: c % p for m, c in cv.items() if c % p}
                Cm.append(cv)
            # express Cm in the basis vecs (it must stay in the span)
            basisM = [[v.get(m, 0) for m in mons] for v in vecs]
            mat = []
            for cv in Cm:
                # solve x * basisM = cv
                from exactlin import solve
                T = [list(col) for col in zip(*basisM)]   # columns = vecs
                sol = solve(T, [cv.get(m, 0) for m in mons], p)
                assert sol is not None, "Casimir leaves the singular span"
                mat.append([int(s) % p for s in sol])
            sing.append((w, vecs, mat))
    sing.sort(key=lambda t: -wf(t[0]))
    squares = [t for t in sing if len(t[1]) == 1 and len(t[1][0]) == 1
               and list(t[1][0])[0][0] == list(t[1][0])[0][1]
               and list(t[1][0])[0][0] != top]
    if squares:
        log("   %d singular vectors are single squares b^2 (b != the top root vector);"
            " Casimir on them: %s" % (len(squares), sorted({t[2][0][0] for t in squares})))
    for w, vecs, mat in sing:
        if any(w is t[0] for t in squares): continue
        if len(vecs) == 1:
            ev_s = str(mat[0][0])
        else:
            ev_s = "matrix " + str(mat)
        lead = sorted(vecs[0].items())[:4]
        log("   singular: wt %s  mult %d  Casimir %s   e.g. %s%s" % (
            tuple(int(x) if F(x).denominator == 1 else str(x) for x in w), len(vecs), ev_s,
            lead, " ..." if len(vecs[0]) > 4 else ""))
    if "--cr" in sys.argv[0:0] or CR:
        nminus = [i for i in range(N) if wf(alg.WT[i]) < 0]
        gens = []
        for w, vecs, mat in sing:
            for v in vecs: gens.append((w, v))
        total = {}
        dims = []
        for w0, v0 in gens:
            M = {w0: [v0]}
            queue = [w0]
            seen = {w0}
            # process weights in decreasing functional order
            import heapq
            heap = [(-wf(w0), w0)]
            done = set()
            while heap:
                _, w = heapq.heappop(heap)
                if w in done: continue
                done.add(w)
                for y in nminus:
                    w2 = tuple(a + b for a, b in zip(w, alg.WT[y]))
                    imgs = [S.act(y, v) for v in M[w]]
                    imgs = [u for u in imgs if u]
                    if not imgs: continue
                    cur = M.get(w2, [])
                    mons = sorted({m for u in cur + imgs for m in u})
                    rows = [[u.get(m, 0) for m in mons] for u in cur + imgs]
                    from exactlin import rref
                    R, piv = rref(rows, p)
                    newb = [{mons[j]: int(r[j]) % p for j in range(len(mons)) if int(r[j]) % p}
                            for r in R[:len(piv)]]
                    if len(newb) > len(cur):
                        M[w2] = newb
                        if w2 in done: done.discard(w2)
                        heapq.heappush(heap, (-wf(w2), w2))
            d = sum(len(b) for b in M.values())
            od = sum(len(b) for w, b in M.items() if b and
                     (alg.PAR[next(iter(b[0]))[0]] + alg.PAR[next(iter(b[0]))[1]]) % 2)
            dims.append("(%d|%d)" % (d - od, od) if any(alg.PAR) else d)
            for w, b in M.items(): total.setdefault(w, []).extend(b)
        SUM = sum(len(b) for b in total.values())
        tot = 0
        for w, b in total.items():
            mons = sorted({m for u in b for m in u})
            tot += rank([[u.get(m, 0) for m in mons] for u in b], p)
        log("   submodules generated by the singular vectors: dims %s, sum %d, dim of their sum %d, dim S^2 %d -> %s" % (
            dims, SUM, tot, len(S.mons),
            "S^2 = their direct sum (completely reducible)" if tot == SUM == len(S.mons) else "NOT the direct sum of them"))
    # Vogel's parameters: C|_g = 2t, C|_{Y(x)} = 4t - 2x  (p != 2), t = sum
    nontriv = [(w, mat[0][0]) for w, vecs, mat in sing if len(vecs) == 1 and any(w)]
    if p != 2 and c_g:
        i2 = pow(2, -1, p)
        t = c_g * i2 % p
        par = [((4 * t - c) * i2) % p for w, c in nontriv]
        if len(par) == 2:
            par.append((t - par[0] - par[1]) % p)
        sym = lambda v: v if v <= p // 2 else v - p
        it = pow(t, -1, p)
        log("   t = %d, (alpha,beta,gamma) = %s ; normalized t=1: %s" % (
            sym(t), [sym(x) for x in par], [sym(x * it % p) for x in par]))
        if len(par) == 3:
            log("   check t = alpha+beta+gamma: %s" % ((sum(par) - t) % p == 0))
    elif p != 2:
        i2 = pow(2, -1, p)
        par = [(-c * i2) % p for w, c in nontriv]
        sym = lambda v: v if v <= p // 2 else v - p
        log("   t = 0, (alpha,beta,...) up to a common factor = %s" % [sym(x) for x in par])
    return c_g, sing


def main():
    out = open(os.path.join(_HERE, "vogele.log"), "a")
    def log(s):
        print(s); out.write(s + "\n"); out.flush()
    what = [a for a in sys.argv[1:] if not a.startswith('--')]
    for job in what:
        name, p = job.split(":")
        p = BIG if p == "0" else int(p)
        quot = name.endswith("/z")
        base = name[:-2] if quot else name
        if base.startswith("eA"):
            alg, npos = e_type(int(base[2:])); base_n = int(base[2:])
            functional = [1] * base_n
        elif base.startswith("e"):
            alg, npos = e_type(int(base[1:]))
            functional = [10 ** (len(E[int(base[1:])]) - i) for i in range(len(E[int(base[1:])]))]
            functional = [1] * len(functional)   # height; e-roots positive iff coords >= 0
        else:
            alg = ab3()
            functional = [F(1000), F(100), F(10), F(1)]
        alg = reduce(alg, p)
        if name.startswith("eA"):
            alg, z, k = with_d(alg, int(base[2:]), p)
            log("   g(A): ker A mod p = %s; d added with alpha_%d(d) = 1" % (z, k + 1))
        if alg.N <= 140:
            bad, cnt = jacobi_bad(alg, p)
        else:
            bad, cnt = jacobi_bad(alg, p, sample=400000)
        log("== %s, p=%s: super-Jacobi on %d triples, failures %d" % (name, p, cnt, bad))
        assert bad == 0
        if quot:
            alg, Z = quotient_by_center(alg, p)
            log("   center: dim %d; working in g/z of dim %d" % (len(Z), alg.N))
        run(alg, p, log, functional, name)


if __name__ == "__main__":
    main()
