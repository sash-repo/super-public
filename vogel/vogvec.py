#!/usr/bin/env python3
"""Where the vectorial Lie algebras with a NIS sit on Vogel's plane.

For a simple Lie algebra g over F_p with a unique (up to a factor) NIS B:
  * C = sum_k ad(b_k) ad(b^k) on g: a scalar c_g = 2t (Vogel's normalization);
  * the split Casimir Omega(a.b) = sum_k [b_k,a].[b^k,b] on S^2(g);
    Vogel: Omega = -x on the summand Y_2(x), x in {alpha, beta, gamma}, Omega = -2t on 1.
  * Omega commutes with the torus, so it is block-diagonal by the additive grading key
    modulo p and modulo the key lam of the NIS (B pairs key mu with lam - mu);
    per block: characteristic and minimal polynomials over F_p (FLINT), factored.
Output: generalized eigenvalues x = -Omega with multiplicities (dims), Jordan blocks,
the value 2t on g.  Lie algebras only (all algebras below are even).
Usage: vogvec.py vect1:3:1 vect1:3:2 ham:3:1,1 ham:5:1,1 my sm2 sby mel ...
Output also appended to vogvec.log."""
import sys, os, io, contextlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from exactlin import nullspace, rank
from flint import nmod_mat, nmod_poly

def binom(n, k):
    if k < 0 or k > n: return 0
    r = 1
    for i in range(k): r = r * (n - i) // (i + 1)
    return r

# ---------------------------------------------------------------- algebras
def alg_vect1(p, N):
    top = p ** N
    basis = list(range(top))                 # x^{(a)} d
    rows = [[{} for _ in basis] for _ in basis]
    for a in basis:
        for b in basis:
            if a + b - 1 < 0 or a + b - 1 >= top: continue
            c = (binom(a + b - 1, a) - binom(a + b - 1, b)) % p
            if c: rows[a][b] = {a + b - 1: c}
    keys = [(a - 1,) for a in basis]
    return rows, keys

def alg_ham2(p, Ns):
    tx, ty = p ** Ns[0], p ** Ns[1]
    mons = [(a, b) for a in range(tx) for b in range(ty) if (a, b) not in ((0, 0), (tx - 1, ty - 1))]
    idx = {m: i for i, m in enumerate(mons)}
    rows = [[{} for _ in mons] for _ in mons]
    for i, (a, b) in enumerate(mons):
        for j, (c, d) in enumerate(mons):
            # {f,g} = f_x g_y - f_y g_x on divided powers
            out = {}
            if a >= 1 and d >= 1:
                m = (a - 1 + c, b + d - 1); co = binom(a - 1 + c, c) * binom(b + d - 1, b)
                out[m] = out.get(m, 0) + co
            if b >= 1 and c >= 1:
                m = (a + c - 1, b - 1 + d); co = binom(a + c - 1, a) * binom(b - 1 + d, d)
                out[m] = out.get(m, 0) - co
            dd = {}
            for m, co in out.items():
                if co % p and m in idx: dd[idx[m]] = co % p
            rows[i][j] = dd
    keys = [(a - 1, b - 1) for (a, b) in mons]
    return rows, keys

def alg_skr(kind, derived):
    from prolong import make, GradedAlg
    with contextlib.redirect_stdout(io.StringIO()):
        g = make(kind, verbose=False)
        if derived:
            from run_derived import derived_basis
            g = GradedAlg(g.V, derived_basis(g), kind + "'")
            if derived == 2:
                g = GradedAlg(g.V, derived_basis(g), kind + "''")
    os.chdir(HERE)
    rows = [g.row(i) for i in range(g.n)]
    keys = [(g.deg[i],) + tuple(g.wt[i]) for i in range(g.n)]
    return rows, keys

def alg_mel(full=True):
    import mel5
    with contextlib.redirect_stdout(io.StringIO()):
        mats, forms = mel5.setup()
        K = mel5.Contact(forms[0], (5, 5, 5, 5, 5))
        gm2 = [mel5.mono(0, 0, 0, 0, 0)]
        gm1 = [{mel5.lin_index(n): 1} for n in range(4)]
        H = [mel5.quad_hamiltonian(K, A) for A in mats]
        t = mel5.mono(1, 0, 0, 0, 0)
        comps = mel5.prolong(K, gm2, gm1, [t] + (H if full else H[:3]))
        G = mel5.Graded(K, comps)
    os.chdir(HERE)
    n = G.n
    rows = [[{} for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            f = K.br(G.basis[i], G.basis[j])
            f = {m: c % 5 for m, c in f.items() if c % 5}
            if f: rows[i][j] = G.coords(f, G.fdeg[i] + G.fdeg[j])
    keys = [(d,) for d in G.fdeg]
    return rows, keys

def quotient_center(rows, keys, p):
    import vogele
    SC = {(i, j): d for i in range(len(rows)) for j, d in enumerate(rows[i]) if d}
    alg = vogele.Alg("x", [0] * len(rows), keys, SC)
    q, Z = vogele.quotient_by_center(alg, p)
    n = q.N
    r2 = [[q.SC.get((i, j), {}) for j in range(n)] for i in range(n)]
    return r2, list(q.WT), len(Z)

# ---------------------------------------------------------------- analysis
def add(a, b): return tuple(x + y for x, y in zip(a, b))

def nis(rows, keys, p):
    n = len(rows)
    sectors = {}
    for i in range(n):
        for j in range(i, n):
            sectors.setdefault(add(keys[i], keys[j]), []).append((i, j))
    eqs = {}
    for x in range(n):
        rx = rows[x]
        for i in range(n):
            for j in range(i, n):
                ci, cj = rx[i], rx[j]
                if not ci and not cj: continue
                lam = add(add(keys[x], keys[i]), keys[j])
                e = {}
                for k, c in ci.items():
                    pr = (k, j) if k <= j else (j, k); e[pr] = (e.get(pr, 0) + c) % p
                for k, c in cj.items():
                    pr = (i, k) if i <= k else (k, i); e[pr] = (e.get(pr, 0) + c) % p
                e = {a: c for a, c in e.items() if c}
                if e: eqs.setdefault(lam, []).append(e)
    out = []
    for lam, unk in sectors.items():
        pos = {pr: t for t, pr in enumerate(unk)}
        M = []
        for e in eqs.get(lam, []):
            r = [0] * len(unk)
            for pr, c in e.items(): r[pos[pr]] = c
            M.append(r)
        ns = nullspace(M, p) if M else [[int(a == b) for a in range(len(unk))] for b in range(len(unk))]
        for v in ns:
            out.append((lam, {unk[t]: int(c) % p for t, c in enumerate(v) if int(c) % p}))
    return out

def classkey(k, lam, p):
    k = [x % p for x in k]; l = [x % p for x in lam]
    piv = next((i for i in range(len(l)) if l[i]), None)
    if piv is not None:
        c = k[piv] * pow(l[piv], -1, p) % p
        k = [(a - c * b) % p for a, b in zip(k, l)]
    return tuple(k)

def analyse(name, rows, keys, p, log):
    t0 = time.time()
    n = len(rows)
    forms = nis(rows, keys, p)
    G = None
    for lam, f in forms:
        M = [[0] * n for _ in range(n)]
        for (i, j), c in f.items(): M[i][j] = c; M[j][i] = c
        if rank(M, p) == n: G = M; LAM = lam; break
    log("== %s over F_%d: dim %d, invariant symmetric forms %d, NIS %s"
        % (name, p, n, len(forms), "yes, sector %s" % (LAM,) if G else "NO"))
    if G is None: return
    Gi = nmod_mat(G, p).inv()
    dual = [{j: int(Gi[k, j]) for j in range(n) if int(Gi[k, j])} for k in range(n)]
    def br(u, v):
        out = {}
        for i, a in u.items():
            for j, b in v.items():
                for k, c in rows[i][j].items(): out[k] = (out.get(k, 0) + a * b * c) % p
        return {k: c for k, c in out.items() if c}
    # adjoint matrices A_k[target][source]
    import numpy as np
    A = np.zeros((n, n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            for k, c in rows[i][j].items(): A[i, k, j] = c
    Ad = np.zeros_like(A)
    for k, dk in enumerate(dual):
        for j, c in dk.items(): Ad[k] = (Ad[k] + c * A[j]) % p
    Cg = sum((A[k] @ Ad[k]) % p for k in range(n)) % p
    diag = set(int(Cg[i, i]) for i in range(n))
    off = int(np.count_nonzero(Cg - np.diag(np.diag(Cg))))
    assert len(diag) == 1 and off == 0, ("C not scalar on g", diag, off)
    cg = diag.pop()
    # S^2 blocks
    mons = [(i, j) for i in range(n) for j in range(i, n)]
    blocks = {}
    for m in mons:
        blocks.setdefault(classkey(add(keys[m[0]], keys[m[1]]), LAM, p), []).append(m)
    tot = {}       # irreducible factor -> total multiplicity (dim)
    jord = {}      # factor -> max exponent in a block minpoly
    for bk, ms in blocks.items():
        pos = {m: r for r, m in enumerate(ms)}
        M = [[0] * len(ms) for _ in ms]
        for c_, (a, b) in enumerate(ms):
            # Omega(b_a . b_b) = sum_k [b_k,b_a].[b^k,b_b]
            X = A[:, :, a]        # X[k] = column [b_k, b_a]
            Y = Ad[:, :, b]       # Y[k] = [b^k, b_b]
            T = (X.T @ Y) % p     # T[u][v] = sum_k X[k][u] Y[k][v]  -> b_u (x) b_v
            S = (T + T.T) % p     # symmetric product coordinates: b_u.b_v (u<v) coef S[u][v]; u=v: T[u][u]
            nzu, nzv = np.nonzero(np.triu(S))
            for u, v in zip(nzu, nzv):
                co = int(T[u, u]) if u == v else int(S[u, v])
                if co % p:
                    M[pos[(u, v)]][c_] = (M[pos[(u, v)]][c_] + co) % p
        Mm = nmod_mat(M, p)
        cp = Mm.charpoly(); mp = Mm.minpoly()
        for f, e in cp.factor()[1]:
            key = str(f); tot[key] = tot.get(key, 0) + e * f.degree()
        for f, e in mp.factor()[1]:
            key = str(f); jord[key] = max(jord.get(key, 0), e)
    ev = []
    for key in sorted(tot):
        ev.append("%s: dim %d%s" % (key, tot[key], ", Jordan block %d" % jord[key] if jord[key] > 1 else ""))
    log("   C on g = 2t = %d (t = %d); dim S^2 = %d in %d blocks" % (cg, cg * pow(2, -1, p) % p, len(mons), len(blocks)))
    log("   factors of the char. polynomial of Omega on S^2 (x <-> Omega; Vogel: Omega=-alpha,-beta,-gamma; -2t on 1):")
    for s in ev: log("      " + s)
    log("   (%.1f s)" % (time.time() - t0))

def main():
    out = open(os.path.join(HERE, "vogvec.log"), "a")
    def log(s): print(s, flush=True); out.write(s + "\n"); out.flush()
    for arg in sys.argv[1:]:
        f = arg.split(":")
        if f[0] == "vect1":
            p, N = int(f[1]), int(f[2]); analyse("vect(1;%d)" % N, *alg_vect1(p, N), p, log)
        elif f[0] == "ham":
            p = int(f[1]); Ns = tuple(int(x) for x in f[2].split(","))
            analyse("h'(2;%s)" % (f[2],), *alg_ham2(p, Ns), p, log)
        elif f[0] == "my":
            analyse("my(6;1)", *alg_skr("my", 0), 3, log)
        elif f[0] == "sm2":
            analyse("smy''(6;1)", *alg_skr("smy", 2), 3, log)
        elif f[0] == "sby":
            r, k = alg_skr("sby", 1); r2, k2, z = quotient_center(r, k, 3)
            log("   sby'(7;1): center of dim %d factored out" % z)
            analyse("sby'(7;1)/c", r2, k2, 3, log)
        elif f[0] == "mel":
            analyse("me(5;1)", *alg_mel(True), 5, log)
        elif f[0] == "g2p5":
            analyse("g(2) [control: (g_-, cgl(2))_* in k(5;1)]", *alg_mel(False), 5, log)

if __name__ == "__main__":
    main()
