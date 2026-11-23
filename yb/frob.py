"""Is g Frobenius in the super sense, i.e., is there an even linear form l such that the
coboundary B(x,y) = l([x,y]) is non-degenerate (Remark 2.4)?  Then r = B^{-1} is a constant
solution of the CYBE satisfying condition (A), with no pole.
Only the coboundaries are tested here (cocyc.py computes all even 2-cocycles): the maximal rank
of l([x,y]) over a few random even l is printed.  Arithmetic mod the prime P of cocyc.py.

Run:  python3 frob.py vect:3 vect:5 svect:5 svectdef:4 h:4 h:7
  (kind:n with kind = vect, svect, svectdef as in cocyc.py, or h = h'(0|n), spanned by the
   Hamiltonian fields H_f with f a monomial of degree 1, ..., n-1)
"""
import sys, random
import os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"), os.path.dirname(os.path.abspath(__file__))]
import cocyc as C
P = C.P
rng = random.Random(11)
try:
    import exactlin
    RANK = lambda B: exactlin.rank(B, P)        # rank over F_P
except Exception:
    RANK = C.rank_modp

def ham_fields(n):
    fs = []
    for k in range(1, n):
        for a in __import__('itertools').combinations(range(n), k):
            f = {a: 1}; s = -1 if len(a) % 2 else 1
            fs.append(tuple({m: (s * c) % P for m, c in C.dleft(i, f).items()} for i in range(n)))
    return fs

def struct(fields, idx):
    """structure constants in the basis `fields` (closure checked)"""
    N = len(fields); vecs = [C.coords(X, idx) for X in fields]
    nv = len(idx)
    A = [[v.get(k, 0) for k in range(nv)] + [1 if j == i else 0 for j in range(N)] for i, v in enumerate(vecs)]
    cols = []; r = 0
    for c in range(nv):
        p = next((i for i in range(r, N) if A[i][c] % P), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; iv = pow(A[r][c], P - 2, P); A[r] = [x * iv % P for x in A[r]]
        for i in range(N):
            if i != r and A[i][c]:
                f = A[i][c]; A[i] = [(x - f * y) % P for x, y in zip(A[i], A[r])]
        cols.append(c); r += 1
    assert r == N
    S = {}
    for i in range(N):
        for j in range(N):
            w = C.coords(C.vf_br(fields[i], fields[j]), idx)
            lam = [0] * N
            for k, c in enumerate(cols):
                if w.get(c):
                    for t in range(N):
                        if A[k][nv + t]: lam[t] = (lam[t] + w[c] * A[k][nv + t]) % P
            back = {}
            for t, l in enumerate(lam):
                if l:
                    for k, c in vecs[t].items(): back[k] = (back.get(k, 0) + l * c) % P
            assert {k: v for k, v in back.items() if v} == {k: v for k, v in w.items() if v % P}, 'not closed'
            S[(i, j)] = {t: l for t, l in enumerate(lam) if l}
    return S

def frob_rank(fields, S, tries=3):
    N = len(fields); par = [C.vf_par(X) for X in fields]; best = 0
    for _ in range(tries):
        l = [rng.randrange(P) if par[k] == 0 else 0 for k in range(N)]
        B = [[sum(l[k] * c for k, c in S[(i, j)].items()) % P for j in range(N)] for i in range(N)]
        best = max(best, RANK(B))
    return best, N, par.count(0), N - par.count(0)

def report(name, fields, idx):
    try:
        S = struct(fields, idx)
    except AssertionError:
        print(name, ': not closed with this sign'); return None
    rk, N, e, o = frob_rank(fields, S)
    print(f"{name}: sdim=({e}|{o}), max rank of l([x,y]) = {rk} of {N} ->",
          "FROBENIUS" if rk == N else "not (for coboundaries)", flush=True)
    return S

if __name__ == '__main__':
    for w in sys.argv[1:]:
        kind, n = w.split(':'); n = int(n)
        vb, idx = C.build(n, 'vect')
        if kind == 'vect':
            N = len(vb)
            S = {(i, j): {k: v for k, v in C.coords(C.vf_br(vb[i], vb[j]), idx).items() if v} for i in range(N) for j in range(N)}
            rk, N, e, o = frob_rank(vb, S, tries=2)
            print(f"vect(0|{n}): sdim=({e}|{o}), max rank of l([x,y]) = {rk} of {N} ->", "FROBENIUS" if rk == N else "not (for coboundaries)", flush=True)
        elif kind == 'h':
            report(f"h'(0|{n})", ham_fields(n), idx)
        else:
            _, _, div = C.build(n, 'svect')
            for sg in (0, 1):
                if kind == 'svect':
                    cond = lambda X, sg=sg: div(X, sg)
                else:
                    top = {tuple(range(n)): 1}
                    cond = lambda X, sg=sg: C.padd(C.vf_apply(X, top), C.pmul(C.padd({(): 1}, top), div(X, sg)))
                g = C.subalg(n, cond, vb, idx)
                if report(f"{kind}(0|{n}) sign{sg}", g, idx) is not None: break
