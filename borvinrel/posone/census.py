"""Statement 6.4.3 of the paper: symmetric integer matrices with A_ii in {1,2} (at least one 1),
indecomposable, violating c) (some positive off-diagonal entry), of type Pos or Z.
Census up to simultaneous permutation, n<=NMAX; lattice Z^m + R identified for Pos."""
import itertools, sys
from fractions import Fraction
from sympy import Matrix

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def det(M):
    n = len(M); A = [[Fraction(x) for x in r] for r in M]; d = Fraction(1)
    for i in range(n):
        p = next((r for r in range(i, n) if A[r][i] != 0), None)
        if p is None: return Fraction(0)
        if p != i: A[i], A[p] = A[p], A[i]; d = -d
        d *= A[i][i]
        for r in range(i+1, n):
            f = A[r][i]/A[i][i]
            if f: A[r] = [a-f*b for a, b in zip(A[r], A[i])]
    return d

def minors_ok(M, strict_full):
    n = len(M)
    for k in range(1, n+1):
        for S in itertools.combinations(range(n), k):
            d = det([[M[i][j] for j in S] for i in S])
            if k < n and d <= 0: return False
            if k == n and ((d <= 0) if strict_full else (d != 0)): return False
    return True

def connected(M):
    n = len(M); seen = {0}; st = [0]
    while st:
        i = st.pop()
        for j in range(n):
            if j not in seen and M[i][j]: seen.add(j); st.append(j)
    return len(seen) == n

def canon(M):
    n = len(M)
    return min(tuple(M[p[i]][p[j]] for i in range(n) for j in range(n))
               for p in itertools.permutations(range(n)))

def lattice(M):
    """m = half the number of norm-1 vectors; R from rank and number of roots orthogonal to them."""
    n = len(M); G = Matrix(M)
    # enumerate short vectors by brute force in a box (Pos only; coordinates small for n<=5)
    B = 3
    n1, roots = [], []
    for c in itertools.product(range(-B, B+1), repeat=n):
        if not any(c): continue
        v = Matrix(c); q = (v.T*G*v)[0]
        if q == 1: n1.append(v)
        elif q == 2: roots.append(v)
    R = [r for r in roots if all((r.T*G*w)[0] == 0 for w in n1)]
    return len(n1)//2, len(R)

for n in range(1, NMAX+1):
    seen = {'Pos': set(), 'Z': set()}
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    for diag in itertools.product((1, 2), repeat=n):
        if 1 not in diag: continue
        for off in itertools.product((-1, 0, 1), repeat=len(pairs)):
            M = [[0]*n for _ in range(n)]
            for i in range(n): M[i][i] = diag[i]
            ok = True
            for (i, j), a in zip(pairs, off):
                if a and diag[i] == 1 and diag[j] == 1 and n > 2: ok = False; break
                M[i][j] = M[j][i] = a
            if not ok or 1 not in off or not connected(M): continue
            for typ, strict in (('Pos', True), ('Z', False)):
                if minors_ok(M, strict): seen[typ].add(canon(M))
    # n=2, Z, with |A12|=2 is outside the box {-1,0,1}: handled by hand in the text
    print(f"n={n}: Pos {len(seen['Pos'])}, Z {len(seen['Z'])}", flush=True)
    if n <= 4:
        from collections import Counter
        C = Counter(lattice([list(c[i*n:(i+1)*n]) for i in range(n)]) for c in seen['Pos'])
        print("   lattices (m, #roots of R):", dict(sorted(C.items())), flush=True)
