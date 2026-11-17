"""The symbol of an invariant operator, and the coinvariant algebra of S_n.

To an operator  D(u_1,...,u_n) = sum_z c_z u_1^(z_1)...u_n^(z_n)  attach its
symbol  P(X) = sum_z c_z X^z  (Malyshev's (14)).  For general weights a_i the
invariance conditions of line.py read, in the symbol,

      sum_i X_i d_i^q P  +  q sum_i a_i d_i^{q-1} P  =  0,     q >= 2,       (*)

and for a_i = 1 they collapse, by C(m,q)+C(m,q-1)=C(m+1,q), into

      E_q P := sum_i d_i^q X_i P = 0,                          q >= 2.       (**)

Tested here, exactly and over Q:
  (1) (*) and (**) reproduce line.py's spaces, so the symbol translation is right;
  (2) the graded dimensions are the Mahonian numbers -- Malyshev's conjecture;
  (3) the symbol of the Wronskian is the Vandermonde determinant;
  (4) how the space sits against the harmonics H_s (the span of the partial
      derivatives of the Vandermonde): same dimension in every degree, and
      whether or not the two subspaces coincide.
"""

import sys
from fractions import Fraction as F
from itertools import permutations
from line import inv_space, midx
from exactlin import rank, nullspace

NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ok, bad = 0, []


def chk(name, cond):
    global ok
    if cond:
        ok += 1
    else:
        bad.append(name)


def _lower(n, s, i, k):
    """matrix of d_i^k : degree s -> degree s-k, as a dict of dicts."""
    src, dst = midx(n, s), midx(n, s - k)
    pos = {z: j for j, z in enumerate(dst)}
    out = []
    for z in src:
        if z[i] >= k:
            f = 1
            for t in range(k):
                f *= z[i] - t
            t = list(z)
            t[i] -= k
            out.append((pos[tuple(t)], f))
        else:
            out.append(None)
    return out, len(dst)


def pk_rows(n, s, k):
    """rows of p_k(d) = sum_i d_i^k from degree s."""
    m = len(midx(n, s))
    nd = len(midx(n, s - k))
    rows = [[F(0)] * m for _ in range(nd)]
    for i in range(n):
        col, _ = _lower(n, s, i, k)
        for j, e in enumerate(col):
            if e:
                rows[e[0]][j] += e[1]
    return rows


def cond_rows(n, s, weights, q):
    """rows of  sum_i X_i d_i^q + q sum_i a_i d_i^{q-1}  from degree s."""
    tgt = s - q + 1
    if tgt < 0:
        return []
    src, dst = midx(n, s), midx(n, tgt)
    pos = {z: j for j, z in enumerate(dst)}
    rows = [[F(0)] * len(src) for _ in dst]
    for j, z in enumerate(src):
        for i in range(n):
            if z[i] >= q:                      # X_i d_i^q
                f = 1
                for t in range(q):
                    f *= z[i] - t
                t = list(z)
                t[i] -= q - 1
                rows[pos[tuple(t)]][j] += F(f)
            if z[i] >= q - 1:                  # q a_i d_i^{q-1}
                f = 1
                for t in range(q - 1):
                    f *= z[i] - t
                t = list(z)
                t[i] -= q - 1
                rows[pos[tuple(t)]][j] += q * weights[i] * F(f)
    return rows


def symbol_space(n, s, weights):
    rows = []
    for q in range(2, s + 2):
        rows += cond_rows(n, s, weights, q)
    m = len(midx(n, s))
    if not rows:
        return [[F(1) if i == j else F(0) for i in range(m)] for j in range(m)]
    return [list(v) for v in nullspace(rows)]


def harmonics(n, s):
    rows = []
    for k in range(1, n + 1):
        if k <= s:
            rows += pk_rows(n, s, k)
    m = len(midx(n, s))
    if not rows:
        return [[F(1) if i == j else F(0) for i in range(m)] for j in range(m)]
    return [list(v) for v in nullspace(rows)]


def same_space(A, B):
    if not A and not B:
        return True
    if not A or not B:
        return False
    return rank(A) == rank(B) == rank(A + B)


def vandermonde(n):
    s = n * (n - 1) // 2
    cols = midx(n, s)
    pos = {z: j for j, z in enumerate(cols)}
    out = [F(0)] * len(cols)
    for sigma in permutations(range(n)):
        inv = sum(1 for i in range(n) for j in range(i + 1, n)
                  if sigma[i] > sigma[j])
        out[pos[tuple(sigma)]] += F((-1) ** inv)
    return out


def mahonian(n):
    poly = [1]
    for r in range(2, n + 1):
        new = [0] * (len(poly) + r - 1)
        for i, c in enumerate(poly):
            for j in range(r):
                new[i + j] += c
        poly = new
    return poly


print(" n   s |  dim symbols   dim harmonics   a_(n-1,s)   same subspace?")
print("-" * 66)
for n in range(1, NMAX + 1):
    top = n * (n - 1) // 2
    mah = mahonian(n)
    w = [F(1)] * n
    for s in range(0, top + 2):
        S = symbol_space(n, s, w)
        H = harmonics(n, s)
        dS, dH = (rank(S) if S else 0), (rank(H) if H else 0)
        want = mah[s] if s < len(mah) else 0
        eq = same_space(S, H) if (S or H) else True
        print(f"{n:2d}  {s:2d} | {dS:12d}   {dH:13d}   {want:9d}   {eq}")
        sys.stdout.flush()
        chk(f"symbols = line.py space n={n} s={s}",
            dS == len(inv_space(tuple(w), s)))
        chk(f"dim = Mahonian n={n} s={s}", dS == want)
        chk(f"dim symbols = dim harmonics n={n} s={s}", dS == dH)

    # the Wronskian's symbol is the Vandermonde
    if n >= 2:
        S = symbol_space(n, top, w)
        V = vandermonde(n)
        chk(f"Wronskian symbol = Vandermonde n={n}",
            len(S) == 1 and rank([S[0], V]) == 1)

# general weights: the symbol form (*) must reproduce line.py there too
for w in [(F(2), F(1), F(1)), (F(1), F(2), F(3)), (F(-2, 3), F(-2, 3)),
          (F(0), F(1)), (F(2), F(1), F(1), F(1))]:
    n = len(w)
    for s in range(0, 8):
        chk(f"symbols = line.py at weights {w} s={s}",
            rank(symbol_space(n, s, list(w))) == len(inv_space(w, s)))

print()
print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  FAIL", b)
