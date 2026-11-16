"""
Odd (isotropic) reflections and breadth-first enumeration of the inequivalent
Cartan matrices of a Lie superalgebra -- i.e. an explicit computation of its
Weyl groupoid.

Implements formulas (oddrefl)/(Boddrefl) of the paper:

    r_k(sigma_j) = -sigma_j                      if j = k,
                   sigma_j + B_kj sigma_k        if j != k,

    B_kj = -2A_kj/A_kk   if A_kk != 0 and -2A_kj/A_kk in Z_+,
           1             if sigma_k odd, A_kk = 0, A_kj != 0,
           0             if A_kj = 0.

If A_kk != 0 and -2A_kj/A_kk is not a non-negative integer the reflection is
undefined -- that is exactly why the reflections form a GROUPOID and not a
group, and the BFS below simply has no outgoing edge there.

Since sigma'_i = sum_j R_ij sigma_j, the new Cartan matrix is  A' = R A R^T.

Parities: reflecting in an odd isotropic sigma_k flips the parity of every
sigma_j with A_kj != 0 (because sigma'_j = sigma_j + sigma_k); sigma_k itself
stays odd.
"""

from fractions import Fraction
from itertools import permutations
from collections import deque

ODD, EVEN = 1, 0


def refl_matrix(A, par, k):
    """R for the reflection in sigma_k, or None if it is not defined."""
    n = len(A)
    if par[k] != ODD or A[k][k] != 0:
        return None                      # we only use isotropic odd ones
    R = [[Fraction(1) if i == j else Fraction(0) for j in range(n)]
         for i in range(n)]
    for j in range(n):
        if j == k:
            continue
        R[j][k] = Fraction(1) if A[k][j] != 0 else Fraction(0)
    for j in range(n):
        R[k][j] = Fraction(-1) if j == k else Fraction(0)
    return R


def apply_refl(A, par, k):
    R = refl_matrix(A, par, k)
    if R is None:
        return None
    n = len(A)
    RA = [[sum(R[i][t] * A[t][j] for t in range(n)) for j in range(n)]
          for i in range(n)]
    A2 = [[sum(RA[i][t] * R[j][t] for t in range(n)) for j in range(n)]
          for i in range(n)]
    par2 = list(par)
    for j in range(n):
        if j != k and A[k][j] != 0:
            par2[j] = par[j] ^ ODD
    return A2, tuple(par2)


def normalize(A, par):
    """Sec. 1.4 normalization: A' = diag(lambda) A with A'_ii = 2 (ev) / 1 (od),
    and a zero-diagonal row scaled so that its first nonzero entry is 1."""
    n = len(A)
    out = []
    for i in range(n):
        row = list(A[i])
        if row[i] != 0:
            lam = Fraction(2 if par[i] == EVEN else 1, 1) / row[i]
        else:
            nz = next((x for x in row if x != 0), None)
            lam = Fraction(1) if nz is None else Fraction(1) / nz
        out.append([lam * x for x in row])
    return out


def canonical(A, par):
    """Least representative under relabelling + the rescaling of Subsec. 3.4 of the paper."""
    n = len(A)
    best = None
    for p in permutations(range(n)):
        Ap = [[A[p[i]][p[j]] for j in range(n)] for i in range(n)]
        pp = tuple(par[p[i]] for i in range(n))
        An = normalize(Ap, pp)
        key = (pp, tuple(tuple(row) for row in An))
        if best is None or key < best:
            best = key
    return best


def bfs(A0, par0, maxnodes=100000):
    """All Cartan matrices reachable by odd reflections, up to relabelling."""
    start = canonical(A0, par0)
    seen = {start: (A0, par0)}
    order = [start]
    q = deque([(A0, par0)])
    edges = []
    while q:
        A, par = q.popleft()
        ca = canonical(A, par)
        for k in range(len(A)):
            res = apply_refl(A, par, k)
            if res is None:
                continue
            A2, par2 = res
            c2 = canonical(A2, par2)
            edges.append((ca, k, c2))
            if c2 not in seen:
                seen[c2] = (A2, par2)
                order.append(c2)
                q.append((A2, par2))
                if len(seen) > maxnodes:
                    raise RuntimeError('orbit too large')
    return order, seen, edges


# ------------------------------------------------------ eps-delta models ---


def word_to_cartan(word):
    """Simple roots sigma_i = w_i - w_{i+1} for a word in 'e' (eps) / 'd' (delta).

    (eps_i,eps_j) = delta_ij,  (delta_i,delta_j) = -delta_ij,  (eps,delta)=0.
    """
    n = len(word) - 1
    vecs = []
    for i in range(n):
        v = [0] * len(word)
        v[i], v[i + 1] = 1, -1
        vecs.append(v)
    sign = [1 if ch == 'e' else -1 for ch in word]

    def ip(u, v):
        return sum(sign[t] * u[t] * v[t] for t in range(len(word)))

    A = [[Fraction(ip(vecs[i], vecs[j])) for j in range(n)] for i in range(n)]
    par = tuple(ODD if word[i] != word[i + 1] else EVEN for i in range(n))
    return A, par


def shuffles(word):
    return sorted(set(''.join(p) for p in permutations(word)))


def fmt(A, par):
    names = {EVEN: 'ev', ODD: 'od'}
    rows = ['[' + ' '.join('%4s' % str(x) for x in row) + ']' for row in A]
    return ' | '.join(rows) + '   parity ' + ','.join(names[p] for p in par)


def report(name, A, par):
    order, seen, edges = bfs(A, par)
    print('\n%s' % name)
    print('  start: %s' % fmt(A, par))
    print('  inequivalent Cartan matrices reachable by odd reflections: %d'
          % len(order))
    for i, key in enumerate(order):
        Ai, pi = seen[key]
        print('   %2d. %s' % (i + 1, fmt(Ai, pi)))
    return order, seen, edges


def classes(word):
    """Shuffles of the word up to reversal of the diagram and, for m = n, also up to
    the interchange eps <-> delta (which only changes the sign of the form, i.e., is
    absorbed by the rescaling of the rows)."""
    swap = str.maketrans('ed', 'de')
    out = set()
    for w in shuffles(word):
        cands = [w, w[::-1]]
        if word.count('e') == word.count('d'):
            cands += [w.translate(swap), w[::-1].translate(swap)]
        out.add(min(cands))
    return out


if __name__ == '__main__':
    print('=' * 78)
    print('Statement 11.6.2: sl(m|n) against shuffles of the eps/delta word')
    print('=' * 78)
    for word in ['eed', 'eeed', 'eedd', 'eeedd', 'eeeedd']:
        A, par = word_to_cartan(word)
        order, seen, edges = bfs(A, par)
        ws = shuffles(word)
        cl = classes(word)
        print('  sl(%d|%d)  shuffles %3d   classes among shuffles %3d   orbit of the search %3d   %s'
              % (word.count('e'), word.count('d'), len(ws), len(cl), len(order),
                 'MATCH' if len(order) == len(cl) else 'MISMATCH'))
    print()
    print('=' * 78)
    print('Statement 11.6.3: the Weyl groupoid of (80), zero diagonal, all odd')
    print('=' * 78)
    F = Fraction
    for (x, y, z) in [(F(1), F(2), F(-5)), (F(2), F(3), F(7)), (F(1), F(1), F(-3)),
                      (F(1), F(1), F(-2))]:
        A = [[F(0), x, y], [x, F(0), z], [y, z, F(0)]]
        report('(a,b,c) = (%s,%s,%s)' % (x, y, z), A, (ODD, ODD, ODD))
