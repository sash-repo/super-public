"""Lemma 3.2 (regular cycles), Theorem 3.4.2 (the odd tabloids), Theorem 4.1 (triples)."""

from itertools import permutations, product
from skew import (Diagram, nu, semi_ordered_of_type, tableaux_of_type,
                  is_semi_ordered, has_cycles_for, hom, tabloid, _sign,
                  _span_dim, _add, dim_S_pair, show)

PASS = FAIL = 0
line = "-" * 70


def check(name, got, want):
    global PASS, FAIL
    ok = got == want
    PASS, FAIL = PASS + ok, FAIL + (not ok)
    print(("  ok   " if ok else "  FAIL ") + f"{name}: got {got}, want {want}")


def diags(maxsize=5):
    """a stock of small skew diagrams, Young and genuinely skew."""
    out = []
    shapes = [(1,), (2,), (1, 1), (3,), (2, 1), (1, 1, 1), (4,), (3, 1), (2, 2),
              (2, 1, 1), (1, 1, 1, 1), (3, 2), (2, 2, 1)]
    for lam in shapes:
        for mu in [()] + [(a,) for a in range(1, lam[0])] + \
                  ([(lam[0], b) for b in range(1, (lam[1] if len(lam) > 1 else 0) + 1)]
                   if len(lam) > 1 else []):
            mu = tuple(x for x in mu)
            if len(mu) > len(lam):
                continue
            if any(mu[i] > lam[i] for i in range(len(mu))):
                continue
            D = Diagram.skew(lam, mu, f"{lam}/{mu}")
            if 1 <= len(D) <= maxsize:
                if D.cells not in [d.cells for d in out]:
                    out.append(D)
    return out


# =========================================================================== #
print(line)
print("Lemma 3.2: a semi-ordered tableau with cycles corresponding to B")
print("           has REGULAR cycles corresponding to B")
print(line)


def has_regular_cycles_for(A, B, tab):
    """same as has_cycles_for, but every cycle must in addition be regular:
    down a cycle the column indices weakly decrease."""
    rb = B.row_number
    pool = {}
    for c in A.cells:
        pool.setdefault(tab[c], []).append(c)
    cols = sorted([sorted(col) for col in B.columns], key=len, reverse=True)
    used, placed = set(), {}

    def place(ci, k):
        if ci == len(cols):
            return True
        col = cols[ci]
        if k == len(col):
            return place(ci + 1, 0)
        cell = col[k]
        prev = placed[col[k - 1]] if (k and len(col) > 1) else None
        for a in pool[rb[cell]]:
            if a in used:
                continue
            if len(col) > 1 and prev is not None:
                if a[0] <= prev[0]:          # strictly lower row
                    continue
                if a[1] > prev[1]:           # regular: not to the right
                    continue
            used.add(a); placed[cell] = a
            if place(ci, k + 1):
                return True
            used.discard(a); del placed[cell]
        return False

    return place(0, 0)


stock = diags(5)
tot = same = 0
for A in stock:
    for B in stock:
        if len(A) != len(B):
            continue
        for t in semi_ordered_of_type(A, B):
            if has_cycles_for(A, B, t):
                tot += 1
                same += has_regular_cycles_for(A, B, t)
check("tableaux with cycles that also have regular ones", same, tot)
print(f"       (over {len(stock)} diagrams, {tot} tableaux with cycles)")

# =========================================================================== #
print(line)
print("Theorem 3.4.2:  nu'(A,B) = dim S'(A,B) = nu(A',B) = nu(A,B')")
print(line)


def _odd_tabloid_key(A, B, g):
    """the odd tabloid [g]' of (3.8): rows of A carry ORDERED tuples up to sign,
    so the class is (sorted content per row, sign of the sorting permutation)."""
    rb = B.row_number
    tab = {g[c]: rb[c] for c in B.cells}
    key, sgn = [], 1
    for r in A.rows:
        vals = [tab[c] for c in r]
        if len(set(vals)) != len(vals):
            return None, 0            # two equal entries in a row: the odd tabloid is 0
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        # sign of the permutation sorting vals
        s, seen = 1, [False] * len(vals)
        for i in range(len(vals)):
            if seen[i]:
                continue
            L, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = order[j]
                L += 1
            if L % 2 == 0:
                s = -s
        sgn *= s
        key.append(tuple(sorted(vals)))
    return tuple(key), sgn


def dim_S_prime(A, B):
    """dim S'(A,B) of Remark 3.4.1, formula (3.9)."""
    CA, CB = A.column_stabilizer(), B.column_stabilizer()
    vecs = []
    for f in hom(A, B):
        acc = {}
        for phi, sa in CA:
            for psi, sb in CB:
                inv = {v: k for k, v in psi.items()}
                g = {c: phi[f[inv[c]]] for c in B.cells}
                key, s = _odd_tabloid_key(A, B, g)
                if key is not None:
                    _add(acc, key, sb * s)
        vecs.append(acc)
    return _span_dim(vecs)


small = [D for D in diags(4)]
for A in small:
    for B in small:
        if len(A) != len(B) or len(A) > 4:
            continue
        d = dim_S_prime(A, B)
        check(f"nu'({A},{B})", d, nu(A.transpose(), B))
        if nu(A.transpose(), B) != nu(A, B.transpose()):
            print(f"    !! nu(A',B) != nu(A,B') for {A},{B}")

# =========================================================================== #
print(line)
print("Theorem 4.1: nu(C,A,B) = sum over partitions of prod nu^{s(i)}(A_i,B_i)")
print(line)


def quasi_ordered_partitions(A, mu):
    """quasi-ordered numberings of A of type mu = the splittings of A into skew
    diagrams A_1,...,A_m of sizes mu_1,...,mu_m (Omega^mu(A))."""
    out = []
    for tab in tableaux_of_type(A, Diagram.skew(tuple(mu))):
        ok = True
        for (i, j) in A.cells:
            if (i, j + 1) in A.index and tab[(i, j)] > tab[(i, j + 1)]:
                ok = False; break
            if (i + 1, j) in A.index and tab[(i, j)] > tab[(i + 1, j)]:
                ok = False; break
        if ok:
            out.append(tab)
    return out


def pieces(A, tab, m):
    return [Diagram([c for c in A.cells if tab[c] == i + 1]) for i in range(m)]


def nu_triple_formula(kinds, mu, A, B):
    """the right-hand side of (4.2)/(4.3).  `kinds[i]` is 0 for a row of C, 1 for
    a column of C; mu[i] is the size of the i-th component."""
    m = len(mu)
    total = 0
    for ta in quasi_ordered_partitions(A, mu):
        Ai = pieces(A, ta, m)
        for tb in quasi_ordered_partitions(B, mu):
            Bi = pieces(B, tb, m)
            p = 1
            for i in range(m):
                if len(Ai[i]) != len(Bi[i]):
                    p = 0; break
                p *= nu(Ai[i], Bi[i]) if kinds[i] == 0 else nu(Ai[i].transpose(), Bi[i])
                if p == 0:
                    break
            total += p
    return total


def make_C(kinds, mu):
    """C: the i-th component is a row (kinds[i]==0) or a column, of mu[i] cells,
    the components being disjoint and staircased so that C is a skew diagram."""
    cells, i0, j0 = [], 1, 1
    for k, size in zip(kinds, mu):
        if k == 0:
            cells += [(i0, j0 + t) for t in range(size)]
            j0 += size
            i0 += 1
        else:
            cells += [(i0 + t, j0) for t in range(size)]
            i0 += size
            j0 += 1
    return Diagram(cells, "C")


def nu_triple_direct(C, A, B):
    """dim S(C,A,B) from the definition: cocycles (u: C->A, v: C->B) modulo R,
    alternated over C_C x C_A x C_B."""
    q = len(C)
    rC, rA, rB = C.row_number, A.row_number, B.row_number
    CC, CA, CB = C.column_stabilizer(), A.column_stabilizer(), B.column_stabilizer()

    def key(u, v):
        d = {}
        for c in C.cells:
            d.setdefault(rC[c], []).append((rA[u[c]], rB[v[c]]))
        return tuple(tuple(sorted(d[k])) for k in sorted(d))

    vecs = []
    for uimg in permutations(A.cells):
        u0 = dict(zip(C.cells, uimg))
        for vimg in permutations(B.cells):
            v0 = dict(zip(C.cells, vimg))
            acc = {}
            for gam, sg in CC:
                inv = {y: x for x, y in gam.items()}
                for phi, sa in CA:
                    for psi, sb in CB:
                        u = {c: phi[u0[inv[c]]] for c in C.cells}
                        v = {c: psi[v0[inv[c]]] for c in C.cells}
                        _add(acc, key(u, v), sg * sa * sb)
            vecs.append(acc)
    return _span_dim(vecs)


cases = [
    ((0,), (3,), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((1,), (3,), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((0, 0), (2, 1), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((0, 1), (2, 1), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((1, 1), (2, 1), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((0, 0), (1, 2), Diagram.skew((3,)), Diagram.skew((2, 1))),
    ((0, 0, 0), (1, 1, 1), Diagram.skew((2, 1)), Diagram.skew((2, 1))),
    ((0, 1), (2, 2), Diagram.skew((2, 2)), Diagram.skew((2, 2))),
    ((0, 0), (2, 2), Diagram.skew((3, 1)), Diagram.skew((2, 2))),
    ((1, 0), (2, 2), Diagram.skew((2, 2)), Diagram.skew((3, 1))),
]
for kinds, mu, A, B in cases:
    C = make_C(kinds, mu)
    d = nu_triple_direct(C, A, B)
    f = nu_triple_formula(kinds, mu, A, B)
    check(f"nu(C{kinds}{mu}, {A.row_sizes}, {B.row_sizes})", d, f)

print(f"\n{PASS} passed, {FAIL} failed")
