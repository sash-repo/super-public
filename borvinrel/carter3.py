"""Our graphs against Carter's Table 3 (Compositio Math. 25 (1972) 1--59, p. 23).

Carter's Table 3 lists every connected admissible diagram together with the
characteristic polynomial of the element w it carries.  Two names on one line --
`D_l(a_k), D_l(b_k)`, `E_7(a_2), E_7(b_2)`, `E_8(a_3), E_8(b_3)`, `E_8(a_5),
E_8(b_5)` -- are two DIAGRAMS with that one polynomial; by his Proposition 25 (the
D case: same signed cycle type and not all cycles even and positive) they carry one
and the same conjugacy class, and they are not among the pairs his Lemma 26 proves
non-conjugate.

Here the multiset of characteristic polynomials our graphs produce is compared with
the multiset Carter's table predicts, lattice by lattice, for 3 <= n <= 8.  The
polynomials are the cyclotomic products of his table, expanded exactly.

Run:  python3 carter3.py
"""

import collections
import pickle
from fractions import Fraction

import vinb
from carter import weyl_element, order, charpoly, _fmt, graph_canon
from lattice import root_system_type
from permcensus import census as perm_census


# ------------------------------------------------------------------ polynomials

def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def cyclo(n):
    """The n-th cyclotomic polynomial, highest degree first, by exact division."""
    # x^n - 1 = prod_{d | n} Phi_d
    num = [1] + [0] * (n - 1) + [-1]          # x^n - 1
    for d in range(1, n):
        if n % d == 0:
            num = pdiv(num, cyclo(d))
    return num


def pdiv(a, b):
    """Exact division of polynomials given highest degree first."""
    a = list(a)
    q = []
    while len(a) >= len(b):
        c = Fraction(a[0], b[0])
        assert c.denominator == 1, (a, b)
        c = int(c)
        q.append(c)
        for i, y in enumerate(b):
            a[i] -= c * y
        assert a[0] == 0
        a.pop(0)
    assert all(x == 0 for x in a), a
    return q


def prod(*ns):
    """Product of cyclotomic polynomials Phi_{n_1} ... Phi_{n_k}."""
    out = [1]
    for n in ns:
        out = pmul(out, cyclo(n))
    return tuple(out)


# ------------------------------------------------- Carter's Table 3, as cyclotomics

def carter_table3(lat, n):
    """(name, characteristic polynomial) for the connected diagrams of rank n."""
    if lat == 'A':
        # A_n : t^n + ... + 1 = prod_{d | n+1, d > 1} Phi_d
        p = [1] * (n + 1)
        return [('A_%d' % n, tuple(p))]
    if lat == 'D':
        # D_n(a_k): (t^{n-k-1} + 1)(t^{k+1} + 1), k = 0 .. [(n-2)/2]; D_n(a_0) = D_n
        if n < 4:                       # D_3 = A_3 is not a separate lattice type
            return []
        out = []
        for k in range((n - 2) // 2 + 1):
            f = pmul([1] + [0] * (n - k - 2) + [1], [1] + [0] * k + [1])
            nm = 'D_%d' % n if k == 0 else 'D_%d(a_%d)' % (n, k)
            out.append((nm, tuple(f)))
            if 2 * (k + 1) == n and k + 1 >= 3:
                out.append(('D_%d(b_%d)' % (n, k), tuple(f)))
        return out
    if (lat, n) == ('E', 6):
        return [('E_6', prod(12, 3)), ('E_6(a_1)', prod(9)),
                ('E_6(a_2)', prod(6, 6, 3))]
    if (lat, n) == ('E', 7):
        return [('E_7', prod(18, 2)), ('E_7(a_1)', prod(14, 2)),
                ('E_7(a_2)', prod(12, 6, 2)), ('E_7(b_2)', prod(12, 6, 2)),
                ('E_7(a_3)', prod(10, 6, 2)), ('E_7(a_4)', prod(6, 6, 6, 2))]
    if (lat, n) == ('E', 8):
        return [('E_8', prod(30)), ('E_8(a_1)', prod(24)), ('E_8(a_2)', prod(20)),
                ('E_8(a_3)', prod(12, 12)), ('E_8(b_3)', prod(12, 12)),
                ('E_8(a_4)', prod(18, 6)),
                ('E_8(a_5)', prod(15)), ('E_8(b_5)', prod(15)),
                ('E_8(a_6)', prod(10, 10)), ('E_8(a_7)', prod(12, 6, 6)),
                ('E_8(a_8)', prod(6, 6, 6, 6))]
    return []


def poly_order(cp):
    """Order of a matrix with this characteristic polynomial, all roots being roots
    of unity: the lcm of the orders of the cyclotomic factors."""
    from math import lcm
    rem = list(cp)
    o = 1
    for d in range(1, 200):
        f = cyclo(d)
        while len(rem) >= len(f):
            try:
                q = pdiv(rem, f)
            except AssertionError:
                break
            rem = q
            o = lcm(o, d)
        if len(rem) == 1:
            break
    assert rem == [1], rem
    return o


# --------------------------------------------------------------------------- main

def ours():
    """Our Carter graphs, by lattice and size: (size, lattice) -> [charpoly]."""
    out = collections.defaultdict(list)
    reps, _ = perm_census(7, verbose=False)
    for n in range(3, 8):
        seen = {}
        for A in reps[n]:
            if vinb.is_connected(A) and vinb.is_bipartite(A):
                seen.setdefault(graph_canon(A), A)
        for A in seen.values():
            out[(n, root_system_type(A)[0][0])].append(tuple(charpoly(weyl_element(A))))
    with open('graphs8.pkl', 'rb') as f:
        info = pickle.load(f)
    for g, (cnt, lat, bip, A) in info.items():
        if bip:
            out[(8, lat[0])].append(tuple(charpoly(weyl_element(A))))
    return out


def main():
    mine = ours()
    allok = True
    for n in range(3, 9):
        for lat in ('A', 'D', 'E'):
            got = mine.get((n, lat))
            want = carter_table3(lat, n)
            if not got and not want:
                continue
            gc = collections.Counter(got or [])
            wc = collections.Counter(p for _, p in want)
            ok = gc == wc
            allok &= ok
            print('n=%d  %s : %d graphs, Carter lists %d   %s'
                  % (n, lat, sum(gc.values()), len(want), 'agree' if ok else 'DIFFER'))
            for nm, p in want:
                print('        %-12s order %-3d %s' % (nm, poly_order(p), _fmt(list(p))))
            if not ok:
                for p in gc:
                    if gc[p] != wc.get(p, 0):
                        print('        ours has %d of %s, Carter %d'
                              % (gc[p], _fmt(list(p)), wc.get(p, 0)))
        print()
    print('every lattice and size agrees with Carter Table 3: %s' % ('yes' if allok else 'NO'))


if __name__ == '__main__':
    main()
