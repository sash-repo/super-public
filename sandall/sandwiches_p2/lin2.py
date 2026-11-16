#!/usr/bin/env python3
r"""
lin2.py --- the small F_2 linear-algebra kit shared by the isomorphism engines
of this directory (`eick.py`, `torus2.py`).

Vectors are Python ints (bit i = coordinate i); a "space" is a list of masks in
row echelon form, sorted by decreasing pivot.
"""


def bits(m):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


def echelon(vectors):
    """Row echelon basis (list of masks, decreasing pivots)."""
    basis = []
    for v in vectors:
        for b in basis:
            if v >> (b.bit_length() - 1) & 1:
                v ^= b
        if v:
            basis.append(v)
            basis.sort(key=lambda u: -u.bit_length())
    return basis


def reduce_vec(v, basis):
    for b in basis:
        if v and v >> (b.bit_length() - 1) & 1:
            v ^= b
    return v


def in_span(v, basis):
    return reduce_vec(v, basis) == 0


def dim(vectors):
    return len(echelon(vectors))


def intersect(A, B, n=None):
    """Zassenhaus: the intersection of two subspaces of F_2^n."""
    A, B = echelon(A), echelon(B)
    if not A or not B:
        return []
    if n is None:
        n = max([a.bit_length() for a in A] + [b.bit_length() for b in B])
    rows = [(a << n) | a for a in A] + [b << n for b in B]
    ech = []
    for r in rows:
        for e in ech:
            if r and r >> (e.bit_length() - 1) & 1:
                r ^= e
        if r:
            ech.append(r)
            ech.sort(key=lambda u: -u.bit_length())
    mask = (1 << n) - 1
    return echelon([r for r in ech if r and r.bit_length() <= n])


def rref(rows):
    """Reduced row echelon form; returns {pivot: row}."""
    piv = {}
    for r in rows:
        while r:
            p = r.bit_length() - 1
            if p in piv:
                r ^= piv[p]
            else:
                piv[p] = r
                break
    # back-substitute
    for p in sorted(piv):
        r = piv[p]
        for q in sorted(piv):
            if q > p and piv[q] >> p & 1:
                piv[q] ^= r
    return piv


def solve_null(rows, nvars):
    """A basis of the null space of the system given by `rows`
    (row i is a mask of the variables occurring in equation i)."""
    piv = rref(rows)
    sols = []
    for u in range(nvars):
        if u in piv:
            continue
        sol = 1 << u
        for p, r in piv.items():
            if r >> u & 1:
                sol |= 1 << p
        sols.append(sol)
    # verify
    for s in sols:
        for r in rows:
            assert bin(r & s).count('1') % 2 == 0, 'null space solver is wrong'
    return sols
