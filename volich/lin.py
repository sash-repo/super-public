"""Minimal exact linear algebra over Q (Fractions), vectors = tuples/lists."""
from fractions import Fraction as F


def zero(n):
    return [F(0)] * n


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def smul(c, a):
    return [F(c) * x for x in a]


def rref(rows):
    """row-reduce; return (list of nonzero rows in rref, pivot columns)."""
    m = [list(r) for r in rows]
    if not m:
        return [], []
    n = len(m[0])
    piv = []
    r = 0
    for c in range(n):
        s = None
        for i in range(r, len(m)):
            if m[i][c] != 0:
                s = i; break
        if s is None:
            continue
        m[r], m[s] = m[s], m[r]
        inv = F(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        piv.append(c); r += 1
        if r == len(m):
            break
    return [row for row in m[:r]], piv


def span(vs):
    b, _ = rref(vs)
    return b


def dim(vs):
    return len(span(vs))


def inspan(v, basis):
    """is v in the span of basis?"""
    b = span(list(basis))
    return dim(list(b) + [list(v)]) == len(b)


def kernel(rows, n):
    """kernel of the matrix whose rows are `rows` (each of length n),
    i.e. {x in Q^n : rows . x = 0}."""
    m, piv = rref(rows)
    free = [c for c in range(n) if c not in piv]
    out = []
    for fc in free:
        x = zero(n)
        x[fc] = F(1)
        for i, pc in enumerate(piv):
            x[pc] = -m[i][fc]
        out.append(x)
    return out


def coords_map(vs, n):
    """matrix (rows) of the projection Q^n -> Q^n / span(vs):
    returns rows R such that R.x = 0  <=>  x in span(vs)."""
    b = span(vs)
    bset, piv = rref(b)
    rows = []
    for c in range(n):
        if c in piv:
            continue
        # linear functional: x_c - sum over pivots ...  use the standard
        # complement: x in span(b) iff reducing x by b kills it.
    # simpler: build R as the kernel-of-transpose (annihilator of span)
    ann = kernel(bset, n) if bset else [[F(1) if i == j else F(0)
                                        for i in range(n)] for j in range(n)]
    # annihilator of span(b) inside the dual = kernel of the matrix b acting
    # on functionals; kernel(bset,n) computes vectors x with b.x=0, which is
    # exactly the annihilator when we identify Q^n with its dual by the
    # standard basis.
    return ann
