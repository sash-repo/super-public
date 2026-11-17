#!/usr/bin/env python3
"""Exact arithmetic over K = GF(2^k) = F_2[t]/(f): elements are bitmasks, the
generator a = t.  Row reduction (rank, kernel) over K.  Used by alsq.py."""

# ---------------------------------------------------------------- field ----
IRRED = {2: 0b111, 3: 0b1011, 4: 0b10011}  # t^2+t+1, t^3+t+1, t^4+t+1


class GF:
    """GF(2^k) element, bitmask representation."""
    k = 2
    mod = IRRED[2]

    __slots__ = ("v",)

    def __init__(self, v=0):
        self.v = v & ((1 << GF.k) - 1) if v < (1 << GF.k) else GF._red(v)

    @staticmethod
    def _red(v):
        m = GF.mod
        deg_m = GF.k
        while v.bit_length() - 1 >= deg_m:
            v ^= m << (v.bit_length() - 1 - deg_m)
        return v

    def __add__(self, o):
        o = _co(o)
        return GF(self.v ^ o.v)

    __radd__ = __add__
    __sub__ = __add__
    __rsub__ = __add__

    def __neg__(self):
        return self

    def __mul__(self, o):
        o = _co(o)
        x, y, r = self.v, o.v, 0
        while y:
            if y & 1:
                r ^= x
            y >>= 1
            x <<= 1
        return GF(GF._red(r))

    __rmul__ = __mul__

    def __pow__(self, n):
        r = GF(1)
        for _ in range(n):
            r = r * self
        return r

    def inv(self):
        # brute force over the (small) field
        for i in range(1, 1 << GF.k):
            if (self * GF(i)).v == 1:
                return GF(i)
        raise ZeroDivisionError

    def __truediv__(self, o):
        return self * _co(o).inv()

    def __eq__(self, o):
        return self.v == _co(o).v

    def __hash__(self):
        return self.v

    def __bool__(self):
        return self.v != 0

    def __repr__(self):
        return f"GF({self.v})"


def _co(o):
    return o if isinstance(o, GF) else GF(o & 1)


def rank(rows, n):
    rows = [r[:] for r in rows]
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, len(rows)):
            if rows[i][c]:
                piv = i
                break
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = rows[r][c].inv()
        rows[r] = [inv * v for v in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c]:
                f = rows[i][c]
                rows[i] = [x + f * y for x, y in zip(rows[i], rows[r])]
        r += 1
        if r == len(rows):
            break
    return r, [row for row in rows[:r]]


def kernel(mat, n):
    """mat: list of rows (images of basis vectors) -> basis of kernel."""
    # kernel of the map e_i -> mat[i]; solve sum c_i mat[i] = 0
    m = len(mat)
    aug = [mat[i][:] + [GF(1) if j == i else GF(0) for j in range(m)] for i in range(m)]
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if aug[i][c]:
                piv = i
                break
        if piv is None:
            continue
        aug[r], aug[piv] = aug[piv], aug[r]
        inv = aug[r][c].inv()
        aug[r] = [inv * v for v in aug[r]]
        for i in range(m):
            if i != r and aug[i][c]:
                f = aug[i][c]
                aug[i] = [x + f * y for x, y in zip(aug[i], aug[r])]
        r += 1
    ker = [aug[i][n:] for i in range(r, m)]
    return ker


def span_rank(vecs, n):
    return rank(vecs, n)[0] if vecs else 0
