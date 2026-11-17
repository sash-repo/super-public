"""Exact linear algebra for the paper engines — fast when FLINT is present.

    from exactlin import rank, rref, nullspace, solve, backend

    rank(M)                 # M = list of rows of int / Fraction
    rank(M, p=3)            # over F_p instead of Q
    rref(M) -> (rows, pivot_columns)
    nullspace(M) -> list of basis vectors, as ROWS
    solve(M, b) -> particular solution, or None if inconsistent

Why: hand-rolled Gaussian elimination over `fractions.Fraction` is slow.
Measured on a laptop, a 120x120 sparse block (the shape of a
Chevalley-Eilenberg differential):

    pure-Python Fraction   1.959 s
    flint fmpq_mat         0.003 s     -> 641x, still exact over Q
    flint nmod_mat         0.003 s     -> 703x, over F_p

`fmpq_mat` is exact rational arithmetic, so the speedup costs no rigour.
Working mod p is faster still, but note that rank over F_p is only a LOWER
BOUND for the rank over Q (equality holds for all but finitely many primes) —
if an engine computes mod p, say so.

Everything degrades gracefully: with no FLINT the pure-Python implementations
below are used, so an engine written against this module still runs anywhere
(just slower). `backend()` reports which path is active.

Conventions worth stating, because they differ between implementations:
  * matrices are lists of rows; rows are lists of int / Fraction
  * `nullspace` returns basis vectors as ROWS (FLINT returns them in columns —
    this module transposes, so callers never see that)
  * results come back as `Fraction` (or int, mod p); no FLINT types leak out
"""
from fractions import Fraction

try:                                    # optional: needs python-flint
    from flint import fmpq_mat, fmpz_mat, nmod_mat, fmpq
    _HAVE_FLINT = True
except Exception:                       # pragma: no cover - fallback path
    _HAVE_FLINT = False


def backend() -> str:
    """'flint' or 'python' — which implementation these calls will use."""
    return "flint" if _HAVE_FLINT else "python"


# ---------------------------------------------------------------- helpers

def _dims(M):
    rows = len(M)
    cols = len(M[0]) if rows else 0
    for r in M:
        if len(r) != cols:
            raise ValueError("ragged matrix: all rows must have equal length")
    return rows, cols


def _clear_denominators(M):
    """Scale each row to integers; exact and lossless for rank/rref/nullspace."""
    out = []
    for row in M:
        fr = [Fraction(x) for x in row]
        den = 1
        for x in fr:
            den = den * x.denominator // _gcd(den, x.denominator)
        out.append([int(x * den) for x in fr])
    return out


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _to_flint(M, p):
    rows, cols = _dims(M)
    if not rows or not cols:
        return None, rows, cols
    if p is None:
        return fmpq_mat([[fmpq(Fraction(x).numerator, Fraction(x).denominator)
                          for x in r] for r in M]), rows, cols
    return nmod_mat([[int(x) % p for x in r] for r in M], p), rows, cols


# ---------------------------------------------------------------- API

def rank(M, p=None):
    """Rank over Q (p=None) or over F_p."""
    rows, cols = _dims(M)
    if not rows or not cols:
        return 0
    if _HAVE_FLINT:
        A, _, _ = _to_flint(M, p)
        return int(A.rank())
    return _py_rref(M, p)[2]


def rref(M, p=None):
    """Reduced row echelon form. Returns (rows, pivot_columns)."""
    rows, cols = _dims(M)
    if not rows or not cols:
        return [row[:] for row in M], []
    if _HAVE_FLINT:
        A, _, _ = _to_flint(M, p)
        R = A.rref()
        R = R[0] if isinstance(R, tuple) else R
        out = _from_flint(R, rows, cols, p)
        return out, _pivots(out, p)
    R, piv, _ = _py_rref(M, p)
    return R, piv


def nullspace(M, p=None):
    """Basis of the (right) kernel, as a list of ROW vectors."""
    rows, cols = _dims(M)
    if not rows or not cols:
        return []
    if _HAVE_FLINT:
        # fmpq_mat has no nullspace(); scaling each row to integers leaves the
        # right kernel unchanged, so go through fmpz_mat over Q.
        if p is None:
            A = fmpz_mat(_clear_denominators(M))
        else:
            A = nmod_mat([[int(x) % p for x in r] for r in M], p)
        X, nullity = A.nullspace()
        nullity = int(nullity)
        # FLINT puts basis vectors in COLUMNS; hand back rows
        return [[(int(X[r, c]) % p if p else Fraction(int(X[r, c])))
                 for r in range(cols)] for c in range(nullity)]
    return _py_nullspace(M, p)


def solve(M, b, p=None):
    """One solution of M x = b, or None when inconsistent."""
    rows, cols = _dims(M)
    aug = [list(M[i]) + [b[i]] for i in range(rows)]
    R, piv = rref(aug, p)
    if cols in piv:                       # pivot in the augmented column
        return None
    zero = 0 if p else Fraction(0)
    x = [zero] * cols
    for i, c in enumerate(piv):
        x[c] = R[i][cols]
    return x


# ---------------------------------------------------------------- internals

def _from_flint(A, rows, cols, p):
    out = []
    for i in range(rows):
        row = []
        for j in range(cols):
            v = A[i, j]
            row.append(int(v) % p if p else Fraction(int(v.numer()), int(v.denom())))
        out.append(row)
    return out


def _pivots(R, p):
    piv = []
    for row in R:
        for j, v in enumerate(row):
            if v:
                piv.append(j)
                break
    return piv


def _py_rref(M, p=None):
    """Reference implementation — also the fallback when FLINT is absent."""
    if p:
        A = [[int(x) % p for x in r] for r in M]
    else:
        A = [[Fraction(x) for x in r] for r in M]
    rows, cols = _dims(A)
    piv_row, pivots = 0, []
    for c in range(cols):
        pr = next((r for r in range(piv_row, rows) if A[r][c]), None)
        if pr is None:
            continue
        A[piv_row], A[pr] = A[pr], A[piv_row]
        inv = pow(A[piv_row][c], p - 2, p) if p else 1 / A[piv_row][c]
        A[piv_row] = [(x * inv) % p if p else x * inv for x in A[piv_row]]
        for r in range(rows):
            if r != piv_row and A[r][c]:
                f = A[r][c]
                A[r] = [((a - f * b) % p if p else a - f * b)
                        for a, b in zip(A[r], A[piv_row])]
        pivots.append(c)
        piv_row += 1
    return A, pivots, piv_row


def _py_nullspace(M, p=None):
    R, pivots, _ = _py_rref(M, p)
    _, cols = _dims(M)
    free = [c for c in range(cols) if c not in pivots]
    one, zero = (1, 0) if p else (Fraction(1), Fraction(0))
    basis = []
    for f in free:
        v = [zero] * cols
        v[f] = one
        for i, c in enumerate(pivots):
            v[c] = (-R[i][f]) % p if p else -R[i][f]
        basis.append(v)
    return basis
