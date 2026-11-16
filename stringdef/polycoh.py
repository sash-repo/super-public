"""The subcomplex of cochains that are DIFFERENTIAL OPERATORS.

In the basis E_i = L_{-i} a weight-w q-cochain is a function c(i_1,...,i_q);
it is (the symbol of) a bidifferential operator exactly when that function is
POLYNOMIAL in the indices.  Polynomials are closed under the differential
    (d gamma)(i,j) = (j+w-i)gamma(j) - (i+w-j)gamma(i) - (j-i)gamma(i+j),
which raises the degree by at most 1 -- so they form a FILTERED subcomplex,
and cutting the filtration at degree D gives a finite computation again.
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sympy as sp
from exactlin import rank                                     # noqa: E402

i, j, k, W = sp.symbols('i j k w')


def basis1(D):
    return [i ** a for a in range(D + 1)]


def basis2(D):
    return [sp.expand(i ** a * j ** b - i ** b * j ** a)
            for a in range(D + 1) for b in range(a) if a + b <= D]


def basis3(D):
    out = []
    for a in range(D + 1):
        for b in range(a):
            for c in range(b):
                if a + b + c > D:
                    continue
                e = 0
                for (p, q, r), s in (((i, j, k), 1), ((j, k, i), 1), ((k, i, j), 1),
                                     ((j, i, k), -1), ((i, k, j), -1), ((k, j, i), -1)):
                    e += s * p ** a * q ** b * r ** c
                out.append(sp.expand(e))
    return out


LAM = sp.Rational(-1)          # lambda of the coefficient module F_lambda


def mu(n, x):
    return n + LAM * x


def d1(g, w):
    return sp.expand(mu(j + w, i) * g.subs(i, j) - mu(i + w, j) * g - (j - i) * g.subs(i, i + j))


def d2(c, w):
    def C(x, y):
        return c.subs({i: x, j: y}, simultaneous=True)
    return sp.expand(mu(j + k + w, i) * C(j, k) - mu(i + k + w, j) * C(i, k)
                     + mu(i + j + w, k) * C(i, j)
                     - (j - i) * C(i + j, k) + (k - i) * C(i + k, j) - (k - j) * C(j + k, i))


def coords(expr, bas, vars_):
    """Coordinates of expr in the basis bas (both polynomials in vars_)."""
    P = sp.Poly(sp.expand(expr), *vars_)
    mons = {}
    for mon, co in zip(P.monoms(), P.coeffs()):
        mons[mon] = co
    Bmon = []
    for b in bas:
        Pb = sp.Poly(sp.expand(b), *vars_)
        Bmon.append(dict(zip(Pb.monoms(), Pb.coeffs())))
    keys = sorted(set(mons) | set().union(*[set(d) for d in Bmon]) if Bmon else set(mons))
    A = [[Bmon[c].get(key, 0) for c in range(len(bas))] for key in keys]
    v = [mons.get(key, 0) for key in keys]
    return A, v


def matrix_of(op, src, tgt, vars_, w):
    """Columns = images of src basis, expressed in tgt basis."""
    Bmon, keys = [], set()
    Tpoly = [sp.Poly(sp.expand(b), *vars_) for b in tgt]
    for Pb in Tpoly:
        d = dict(zip(Pb.monoms(), Pb.coeffs()))
        Bmon.append(d)
        keys |= set(d)
    cols = []
    imgs = []
    for s in src:
        e = sp.expand(op(s, w))
        P = sp.Poly(e, *vars_) if e != 0 else None
        d = dict(zip(P.monoms(), P.coeffs())) if P else {}
        imgs.append(d)
        keys |= set(d)
    keys = sorted(keys)
    M = [[0] * (len(src) + len(tgt)) for _ in keys]
    for r, key in enumerate(keys):
        for c in range(len(src)):
            M[r][c] = int(imgs[c].get(key, 0))
        for c in range(len(tgt)):
            M[r][len(src) + c] = int(Bmon[c].get(key, 0))
    return M, keys


def solve_in_basis(op, src, tgt, vars_, w):
    """Matrix of op: src -> tgt in the given bases (tgt must span the image)."""
    M, keys = matrix_of(op, src, tgt, vars_, w)
    ns, nt = len(src), len(tgt)
    # solve  T x = img_c  for each column c
    T = [[row[ns + t] for t in range(nt)] for row in M]
    out = []
    for c in range(ns):
        b = [row[c] for row in M]
        x = _solve(T, b)
        if x is None:
            raise RuntimeError("image left the target basis (raise the cutoff)")
        out.append(x)
    return [[out[c][r] for c in range(ns)] for r in range(nt)]


def _solve(T, b):
    from exactlin import solve
    return solve(T, b)


def poly_cohomology(w, D):
    B1, B2, B3 = basis1(D + 1), basis2(D + 2), basis3(D + 3)
    A1 = solve_in_basis(d1, B1, B2, (i, j), w)          # C^1_{<=D+1} -> C^2
    A2 = solve_in_basis(d2, B2, B3, (i, j, k), w)       # C^2_{<=D+2} -> C^3
    dimZ2 = len(B2) - rank(A2)
    dimB2 = rank(A1)
    dimZ1 = len(B1) - rank(A1)
    return dimZ1, dimZ2, dimB2, dimZ2 - dimB2, len(B1), len(B2)


if __name__ == "__main__":
    print("cochains that are differential operators (polynomial coefficients)")
    print("module: witt itself (lambda = -1)\n")
    print("   w   D   dimC^1 dimC^2   Z^1  Z^2   B^2   H^2")
    for w in (-3, -1, 0, 1, 2, 5):
        for D in (3, 5, 7):
            z1, z2, b2, h2, c1, c2 = poly_cohomology(w, D)
            print("%4d %3d %7d %6d %5d %4d %5d %5d" % (w, D, c1, c2, z1, z2, b2, h2))
        print()
