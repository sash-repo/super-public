r"""
sle.py -- the divergence-free subalgebra  SLe(n) = Le(n) \cap S(n,n)  of le(n),
and the search for singular vectors in the induced SLe(n)-modules I(V).

    Delta f = sum_i d^2 f / d x_i d xi_i          (the odd Laplacian / BV operator)

For D_f as normalised in le.py one has  div D_f = 2 (-1)^{p(f)} Delta f, so

    sle(n) = { D_f : Delta f = 0 },      L_j = sle(n) \cap W_j = Ker Delta|_{deg = j+2}.

Consequences used below and verified by the self-test at the bottom:

  * Ker Delta is closed under the antibracket (Delta is an odd second-order derivation:
    Delta{f,g} = {Delta f, g} + (-1)^{p(f)+1} {f, Delta g});
  * L_{-1} = <x_i, xi_i> is the whole of le(n)_{-1} (linear f have Delta f = 0);
  * L_0 = spe(n) = S^2 + sl(n) + Lambda^2, because Delta(x_i xi_j) = delta_{ij}:
    the trace  z = sum_i x_i xi_i  is the one element of pe(n) that drops out.
    Hence n_+ and n_- of Kochetkov's Borel are the SAME for pe(n) and spe(n)
    (his B_+ = <x_i x_j, i <= j;  xi_i x_j, i < j>), only the Cartan shrinks:
    H = <x_i xi_i - x_{i+1} xi_{i+1}, 1 <= i <= n-1>, of dimension n-1.

Therefore a finite-dimensional irreducible spe(n)-module is V(lambda) computed by
pemod.py for ANY lift lambda of the weight, the maximal submodule being defined by
n_+ alone; we always use the lift with lambda_n = 0.  Kochetkov's weight is the
(n-1)-tuple  (lambda_1 - lambda_2, ..., lambda_{n-1} - lambda_n).

Since Delta preserves the eps-weight, both n_+ and L_1 have bases of weight vectors,
and the trace element z acts on I(V) diagonally with [z, .] diagonalisable on n_+ and
on L_1; so the equations "n_+ v = 0, L_1 v = 0" decouple over the pe(n)-weight spaces.
An H-eigenvector that is singular is therefore a sum of pe(n)-weight singular vectors,
and searching pe(n)-weight space by pe(n)-weight space (as we do) loses nothing.
"""

import sys
from fractions import Fraction as F
from itertools import combinations

from le import dx, dxi, padd, pmul, pscal, abr, par_le, pwt, pdeg, nullspace
from pemod import mon
from imod import Le, Induced


# ---------------------------------------------------------------- the odd Laplacian
def lap(f, n):
    """Delta f = sum_i d/dx_i d/dxi_i f"""
    out = {}
    for i in range(n):
        out = padd(out, dx(dxi(f, i), i))
    return out


def monos(n, d):
    """all monomials x^a xi^S of total degree d"""
    out = []
    for k in range(min(d, n) + 1):
        for S in combinations(range(n), k):
            for a in _comps(d - k, n):
                out.append(mon(n, a, S))
    return out


def _comps(r, n):
    if n == 0:
        return [()] if r == 0 else []
    if n == 1:
        return [(r,)]
    out = []
    for e in range(r + 1):
        for rest in _comps(r - e, n - 1):
            out.append((e,) + rest)
    return out


def kerdelta(n, d):
    """a weight-homogeneous basis of { f of degree d : Delta f = 0 }"""
    groups = {}
    for f in monos(n, d):
        groups.setdefault(pwt(f, n), []).append(f)
    out = []
    for w in sorted(groups):
        cols = groups[w]
        idx = {i: i for i in range(len(cols))}
        rows = {}
        for ci, f in enumerate(cols):
            for m, c in lap(f, n).items():
                rows.setdefault(m, {})[ci] = c
        ker = nullspace(list(rows.values()), list(range(len(cols))))
        for sol in ker:
            p = {}
            for ci, c in sol.items():
                p = padd(p, pscal(cols[ci], c))
            if p:
                out.append(p)
    return out


# ---------------------------------------------------------------- the induced module
class InducedSle(Induced):
    """I(V) over sle(n); V = V(lambda) irreducible over L_0 = spe(n)."""

    def __init__(self, le, lam):
        Induced.__init__(self, le, lam)
        self.pos1 = kerdelta(self.n, 3)          # L_1 of sle(n)

    def singular(self, m, nu):
        cols = []
        for u in self.umonos(m):
            w = self.uwt(u)
            mu = tuple(a - b for a, b in zip(nu, w))
            for i in range(len(self.V.wtbasis(mu))):
                cols.append((u, mu, i))
        if not cols:
            return []
        eqs = []
        ops = list(self.pos1) + [self.pe.basis[len(self.pe.nneg) + self.n + t]
                                 for t in range(len(self.pe.npos))]
        for f in ops:
            img = {}
            for key in cols:
                for k2, c2 in self._act_one(f, key).items():
                    row = img.setdefault(k2, {})
                    row[key] = row.get(key, 0) + c2
            eqs.extend(img.values())
        return nullspace(eqs, cols)


# ---------------------------------------------------------------- self-tests
def _span_reduce(basis_rows, cols):
    """row-reduce a list of dict(col->val); returns list of pivots"""
    from le import rref
    idx = {c: i for i, c in enumerate(cols)}
    rows = [{idx[k]: v for k, v in r.items()} for r in basis_rows]
    piv, red = rref([r for r in rows if r], len(cols))
    return piv, red


def _as_row(p, cols):
    idx = {}
    for m, c in p.items():
        idx[m] = c
    return idx


def selftest(nmax=5, dmax=5):
    ok = True
    for n in range(2, nmax + 1):
        # 1. dimensions and Delta-closure of the bracket
        for d in range(1, dmax + 1):
            K = kerdelta(n, d)
            allm = monos(n, d)
            print('n=%d  deg %d:  dim F_d = %3d,  dim Ker Delta = %3d' % (n, d, len(allm), len(K)))
            for f in K:
                assert not lap(f, n), 'kerdelta broken'
        # 2. closure under the antibracket
        K2, K3 = kerdelta(n, 2), kerdelta(n, 3)
        for f in K2 + K3:
            for g in K2 + K3:
                b = abr(f, g, n)
                if b and lap(b, n):
                    print('  !! Ker Delta NOT closed at n=%d' % n)
                    ok = False
        # 3. L_0 = spe(n): dim = 2n^2 - 1
        assert len(K2) == 2 * n * n - 1, (n, len(K2))
        # 4. is L_1 generating?  [L_1,L_1] = L_2 ?
        for d in (4, 5):
            src = kerdelta(n, 3) if d == 4 else None
            tgt = kerdelta(n, d)
            cols = [m for f in monos(n, d) for m in f]
            if d == 4:
                prods = [abr(f, g, n) for f in K3 for g in K3]
            else:
                K4 = kerdelta(n, 4)
                prods = [abr(f, g, n) for f in K3 for g in K4]
            prods = [p for p in prods if p]
            piv, _ = _span_reduce([_as_row(p, cols) for p in prods], cols)
            print('n=%d  [L_1,L_%d] spans %d of dim L_%d = %d  -> %s'
                  % (n, d - 3, len(piv), d - 2, len(tgt),
                     'generating' if len(piv) == len(tgt) else 'NOT GENERATING'))
            if len(piv) != len(tgt):
                if d - 2 == n - 2 and len(tgt) - len(piv) == 1:
                    # expected: L_1 misses exactly D_{xi_1...xi_n}, which lies in L_{n-2}
                    print('      (one dimension short in L_{n-2}: the field D_{xi_1...xi_n}, as expected)')
                else:
                    ok = False
    print('selftest:', 'OK' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    selftest(nmax)
