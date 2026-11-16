"""sb.py -- the special Buttin superalgebra  sb(n) = { f in b(n) : Delta f = 0 },
the central extension of sle(n), and the singular vectors in its induced modules.

Set-up (the survey [LSh], arXiv:2406.16760):

    b(n)  = Pi(Lambda(x_1..x_n, xi_1..xi_n)) with the Buttin antibracket,
            graded by deg f - 2, so the constants sit in degree -2;
    le(n) = b(n)/C.1,        sb(n) = Ker Delta subset b(n),   sle(n) = sb(n)/C.1.

Inside M(n) (mn.py) the tau-free generating functions are exactly b(n) = M_{0,b}(n),
on which the M-bracket restricts to the antibracket; so b(n) and sb(n) are handled by
the M-machinery of mimod.py with

    ops = b(n)_1  resp.  sb(n)_1 = Ker Delta cap b(n)_1 .

What is checked here:

  1. b(n)_0 = pe(n) and sb(n)_0 = spe(n) (dimensions 2n^2 and 2n^2 - 1);
     b(n)_{-1} = sb(n)_{-1} = <x_i, xi_i>, and D = '1/2 spans b(n)_{-2} = sb(n)_{-2};
  2. D is CENTRAL in b(n) and in sb(n) (it is not central in m(n));
  3. hence, for every V, the vector D (x) m_1 is singular in I(V), of degree 2:
     no induced module over b(n) or over sb(n) is irreducible;
  4. z |--> D z carries singular vectors of I_{sle(n)}(V) to singular vectors of
     I_{sb(n)}(V), and every singular vector of I_{sb(n)}(V) is one of the two kinds
     -- which is, verbatim, item 5) of the b_{a,b}(n)-Theorem for a = 0, with Ker Delta
     in place of the whole of b(n).

Run:  python3 sb.py
"""
from fractions import Fraction as F
from itertools import combinations

from mn import mono, one, mbr, mdeg, layer, par_m
from mimod import Mn, MInduced
from sle import lap
from le import nullspace


def taufree(f):
    """is the generating function f free of tau (= the odd index n)?"""
    return all(n_ not in S for (a, S) in f for n_ in (len(a),))


def blayer(mn, j):
    """a basis of b(n)_j = { f in M(n)_j : f is tau-free }"""
    n = mn.n
    return [f for f in layer(n, j)
            if all(n not in S for (a, S) in f)]


def _combine(bas, sol):
    out = {}
    for k, c in sol.items():
        for key, v in bas[k].items():
            out[key] = out.get(key, 0) + c * v
    return {k: v for k, v in out.items() if v}


def sblayer(mn, j):
    """a basis of sb(n)_j = Ker Delta cap b(n)_j (the kernel as a SUBSPACE)"""
    n = mn.n
    bas = blayer(mn, j)
    if not bas:
        return []
    rows = {}
    for k, f in enumerate(bas):
        for key, c in lap(f, n).items():
            rows.setdefault(key, {})[k] = rows.setdefault(key, {}).get(k, 0) + c
    eqs = [r for r in rows.values() if r]
    if not eqs:
        return list(bas)
    return [_combine(bas, s) for s in nullspace(eqs, list(range(len(bas))))]


def report(n):
    mn = Mn(n)
    print('=== n = %d ===' % n)
    for j in (-2, -1, 0, 1):
        b, s = blayer(mn, j), sblayer(mn, j)
        print('   dim b(n)_%-2d = %-3d   dim sb(n)_%-2d = %-3d' % (j, len(b), j, len(s)))
    assert len(blayer(mn, 0)) == 2 * n * n, 'b(n)_0 = pe(n)'
    assert len(sblayer(mn, 0)) == 2 * n * n - 1, 'sb(n)_0 = spe(n)'
    assert len(blayer(mn, -2)) == 1 and len(sblayer(mn, -2)) == 1
    assert len(blayer(mn, -1)) == 2 * n and len(sblayer(mn, -1)) == 2 * n

    # ---- 2. D is central in b(n) and in sb(n), and is NOT central in m(n)
    D = one(n)                              # the constant, = 2D
    cen_b = cen_sb = True
    for j in range(-2, 4):
        for f in blayer(mn, j):
            if mbr(D, f, n):
                cen_b = False
        for f in sblayer(mn, j):
            if mbr(D, f, n):
                cen_sb = False
    noncen_m = any(mbr(D, f, n) for j in range(-2, 4) for f in layer(n, j))
    print('   D central in b(n): %s ; in sb(n): %s ; in m(n): %s'
          % (cen_b, cen_sb, not noncen_m))
    assert cen_b and cen_sb and noncen_m

    # ---- 3./4. singular vectors, degree by degree
    #  the four weights of the le(n)-list, in the pe(n)-basis
    lams = [tuple([0] * n), tuple([1] * n), tuple([2] + [1] * (n - 1)),
            tuple([0] * (n - 1) + [-1])]
    algs = (('b(n)  over le(n) ', blayer(mn, 1)), ('sb(n) over sle(n)', sblayer(mn, 1)))
    for lam in lams:
        for algname, ops in algs:
            I = MInduced(mn, lam, 0, ops=ops)
            dimS, dimQ = {}, {}     # full singular space / singular space MODULO D
            for m in range(0, 5):
                wts = sorted({tuple(x + y for x, y in zip(lam, I.uwt(u)))
                              for u in I.umonos(m)})
                for nu in wts:
                    if m == 0:
                        if nu == lam:
                            dimS[(m, nu)] = dimQ[(m, nu)] = 1     # the generator
                        continue
                    ns = len(I.singular(m, nu))
                    nq = len(singular_mod_D(I, m, nu))
                    if ns:
                        dimS[(m, nu)] = ns
                    if nq:
                        dimQ[(m, nu)] = nq
            pairs = set(list(dimS) + [(m + 2, nu) for (m, nu) in dimQ if m + 2 <= 4])
            bad = [(m, nu, dimS.get((m, nu), 0), dimQ.get((m, nu), 0),
                    dimQ.get((m - 2, nu), 0))
                   for (m, nu) in pairs
                   if dimS.get((m, nu), 0) != dimQ.get((m, nu), 0) + dimQ.get((m - 2, nu), 0)]
            print('   lam=%-13s %s: dim S(m,nu) = dim S_quot(m,nu) + dim S_quot(m-2,nu): %s'
                  % (str(lam), algname, 'OK' if not bad else 'FAILS %s' % (bad,)))
            for (m, nu) in sorted(dimS, key=str):
                print('        deg %d  wt %-16s dim S = %d,  mod D: %d,  D.(m-2): %d'
                      % (m, str(nu), dimS[(m, nu)], dimQ.get((m, nu), 0),
                         dimQ.get((m - 2, nu), 0)))
            assert not bad


def singular_mod_D(I, m, nu):
    """the singular vectors of I(V) MODULO the central D, i.e. those of the induced
    module over le(n) (resp. over sle(n)): the D-free monomials, the conditions read
    in the quotient I(V)/D.I(V)."""
    cols = []
    for u in I.umonos(m):
        if u[2]:                      # drop the monomials carrying D
            continue
        w = I.uwt(u)
        mu = tuple(x - y for x, y in zip(nu[:I.n], w))
        for i in range(len(I.V.wtbasis(mu))):
            cols.append((u, mu, i))
    if not cols:
        return []
    eqs = []
    ops = list(I.ops) + [I.pe.basis[len(I.pe.nneg) + I.n + q]
                         for q in range(len(I.pe.npos))]
    for f in ops:
        img = {}
        for key in cols:
            for k2, c2 in I._act_one(f, key).items():
                if k2[0][2]:          # the D-part of the image is read modulo D
                    continue
                row = img.setdefault(k2, {})
                row[key] = row.get(key, 0) + c2
        eqs.extend(img.values())
    return nullspace(eqs, cols)


def _rref_rows(rows, keys):
    """the rank of a list of dict-rows over the ordered key list `keys`"""
    from fractions import Fraction as Fr
    mat = [[Fr(r.get(k, 0)) for k in keys] for r in rows]
    piv = []
    r = 0
    for c in range(len(keys)):
        p = next((i for i in range(r, len(mat)) if mat[i][c]), None)
        if p is None:
            continue
        mat[r], mat[p] = mat[p], mat[r]
        pv = mat[r][c]
        mat[r] = [v / pv for v in mat[r]]
        for i in range(len(mat)):
            if i != r and mat[i][c]:
                f = mat[i][c]
                mat[i] = [a - f * b for a, b in zip(mat[i], mat[r])]
        piv.append(c)
        r += 1
        if r == len(mat):
            break
    return piv


if __name__ == '__main__':
    for n in (2, 3):
        report(n)
