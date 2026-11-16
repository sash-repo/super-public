"""
layer2.py -- the case V = the 1-dimensional L_0-module of weight (1,...,1).

Here V is one-dimensional, so I(V) = U(L_{-1}) = C[p_1..p_n] (x) Lambda(d_1..d_n) and the
search for singular vectors is COMPLETE: there is only one weight of V, so every weight of
I(V)_m is enumerated and no height cutoff enters.

Kochetkov's Theorem (Yaroslavl' 1989, last page) asserts that I_2(V) = I(V)/Z_1 is
irreducible for this V ("It remains to consider the case of a one-dimensional V.  This is a
simple computation").  We test that claim.
"""

import sys
from fractions import Fraction as F
from le import rref
from imod import Le, Induced
from report import umono_name


def run(n, maxdeg=None):
    maxdeg = maxdeg or n + 1
    le = Le(n, odd_up=False)
    I = Induced(le, tuple([1] * n))
    lam = I.lam
    print('=== n = %d,  V = the 1-dimensional module of weight (1,...,1)' % n)

    # ---- all singular vectors of I(V), degrees 1..maxdeg (complete: dim V = 1)
    sing = []
    for m in range(1, maxdeg + 1):
        cand = {tuple(a + b for a, b in zip(lam, I.uwt(u))) for u in I.umonos(m)}
        for nu in sorted(cand):
            for s in I.singular(m, nu):
                sing.append((m, nu, s))
    print('  singular vectors of I(V) in degrees 1..%d: %d' % (maxdeg, len(sing)))
    for m, nu, s in sing:
        print('     d=%d : %s' % (m, ' + '.join(
            '%s%s' % ('' if c == 1 else ('-' if c == -1 else str(c) + ' '), umono_name(n, u))
            for (u, mu, i), c in sorted(s.items()))))

    # ---- Z_1 = U(g) . (the singular vectors), degree by degree
    L0 = list(le.pe.basis)
    layers = {}
    for m, nu, s in sing:
        layers.setdefault(m, []).append(s)
    Z = {}
    for m in sorted(layers):
        Z.setdefault(m, [])
        Z[m].extend(layers[m])
        # close under L_0 within the degree
        Z[m] = close(I, Z[m], L0)
        # push up with L_{-1}
        d = m
        while d < maxdeg + 1:
            nxt = [I.lmul(k, v) for k in range(2 * n) for v in Z[d]]
            nxt = [v for v in nxt if v]
            Z.setdefault(d + 1, [])
            Z[d + 1] = close(I, Z[d + 1] + nxt, L0)
            d += 1
    for d in sorted(Z):
        Z[d] = independent(Z[d])
    print('  dim Z_1 by degree:', {d: len(Z[d]) for d in sorted(Z) if Z[d]})

    # ---- the candidate second-layer vector delta_1...delta_n (x) m1
    zt = {(((0,) * n, tuple(range(n))), lam, 0): F(1)}
    deg = n
    inZ = member(Z.get(deg, []), zt)
    imgs = [I.act(f, zt) for f in le.pos1]
    imgs = [v for v in imgs if v]
    allin = all(member(Z.get(deg - 1, []), v) for v in imgs)
    npos = [le.pe.basis[len(le.pe.nneg) + n + t] for t in range(len(le.pe.npos))]
    nplus = [v for v in (I.act(b, zt) for b in npos) if v]
    print('  z = %s :' % umono_name(n, ((0,) * n, tuple(range(n)))))
    print('     n_+ z = 0                      :', not nplus)
    print('     L_1 z subset Z_1               :', allin, '(%d nonzero images)' % len(imgs))
    print('     z in Z_1                       :', inZ)
    print('     => I_2(V) = I(V)/Z_1 REDUCIBLE :', (not nplus) and allin and not inZ)


def close(I, vecs, ops):
    vecs = independent([v for v in vecs if v])
    changed = True
    while changed:
        changed = False
        for f in ops:
            for v in list(vecs):
                w = I.act(f, v)
                if w and not member(vecs, w):
                    vecs.append(w)
                    vecs = independent(vecs)
                    changed = True
    return vecs


def _mat(vecs):
    keys = sorted({k for v in vecs for k in v}, key=str)
    idx = {k: i for i, k in enumerate(keys)}
    return idx, [{idx[k]: c for k, c in v.items()} for v in vecs]


def independent(vecs):
    vecs = [v for v in vecs if v]
    if not vecs:
        return []
    out = []
    for v in vecs:
        if not member(out, v):
            out.append(v)
    return out


def member(vecs, w):
    if not w:
        return True
    if not vecs:
        return False
    idx, rows = _mat(list(vecs) + [w])
    piv, _ = rref(rows[:-1], len(idx))
    piv2, _ = rref(rows, len(idx))
    return len(piv2) == len(piv)


if __name__ == '__main__':
    for n in (2, 3, 4):
        run(n)
        sys.stdout.flush()
