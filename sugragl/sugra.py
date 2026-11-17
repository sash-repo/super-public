#!/usr/bin/env python3
"""
sugra.py -- exact checks for "From Supergravity to Ballbearings"
(Grozman & Leites).

Two independent jobs.

(1) Par-strings.  The Table lists its columns by a parity string Par only; the
    marking of the simple roots (which fixes the Z-grading, hence g_- and g_0,
    hence everything) is printed only for N=1 (diagram (5)) and for the two
    N=8 candidates.  We reconstruct the marking from the rule "cut Par into
    consecutive blocks of size <= 2, a parity change forcing a cut" and test
    the reconstruction against an invariant that IS printed: the number of
    entries in each weight tuple of the column equals the number of sl(2)
    factors of g_0, i.e. the number of blocks of size 2.
    Then g_- is the strictly-lower block-triangular part and we report sdim.

(2) The Par=(0000) column, i.e. N=0, is the classical Riemann tensor and can
    be recomputed from scratch: Spencer cohomology H^{2,2}(g_{-1}; (g_{-1},g_0)_*)
    for g_0 = o(4) ("Riemannian") and g_0 = co(4) ("conformal"), decomposed
    under sl_L(2) x sl_R(2).  This is the paper's own anchor column, so it
    tests our reading of the Table's conventions (weights, boxing).

Exact arithmetic throughout (Fraction), no external dependencies.
"""

from fractions import Fraction
from itertools import combinations

# --------------------------------------------------------------------------
# exact linear algebra
# --------------------------------------------------------------------------


def rref(rows, ncols):
    """Row-reduce a list of rows (lists of Fraction). Returns (rref, pivots)."""
    m = [list(map(Fraction, r)) for r in rows]
    pivots = []
    r = 0
    for c in range(ncols):
        piv = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        inv = Fraction(1) / m[r][c]
        m[r] = [x * inv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    return m[:r], pivots


def rank(rows, ncols):
    if not rows:
        return 0
    return len(rref(rows, ncols)[1])


def nullspace(rows, ncols):
    """Basis of {x : A x = 0} where rows are the rows of A."""
    if not rows:
        return [[Fraction(1) if i == j else Fraction(0) for i in range(ncols)]
                for j in range(ncols)]
    red, piv = rref(rows, ncols)
    free = [c for c in range(ncols) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * ncols
        v[f] = Fraction(1)
        for i, p in enumerate(piv):
            v[p] = -red[i][f]
        basis.append(v)
    return basis


# --------------------------------------------------------------------------
# (1) Par strings -> blocks -> sdim g_-
# --------------------------------------------------------------------------


def blocks_of(par):
    """Cut Par into consecutive blocks of size <= 2; a parity change forces a cut.
    Returns a list of (size, parity)."""
    out = []
    i = 0
    while i < len(par):
        if i + 1 < len(par) and par[i + 1] == par[i]:
            out.append((2, int(par[i])))
            i += 2
        else:
            out.append((1, int(par[i])))
            i += 1
    return out


def gminus_sdim(blocks):
    """g_- = strictly lower block-triangular part.  Returns (even, odd)."""
    ev = od = 0
    for j in range(len(blocks)):
        for i in range(j + 1, len(blocks)):
            d = blocks[i][0] * blocks[j][0]
            if (blocks[i][1] + blocks[j][1]) % 2 == 0:
                ev += d
            else:
                od += d
    return ev, od


def minkowski_split(blocks):
    """The block-pair (i,j) sits in degree -(i-j) of the Z-grading (one unit per
    marked root crossed).  A 'copy of the Minkowski space' is an even piece
    id (x) id^* in the LOWEST degree, i.e. an adjacent pair of size-2 blocks of
    equal parity; every other even piece is the 'auxiliary space'."""
    deg1 = []      # 4-dim even pieces in degree -1 (adjacent blocks)
    allp = []      # all 4-dim even pieces id (x) id^*, any degree
    aux = 0
    for j in range(len(blocks)):
        for i in range(j + 1, len(blocks)):
            if (blocks[i][1] + blocks[j][1]) % 2:
                continue
            d = blocks[i][0] * blocks[j][0]
            if blocks[i][0] == 2 and blocks[j][0] == 2:
                allp.append((j, i, -(i - j), "odd blocks" if blocks[i][1]
                             else "even blocks"))
                if i - j == 1:
                    deg1.append((j, i, "odd" if blocks[i][1] else "even"))
                else:
                    aux += d
            else:
                aux += d
    return deg1, allp, aux


def depth(blocks):
    """Depth of the Z-grading = largest degree drop = len(blocks) - 1."""
    return len(blocks) - 1


# Par -> (N, number of weight-components actually printed in that Table column)
COLUMNS = [
    ("0000",         0, 2),
    ("00100",        1, 2),
    ("00001",        1, 2),
    ("001100",       2, 3),
    ("100001",       2, 2),
    ("000011",       2, 3),
    ("00111100",     4, 4),
    ("00001111",     4, 4),
    ("111100001111", 8, 6),
    # the N=8 candidate displayed in the text but ABSENT from the Table:
    ("001111111100", 8, None),
]


def part1():
    print("=" * 74)
    print("(1) Par -> blocks -> sdim g_-   [g_- = strictly lower block-triangular]")
    print("=" * 74)
    print(f"{'Par':<14} {'N':>2} {'blocks':<20} {'#sl2':>4} {'wts':>4} "
          f"{'depth':>5} {'sdim g_-':>10}  {'M(deg -1)':>9} {'M(all)':>6}")
    ok = True
    rows = []
    for par, N, printed in COLUMNS:
        b = blocks_of(par)
        nsl2 = sum(1 for s, _ in b if s == 2)
        ev, od = gminus_sdim(b)
        deg1, allp, aux = minkowski_split(b)
        bs = "|".join(f"{s}{'o' if p else 'e'}" for s, p in b)
        if printed is None:
            flag = "  (not in Table)"
        elif nsl2 != printed:
            flag, ok = "  <-- MISMATCH", False
        else:
            flag = ""
        print(f"{par:<14} {N:>2} {bs:<20} {nsl2:>4} {str(printed):>4} "
              f"{depth(b):>5} {f'({ev}|{od})':>10}  {len(deg1):>9} "
              f"{len(allp):>6}{flag}")
        rows.append((par, N, deg1, allp, aux, ev, od))
    print()
    print("marking reconstruction consistent with every printed weight tuple:",
          "YES" if ok else "NO")
    print()
    print("--- 'copies of the Minkowski space' in detail --------------------")
    for par, N, deg1, allp, aux, ev, od in rows:
        if N < 2:
            continue
        kinds = ", ".join(f"blocks {i+1}{j+1} ({k})" for i, j, k in deg1)
        degs = ", ".join(f"deg {d}" for _, _, d, _ in allp)
        print(f"  Par={par} (N={N}): even part {ev} = 4*{len(allp)} pieces "
              f"id(x)id^* at [{degs}]")
        print(f"      of them in degree -1: {len(deg1)}"
              + (f"  [{kinds}]" if deg1 else ""))
    return ok


# --------------------------------------------------------------------------
# (2) Spencer cohomology H^{2,2} for the Par=(0000) column
# --------------------------------------------------------------------------

# g_{-1} = C^2 (x) C^2, basis v[a][b], a,b in {0,1}.
# h_L-weight of index a: +1 if a=0 else -1;  same for h_R.
VBASIS = [(a, b) for a in range(2) for b in range(2)]
VIDX = {ab: i for i, ab in enumerate(VBASIS)}
NV = 4


def vweight(i):
    a, b = VBASIS[i]
    return (1 - 2 * a, 1 - 2 * b)


def mat_from_action(f):
    """Build the 4x4 matrix of a linear map given as f(i) -> dict{j: coeff}."""
    M = [[Fraction(0)] * NV for _ in range(NV)]
    for i in range(NV):
        for j, c in f(i).items():
            M[j][i] += Fraction(c)
    return M


def sl2_gens(side):
    """e, f, h of sl(2) acting on the `side` factor (0 = L, 1 = R)."""
    def act(kind):
        def f(i):
            ab = list(VBASIS[i])
            x = ab[side]
            out = {}
            if kind == "e":          # e: index 1 -> index 0
                if x == 1:
                    ab2 = ab[:]
                    ab2[side] = 0
                    out[VIDX[tuple(ab2)]] = 1
            elif kind == "f":        # f: index 0 -> index 1
                if x == 0:
                    ab2 = ab[:]
                    ab2[side] = 1
                    out[VIDX[tuple(ab2)]] = 1
            else:                    # h
                out[i] = 1 if x == 0 else -1
            return out
        return f
    return [mat_from_action(act(k)) for k in ("e", "f", "h")]


def identity_mat():
    return [[Fraction(1) if i == j else Fraction(0) for j in range(NV)]
            for i in range(NV)]


def apply_mat(M, i):
    """M applied to basis vector i -> coefficient vector."""
    return [M[j][i] for j in range(NV)]


def spencer_H22(g0, name):
    """g0: list of 4x4 weight-homogeneous matrices (a basis).
    Everything below is computed weight space by weight space, so the answer
    comes out as a character of sl_L(2) x sl_R(2)."""
    d0 = len(g0)
    g0w = G0_WEIGHTS[name]

    # ---- g_1 = { X in V^* (x) g_0 : X(u)v = X(v)u } -------------------------
    # coordinates of X: X[i*d0 + alpha];  weight of that coordinate direction
    ncoord = NV * d0

    def coord_weight(idx):
        i, al = divmod(idx, d0)
        wv, wg = vweight(i), g0w[al]
        return (-wv[0] + wg[0], -wv[1] + wg[1])

    rows = []
    for i, j in combinations(range(NV), 2):
        # X(v_i) v_j - X(v_j) v_i = 0   (a vector in V: 4 equations)
        for comp in range(NV):
            row = [Fraction(0)] * ncoord
            for al in range(d0):
                row[i * d0 + al] += apply_mat(g0[al], j)[comp]
                row[j * d0 + al] -= apply_mat(g0[al], i)[comp]
            rows.append(row)
    # solve weight block by weight block so that the basis of g_1 is homogeneous
    g1 = []
    blocks = {}
    for idx in range(ncoord):
        blocks.setdefault(coord_weight(idx), []).append(idx)
    for w, idxs in sorted(blocks.items()):
        sub = [[r[c] for c in idxs] for r in rows]
        sub = [r for r in sub if any(x != 0 for x in r)]
        for v in nullspace(sub, len(idxs)):
            full = [Fraction(0)] * ncoord
            for pos, c in zip(idxs, v):
                full[pos] = c
            g1.append(full)

    # ---- chain spaces -------------------------------------------------------
    # C1 = V^* (x) g_1        basis: g1 vectors, each an element of V^* (x) g_0
    # C2 = Lambda^2 V^* (x) g_0
    # C3 = Lambda^3 V^* (x) V
    pairs = list(combinations(range(NV), 2))
    triples = list(combinations(range(NV), 3))
    dimC2 = len(pairs) * d0
    dimC3 = len(triples) * NV

    def c2_index(pi, al):
        return pi * d0 + al

    def c3_index(ti, comp):
        return ti * NV + comp

    # ---- d1 : C1 -> C2,  (dX)(v_i,v_j) = X(v_i)|_{g_0} ... -----------------
    # For X in g_1 (subset of V^* (x) g_0): [X, v_j] = X(v_j) in g_0.
    # (dX)(v_i, v_j) = [X(v_i), v_j] - [X(v_j), v_i] is WRONG for this bidegree;
    # the bracket g_1 x g_{-1} -> g_0 is [X, v] = X(v), so
    # (dX)(v_i, v_j) = [X, v_i] evaluated ... no: omega in C^1 = V^* (x) g_1,
    # omega(v) in g_1, and (d omega)(v_i,v_j) = [omega(v_i), v_j] - [omega(v_j), v_i].
    # Here we take omega = the identity inclusion of the degree-2 piece, i.e.
    # C^{1,2} = V^* (x) g_1 with omega(v_i) = sum_k Y[i][k] * g1[k].
    d1_cols, d1_wts = [], []
    for k, X in enumerate(g1):
        wX = next(coord_weight(t) for t in range(ncoord) if X[t] != 0)
        for i in range(NV):
            d1_wts.append((-vweight(i)[0] + wX[0], -vweight(i)[1] + wX[1]))
            # omega(v_i) = g1[k]  placed in slot i ; compute d omega
            col = [Fraction(0)] * dimC2
            for pi, (p, q) in enumerate(pairs):
                # [omega(v_p), v_q] - [omega(v_q), v_p]
                if p == i:
                    for al in range(d0):
                        col[c2_index(pi, al)] += X[q * d0 + al]
                if q == i:
                    for al in range(d0):
                        col[c2_index(pi, al)] -= X[p * d0 + al]
            d1_cols.append(col)

    # ---- d2 : C2 -> C3 ------------------------------------------------------
    d2_cols = []
    for pi, (p, q) in enumerate(pairs):
        for al in range(d0):
            col = [Fraction(0)] * dimC3
            for ti, (x, y, z) in enumerate(triples):
                # (d omega)(x,y,z) = [w(y,z),x] - [w(x,z),y] + [w(x,y),z]
                for (a, b, c, sgn) in ((y, z, x, 1), (x, z, y, -1), (x, y, z, 1)):
                    if (a, b) == (p, q):
                        vec = apply_mat(g0[al], c)
                        for comp in range(NV):
                            col[c3_index(ti, comp)] += sgn * vec[comp]
            d2_cols.append(col)

    # ---- weights ------------------------------------------------------------
    def g0_weight(al):
        """g_0 basis vectors are weight vectors by construction."""
        return G0_WEIGHTS[name][al]

    def c2_weight(idx):
        pi, al = divmod(idx, d0)
        p, q = pairs[pi]
        wp, wq = vweight(p), vweight(q)
        wg = g0_weight(al)
        # a cochain in Lambda^2 V^* (x) g_0 has weight -w_p - w_q + w_g
        return (-wp[0] - wq[0] + wg[0], -wp[1] - wq[1] + wg[1])

    # ---- dims, weight by weight --------------------------------------------
    wts = {}
    for idx in range(dimC2):
        wts.setdefault(c2_weight(idx), []).append(idx)

    char, tot_ker, tot_im = {}, 0, 0
    for w, idxs in sorted(wts.items()):
        # d2 restricted to the weight space (columns = idxs)
        A = [[d2_cols[j][r] for j in idxs] for r in range(dimC3)]
        A = [r for r in A if any(x != 0 for x in r)]
        k = len(idxs) - rank(A, len(idxs))
        # d1 columns of this weight
        B = [[col[r] for r in idxs]
             for col, cw in zip(d1_cols, d1_wts) if cw == w]
        im = rank(B, len(idxs)) if B else 0
        tot_ker += k
        tot_im += im
        if k - im:
            char[w] = k - im

    # cross-check the weightwise answer against the global one
    ker_g = len(nullspace([[col[r] for col in d2_cols] for r in range(dimC3)],
                          dimC2))
    im_g = rank([list(c) for c in d1_cols], dimC2)
    assert (tot_ker, tot_im) == (ker_g, im_g), \
        f"weightwise {tot_ker},{tot_im} != global {ker_g},{im_g}"
    return len(g1), tot_ker, tot_im, tot_ker - tot_im, char


G0_WEIGHTS = {}


def build_g0(kind):
    """Return (basis, weights) for o(4) or co(4)."""
    eL, fL, hL = sl2_gens(0)
    eR, fR, hR = sl2_gens(1)
    basis = [eL, fL, hL, eR, fR, hR]
    weights = [(2, 0), (-2, 0), (0, 0), (0, 2), (0, -2), (0, 0)]
    if kind == "co":
        basis = basis + [identity_mat()]
        weights = weights + [(0, 0)]
    return basis, weights


def decompose(char):
    """Decompose a character (dict weight -> mult) into sl2 x sl2 irreps (m,n)."""
    ch = dict(char)
    out = []
    while any(v for v in ch.values()):
        hi = max((w for w, v in ch.items() if v), key=lambda w: (w[0] + w[1], w[0]))
        m, n = hi
        if m < 0 or n < 0:
            raise RuntimeError(f"bad highest weight {hi}")
        out.append((m, n))
        for i in range(m + 1):
            for j in range(n + 1):
                w = (m - 2 * i, n - 2 * j)
                ch[w] = ch.get(w, 0) - 1
                if ch[w] == 0:
                    del ch[w]
        if any(v < 0 for v in ch.values()):
            raise RuntimeError("negative multiplicity - not a character")
    return sorted(out, reverse=True)


def part2():
    print()
    print("=" * 74)
    print("(2) Par=(0000): Spencer H^{2,2} under sl_L(2) x sl_R(2)")
    print("=" * 74)
    for kind, label in (("o", "o(4)   'Riemannian'"), ("co", "co(4)  'conformal'")):
        basis, weights = build_g0(kind)
        G0_WEIGHTS[kind] = weights
        g1, kerd, imd, dimH, char = spencer_H22(basis, kind)
        irreps = decompose(char)
        pretty = " + ".join(f"({m},{n})" for m, n in irreps)
        dims = " + ".join(str((m + 1) * (n + 1)) for m, n in irreps)
        total = sum((m + 1) * (n + 1) for m, n in irreps)
        print(f"  g_0 = {label}:  dim g_0 = {len(basis)}, dim g_1 = {g1}")
        print(f"     dim ker d2 = {kerd}, dim im d1 = {imd}, dim H^(2,2) = {dimH}")
        print(f"     H^(2,2) = {pretty}")
        print(f"             = {dims} = {total}   "
              f"[{'consistent' if total == dimH else 'INCONSISTENT'}]")
        print()


if __name__ == "__main__":
    part1()
    part2()
