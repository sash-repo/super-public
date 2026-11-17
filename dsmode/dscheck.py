#!/usr/bin/env python3
"""
Independent recomputation of the DS-homology tables of Sections 3 and 4 of the
text: tables (54)-(56) (gl, sl, psl(n|n), n = 2, 3, 4) and the tables of
Section 4 (gl, psl(n|n+pk), p = 2, 3, 5).

For gl(a|b) in the *alternating* format of parities (the one used in the text),
the odd simple root vectors are E_{12}, E_{34}, E_{56}, ... ; the elements
x = x_1, y = x_1+x_3, z = x_1+x_3+x_5, v = x_1+x_3+x_5+x_7 of the tables are
x_(r) := sum_{1<=j<=r} E_{2j-1,2j}.

We compute rank(ad_x) on gl, sl and psl over F_p (p=0 means Q, done modulo
a large prime), and dim g_x = dim g - 2 rank(ad_x), formula (15) of the text,
which holds because (ad_x)^2 = ad_{x^2} = 0.  For Section 4 the script also
lists the gl(i|j) / psl(i|j) whose dimension equals dim g_x.

Usage:  python3 dscheck.py
"""

BIG = (1 << 61) - 1  # prime, stands in for characteristic 0


def parities(a, b):
    """Alternating format: (ev,od,ev,od,...) with the surplus at the end."""
    n = a + b
    par = [None] * n
    ev = od = 0
    i = 0
    # interleave while both remain
    while ev < a and od < b:
        par[i] = 0; ev += 1; i += 1
        par[i] = 1; od += 1; i += 1
    while ev < a:
        par[i] = 0; ev += 1; i += 1
    while od < b:
        par[i] = 1; od += 1; i += 1
    return par


def rank_mod(rows, ncols, p):
    """Gaussian elimination over F_p; rows is a list of lists."""
    M = [[c % p for c in r] for r in rows]
    rank = 0
    row = 0
    for col in range(ncols):
        piv = None
        for r in range(row, len(M)):
            if M[r][col] % p:
                piv = r
                break
        if piv is None:
            continue
        M[row], M[piv] = M[piv], M[row]
        inv = pow(M[row][col], p - 2, p)
        M[row] = [(v * inv) % p for v in M[row]]
        for r in range(len(M)):
            if r != row and M[r][col]:
                f = M[r][col]
                M[r] = [(M[r][k] - f * M[row][k]) % p for k in range(ncols)]
        row += 1
        rank += 1
        if row == len(M):
            break
    return rank


def ad_matrix(a, b, r, p, algebra):
    """
    Matrix of ad_x on `algebra` in ('gl','sl','psl'), x = sum_{j<=r} E_{2j-1,2j}.
    Basis of gl: E_{ij}. sl: str = 0. psl: sl modulo the identity.
    Returns (matrix as list of columns, dim of the algebra).
    """
    n = a + b
    par = parities(a, b)
    # x as a list of (i,j) with coefficient 1, 0-indexed
    xs = [(2 * j, 2 * j + 1) for j in range(r)]
    for (i, j) in xs:
        assert par[i] != par[j], "x must be odd"

    # --- basis of the algebra ---
    if algebra == 'gl':
        basis = [(i, j) for i in range(n) for j in range(n)]
    else:
        # sl: E_ij for i!=j, plus H_i = E_ii - (-1)^{par_i+par_{i+1}} ... simpler:
        # supertrace of E_ii is (-1)^{par_i}. Basis of str=0 part:
        # all E_ij with i!=j, and E_ii - eps_i*eps_0*E_00 for i>=1 where eps=(-1)^par.
        basis = [(i, j) for i in range(n) for j in range(n) if i != j]
        # diagonal part handled separately below
    # We work in the full gl coordinate space and use subspace bases as vectors.

    def vec_of(mat):
        return [mat.get((i, j), 0) % p for i in range(n) for j in range(n)]

    eps = [1 if par[i] == 0 else p - 1 for i in range(n)]  # (-1)^{par_i} mod p

    # explicit basis vectors (as dicts) of the algebra
    B = []
    if algebra == 'gl':
        for i in range(n):
            for j in range(n):
                B.append({(i, j): 1})
    else:
        for i in range(n):
            for j in range(n):
                if i != j:
                    B.append({(i, j): 1})
        # diagonal, supertrace zero: D_i = E_ii - eps_i*eps_0^{-1} E_00, i>=1
        inv_eps0 = pow(eps[0], p - 2, p)
        for i in range(1, n):
            B.append({(i, i): 1, (0, 0): (p - (eps[i] * inv_eps0) % p) % p})
        if algebra == 'psl':
            # identity must lie in sl: str(1) = a - b must be 0 mod p
            assert (a - b) % p == 0, "psl needs p | a-b"

    def bracket_with_x(Y, parY):
        """ad_x(Y) = x Y - (-1)^{parY} Y x, x odd."""
        out = {}
        for (i, j) in xs:
            # (E_ij Y)_{i,l} = Y_{j,l}
            for (k, l), v in Y.items():
                if k == j:
                    out[(i, l)] = (out.get((i, l), 0) + v) % p
            # -(-1)^{parY} (Y E_ij)_{k,j} = Y_{k,i}
            s = (p - 1) if parY == 0 else 1    # coefficient -(-1)^{parY}
            for (k, l), v in Y.items():
                if l == i:
                    out[(k, j)] = (out.get((k, j), 0) + s * v) % p
        return {kk: vv % p for kk, vv in out.items() if vv % p}

    cols = []
    for Y in B:
        # parity of Y (all basis elements are parity-homogeneous)
        (i0, j0) = next(iter(Y))
        parY = (par[i0] + par[j0]) % 2
        img = bracket_with_x(Y, parY)
        cols.append(vec_of(img))

    dim = len(B)
    if algebra == 'psl':
        dim -= 1  # quotient by the identity, which is central hence in ker
    return cols, dim


def analyse(a, b, r, p, algebra):
    pp = BIG if p == 0 else p
    cols, dim = ad_matrix(a, b, r, pp, algebra)
    n = a + b
    if algebra == 'psl':
        # ad_x on psl = induced map; rank is unchanged by quotienting by the
        # central identity (identity is in ker ad_x and maps to 0), but the image
        # must be taken modulo the identity too.  Image of ad_x on sl: does it
        # contain the identity?  Compute rank of image mod <1>.
        ident = [1 if i == j else 0 for i in range(n) for j in range(n)]
        rank_sl = rank_mod(cols, n * n, pp)
        rank_psl = rank_mod(cols + [ident], n * n, pp) - 1
        rank = rank_psl
    else:
        rank = rank_mod(cols, n * n, pp)
    return rank, dim - 2 * rank


def name(d, p):
    """Guess gl(m|k)/sl/psl of the given dimension - printed for eyeballing."""
    return d


if __name__ == '__main__':
    print("=== Section 3:  gl, sl, psl(n|n),  tables (54), (55), (56) ===")
    print(f"{'g':>12} {'r':>2} {'p':>3} {'rank':>5} {'dim g_x':>8}")
    for nn in (2, 3, 4):
        for p in (0, 2, 3, 5):
            for r in range(1, nn + 1):
                for alg in ('gl', 'sl', 'psl'):
                    rank, dgx = analyse(nn, nn, r, p, alg)
                    print(f"{alg+f'({nn}|{nn})':>12} {r:>2} {p:>3} {rank:>5} {dgx:>8}")
        print()

    print("=== Section 4:  gl, psl(a|a+kp) ===")
    # (p, a, b) exactly as tabulated in Section 4
    S4 = [(2, 2, 4), (2, 2, 6), (2, 2, 8),
          (2, 3, 5), (2, 3, 7), (2, 3, 9),
          (2, 4, 6), (2, 4, 8), (2, 4, 10),
          (3, 2, 5), (3, 2, 8), (3, 2, 11),
          (3, 3, 6), (3, 3, 9), (3, 3, 12),
          (3, 4, 7), (3, 4, 10),
          (5, 1, 6), (5, 1, 11), (5, 2, 7), (5, 2, 12), (5, 3, 8), (5, 4, 9)]

    def label(d, p):
        """Name the algebra of dimension d as gl(m|k) / psl(m|k) if possible."""
        out = []
        for m in range(0, 12):
            for k in range(0, 14):
                if (m + k) ** 2 == d:
                    out.append(f"gl({m}|{k})")
                if (m - k) % p == 0 and (m + k) ** 2 - 2 == d and m + k > 0:
                    out.append(f"psl({m}|{k})")
        return ",".join(out) if out else "?"

    for (p, a, b) in S4:
        for alg in ('gl', 'psl'):
            row = []
            for r in range(1, a + 1):
                rank, dgx = analyse(a, b, r, p, alg)
                row.append(f"r={r}: rank {rank}, dim g_x {dgx} [{label(dgx, p)}]")
            print(f"p={p}  {alg}({a}|{b}):  " + " | ".join(row))
        print()
