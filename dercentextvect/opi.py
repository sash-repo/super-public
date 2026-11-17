"""
o_Pi(2n) in characteristic 2, with exact F_2 linear algebra (Subsecs. 3.3, 3.4
of the text):

    1) dim out der (o^{(2)}_Pi(8)/c) = 26 ?
    2) psl(4) ~ o^{(2)}_Pi(6), via the correspondence (44) ?
    3) dim out der o^{(2)}_Pi(6) = 7 ?

Everything is over F_2.  A matrix of gl(N) is an int, bit i*N+j = entry (i,j);
addition is XOR, so a subspace is handled by Gaussian elimination on ints.

An independent check of the answers quoted in [BGLL1] = arXiv:1307.1858,
Subsec. 3.1.
"""

from itertools import combinations


# ---------------------------------------------------------------- gl(N)/F_2

def E(N, i, j):
    """Matrix unit E^{i,j}, indices 1-based."""
    return 1 << ((i - 1) * N + (j - 1))


def mul(N, X, Y):
    """Matrix product over F_2."""
    Z = 0
    for i in range(N):
        row = (X >> (i * N)) & ((1 << N) - 1)
        if not row:
            continue
        acc = 0
        for k in range(N):
            if (row >> k) & 1:
                acc ^= (Y >> (k * N)) & ((1 << N) - 1)
        Z |= acc << (i * N)
    return Z


def bracket(N, X, Y):
    """[X,Y] = XY + YX over F_2."""
    return mul(N, X, Y) ^ mul(N, Y, X)


def transpose(N, X):
    T = 0
    for i in range(N):
        for j in range(N):
            if (X >> (i * N + j)) & 1:
                T |= 1 << (j * N + i)
    return T


# ------------------------------------------------------ linear algebra /F_2

def echelon(vectors):
    """Row-reduce a list of ints; return the list of nonzero pivot rows."""
    basis = []          # (pivot bit, row)
    for v in vectors:
        for p, b in basis:
            if (v >> p) & 1:
                v ^= b
        if v:
            p = v.bit_length() - 1
            basis.append((p, v))
            basis.sort(key=lambda t: -t[0])
    return [b for _, b in basis]


def reduce_by(basis, v):
    """Reduce v modulo an echelon basis; 0 iff v lies in the span."""
    for b in basis:
        p = b.bit_length() - 1
        if (v >> p) & 1:
            v ^= b
    return v


def coords(basis, v):
    """Coordinates of v in `basis` (a list of ints, assumed independent).

    Returns None if v is not in the span.  Straightforward: run elimination
    keeping track of which basis vectors were used.
    """
    # build echelon form with provenance
    rows = []
    for k, b in enumerate(basis):
        vec, prov = b, 1 << k
        for p, r, pr in rows:
            if (vec >> p) & 1:
                vec ^= r
                prov ^= pr
        if vec:
            rows.append((vec.bit_length() - 1, vec, prov))
            rows.sort(key=lambda t: -t[0])
    vec, prov = v, 0
    for p, r, pr in rows:
        if (vec >> p) & 1:
            vec ^= r
            prov ^= pr
    if vec:
        return None
    return prov          # bitmask of basis indices


def rank(vectors):
    return len(echelon(vectors))


def same_span(A, B):
    """True iff the two lists of vectors span the same subspace."""
    ea, eb = echelon(A), echelon(B)
    if len(ea) != len(eb):
        return False
    return all(reduce_by(ea, v) == 0 for v in eb)


def nullspace(rows, nvars):
    """Basis of {x in F_2^nvars : rows . x = 0}, rows given as int bitmasks."""
    piv = {}                            # pivot column -> row with that leader
    for r in rows:
        while r:
            c = r.bit_length() - 1
            if c in piv:
                r ^= piv[c]
            else:
                piv[c] = r
                break
    # back-substitute to reduced row echelon form
    for c in sorted(piv):
        r = piv[c]
        for c2 in sorted(piv):
            if c2 > c and (r >> c2) & 1:
                r ^= piv[c2]
        piv[c] = r
    pivots = set(piv)
    free = [c for c in range(nvars) if c not in pivots]
    out = []
    for f in free:
        x = 1 << f
        for c, r in piv.items():
            if (r >> f) & 1:
                x |= 1 << c
        out.append(x)
    return out


# ------------------------------------------------------------- Lie algebras

class LieAlg:
    """A Lie algebra over F_2 given by a basis of matrices in gl(N)."""

    def __init__(self, N, basis, name=""):
        self.N = N
        self.basis = echelon(basis)
        self.dim = len(self.basis)
        self.name = name
        self._sc = None

    def struct(self):
        """Structure constants: sc[i][j] = bitmask of basis indices."""
        if self._sc is not None:
            return self._sc
        n, N, B = self.dim, self.N, self.basis
        sc = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                c = coords(B, bracket(N, B[i], B[j]))
                assert c is not None, "not closed under the bracket"
                sc[i][j] = sc[j][i] = c        # p = 2: no sign
        return sc

    def derived(self):
        vs = [bracket(self.N, x, y) for x, y in combinations(self.basis, 2)]
        return LieAlg(self.N, vs, self.name + "'")

    def center(self):
        sc = self.struct()
        n = self.dim
        rows = []
        for j in range(n):                      # one equation per basis index
            for k in range(n):                  # ... and per output coordinate
                r = 0
                for i in range(n):
                    if (sc[i][j] >> k) & 1:
                        r |= 1 << i
                if r:
                    rows.append(r)
        return nullspace(rows, n)

    def quotient_by_center(self):
        """g/c as an abstract Lie algebra, given by structure constants."""
        c = self.center()
        keep = []                               # complement of the center
        cbasis = echelon(c)
        for k in range(self.dim):
            v = 1 << k
            if reduce_by(echelon(cbasis + keep), v):
                keep.append(v)
        return keep, cbasis


def der_dim(sc, n):
    """dim Der(g) for a Lie algebra of dim n with structure constants sc.

    Unknowns: D_{ab}, the (a,b) entry of D in the given basis, index a*n+b
    meaning  D(e_b) = sum_a D_{ab} e_a.
    Equation, for each pair b<c and each output index a:
        sum_k sc[b][c]_k D_{ak} + sum_m (D_{mb} sc[m][c]_a + D_{mc} sc[b][m]_a) = 0
    """
    rows = []
    for b in range(n):
        for c in range(b + 1, n):
            for a in range(n):
                r = 0
                for k in range(n):
                    if (sc[b][c] >> k) & 1:
                        r ^= 1 << (a * n + k)
                for m in range(n):
                    if (sc[m][c] >> a) & 1:
                        r ^= 1 << (m * n + b)
                    if (sc[b][m] >> a) & 1:
                        r ^= 1 << (m * n + c)
                if r:
                    rows.append(r)
    return len(nullspace(rows, n * n))


def der_brute(sc, n):
    """Same, by enumerating every matrix.  Only for n <= 4."""
    assert n <= 4
    good = []
    for D in range(1 << (n * n)):
        ok = True
        for b in range(n):
            for c in range(b + 1, n):
                lhs = 0
                for k in range(n):
                    if (sc[b][c] >> k) & 1:
                        for a in range(n):
                            if (D >> (a * n + k)) & 1:
                                lhs ^= 1 << a
                for m in range(n):
                    if (D >> (m * n + b)) & 1:
                        lhs ^= sc[m][c]
                    if (D >> (m * n + c)) & 1:
                        lhs ^= sc[b][m]
                if lhs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            good.append(D)
    return rank(good)


# --------------------------------------------------- the algebras in question

def o_Pi(n):
    """o_Pi(2n) = {X in gl(2n) : X^T Pi + Pi X = 0}, Pi = [[0,1],[1,0]]."""
    N = 2 * n
    Pi = 0
    for i in range(1, n + 1):
        Pi |= E(N, i, n + i) | E(N, n + i, i)
    vs = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            vs.append(E(N, i, j))
    # solve the linear condition by brute force over the matrix units
    sol = []
    # X^T Pi + Pi X = 0 is linear; collect its matrix and take the nullspace
    rows = []
    for a in range(N):
        for b in range(N):
            r = 0
            for k, X in enumerate(vs):
                Y = mul(N, transpose(N, X), Pi) ^ mul(N, Pi, X)
                if (Y >> (a * N + b)) & 1:
                    r |= 1 << k
            if r:
                rows.append(r)
    for x in nullspace(rows, len(vs)):
        M = 0
        for k in range(len(vs)):
            if (x >> k) & 1:
                M ^= vs[k]
        sol.append(M)
    return LieAlg(N, sol, "o_Pi(%d)" % N)


def chevalley_span(n):
    """The span of the Chevalley elements (42), (43), reduced mod 2."""
    N = 2 * n
    vs = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i != j:
                vs.append(E(N, i, j) ^ E(N, n + j, n + i))
                vs.append(E(N, i, n + j) ^ E(N, j, n + i))
                vs.append(E(N, n + i, j) ^ E(N, n + j, i))
    for i in range(1, n):
        vs.append(E(N, i, i) ^ E(N, i + 1, i + 1)
                  ^ E(N, n + i, n + i) ^ E(N, n + i + 1, n + i + 1))
    vs.append(E(N, n - 1, n - 1) ^ E(N, n, n)
              ^ E(N, 2 * n - 1, 2 * n - 1) ^ E(N, 2 * n, 2 * n))
    return LieAlg(N, vs, "Chev(D_%d)" % n)


def psl4():
    """psl(4) over F_2, as structure constants on a chosen basis of sl(4)/<I>."""
    N = 4
    sl, names = [], []
    for i in range(1, 5):
        for j in range(1, 5):
            if i != j:
                sl.append(E(N, i, j))
                names.append("E^{%d,%d}" % (i, j))
    sl.append(E(N, 1, 1) ^ E(N, 2, 2))
    names.append("E^{1,1}+E^{2,2}")
    sl.append(E(N, 2, 2) ^ E(N, 3, 3))
    names.append("E^{2,2}+E^{3,3}")
    I = E(N, 1, 1) ^ E(N, 2, 2) ^ E(N, 3, 3) ^ E(N, 4, 4)
    return N, sl, names, I
