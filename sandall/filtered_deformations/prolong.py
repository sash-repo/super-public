#!/usr/bin/env python3
"""The Skryabin algebras my(6; 1), smy^(1)(6; 1), by(7; 1), sby^(1)(7; 1) over F_3
as generalized Cartan prolongs, realized inside vect(6; 1) resp. vect(7; 1).

    my(6; un) = (E^2(V), V, gl(V))_{*,un},      dim V = 3,
    by(7; un) = (E^3(V), E^2(V), V, gl(V))_{*,un},

see the list of algebras in the text.  g_{-1} = V, g_{-2} = E^2(V), g_{-3} = E^3(V);
the coordinates dual to them are x_1..x_3 (degree 1), y_1..y_3 (degree 2, y_k dual
to e_i^e_j with (i,j,k) cyclic) and z (degree 3).

The nilpotent g_- is realized by the left-invariant fields written out below; the
brackets are verified by assert, so the realization is checked, not asserted.
Then g_0 is computed as the normalizer of g_- in vect(n; 1)_0 and the positive
part by the usual recursion g_k = {D in vect_k : [D, g_{-1}] subset g_{k-1}}.
"""
import numpy as np
from gvect import P, Vect, rref, nullspace, left_nullspace

EPS = {}
for (i, j, k), s in {(0, 1, 2): 1, (1, 2, 0): 1, (2, 0, 1): 1,
                     (0, 2, 1): -1, (2, 1, 0): -1, (1, 0, 2): -1}.items():
    EPS[(i, j, k)] = s


def eps(i, j, k):
    return EPS.get((i, j, k), 0)


class GradedAlg:
    """a graded subalgebra of vect(n; 1), given by a basis of homogeneous fields."""

    def __init__(self, V, basis, name=""):
        self.V = V
        self.name = name
        self.basis = list(basis)          # list of dicts (vector fields)
        self.n = len(self.basis)
        self.deg = [self._deg(D) for D in self.basis]
        self.wt = [self._wt(D) for D in self.basis]
        self.names = ["b%d" % i for i in range(self.n)]
        self._index()
        self._c = {}

    def _deg(self, D):
        ds = {self.V.deg(k) for k in D}
        assert len(ds) == 1, "not homogeneous"
        return ds.pop()

    def _wt(self, D):
        ws = {self.V.wt(k) for k in D}
        assert len(ws) == 1, "not weight-homogeneous"
        return ws.pop()

    def _index(self):
        """the basis is (degree, weight)-homogeneous, and so is every bracket, so
        the coordinate problem splits into one small block per (degree, weight)."""
        self.block = {}
        for i, D in enumerate(self.basis):
            self.block.setdefault((self.deg[i], self.wt[i]), []).append(i)
        self.solver = {}
        for key, idx in self.block.items():
            keys = sorted({k for i in idx for k in self.basis[i]})
            kpos = {k: t for t, k in enumerate(keys)}
            M = np.zeros((len(idx), len(keys)), dtype=np.int64)
            for r, i in enumerate(idx):
                for k, c in self.basis[i].items():
                    M[r, kpos[k]] = c % P
            aug = np.concatenate([M, np.eye(len(idx), dtype=np.int64)], axis=1)
            Ra, _ = rref(aug)
            R, piv = rref(M)
            assert len(R) == len(idx), "the given basis is dependent"
            self.solver[key] = (keys, kpos, R, piv, Ra[:len(R), len(keys):], idx)

    def coords(self, D):
        """coordinates of the field D in self.basis (raises if D is outside)."""
        out = np.zeros(self.n, dtype=np.int64)
        if not D:
            return out
        key = (self.V.fdeg(D), self.V.fwt(D))
        if key not in self.solver:
            raise ValueError("field outside the algebra (empty block %s)" % (key,))
        keys, kpos, R, piv, T, idx = self.solver[key]
        v = np.zeros(len(keys), dtype=np.int64)
        for k, c in D.items():
            if k not in kpos:
                raise ValueError("field outside the algebra")
            v[kpos[k]] = c % P
        co = np.zeros(len(R), dtype=np.int64)
        for i, c in enumerate(piv):
            if v[c]:
                co[i] = v[c] % P
                v = (v - co[i] * R[i]) % P
        if v.any():
            raise ValueError("field outside the algebra")
        loc = (co @ T) % P
        for t, i in enumerate(idx):
            out[i] = loc[t]
        return out

    def row(self, i):
        """structure constants [b_i, b_m] for all m, as a list of dicts."""
        if i not in self._c:
            out = []
            for m in range(self.n):
                br = self.V.bracket(self.basis[i], self.basis[m])
                co = self.coords(br) if br else np.zeros(self.n, dtype=np.int64)
                out.append({j: int(c) for j, c in enumerate(co) if c})
            self._c[i] = out
        return self._c[i]

    def brk(self, i, m):
        return self.row(i)[m]


# ------------------------------------------------------------------ the algebras
def build(kind, real=0):
    """kind in {'my', 'smy', 'by', 'sby'}; returns (V, gminus, layers).

    `real` selects one of the (equivalent) realizations of g_- in the by/sby case:
    a_i = lam*y_i + h_i, b_j = mu*x_j with mu - lam = 1 and
    d_i h_j - d_j h_i = eps_{ijl} x_l; the prolong must not depend on the choice.
    """
    three = (1, 1, 1)
    e = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if kind in ("my", "smy"):
        degs = [1, 1, 1, 2, 2, 2]
        wts = [tuple(-a for a in e[i]) for i in range(3)] + \
              [tuple(-(three[t] - e[i][t]) for t in range(3)) for i in range(3)]
        names = ["x1", "x2", "x3", "y1", "y2", "y3"]
    else:
        degs = [1, 1, 1, 2, 2, 2, 3]
        wts = [tuple(-a for a in e[i]) for i in range(3)] + \
              [tuple(-(three[t] - e[i][t]) for t in range(3)) for i in range(3)] + \
              [(-1, -1, -1)]
        names = ["x1", "x2", "x3", "y1", "y2", "y3", "z"]
    V = Vect(degs, wts, names)
    n = V.n
    Z0 = tuple([0] * n)

    def mon(**kw):
        m = [0] * n
        for a, v in kw.items():
            m[names.index(a)] = v
        return tuple(m)

    ix = lambda a: names.index(a)

    # g_{-1}: X_i = d_{x_i} + sum eps(i,j,k) x_j d_{y_k}  [ + h_i d_z ]
    X = []
    for i in range(3):
        D = {(Z0, ix("x%d" % (i + 1))): 1}
        for j in range(3):
            for k in range(3):
                s = eps(i, j, k)
                if s:
                    key = (mon(**{"x%d" % (j + 1): 1}), ix("y%d" % (k + 1)))
                    D[key] = (D.get(key, 0) + s) % P
        X.append({k: v for k, v in D.items() if v})
    Y = [{(Z0, ix("y%d" % (k + 1))): 1} for k in range(3)]

    if kind in ("by", "sby"):
        # h_1 = 0, h_2 = x_1 x_3, h_3 = 2 x_1 x_2 ;  Y_k gets + x_k d_z
        # h_1 = A x2 x3, h_2 = B x3 x1, h_3 = C x1 x2 with B-A = C-B = A-C = 1
        ABC, lam = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][real % 3], (real // 3) % 3
        mu = (lam + 1) % P
        H = [{mon(x2=1, x3=1): ABC[0]}, {mon(x3=1, x1=1): ABC[1]},
             {mon(x1=1, x2=1): ABC[2]}]
        for i in range(3):
            if lam:
                key = (mon(**{"y%d" % (i + 1): 1}), ix("z"))
                X[i][key] = (X[i].get(key, 0) + lam) % P
            for m, c in H[i].items():
                if c % P:
                    key = (m, ix("z"))
                    X[i][key] = (X[i].get(key, 0) + c) % P
        X[:] = [{k: v % P for k, v in D.items() if v % P} for D in X]
        for k in range(3):
            if mu:
                Y[k][(mon(**{"x%d" % (k + 1): 1}), ix("z"))] = mu
        Zf = [{(Z0, ix("z")): 1}]
    else:
        Zf = []

    gminus = X + Y + Zf

    # ---- verify the brackets of g_-
    def cmp_(a, b):
        return all((a.get(k, 0) - b.get(k, 0)) % P == 0 for k in set(a) | set(b))

    for i in range(3):
        for j in range(3):
            want = {}
            for k in range(3):
                s = eps(i, j, k)
                if s:
                    for kk, c in Y[k].items():
                        want[kk] = (want.get(kk, 0) + s * c) % P
            assert cmp_(V.bracket(X[i], X[j]), want), ("XX", i, j)
            if kind in ("by", "sby"):
                want = {kk: c for kk, c in Zf[0].items()} if i == j else {}
                assert cmp_(V.bracket(X[i], Y[j]), want), ("XY", i, j)
            else:
                assert not V.bracket(X[i], Y[j]), ("XY", i, j)
            assert not V.bracket(Y[i], Y[j]), ("YY", i, j)
    if kind in ("by", "sby"):
        for i in range(3):
            assert not V.bracket(X[i], Zf[0]) and not V.bracket(Y[i], Zf[0])

    # ---- g_0 = normalizer of g_- among the degree-0 fields
    layers = {}
    dmin = -3 if kind in ("by", "sby") else -2
    for d in range(dmin, 0):
        layers[d] = [D for D in gminus if V.fdeg(D) == d]

    g0 = normalizer(V, layers, gminus)
    if kind in ("smy", "sby"):
        g0 = trace_free(V, g0, layers[-1])
    layers[0] = g0
    return V, gminus, layers


def _tomat(V, fields, keys):
    kpos = {k: i for i, k in enumerate(keys)}
    M = np.zeros((len(fields), len(keys)), dtype=np.int64)
    for r, D in enumerate(fields):
        for k, c in D.items():
            M[r, kpos[k]] = c % P
    return M


def normalizer(V, layers, gminus):
    """{D in vect_0 : [D, g_i] subset g_i for all i<0}."""
    src = V.basis_of_degree(0)
    spos = {k: i for i, k in enumerate(src)}
    rows = []
    for d, fields in layers.items():
        tgt = V.basis_of_degree(d)
        tpos = {k: i for i, k in enumerate(tgt)}
        Msub = _tomat(V, fields, tgt)
        A = nullspace(Msub, len(tgt))               # functionals vanishing on g_d
        for E in fields:
            col = np.zeros((len(src), len(tgt)), dtype=np.int64)
            for i, s in enumerate(src):
                br = V.bracket({s: 1}, E)
                for k, c in br.items():
                    col[i, tpos[k]] = c % P
            rows.append((A @ col.T) % P)            # (|A|, |src|)
    Mall = np.concatenate(rows, axis=0) if rows else np.zeros((0, len(src)), dtype=np.int64)
    NS = nullspace(Mall, len(src))
    return [{s: int(c) for s, c in zip(src, v) if c} for v in NS]


def trace_free(V, g0, gm1):
    """the sl(V) part of g_0: those D whose action on g_{-1} has trace 0."""
    tgt = V.basis_of_degree(-1)
    tpos = {k: i for i, k in enumerate(tgt)}
    Mm1 = _tomat(V, gm1, tgt)
    R, piv = rref(Mm1)
    aug = np.concatenate([Mm1, np.eye(len(gm1), dtype=np.int64)], axis=1)
    Ra, _ = rref(aug)
    T = Ra[:len(R), len(tgt):]

    def coords(br):
        v = np.zeros(len(tgt), dtype=np.int64)
        for k, c in br.items():
            v[tpos[k]] = c % P
        co = np.zeros(len(R), dtype=np.int64)
        w = v.copy()
        for i, c in enumerate(piv):
            if w[c]:
                co[i] = w[c] % P
                w = (w - co[i] * R[i]) % P
        assert not w.any()
        return (co @ T) % P

    tr = []
    for D in g0:
        t = 0
        for a, E in enumerate(gm1):
            t += int(coords(V.bracket(D, E))[a])
        tr.append(t % P)
    keys = sorted({k for D in g0 for k in D})
    M = _tomat(V, g0, keys)
    NS = nullspace(np.array([tr], dtype=np.int64), len(g0))
    out = []
    for v in NS:
        D = {}
        for c, E in zip(v, g0):
            if c:
                for k, cc in E.items():
                    D[k] = (D.get(k, 0) + int(c) * cc) % P
        out.append({k: c for k, c in D.items() if c})
    return out


def prolong(V, layers, verbose=True):
    """extend layers to all positive degrees; returns the full list of basis fields."""
    gm1 = layers[-1]
    d = 1
    while True:
        src = V.basis_of_degree(d)
        if not src:
            break
        prev = layers.get(d - 1, [])
        tgt = V.basis_of_degree(d - 1)
        if not prev:
            break
        tpos = {k: i for i, k in enumerate(tgt)}
        Mprev = _tomat(V, prev, tgt)
        A = nullspace(Mprev, len(tgt))
        rows = []
        for E in gm1:
            col = np.zeros((len(src), len(tgt)), dtype=np.int64)
            for i, s in enumerate(src):
                for k, c in V.bracket({s: 1}, E).items():
                    col[i, tpos[k]] = c % P
            rows.append((A @ col.T) % P)
        Mall = np.concatenate(rows, axis=0)
        NS = nullspace(Mall, len(src))
        cur = [{s: int(c) for s, c in zip(src, v) if c} for v in NS]
        if not cur:
            break
        layers[d] = cur
        if verbose:
            print("   g_%-3d dim %d" % (d, len(cur)))
        d += 1
    return layers


def assemble(V, layers, name):
    basis = []
    for d in sorted(layers):
        basis.extend(layers[d])
    g = GradedAlg(V, basis, name)
    nm = []
    for d in sorted(layers):
        for i in range(len(layers[d])):
            nm.append("%d.%d" % (d, i))
    g.names = nm
    g.layers = layers
    return g


def make(kind, verbose=True, real=0):
    V, gminus, layers = build(kind, real)
    if verbose:
        for d in sorted(layers):
            print("   g_%-3d dim %d" % (d, len(layers[d])))
    prolong(V, layers, verbose)
    g = assemble(V, layers, kind)
    if verbose:
        print("   %s(%d; 1): dim = %d, degrees %d..%d"
              % (kind, V.n, g.n, min(layers), max(layers)))
    return g


if __name__ == "__main__":
    import sys
    for kind in (sys.argv[1:] or ["my", "by"]):
        print("=" * 60)
        print(kind)
        make(kind)
