"""lanes.py -- Phi evaluated at many points at once, in floating
point: every coefficient of the ring is a numpy vector of 'lanes'.  Used for
the Jacobian (holomorphic 4-point stencil, exact for degree <= 4 up to
rounding) and for a Levenberg-Marquardt search for zeros of Phi.
The exact engine (n4elim.py, n4phi2.py) remains the reference: lanes.check() compares."""
import os, sys, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import n4emb, n4noeth, n4rec
import fast
from n4loop import step_of
OldC = n4emb.C

class V:
    __slots__ = ("a",)
    def __init__(self, re=0, im=0):
        self.a = re if isinstance(re, np.ndarray) else complex(float(re), float(im))
    @staticmethod
    def w(a):
        v = V.__new__(V); v.a = a; return v
    def __add__(s, o): return V.w(s.a + cx(o).a)
    __radd__ = __add__
    def __neg__(s): return V.w(-s.a)
    def __sub__(s, o): return V.w(s.a - cx(o).a)
    def __rsub__(s, o): return V.w(cx(o).a - s.a)
    def __mul__(s, o): return V.w(s.a * cx(o).a)
    __rmul__ = __mul__
    def __bool__(s):
        a = s.a
        return bool(np.any(a != 0)) if isinstance(a, np.ndarray) else a != 0
    def __eq__(s, o): return not bool(s - o)

def cx(z):
    if isinstance(z, V): return z
    if isinstance(z, OldC): return V(z.re, z.im)
    return V(z)

def recip(z): return V.w(1 / cx(z).a)

def patch():
    fast.patch()
    for mod in (n4emb, n4noeth, n4rec, fast):
        mod.C = V
        if hasattr(mod, "cx"): mod.cx = cx
        if hasattr(mod, "recip"): mod.recip = recip
    for mod in (n4noeth, n4rec):
        for name, val in list(vars(mod).items()):
            if isinstance(val, OldC):
                setattr(mod, name, cx(val))
            elif isinstance(val, dict) and val and all(isinstance(x, OldC) for x in val.values()):
                setattr(mod, name, {k: cx(x) for k, x in val.items()})

def conv_q(q):
    return [[{k: cx(v) for k, v in q[i][j].items()} for j in (0, 1)] for i in (0, 1)]

class Engine:
    def __init__(self, tag):
        B = pickle.load(open(os.path.join(HERE, "build-%s.pkl" % tag), "rb"))
        Cc = pickle.load(open(os.path.join(HERE, "coker-%s.pkl" % tag), "rb"))
        patch()
        c = lambda z: complex(float(z.re), float(z.im))
        self.q0 = conv_q(B["q0"])
        self.x0 = np.array([c(v) for v in B["x0"]])
        self.K = np.array([[c(v) for v in kv] for kv in B["ker0"]]).T  # 896 x 448
        # level-2 part: u[i][j][key] = sum_d co_d * d2_d[i][j][key]
        self.ukeys = {}
        for d, D in enumerate(B["d2"]):
            for i in (0, 1):
                for j in (0, 1):
                    for k, v in D[i][j].items():
                        self.ukeys.setdefault((i, j, k), []).append((d, c(v)))
        self.covered = Cc["covered"]
        self.Ylist = Cc["Y"]
        self.yidx = None
    def q_of(self, lamL):
        """lamL: 448 x L complex array -> q with lane coefficients"""
        co = self.x0[:, None] + self.K @ lamL
        q = [[dict((k, V.w(v.a * np.ones(co.shape[1]))) for k, v in self.q0[i][j].items())
              for j in (0, 1)] for i in (0, 1)]
        for (i, j, k), lst in self.ukeys.items():
            ds = [d for d, _ in lst]; cs = np.array([x for _, x in lst])
            val = cs @ co[ds, :]
            if k in q[i][j]:
                q[i][j][k] = V.w(q[i][j][k].a + val)
            else:
                q[i][j][k] = V.w(val)
        return q
    def phi(self, lamL):
        R = n4rec.resid_full(self.q_of(lamL))
        R1 = {k: v.a for k, v in R.items() if step_of(k) == 1}
        L = lamL.shape[1]
        if self.yidx is None:
            keys = sorted({k for y in self.Ylist for k in y}, key=str)
            self.ykeys = keys
            kidx = {k: n for n, k in enumerate(keys)}
            import scipy.sparse as sp
            rows, cols, vals = [], [], []
            for n, y in enumerate(self.Ylist):
                for k, v in y.items():
                    rows.append(n); cols.append(kidx[k]); vals.append(complex(float(v.re), float(v.im)))
            self.Ymat = sp.csr_matrix((vals, (rows, cols)), shape=(len(self.Ylist), len(keys)))
            self.yidx = kidx
        Rm = np.zeros((len(self.ykeys), L), complex)
        free = {}
        for k, a in R1.items():
            if k in self.yidx:
                Rm[self.yidx[k], :] = a
            elif k not in self.covered:
                free[k] = a
        out = self.Ymat @ Rm
        return out, free

if __name__ == "__main__":
    # check against the exact engine at the particular solution and one point
    import random
    E = Engine("mono1-w-10_7")
    rng = random.Random(5)
    lam = np.array([complex(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in range(448)])
    L = np.stack([np.zeros(448), lam], 1)
    import time; t = time.time()
    Y, F = E.phi(L)
    print("lanes eval %.1fs" % (time.time() - t))
    print("|Phi| at 0:", np.linalg.norm(Y[:, 0]), np.linalg.norm([v[0] for v in F.values()]))
    print("|Phi| at lam:", np.linalg.norm(Y[:, 1]), np.linalg.norm([v[1] for v in F.values()]))
    print("nonzero coker comps at lam:", int(np.sum(np.abs(Y[:, 1]) > 1e-6)), "free nonzero:", sum(abs(v[1]) > 1e-6 for v in F.values()))


class Engine2(Engine):
    """As Engine, but from any build pickle, with an ORTHONORMAL basis of the
    cokernel of M4 computed in floating point component by component, so
    that |Phi(lam)| = Euclidean distance from the level-1 residual to the
    image of M4 (in the monomial basis of the rows)."""
    def __init__(self, path):
        import n4noeth as N0
        B = pickle.load(open(path, "rb"))
        comp = N0._components(B["M4"])
        patch()
        c = lambda z: complex(float(z.re), float(z.im))
        self.q0 = conv_q(B["q0"])
        self.x0 = np.array([c(v) for v in B["x0"]])
        self.K = np.array([[c(v) for v in kv] for kv in B["ker0"]]).T
        self.nker = self.K.shape[1]
        self.ukeys = {}
        for d, D in enumerate(B["d2"]):
            for i in (0, 1):
                for j in (0, 1):
                    for k, v in D[i][j].items():
                        self.ukeys.setdefault((i, j, k), []).append((d, c(v)))
        import scipy.sparse as sp
        keys, rows, cols, vals, nq = [], [], [], [], 0
        covered = set()
        for root, mem in comp.items():
            ks = sorted([m[1] for m in mem if m[0] == "k"], key=str)
            js = [m[1] for m in mem if m[0] == "c"]
            if not ks:
                continue
            covered |= set(ks)
            kidx = {k: n for n, k in enumerate(ks)}
            A = np.zeros((len(ks), len(js)), complex)
            for n, j in enumerate(js):
                for k, v in B["M4"][j].items():
                    A[kidx[k], n] = c(v)
            U, s, _ = np.linalg.svd(A, full_matrices=True)
            r = int(np.sum(s > (s[0] if len(s) else 1) * 1e-10)) if len(s) else 0
            Qc = U[:, r:]
            base = len(keys)
            keys += ks
            for a in range(Qc.shape[1]):
                for b in range(Qc.shape[0]):
                    if abs(Qc[b, a]) > 1e-14:
                        rows.append(nq + a); cols.append(base + b); vals.append(Qc[b, a].conjugate())
            nq += Qc.shape[1]
        self.ykeys = keys
        self.yidx = {k: n for n, k in enumerate(keys)}
        self.Ymat = sp.csr_matrix((vals, (rows, cols)), shape=(nq, len(keys)))
        self.covered = covered
        print("Engine2: %d level-2 dirs, kernel %d, M4 %d cols, coker dim %d inside %d row keys"
              % (len(B["d2"]), self.nker, len(B["M4"]), nq, len(keys)), flush=True)
