"""grading.py -- which row power n_z each term of
Phi = c + L lam + Q(lam,lam) reaches, lam = (lam_n), n=-3..4, 56 each.
Rows are pulled back to monomial row keys (Ymat has orthonormal rows, so
norms are preserved)."""
import sys, os, pickle, numpy as np, collections
sys.argv = [sys.argv[0], sys.argv[1]]
import lm2
E, N = lm2.E, lm2.N
nb = N // 56; ny = E.Ymat.shape[0]
_zd = collections.defaultdict(set)
for (i, j, k), lst in E.ukeys.items():
    for d, _ in lst: _zd[d].add(k[0])
KZ = []
for col in E.K.T:
    s_ = set()
    for d in np.nonzero(np.abs(col) > 1e-12)[0]: s_ |= _zd[d]
    assert len(s_) == 1, s_; KZ.append(s_.pop())
KZ = np.array(KZ)
zrow = np.zeros(len(E.yidx), int)
for k, n in E.yidx.items(): zrow[n] = k[2][0]
def rowz(R):
    """R: comps x L -> per-z squared norms, for each lane"""
    _ = lm2.resid(np.zeros((N,1)))
    rv = E.Ymat.conj().T @ R[:ny]
    out = collections.defaultdict(lambda: np.zeros(R.shape[1]))
    for z in set(zrow): out[z] += np.sum(np.abs(rv[zrow == z])**2, 0)
    for i, k in enumerate(E.fkeys): out[k[2][0]] += np.abs(R[ny+i])**2
    return out
rng = np.random.default_rng(5)
blocks = sorted(set(KZ)); nb = len(blocks)
vec = {}
for b in range(nb):
    v = np.zeros(N, complex); m_ = KZ == blocks[b]; v[m_] = rng.standard_normal(m_.sum()) + 1j*rng.standard_normal(m_.sum()); vec[b] = v
cols = [np.zeros(N)]; lab = ["c"]
for b in range(nb):
    cols += [vec[b], -vec[b]]; lab += ["+%d"%b, "-%d"%b]
for a in range(nb):
    for b in range(a+1, nb):
        cols.append(vec[a]+vec[b]); lab.append("%d,%d"%(a,b))
R = lm2.resid(np.array(cols).T)
ix = {l:i for i,l in enumerate(lab)}
c = R[:, 0]
def show(name, x):
    d = rowz(x[:, None]); tot = sum(v[0] for v in d.values())
    zs = sorted(z for z in d if d[z][0] > 1e-12*max(tot,1))
    print("%-10s |.|=%9.3g  rows n_z: %s" % (name, np.sqrt(tot), [int(z) for z in zs]), flush=True)
show("const", c)
lin, quad = {}, {}
for b in range(nb):
    p, m = R[:, ix["+%d"%b]], R[:, ix["-%d"%b]]
    lin[b] = (p - m)/2; quad[(b,b)] = (p + m)/2 - c
    show("L lam_%d" % blocks[b], lin[b])
for b in range(nb): show("Q %d.%d" % (blocks[b], blocks[b]), quad[(b,b)])
for a in range(nb):
    for b in range(a+1, nb):
        x = R[:, ix["%d,%d"%(a,b)]] - c - lin[a] - lin[b] - quad[(a,a)] - quad[(b,b)]
        show("Q %d.%d" % (blocks[a], blocks[b]), x)
