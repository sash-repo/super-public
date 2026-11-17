"""In-silico cohort for the prognostic conjecture of the text (the simulated cohort).

The model.  A nodule is a closed star-shaped surface r(u) = R s exp(h(u)), u in S^2, with
h a random sum of K von Mises-Fisher bumps (lobulation).  Visit 2 is the material point
of direction u carried radially, the whole lesion scaled by the growth factor s.  The two
classes differ only in what the growth does to the PATTERN of asymmetry:

  benign      the follow-up keeps the pattern (lambda or theta small);
  malignant   the pattern changes:
     design A (exactly matched): the follow-up is CONGRUENT to the baseline, the bumps
              having moved over the tissue by a rotation of angle theta -- every separable
              descriptor is the same in the two classes, by construction;
     design B (generic):  h2 = cos(lam) h1 + sin(lam) h_new, new lobes replacing old ones;
              the per-visit distributions agree, but the pair of them need not.

The size (R, s) is drawn from one and the same law in both classes.  Noise: independent
relative noise sigma on each second fundamental form at each of N sample points, a
registration error of the frame transport (constant + smooth part), and a small error
of the point correspondence.  Each class has n lesions; the AUC of rho, of the best
separable classifier (ridge logistic regression on size + per-visit curvature
quantiles + their deltas, 5-fold cross-validated) and of both together are reported.

Run:  python3 cohort.py
"""
import numpy as np
import sys

RNG = np.random.default_rng(20261003)
KBUMPS, KAPPA, AMP = 12, 8.0, 0.12
NPTS = 300


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


U = fib_sphere(NPTS)
DOMEGA = 4 * np.pi / NPTS


def rand_unit(k):
    v = RNG.normal(size=(k, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rot_axis(axis, ang):
    a = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def new_field():
    return rand_unit(KBUMPS), RNG.normal(scale=AMP, size=KBUMPS)


def h_eval(field, u):
    V, a = field
    return np.exp(KAPPA * (u @ V.T - 1)) @ a


class Surf:
    def __init__(self, field, scale):
        self.field, self.scale = field, scale

    def F(self, x):
        r = np.linalg.norm(x, axis=-1)
        return r - self.scale * np.exp(h_eval(self.field, x / r[..., None]))

    def point(self, u):
        return (self.scale * np.exp(h_eval(self.field, u)))[:, None] * u

    def grad_hess(self, p, d):
        n = len(p)
        E = np.eye(3) * d
        g = np.zeros((n, 3))
        Hs = np.zeros((n, 3, 3))
        f0 = self.F(p)
        for i in range(3):
            fp, fm = self.F(p + E[i]), self.F(p - E[i])
            g[:, i] = (fp - fm) / (2 * d)
            Hs[:, i, i] = (fp - 2 * f0 + fm) / d ** 2
            for j in range(i + 1, 3):
                v = (self.F(p + E[i] + E[j]) - self.F(p + E[i] - E[j])
                     - self.F(p - E[i] + E[j]) + self.F(p - E[i] - E[j])) / (4 * d * d)
                Hs[:, i, j] = Hs[:, j, i] = v
        return g, Hs


def tangent_basis(n):
    a = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e1 = a - (a * n).sum(1, keepdims=True) * n
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(n, e1)


def II_in(Hs, g, f1, f2):
    gn = np.linalg.norm(g, axis=1)
    A = np.empty((len(g), 2, 2))
    F = [f1, f2]
    for i in range(2):
        for j in range(2):
            A[:, i, j] = np.einsum('ni,nij,nj->n', F[i], Hs, F[j]) / gn
    return A


def a_alpha(A):
    return (A[:, 0, 0] + A[:, 1, 1]) / 2, (A[:, 0, 0] - A[:, 1, 1]) / 2 - 1j * A[:, 0, 1]


def corr_noise(shape, u=None):
    """Smooth noise: unit-variance random field of correlation angle ~ 1/sqrt(KN)."""
    KN, NB = 20.0, 60
    out = np.empty(shape)
    for i in range(2):
        for j in range(2):
            V = rand_unit(NB)
            z = np.exp(KN * (U @ V.T - 1)) @ RNG.normal(size=NB)
            out[:, i, j] = z / z.std()
    return out


CORR = False


def noisy(A, sigma, scale):
    Z = corr_noise(A.shape) if CORR else RNG.normal(size=A.shape)
    Z = (Z + np.transpose(Z, (0, 2, 1))) / 2
    return A + sigma * scale * Z


def lesion(design, par, sigma, reg_err=0.05, pt_err=0.02, world_rot=True):
    """Return (rho, separable feature vector) of one simulated lesion."""
    R = RNG.uniform(4.0, 15.0)
    s = RNG.uniform(1.05, 1.4)
    f1 = new_field()
    if design == 'A':
        Q = rot_axis(rand_unit(1)[0], par)
        f2 = (f1[0] @ Q.T, f1[1])
    else:
        fn = new_field()
        f2 = (np.vstack([f1[0], fn[0]]), np.concatenate([np.cos(par) * f1[1], np.sin(par) * fn[1]]))
    S1, S2 = Surf(f1, R), Surf(f2, R * s)
    G = rot_axis(rand_unit(1)[0], RNG.uniform(0, np.pi)) if world_rot else np.eye(3)  # patient positioning
    Qerr = rot_axis(rand_unit(1)[0], RNG.normal(scale=pt_err))  # point-correspondence error

    p1 = S1.point(U)
    g1, H1 = S1.grad_hess(p1, 1e-4 * R)
    n1 = g1 / np.linalg.norm(g1, axis=1, keepdims=True)
    e1, e2 = tangent_basis(n1)
    A = II_in(H1, g1, e1, e2)

    U2 = U @ Qerr.T
    p2 = S2.point(U2)
    g2, H2 = S2.grad_hess(p2, 1e-4 * R)
    # visit 2 in the scanner's own coordinates: rotated by G
    g2w, H2w = g2 @ G.T, np.einsum('ab,nbc,dc->nad', G, H2, G)
    n2 = g2w / np.linalg.norm(g2w, axis=1, keepdims=True)
    q1, q2 = tangent_basis(n2)

    # d(phi): material map u -> G p2(Qerr u), read in ON frames; polar factor R
    w1, w2 = tangent_basis(U)
    eps = 1e-5

    def push(S, w, M):
        up = U + eps * w
        um = U - eps * w
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        um /= np.linalg.norm(um, axis=1, keepdims=True)
        return (S.point(up @ M.T) - S.point(um @ M.T)) @ (G.T if S is S2 else np.eye(3)) / (2 * eps)

    d1 = [push(S1, w, np.eye(3)) for w in (w1, w2)]
    d2 = [push(S2, w, Qerr) for w in (w1, w2)]
    M1 = np.stack([np.stack([(e * d).sum(1) for d in d1], 1) for e in (e1, e2)], 1)
    M2 = np.stack([np.stack([(q * d).sum(1) for d in d2], 1) for q in (q1, q2)], 1)
    D = M2 @ np.linalg.inv(M1)
    Us, _, Vt = np.linalg.svd(D)
    Rp = Us @ Vt
    # registration error of the frame transport: constant + smooth part
    e0 = RNG.normal(scale=reg_err)
    ax = rand_unit(1)[0]
    eps_field = e0 + reg_err * (U @ ax)
    c, sn = np.cos(eps_field), np.sin(eps_field)
    Rerr = np.stack([np.stack([c, -sn], 1), np.stack([sn, c], 1)], 1)
    Rp = Rp @ Rerr
    E1 = q1 * Rp[:, 0, 0:1] + q2 * Rp[:, 1, 0:1]
    E2 = q1 * Rp[:, 0, 1:2] + q2 * Rp[:, 1, 1:2]
    B = II_in(H2w, g2w, E1, E2)

    A = noisy(A, sigma, 1 / R)
    B = noisy(B, sigma, 1 / (R * s))
    a, al = a_alpha(A)
    b, be = a_alpha(B)
    w = (np.linalg.norm(p1, axis=1) ** 2 / (n1 * U).sum(1)) * DOMEGA
    rho = abs((w * np.conj(al) * be).sum()) / (w * abs(al) * abs(be)).sum()

    def sep(Sx, p, aa, alx):
        r = np.linalg.norm(p, axis=1)
        V = (r ** 3).sum() * DOMEGA / 3
        Dmax = (r + Sx.scale * np.exp(h_eval(Sx.field, -U))).max()
        K = aa ** 2 - abs(alx) ** 2
        Rs = V ** (1 / 3)  # make curvature features dimensionless by the size
        q = lambda x: np.quantile(x, [0.1, 0.5, 0.9])
        return np.concatenate([[np.log(V), np.log(Dmax)], q(aa * Rs), q(K * Rs ** 2), q(abs(alx) * Rs)])

    s1 = sep(S1, p1, a, al)
    s2 = sep(S2, S2.point(U2), b, be)
    return rho, np.concatenate([s1, s2, s2 - s1])


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def logreg_cv(X, y, lam=1.0, folds=5):
    n = len(y)
    idx = RNG.permutation(n)
    out = np.empty(n)
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Xt = np.hstack([np.ones((len(tr), 1)), (X[tr] - mu) / sd])
        Xe = np.hstack([np.ones((len(te), 1)), (X[te] - mu) / sd])
        wv = np.zeros(Xt.shape[1])
        for _ in range(50):
            p = 1 / (1 + np.exp(-Xt @ wv))
            gr = Xt.T @ (p - y[tr]) + lam * np.r_[0, wv[1:]]
            Hm = Xt.T @ (Xt * (p * (1 - p))[:, None]) + lam * np.diag(np.r_[0, np.ones(len(wv) - 1)])
            wv -= np.linalg.solve(Hm, gr)
        out[te] = Xe @ wv
    return out


def cohort(design, ben, mal, sigma, n=300, **kw):
    rows = []
    for cls, rng in ((0, ben), (1, mal)):
        for _ in range(n):
            rows.append((cls,) + lesion(design, RNG.uniform(*rng), sigma, **kw))
    y = np.array([r[0] for r in rows])
    rho = np.array([r[1] for r in rows])
    X = np.array([r[2] for r in rows])
    a_rho = auc(-rho, y)
    a_size = auc(logreg_cv(X[:, [0, 1, 11, 12, 22, 23]], y), y)
    a_sep = auc(logreg_cv(X, y), y)
    a_all = auc(logreg_cv(np.hstack([X, rho[:, None]]), y), y)
    return a_rho, a_size, a_sep, a_all, rho[y == 0].mean(), rho[y == 1].mean()


def selftest():
    # noiseless, no registration error, theta = 0: pure similar growth -> rho = 1
    r, _ = lesion('A', 0.0, 0.0, reg_err=0.0, pt_err=0.0, world_rot=True)
    print(f"selftest: similar growth, scanner rotated, no noise: rho = {r:.10f}")
    assert abs(r - 1) < 1e-6
    # design A is exactly matched: per-visit features of the follow-up equal those of a
    # congruent copy, so the separable AUC must be ~0.5 (checked in the table)
    r, _ = lesion('A', 0.3, 0.0, reg_err=0.0, pt_err=0.0)
    print(f"selftest: moved pattern theta=0.3, no noise: rho = {r:.4f} (< 1 expected)")
    assert r < 0.999


def main():
    selftest()
    print(f"N = {NPTS} sample points per surface, n = 300 lesions per class\n")
    print("design  benign       malignant     sigma | AUC rho  size  separable  sep+rho | mean rho B / M")
    for design, ben, mals in (('A', (0.0, 0.05), [(0.10, 0.20), (0.20, 0.40), (0.40, 0.80)]),
                              ('B', (0.0, 0.15), [(0.15, 0.30), (0.30, 0.60), (0.60, 1.20)])):
        for mal in mals:
            for sigma in (0.10, 0.25):
                res = cohort(design, ben, mal, sigma)
                print(f"  {design}    {ben}  {mal}  {sigma:.2f} |  {res[0]:.3f}  {res[1]:.3f}   {res[2]:.3f}    {res[3]:.3f}  | {res[4]:.3f} / {res[5]:.3f}")
                sys.stdout.flush()
    global CORR
    CORR = True
    print("\nspatially correlated curvature noise (correlation angle ~0.22 rad), sigma = 0.25:")
    for design, ben, mal in (('A', (0.0, 0.05), (0.20, 0.40)), ('A', (0.0, 0.05), (0.40, 0.80)),
                             ('B', (0.0, 0.15), (0.30, 0.60)), ('B', (0.0, 0.15), (0.60, 1.20))):
        res = cohort(design, ben, mal, 0.25)
        print(f"  {design}    {ben}  {mal}  0.25 |  {res[0]:.3f}  {res[1]:.3f}   {res[2]:.3f}    {res[3]:.3f}  | {res[4]:.3f} / {res[5]:.3f}")
    CORR = False
    # negative control: both classes from the benign law
    for design, ben in (('A', (0.0, 0.05)), ('B', (0.0, 0.15))):
        res = cohort(design, ben, ben, 0.10)
        print(f"  control {design}: both classes {ben}: AUC rho {res[0]:.3f}, separable {res[2]:.3f}, sep+rho {res[3]:.3f}")


if __name__ == '__main__':
    main()
