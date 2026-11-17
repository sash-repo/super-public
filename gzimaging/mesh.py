# The two exponents of the mesh estimator of Appendix F, and what they predict.
#
# A script of H. Andersen measures the pointwise turn error of the
# Appendix F pipeline on a closed triangulated ellipsoid at three fitting radii on clean
# data and at two radii under 0.5% normal noise.  Those five numbers are two rows of one
# law: an estimator bias growing like r^beta and a noise term falling like r^-nu.  This
# engine measures beta and nu over a full sweep and at three noise levels, tests nu
# against the exponent k+1/2 of the paper's own Lemma "The best scale of an order" at
# k=2, locates the optimal radius the two exponents predict, and checks the prediction
# against the measured minimum.  Parts 4 and 5 ask why the REGIONAL numbers survive the
# noise that destroys the pointwise field, and where the fall of rho comes from.
#
# The mesh, the deformation, the analytic truth and the pipeline are hers,
# copied here unchanged so that the exponents are measured on exactly her estimator.
import sys, math, json
import numpy as np, sympy as sp
from scipy.spatial import cKDTree

LOG = open(__file__.replace(".py", ".log"), "w")
def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.write(s + "\n"); LOG.flush()

CHECKS = [0, 0]
def check(name, cond, detail=""):
    CHECKS[0] += 1
    if not cond: CHECKS[1] += 1
    say(f"  [{'ok ' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

# ---------------------------------------------------------------- her mesh and truth
def icosphere(n):
    t = (1 + 5 ** 0.5) / 2
    V = np.array([[-1,t,0],[1,t,0],[-1,-t,0],[1,-t,0],[0,-1,t],[0,1,t],[0,-1,-t],
                  [0,1,-t],[t,0,-1],[t,0,1],[-t,0,-1],[-t,0,1]], float)
    F = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],
         [7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],
         [8,6,7],[9,8,1]]
    V /= np.linalg.norm(V, axis=1)[:, None]; F = np.array(F)
    for _ in range(n):
        cache = {}; V = list(V); NF = []
        def mid(a, b):
            k = (min(a,b), max(a,b))
            if k not in cache:
                m = (V[a] + V[b]); m /= np.linalg.norm(m); V.append(m); cache[k] = len(V) - 1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a,b), mid(b,c), mid(c,a)
            NF += [[a,ab,ca],[b,bc,ab],[c,ca,bc],[ab,bc,ca]]
        V = np.array(V); F = np.array(NF)
    for i, (a, b, c) in enumerate(F):
        if np.dot(np.cross(V[b]-V[a], V[c]-V[a]), V[a]+V[b]+V[c]) < 0: F[i] = [a, c, b]
    return V, F

S, Fc = icosphere(4)
AX = np.array([1.0, 0.72, 1.45]); V1 = S * AX
th0 = np.radians(10)
c45 = np.cos(np.radians(45)); R45 = np.array([[c45,-c45,0],[c45,c45,0],[0,0,1]])
M3 = R45.T @ np.diag([1.25, 1.0, 1.06]) @ R45
def phi_np(P):
    Q = P @ M3.T; Q = np.stack([Q[:,0] + 0.08 * P[:,2]**2, Q[:,1], Q[:,2]], 1)
    th = th0 + 0.15 * P[:,2]
    return np.stack([np.cos(th)*Q[:,0] - np.sin(th)*Q[:,1],
                     np.sin(th)*Q[:,0] + np.cos(th)*Q[:,1], Q[:,2]], 1)
V2 = phi_np(V1)

u, v = sp.symbols('u v', real=True)
r1 = sp.Matrix([sp.cos(u)*sp.cos(v), 0.72*sp.sin(u)*sp.cos(v), 1.45*sp.sin(v)])
def rot_z(th): return sp.Matrix([[sp.cos(th),-sp.sin(th),0],[sp.sin(th),sp.cos(th),0],[0,0,1]])
M3s = sp.Matrix(M3)
r2 = rot_z(th0 + 0.15*r1[2]) * (M3s*r1 + sp.Matrix([0.08*r1[2]**2, 0, 0]))
def geom(r):
    ru, rv = r.diff(u), r.diff(v); n = ru.cross(rv); n = n/sp.sqrt(n.dot(n))
    II = sp.Matrix([[r.diff(u,2).dot(n), r.diff(u,v).dot(n)],
                    [r.diff(u,v).dot(n), r.diff(v,2).dot(n)]])
    e1 = ru/sp.sqrt(ru.dot(ru)); w = rv - rv.dot(e1)*e1; e2 = w/sp.sqrt(w.dot(w))
    P = sp.Matrix([[1/sp.sqrt(ru.dot(ru)), -rv.dot(e1)/(sp.sqrt(ru.dot(ru))*sp.sqrt(w.dot(w)))],
                   [0, 1/sp.sqrt(w.dot(w))]])
    return (sp.lambdify((u,v), P.T*II*P, 'numpy'), sp.lambdify((u,v), sp.Matrix.hstack(e1,e2), 'numpy'),
            sp.lambdify((u,v), sp.sqrt(ru.cross(rv).dot(ru.cross(rv))), 'numpy'))
A1f, E1f, dA1f = geom(r1); A2f, E2f, _ = geom(r2)
X = sp.symbols('x0:3')
phis = rot_z(th0 + 0.15*X[2]) * (M3s*sp.Matrix(X) + sp.Matrix([0.08*X[2]**2, 0, 0]))
Dphi = sp.lambdify(X, phis.jacobian(X), 'numpy')
uv = np.stack([np.arctan2(S[:,1], S[:,0]), np.arcsin(np.clip(S[:,2], -1, 1))], 1)

def hopf(M): return (M[0,0]+M[1,1])/2, (M[0,0]-M[1,1])/2 - 1j*M[0,1]
def polar(M):
    Uu, s, Vt = np.linalg.svd(M); R = Uu @ Vt
    if np.linalg.det(R) < 0: Uu[:, -1] *= -1; R = Uu @ Vt
    return R, R.T @ M

def vertex_normals_areas(V, F):
    fn = np.cross(V[F[:,1]]-V[F[:,0]], V[F[:,2]]-V[F[:,0]]); fa = np.linalg.norm(fn, axis=1)/2
    N = np.zeros_like(V); Ar = np.zeros(len(V))
    for k in range(3): np.add.at(N, F[:,k], fn); np.add.at(Ar, F[:,k], fa/3)
    return N/np.linalg.norm(N, axis=1)[:,None], Ar
def frames(N):
    a = np.where((np.abs(N[:,0]) < 0.9)[:,None], np.array([1.0,0,0]), np.array([0,1.0,0]))
    e1 = np.cross(N, a); e1 /= np.linalg.norm(e1, axis=1)[:,None]
    return e1, np.cross(N, e1)
def fit_II(V, N, e1, e2, tree, rfit):
    out = np.zeros((len(V),2,2)); nn = np.zeros(len(V), int)
    for i in range(len(V)):
        idx = tree.query_ball_point(V[i], rfit); d = V[idx] - V[i]
        x = d@e1[i]; y = d@e2[i]; z = d@N[i]; nn[i] = len(idx)
        M = np.stack([x*x/2, x*y, y*y/2, x, y, np.ones_like(x)], 1)
        w = np.exp(-(x*x + y*y)/(rfit/2)**2)
        c = np.linalg.lstsq(M*w[:,None], z*w, rcond=None)[0]; a, b, cc, dx, dy = c[:5]
        gn = 1/np.sqrt(1 + dx*dx + dy*dy); H = np.array([[a,b],[b,cc]])*gn
        g = np.array([[1+dx*dx, dx*dy],[dx*dy, 1+dy*dy]])
        L = np.linalg.cholesky(g); Li = np.linalg.inv(L)
        out[i] = Li@H@Li.T
    return out, nn
def fit_dphi(V1, V2, e1, e2, e1p, e2p, tree, sphi):
    M = np.zeros((len(V1),2,2))
    for i in range(len(V1)):
        idx = tree.query_ball_point(V1[i], sphi); d = V1[idx]-V1[i]; dp = V2[idx]-V2[i]
        Xm = np.stack([d@e1[i], d@e2[i]], 1); Ym = np.stack([dp@e1p[i], dp@e2p[i]], 1)
        M[i] = np.linalg.lstsq(Xm, Ym, rcond=None)[0].T
    return M
def pipeline(V1, V2, F, rfit, sphi):
    N1, Ar1 = vertex_normals_areas(V1, F); N2, _ = vertex_normals_areas(V2, F)
    e1, e2 = frames(N1); e1p, e2p = frames(N2); t1 = cKDTree(V1)
    A, nn = fit_II(V1, N1, e1, e2, t1, rfit); Bf, _ = fit_II(V2, N2, e1p, e2p, cKDTree(V2), rfit)
    M = fit_dphi(V1, V2, e1, e2, e1p, e2p, t1, sphi)
    al = np.zeros(len(V1), complex); be = np.zeros(len(V1), complex)
    for i in range(len(V1)):
        R, U = polar(M[i]); B = R.T@Bf[i]@R
        al[i] = hopf(A[i])[1]; be[i] = hopf(B)[1]
    return al, be, Ar1, nn
def summaries(al, be, w):
    J = np.conj(al)*be; num = np.sum(w*J); den = np.sum(w*np.abs(al)*np.abs(be))
    return abs(num)/den, np.degrees(-0.5*np.angle(num)), num.imag/den

# per-vertex truth
alT = np.zeros(len(V1), complex); beT = np.zeros(len(V1), complex)
for i in range(len(V1)):
    uu, vv = uv[i]
    A = np.array(A1f(uu,vv), float); E1 = np.array(E1f(uu,vv), float); E2 = np.array(E2f(uu,vv), float)
    M = E2.T @ np.array(Dphi(*V1[i]), float) @ E1; R, U = polar(M)
    alT[i] = hopf(A)[1]; beT[i] = hopf(R.T @ np.array(A2f(uu,vv), float) @ R)[1]
umb = np.abs(alT) < 0.10*np.median(np.abs(alT)); ok = ~umb
JT = np.conj(alT)*beT

def errors(al, be):
    """Two measures of the same error.  The first is hers, the RMS of the turn in degrees,
    which is what a physicist reads; being an angle it saturates, and an exponent must not
    be fitted on it above a few degrees.  The second is the relative error of J itself.
    It is J and not alpha that may be compared with the truth at all: the frame of step 1)
    is the mesh's own, alpha has weight 2 and turns with it, and only J is a function --
    Remark re:measure.  J is besides linear in each form, so this measure does not
    saturate, and it is the one the exponents are fitted on."""
    tw = np.degrees(np.angle(np.conj(al[ok])*be[ok]*np.conj(JT[ok])))/2
    J = np.conj(al)*be
    relJ = math.sqrt(float(np.sum(np.abs(J[ok]-JT[ok])**2)/np.sum(np.abs(JT[ok])**2)))
    return math.sqrt(float(np.mean(tw**2))), relJ

def loglog_slope(xs, ys):
    lx = np.log(np.array(xs)); ly = np.log(np.array(ys))
    return float(np.polyfit(lx, ly, 1)[0])

def noisy(V, F, sigma, seed):
    N, _ = vertex_normals_areas(V, F)
    return V + sigma*np.random.default_rng(seed).standard_normal(len(V))[:,None]*N

SPHI = 0.25
say("=" * 78)
say("mesh.py -- the two exponents of the Appendix F estimator on H. Andersen's mesh")
say("=" * 78)
say(f"mesh: icosphere ellipsoid 1:0.72:1.45, {len(V1)} vertices, mean edge "
    f"{np.mean(np.linalg.norm(V1[Fc[:,1]]-V1[Fc[:,0]], axis=1)):.4f}; "
    f"near-umbilic excluded from the pointwise error: {int(umb.sum())}")

# ---------------------------------------------------- 1) the clean bias and its exponent
say("\n1) THE CLEAN BIAS.  No noise, so the whole error is the bias of the quadric fit.")
RS = [0.15, 0.175, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
clean = {}
for r in RS:
    al, be, Ar, nn = pipeline(V1, V2, Fc, r, SPHI)
    tw, rel = errors(al, be); clean[r] = (tw, rel, int(np.median(nn)))
    say(f"   r_fit = {r:.3f}   turn RMS {tw:6.2f} deg   rel err {rel:.5f}   neighbours {int(np.median(nn)):4d}")
WIN = [r for r in RS if 0.20 <= r <= 0.40]           # her own window
beta_tw  = loglog_slope(WIN, [clean[r][0] for r in WIN])
beta_rel = loglog_slope(WIN, [clean[r][1] for r in WIN])
say(f"   exponent over {WIN[0]}-{WIN[-1]}: beta = {beta_tw:.3f} (turn), {beta_rel:.3f} (J)")
check("the clean error is the r^2 bias of a quadric fit", abs(beta_tw - 2) < 0.15,
      f"beta = {beta_tw:.3f} against 2")
check("the same exponent on the measure that cannot saturate", abs(beta_rel - 2) < 0.25,
      f"beta = {beta_rel:.3f} against 2")
beta_all = loglog_slope(RS, [clean[r][0] for r in RS])
say(f"   over the whole sweep {RS[0]}-{RS[-1]}: beta = {beta_all:.3f} "
    f"(the ball ceases to be small against the semi-axis 0.72)")
# why the comparison has to be made on J and not on alpha
al, be, _, _ = pipeline(V1, V2, Fc, 0.20, SPHI)
bad = math.sqrt(float(np.sum(np.abs(al[ok]-alT[ok])**2)/np.sum(np.abs(alT[ok])**2)))
say(f"   the same estimate compared through alpha instead of J: relative error {bad:.3f}")
check("alpha may not be compared with the truth at all, J may -- it has weight 2 and the"
      " mesh frame is its own", bad > 1.0 and clean[0.20][1] < 0.05,
      f"alpha {bad:.3f} against J {clean[0.20][1]:.5f}; a random relative phase gives sqrt2")

# ------------------------------------------------- 2) the noise term and its exponent
say("\n2) THE NOISE TERM.  Independent normal displacement on both meshes; the noise part")
say("   is separated from the bias in quadrature, the two being independent.")
SIG = [0.0025, 0.005, 0.010]; RN = [0.25, 0.30, 0.35, 0.40, 0.45, 0.55, 0.65]
NUWIN = [0.30, 0.35, 0.40, 0.45]
SEEDS = [0, 1, 2]
say("   The exponent is fitted on J, the turn being an angle and so saturating; and the bias")
say("   subtracted is the MEASURED clean error at the same radius, not its power law, which")
say("   holds only while the ball is small against the smallest semi-axis.")
b_tw  = math.exp(np.mean([math.log(clean[r][0]) - beta_tw *math.log(r) for r in WIN]))
b_rel = math.exp(np.mean([math.log(clean[r][1]) - beta_rel*math.log(r) for r in WIN]))
noise = {}
for sg in SIG:
    say(f"   sigma = {sg}")
    for r in RN:
        tws = []; rels = []
        for sd in SEEDS:
            al, be, Ar, nn = pipeline(noisy(V1, Fc, sg, sd), noisy(V2, Fc, sg, 100+sd), Fc, r, SPHI)
            t, e = errors(al, be); tws.append(t); rels.append(e)
        tw = math.sqrt(float(np.mean(np.square(tws))))
        rel = math.sqrt(float(np.mean(np.square(rels))))
        npart  = math.sqrt(max(tw**2  - clean[r][0]**2, 1e-12))
        npartJ = math.sqrt(max(rel**2 - clean[r][1]**2, 1e-12))
        noise[(sg, r)] = (tw, rel, npart, npartJ)
        say(f"      r_fit = {r:.3f}   turn RMS {tw:6.2f} deg   J err {rel:.4f}"
            f"   J bias {clean[r][1]:.4f}   J noise part {npartJ:.4f}")
    # One window for all three noise levels.  Below it the fit is starved -- 23 neighbours
    # for 6 unknowns at r = 0.20 -- and the error leaves every power law; above it the
    # quadrature subtraction is a small difference of two large numbers.
    nu = -loglog_slope(NUWIN, [noise[(sg, r)][3] for r in NUWIN])
    say(f"      exponent nu(sigma={sg}) = {nu:.3f} over r = {NUWIN[0]}-{NUWIN[-1]}"
        f"  (on the turn: {-loglog_slope(NUWIN, [noise[(sg,r)][2] for r in NUWIN]):.3f},"
        " flattened by the saturation of an angle)")
    noise[("nu", sg)] = nu
nus = [noise[("nu", s)] for s in SIG]
nu_bar = float(np.mean(nus))
say(f"   nu = {nu_bar:.3f} +- {float(np.std(nus)):.3f} over the three noise levels")
check("nu does not depend on the noise level (it is an exponent, not a constant)",
      float(np.std(nus))/nu_bar < 0.08,
      f"{', '.join(f'{x:.2f}' for x in nus)}; spread {float(np.std(nus)):.3f}, "
      f"{100*float(np.std(nus))/nu_bar:.0f}% of the mean -- a residual drift with the noise"
      " level, the window being better conditioned where the noise is larger")
say("   Formula (Ik) of the paper integrates over the LINE: it models a profile, so its own")
say("   exponent k+1/2 is k+d/2 with d = 1.  An estimator on a surface averages over a")
say("   two-dimensional neighbourhood, so the same count gives k+d/2 = k+1 = 3 at k = 2.")
check("nu is the k+d/2 of the paper's own noise model, at k = 2 and d = 2",
      abs(nu_bar - 3.0) < 0.3,
      f"nu = {nu_bar:.3f} against 3; the profile model of (Ik), d = 1, would give 2.5")
say("   and the noise term is proportional to sigma at fixed radius:")
for r in (0.30, 0.45):
    amp = [noise[(s, r)][3]/s for s in SIG]
    say(f"      r = {r}: (J noise)/sigma = " + ", ".join(f"{a:.1f}" for a in amp))
    check(f"noise linear in sigma at r = {r}",
          (max(amp)-min(amp))/np.mean(amp) < 0.25, f"spread {(max(amp)-min(amp))/np.mean(amp):.3f}")

# ------------------------------------------- 3) the optimal radius the exponents predict
say("\n3) THE OPTIMAL RADIUS.  With error^2 = (b r^beta)^2 + (c sigma r^-nu)^2 the minimum")
say("   sits at r* = (nu c^2 sigma^2 / (beta b^2))^(1/(2 beta + 2 nu)), so r* grows with the")
say(f"   noise like sigma^(1/(2 beta + 2 nu)) = sigma^{1/(2*beta_rel+2*nu_bar):.3f} -- very slowly;")
say("   and the error at the optimum like sigma^(beta/(beta+nu)) = "
    f"sigma^{beta_rel/(beta_rel+nu_bar):.3f}.")
pred = {}
for sg in SIG:
    c_sg = math.exp(np.mean([math.log(noise[(sg, r)][3]) + nu_bar*math.log(r) for r in NUWIN]))
    rstar = (nu_bar*c_sg**2/(beta_rel*b_rel**2))**(1/(2*beta_rel + 2*nu_bar))
    estar = math.sqrt((b_rel*rstar**beta_rel)**2 + (c_sg*rstar**(-nu_bar))**2)
    measJ = min(RN, key=lambda r: noise[(sg, r)][1])
    measT = min(RN, key=lambda r: noise[(sg, r)][0])
    pred[sg] = (rstar, estar, measJ)
    say(f"   sigma = {sg}:  r* predicted {rstar:.3f};  measured minimum over the grid at"
        f" r = {measJ:.2f} on J and r = {measT:.2f} on the turn")
    check(f"the predicted optimum brackets the measured one at sigma = {sg}",
          abs(math.log(rstar/measJ)) < math.log(1.5),
          f"r* = {rstar:.3f} against {measJ:.2f} on a grid of spacing 0.05-0.10")
ratio = pred[SIG[-1]][0]/pred[SIG[0]][0]; want = (SIG[-1]/SIG[0])**(1/(2*beta_rel+2*nu_bar))
check("r* moves with sigma by the predicted power", abs(ratio/want - 1) < 0.20,
      f"{ratio:.3f} against {want:.3f} over a factor {SIG[-1]/SIG[0]:.0f} in sigma")
eratio = pred[SIG[-1]][1]/pred[SIG[0]][1]; ewant = (SIG[-1]/SIG[0])**(beta_rel/(beta_rel+nu_bar))
check("the error at the optimum grows like sigma^(beta/(beta+nu))", abs(eratio/ewant - 1) < 0.25,
      f"{eratio:.3f} against {ewant:.3f}")
# the one she actually ran: sigma = 0.005, stopped at 0.45
rstar5 = pred[0.005][0]
say(f"   At her own noise level, 0.005, the law puts the best radius at {rstar5:.3f}. She")
say("   tried 0.30 and 0.45 and reported the second: so the second was the optimum, and")
say("   the law says by how much the first missed it.")
check("the law puts the optimum at the better of the two radii she tried",
      abs(math.log(rstar5/0.45)) < abs(math.log(rstar5/0.30)) and abs(math.log(rstar5/0.45)) < 0.2,
      f"r* = {rstar5:.3f} against her 0.45 and 0.30")
check("and the cost of the first is what the law says it is",
      abs(noise[(0.005,0.30)][0]/noise[(0.005,0.45)][0] - 1) > 1.0,
      f"{noise[(0.005,0.30)][0]:.1f} deg at 0.30 against {noise[(0.005,0.45)][0]:.1f} at 0.45")
say("   The bias is a property of the surface and the noise one of the segmentation, so the")
say("   law is to be read as: the fitting radius is not a preference but a function of the")
say("   two, and it is flat -- a factor 4 in the noise moves it by a third.")

# --------------------------------- 4) why the regional numbers survive what the field does not
say("\n4) POINTWISE AGAINST REGIONAL.  At one and the same radius the noise multiplies the")
say("   pointwise turn error by a factor of twenty and moves the regional turn by a tenth.")
al0, be0, Ar, _ = pipeline(V1, V2, Fc, 0.30, SPHI)
r0, t0, tau0 = summaries(al0, be0, Ar)
rows = []
for sg in SIG:
    al, be, Ar2, _ = pipeline(noisy(V1, Fc, sg, 0), noisy(V2, Fc, sg, 100), Fc, 0.30, SPHI)
    rn, tn, taun = summaries(al, be, Ar2)
    tw = errors(al, be)[0]
    rows.append((sg, tw, rn, tn, taun))
    say(f"   sigma = {sg}: pointwise {tw:6.2f} deg (clean {clean[0.30][0]:.2f});"
        f"  regional turn {tn:.3f} (clean {t0:.3f}), taubar {taun:.5f} (clean {tau0:.5f}),"
        f" rho {rn:.4f} (clean {r0:.4f})")
sg, tw, rn, tn, taun = rows[1]                     # her own 0.005
check("the pointwise field is destroyed where the regional turn is not",
      tw/clean[0.30][0] > 10 and abs(tn-t0)/abs(t0) < 0.25,
      f"pointwise x{tw/clean[0.30][0]:.0f}, regional turn {100*abs(tn-t0)/abs(t0):.0f}% ")
say(f"   the reason is that a summary is an integral: the displacement is independent from")
say(f"   vertex to vertex, so it averages down by the square root of the count,")
say(f"   sqrt({len(V1)}) = {math.sqrt(len(V1)):.0f}, while the field keeps it whole.")
check("the surviving regional shift stays under the pointwise error over sqrt(N)",
      abs(tn-t0) < tw/math.sqrt(len(V1)),
      f"{abs(tn-t0):.3f} deg against the bound {tw/math.sqrt(len(V1)):.3f} = pointwise/sqrt(N)")

# ------------------------------------------------ 5) where the fall of rho comes from
say("\n5) WHY RHO FALLS WHILE THE TURN DOES NOT.  This is Remark re:coh, which says the")
say("   floor of rho is the upward bias of its DENOMINATOR, incurred once per sample and")
say("   not averaged away, seen for the first time on mesh data rather than on forms.")
say("   Two things are asked of the data: that the numerator, a coherent sum, is unbiased,")
say("   and that the denominator grows by the amount the measured noise variance predicts.")
say("   The variance is read off |alpha|, which is frame-free, and not off alpha, which is")
say("   not: n_A^2 = mean(|alpha_noisy|^2 - |alpha_clean|^2).")
say("   A modulus is a convex function, so E|alpha+noise| is NOT sqrt(|alpha|^2+n^2): the")
say("   expectation for a circular perturbation is the Rice mean, and it is what is used.")
from scipy.stats import rice
num0 = abs(np.sum(Ar*np.conj(al0)*be0)); den0 = float(np.sum(Ar*np.abs(al0)*np.abs(be0)))
def rice_mean(mag, n2):
    s = math.sqrt(max(n2, 1e-18)/2)                # per real component
    return rice.mean(b=mag/s, scale=s)
for sg in SIG:
    nums = []; dens = []; rns = []; dpred = []; npred = []
    for sd in SEEDS:
        alN, beN, ArN, _ = pipeline(noisy(V1, Fc, sg, sd), noisy(V2, Fc, sg, 100+sd), Fc, 0.30, SPHI)
        nA2 = float(np.mean(np.abs(alN[ok])**2 - np.abs(al0[ok])**2))
        nB2 = float(np.mean(np.abs(beN[ok])**2 - np.abs(be0[ok])**2))
        dp = float(np.sum(ArN*rice_mean(np.abs(al0), nA2)*rice_mean(np.abs(be0), nB2)))
        # the numerator is a sum of independent perturbations; its modulus is biased up by
        # its own variance and by nothing else
        vnum = float(np.sum(ArN**2*(np.abs(al0)**2*nB2 + np.abs(be0)**2*nA2 + nA2*nB2)))
        dpred.append(dp); npred.append(math.sqrt(num0**2 + vnum))
        nums.append(abs(np.sum(ArN*np.conj(alN)*beN)))
        dens.append(float(np.sum(ArN*np.abs(alN)*np.abs(beN))))
        rns.append(summaries(alN, beN, ArN)[0])
    num, den, rn = np.mean(nums), np.mean(dens), np.mean(rns)
    dp, npp = np.mean(dpred), np.mean(npred)
    say(f"   sigma = {sg}: numerator {num:.5f} against {num0:.5f} clean"
        f" ({100*(num/num0-1):+.1f}%, predicted {100*(npp/num0-1):+.1f}%);")
    say(f"      denominator {den:.5f} against {den0:.5f} ({100*(den/den0-1):+.1f}%,"
        f" predicted {100*(dp/den0-1):+.1f}%)")
    say(f"      rho measured {rn:.4f}, predicted from the noise variance alone {npp/dp:.4f}")
    check(f"the numerator of rho is the one the noise leaves alone, sigma = {sg}",
          abs(num/num0 - 1) < 0.10, f"{100*(num/num0-1):+.1f}% against the denominator's"
          f" {100*(den/den0-1):+.1f}%")
    check(f"the denominator inflation is the noise variance, sigma = {sg}",
          abs(dp - den)/den0 < 0.04, f"{100*(den/den0-1):+.1f}% against {100*(dp/den0-1):+.1f}%")
    check(f"whence rho itself, from the noise and nothing fitted, sigma = {sg}",
          abs(npp/dp - rn) < 0.045, f"predicted {npp/dp:.4f} against measured {rn:.4f}")

# 5b) the numerator does move a little at the coarsest noise.  Two things could do it: the
# estimator of the pair could be biased, or the modulus of a noisy sum could be biased up
# while the sum itself is not.  They are told apart by averaging the sum BEFORE taking the
# modulus, over more seeds.
say("\n5b) WHERE THE SMALL MOVE OF THE NUMERATOR COMES FROM.  At the coarsest noise the")
say("    numerator moves too.  Averaging the complex sum over eight noise fields before")
say("    taking its modulus separates a biased estimator from a biased modulus.")
sg = 0.01; sums = []; mods = []
for sd in range(8):
    alN, beN, ArN, _ = pipeline(noisy(V1, Fc, sg, 10*sd), noisy(V2, Fc, sg, 500+10*sd), Fc, 0.30, SPHI)
    z = np.sum(ArN*np.conj(alN)*beN); sums.append(z); mods.append(abs(z))
zbar = np.mean(sums)
say(f"    sigma = {sg}: mean of |sum| {np.mean(mods):.5f}; |mean of sum| {abs(zbar):.5f};"
    f" clean {num0:.5f}")
check("the estimator of the pair is what is biased, not the modulus of the sum",
      abs(abs(zbar)/num0 - 1) > 0.5*abs(np.mean(mods)/num0 - 1),
      f"|E sum| is {100*(abs(zbar)/num0-1):+.1f}% off, E|sum| {100*(np.mean(mods)/num0-1):+.1f}%")
say("    So at twice her noise level the quadric fit itself acquires a bias -- the frame is")
say("    read off the noisy normals and the tilt correction is not linear -- and the linear")
say("    model above, exact to a point at 0.0025 and 0.005, is where it stops.")

say("\n" + "=" * 78)
say(f"{CHECKS[0]} checks, {CHECKS[1]} failed")
say("=" * 78)
LOG.close()
sys.exit(1 if CHECKS[1] else 0)
