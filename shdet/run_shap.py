"""Shapovalov determinants of po(0|4) and po(0|6), against [GL] and [GS].

[GL]  Grozman-Leites, JNMP 8 (2001) 220-228, Theorem 2:
      M^a is irreducible iff 2(a+rho,beta) != (beta,beta) for every QUASIROOT
      beta, where rho = (-2)^{k-2} H_{I-{k}} for g = po(0|2k).
[GS]  Gorelik-Serganova, Contemp. Math. 391 (2005) 111-121:
      Thm 3.1 (n>2): det S_nu is a product of the linear forms h_alpha over the
      ROOTS alpha; Cor 3.2: M(lambda) is simple iff h_alpha(lambda) != 0 for
      every positive root; Subsec 5.3: the explicit answer for po(0|4).
"""
import itertools
import sympy
from poi import Po, Verma, parity

LOG = open("run-shap.log", "w")
CHECKS = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, note=""):
    CHECKS.append((name, ok))
    say(f"  [{'OK  ' if ok else 'FAIL'}] {name}{('  -- ' + note) if note else ''}")


# ---------------------------------------------------------------- utilities
def pos_roots(g):
    return sorted(w for w in g.roots() if g.positive(w))


def h_alpha(V, alpha):
    """[GS] Subsec 3: h_alpha = sum_j alpha_j h_{I minus j}, as a polynomial."""
    n = V.g.n
    return sum(sympy.Integer(s) * V.hsym[tuple(j for j in range(n) if j != i)]
               for i, s in enumerate(alpha) if s)


def det_S(V, nu):
    M, rows, cols = V.shap_matrix(nu)
    return sympy.factor(sympy.expand(M.det())), len(rows)


# ================================================================ po(0|4)
say("=" * 70)
say("po(0|4):  n = 2,  the case [GS] treat separately in their Sect. 5")
say("=" * 70)
g2 = Po(2)
V2 = Verma(g2)
l1, l2 = V2.hsym[(0,)], V2.hsym[(1,)]
say("positive roots:", pos_roots(g2))

# [GS] 5.3: det S_nu = prod_k (h2-h1+k-1)^{tau(nu-k(e1-e2))} (h2+h1-k-1)^{tau(nu-k(e1+e2))}
#                      h1^{2 sum (-1)^{m+1} tau(nu-m e2)} (h2-1)^{2 sum (-1)^{m+1} tau(nu-m e1)}
KOSTANT2 = {}


def tau(V, mu):
    """Kostant partition function: dim U(n_-)_{-mu}."""
    key = tuple(mu)
    if key not in KOSTANT2:
        try:
            KOSTANT2[key] = len(V.env.basis_of_weight(mu))
        except Exception:
            KOSTANT2[key] = 0
    return KOSTANT2[key]


def gs53(V, nu, kmax=8, mmax=8):
    e1e2m, e1e2p = (1, -1), (1, 1)
    f = sympy.Integer(1)
    for k in range(1, kmax + 1):
        t = tau(V, tuple(a - k * b for a, b in zip(nu, e1e2m)))
        if t:
            f *= (l2 - l1 + k - 1) ** t
        t = tau(V, tuple(a - k * b for a, b in zip(nu, e1e2p)))
        if t:
            f *= (l2 + l1 - k - 1) ** t
    c2 = sum((-1) ** (m + 1) * tau(V, tuple(a - m * b for a, b in zip(nu, (0, 1))))
             for m in range(1, mmax + 1))
    c1 = sum((-1) ** (m + 1) * tau(V, tuple(a - m * b for a, b in zip(nu, (1, 0))))
             for m in range(1, mmax + 1))
    return sympy.factor(f * l1 ** (2 * c2) * (l2 - 1) ** (2 * c1))


NUS2 = [(0, 1), (1, -1), (1, 1), (0, 2), (1, 0), (2, -2), (1, 2)]
say("\n--- det S_nu computed from scratch, against [GS] Subsec. 5.3")
for nu in NUS2:
    try:
        d, size = det_S(V2, nu)
    except AssertionError as e:
        say(f"  nu={nu}: skipped ({e})")
        continue
    pred = gs53(V2, nu)
    # compare up to a non-zero scalar
    r = sympy.simplify(sympy.cancel(d / pred)) if pred != 0 else None
    agree = r is not None and r.is_number and r != 0
    say(f"  nu={nu}  size {size}x{size}")
    say(f"     det   = {d}")
    say(f"     [GS]  = {pred}")
    check(f"po(0|4) det S_{nu} = [GS] 5.3 up to a scalar", bool(agree),
          f"ratio {r}")

# the determinant of po(0|4) is NOT homogeneous: [GS] Thm 3.1 is stated for n>2
d, _ = det_S(V2, (1, 0))
check("po(0|4): det S_nu is not homogeneous (so [GS] Thm 3.1 needs n>2)",
      not sympy.Poly(sympy.expand(d), l1, l2).is_homogeneous,
      str(d))

# ================================================================ po(0|6)
say("\n" + "=" * 70)
say("po(0|6):  n = 3,  the case of [GS] Thm 3.1 and of [GL] Thm 2 with k=3")
say("=" * 70)
g3 = Po(3)
V3 = Verma(g3)
pr3 = pos_roots(g3)
say("positive roots:", pr3)
say("simple roots per [GS] 2.6.3:", [(1, -1, -1), (0, 1, -1), (0, 0, 1)])

# the three simple roots of [GS] 2.6.3, and two composite weights that stay
# small enough for an exact symbolic determinant in eight variables
NUS3 = [(0, 0, 1), (0, 1, -1), (1, -1, -1), (0, 0, 2), (0, 1, -2)]
dets3 = {}
say("\n--- det S_nu computed from scratch")
for nu in NUS3:
    try:
        d, size = det_S(V3, nu)
    except AssertionError as e:
        say(f"  nu={nu}: skipped ({e})")
        continue
    dets3[nu] = d
    say(f"  nu={nu}  size {size}x{size}:  det = {d}")

# 1) every irreducible factor is one of the h_alpha, alpha a positive ROOT
halphas = {a: sympy.expand(h_alpha(V3, a)) for a in pr3}
ok, bad = True, []
for nu, d in dets3.items():
    for fac, _mult in sympy.factor_list(d)[1]:
        f = sympy.expand(fac)
        if f.is_number:
            continue
        hit = any(sympy.simplify(sympy.cancel(f / h)).is_number for h in halphas.values())
        if not hit:
            ok = False
            bad.append((nu, fac))
check("po(0|6): every factor of det S_nu is an h_alpha with alpha a ROOT", ok, str(bad))

# at nu = alpha a root, the exponent must be dim g_alpha ([GS] Thm 4.2)
mult_ok, note = True, []
for nu, d in dets3.items():
    if tuple(nu) in halphas:
        e = sum(m for f, m in sympy.factor_list(d)[1]
                if not sympy.expand(f).is_number)
        want = len(g3.roots()[tuple(nu)])
        note.append(f"alpha={nu}: exponent {e}, dim g_alpha = {want}")
        mult_ok &= (e == want)
check("po(0|6): at nu = a root, the exponent of h_alpha is dim g_alpha", mult_ok,
      "; ".join(note))

# 2) the determinant is homogeneous (no rho-shift), as [GS] Thm 3.1 says
allvars = sorted(set().union(*[set(sympy.expand(d).free_symbols) for d in dets3.values()]),
                 key=str)
hom = all(sympy.Poly(sympy.expand(d), *allvars).is_homogeneous for d in dets3.values())
check("po(0|6): det S_nu is homogeneous -- the rho of [GL] Thm 2 drops out", hom,
      "variables " + ", ".join(map(str, allvars)))

# 3) the two criteria compared.  For n>=3 the terms (beta,beta) and 2(rho,beta)
#    of [GL] both vanish, so [GL] says: reducible iff h_beta(a)=0 for some
#    QUASIROOT beta; [GS] say: for some ROOT.  Exhibit a weight where the two
#    disagree and decide it by the determinants.
# ---------------------------------------------------------------------------
# The decisive test: a weight at which NO root vanishes but a non-root
# quasiroot does.  [GL] Thm 2 then calls M(lambda) reducible, [GS] Cor 3.2
# calls it simple.  Substituting lambda first makes the big weight space
# affordable: the Shapovalov matrix becomes an integer matrix.
say("\n--- the decisive weight")
X = {(1, 2): 5, (0, 2): 1, (0, 1): 2}       # h23 = 5, h13 = 1, h12 = 2
vals = {a: sum(s * X[tuple(j for j in range(3) if j != i)]
               for i, s in enumerate(a) if s) for a in pr3}
say("  lambda: h23 = 5, h13 = 1, h12 = 2")
say("  h_alpha(lambda) over the positive roots:", vals)
check("no positive ROOT alpha has h_alpha(lambda) = 0",
      all(v != 0 for v in vals.values()), str([a for a, v in vals.items() if v == 0]))

beta = (0, 2, -1)
say("  beta =", beta, "= (0,1,-1) + (0,1,0), a sum of two positive roots, not a root")
hb = 2 * X[(0, 2)] - 1 * X[(0, 1)]
check("h_beta(lambda) = 0 for that non-root quasiroot beta", hb == 0, f"h_beta = {hb}")

V3n = Verma(g3)
V3n.set_lambda(X)
M, rows, cols = V3n.shap_matrix(beta)
say(f"  the Shapovalov matrix at nu = beta is {len(rows)}x{len(cols)}, entries in Z")
dv = M.det()
say(f"  det S_beta(lambda) = {dv}")
check("det S_beta(lambda) != 0: M(lambda) has NO singular vector of weight "
      "lambda - beta", dv != 0,
      "so [GS] Cor 3.2 is right and the 'only if' half of [GL] Thm 2 is too strong")

say("\n" + "=" * 70)
say(f"{sum(1 for _, o in CHECKS if o)}/{len(CHECKS)} checks pass")
for name, o in CHECKS:
    if not o:
        say("  FAILED:", name)
