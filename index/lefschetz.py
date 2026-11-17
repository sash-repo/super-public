"""Checks for the super holomorphic Lefschetz formula (Subsec. 2.9 of the text:
Example 2.9.2, Corollary 2.9.3, Remark 2.9.4):
1) Ber(1 - X) = 1 for every Q-type supermatrix X = (A B; B A) with B odd (Grassmann entries),
   n = 1, 2, 3 (A generic even scalar matrix, B with distinct odd generators);
2) GL(1|1): the flag supermanifold is C^{0|1}; the formula at its single point gives the Kac
   supercharacter e^lam (1 - e^{-beta}); compare with str of the Kac module K(lam) of gl(1|1);
3) GL(2|1): the fixed-point sum over W_0 = S_2 equals sch K(lam) = sch(Weyl module of gl(2)) *
   (1 - e^{-beta_1})(1 - e^{-beta_2}) for several lam (exact, sympy);
4) Q(2): for the natural module V and its dual V*, the sum over the two fixed points of the odd
   characters of the fibers equals psi_V, psi_{V*} (the qet-correction multiplies str(phi_p)=0)."""
import sympy as sp, itertools
# ---------- 1) Ber of Q-type matrices
def grassmann_test(n):
    # exact: entries of B are odd generators b_ij; compute det(1 - Y^2)-type Ber via sympy with
    # anticommuting symbols modeled by a small Grassmann algebra
    from functools import reduce
    gens = [(i, j) for i in range(n) for j in range(n)]
    idx = {g: k for k, g in enumerate(gens)}
    def gmul(x, y):
        out = {}
        for ka, va in x.items():
            for kb, vb in y.items():
                if set(ka) & set(kb): continue
                seq = list(ka) + list(kb); s = 1
                for i in range(len(seq)):
                    for j in range(len(seq) - 1 - i):
                        if seq[j] > seq[j + 1]: seq[j], seq[j + 1] = seq[j + 1], seq[j]; s = -s
                k = tuple(seq); out[k] = sp.expand(out.get(k, 0) + s * va * vb)
        return {k: v for k, v in out.items() if v != 0}
    def gadd(*xs):
        out = {}
        for x in xs:
            for k, v in x.items(): out[k] = sp.expand(out.get(k, 0) + v)
        return {k: v for k, v in out.items() if v != 0}
    import random; random.seed(n)
    a = [sp.Rational(random.randint(1, 9), random.randint(2, 11)) for _ in range(n * n)]   # generic rational A
    A = [[{(): (1 if i == j else 0) - a[i * n + j]} for j in range(n)] for i in range(n)]   # 1 - A
    B = [[{(idx[(i, j)],): -1} for j in range(n)] for i in range(n)]                       # -B
    # Ber(1-X) = det(D - C A^{-1} B)^{-1} det(A) with A=D=1-A0, B=C=-B0  -> check det(1 - (A^{-1}B)^2) = 1
    Am = sp.Matrix(n, n, lambda i, j: A[i][j].get((), 0)); Ai = Am.inv()
    # Y = A^{-1} B (odd matrix with Grassmann entries)
    Y = [[gadd(*[{k: sp.expand(Ai[i, l] * v) for k, v in B[l][j].items()} for l in range(n)]) for j in range(n)] for i in range(n)]
    def mm(P, Q): return [[gadd(*[gmul(P[i][k], Q[k][j]) for k in range(n)]) for j in range(n)] for i in range(n)]
    Y2 = mm(Y, Y)
    # det(1 - Y2) = exp(-sum tr(Y2^k)/k); tr(Y^{2k}) must vanish
    P = Y2; ok = True
    for k in range(1, n * n + 1):
        tr = gadd(*[P[i][i] for i in range(n)])
        tr = {kk: sp.simplify(v) for kk, v in tr.items() if sp.simplify(v) != 0}
        if tr: ok = False; break
        P = mm(P, Y2)
        if all(not P[i][j] for i in range(n) for j in range(n)): break
    return ok
for n in (1, 2, 3):
    print("Q-type, n=%d: tr((A^-1 B)^(2k)) = 0 for all k, hence Ber(1-X) = 1:" % n, grassmann_test(n))
# ---------- 2), 3) GL(m|1) via fixed points
t1, t2, s1 = sp.symbols('t1 t2 s1')
def weyl_gl2(l1, l2):   # sum over S_2 of e^{w lam}/(1 - e^{-w alpha}), alpha = e1 - e2
    E = sp.exp
    return sp.simplify(E(l1 * t1 + l2 * t2) / (1 - E(t2 - t1)) + E(l1 * t2 + l2 * t1) / (1 - E(t1 - t2)))
def schur_gl2(l1, l2):
    return sum(sp.exp(k * t1 + (l1 + l2 - k) * t2) for k in range(l2, l1 + 1))
for lam in [(0, 0), (1, 0), (2, 1), (3, 0)]:
    for mu in (0, 1, 2):
        # odd positive roots of gl(2|1) (distinguished Borel): e_i - d, i = 1, 2; W_0-invariant product
        oddf = (1 - sp.exp(s1 - t1)) * (1 - sp.exp(s1 - t2))
        fp = sp.simplify(oddf * weyl_gl2(*lam) * sp.exp(mu * s1))
        kac = sp.expand(oddf * schur_gl2(*lam) * sp.exp(mu * s1))
        assert sp.simplify(fp - kac) == 0
print("GL(2|1): fixed-point sum = sch of the Kac module for lam in {(0,0),(1,0),(2,1),(3,0)}, mu in {0,1,2}")
# GL(1|1): flag = C^{0|1}, one point; local term e^lam det(1 - D), D = e^{-beta} on the odd tangent
lam, beta = sp.symbols('lam beta')
print("GL(1|1): local term at the single point:", sp.exp(lam) * (1 - sp.exp(-beta)), "= sch K(lam)")
# ---------- 4) Q(2): natural and dual modules
th1, th2 = sp.symbols('theta1 theta2')   # stand-ins: the odd characters are linear in them
psiV = 2 * (th1 * sp.exp(t1) + th2 * sp.exp(t2))
fibers = [2 * th1 * sp.exp(t1), 2 * th2 * sp.exp(t2)]          # str(J phi_p) at the two fixed points
print("Q(2), V : sum over fixed points == psi_V:", sp.simplify(sum(fibers) - psiV) == 0)
psiVd = 2 * sp.I * (th1 * sp.exp(-t1) + th2 * sp.exp(-t2))
fibd = [2 * sp.I * th1 * sp.exp(-t1), 2 * sp.I * th2 * sp.exp(-t2)]
print("Q(2), V*: sum over fixed points == psi_V*:", sp.simplify(sum(fibd) - psiVd) == 0)
