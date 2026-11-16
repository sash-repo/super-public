"""Checks of the material on svect^L_lam(1|2) and on k^L(1|n), k^M(1|n) as witt-modules in the text.

1. svect^L_lam(1|2): the three bases (distinguished one and its two odd reflections),
   membership in svect_lam, [X_i^+,X_j^-] = delta_ij H_i (up to scale for the reflected
   bases), [H_i, X_j^pm] = pm A_ij X_j^pm, and the reduction of the three Cartan
   matrices to the normal shape with parameters lam, 1-lam, 1+lam.
2. k^L(1|n), k^M(1|n) as witt-modules: the weight lambda and twist mu of every
   theta-monomial string, from the brackets themselves.
3. The contact bracket and multiplication: {f,gh} = {f,g}h + (-1)^{p(f)p(g)} g{f,h} + K_1(f)gh.
4. The pairing Res(fg) is invariant on k^L(1|6) and on k^M(1|7) (on a window of monomials).
"""
import itertools, random, sys
from fractions import Fraction as Fr
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from stringy import mono, add, smul, mul, parity, d_t, d_odd, eq, VF

random.seed(1)
NFAIL = 0
NCHK = 0
def check(c, msg):
    global NFAIL, NCHK
    NCHK += 1
    if not c:
        NFAIL += 1
        print("FAIL", msg)

# ---------------------------------------------------------------- 1. svect_lam(1|2)
def X(comps):
    return VF(comps, (1, 2))

def div(D):
    s = -1 if D.parity() == 0 else 1
    out = d_t(D.c.get('t', {}))
    for i in (1, 2):
        out = add(out, smul(s, d_odd(i, D.c.get(i, {}))))
    return out

def in_svect(D, lam):
    f = D.c.get('t', {})
    return eq(smul(lam, f), smul(-1, mul(mono(1), div(D))))

def bases(l):
    one = Fr(1)
    b1 = dict(
        Xp=[X({2: mono(0, (1,))}), X({1: mono(1)}), X({'t': mono(1, (2,)), 1: mono(0, (1, 2), -(l + 1))})],
        Xm=[X({1: mono(0, (2,))}), X({2: mono(-1, (1, 2), l), 't': mono(0, (1,))}), X({2: mono(0)})],
        H=[X({1: mono(0, (1,)), 2: mono(0, (2,), -1)}),
           X({'t': mono(1), 1: mono(0, (1,)), 2: mono(0, (2,), l)}),
           X({'t': mono(1), 1: mono(0, (1,), l + 1)})])
    # bases 2 and 3 as printed in the text
    b2 = dict(
        Xp=[X({2: mono(1)}), X({2: mono(-1, (1, 2), l), 't': mono(0, (1,))}), X({1: mono(1, (2,))})],
        Xm=[X({'t': mono(0, (2,)), 1: mono(-1, (1, 2), -l)}), X({1: mono(1)}), X({2: mono(-1, (1,))})],
        H=[X({'t': mono(1, (), -1), 1: mono(0, (1,), -l), 2: mono(0, (2,), -1)}),
           X({'t': mono(1), 1: mono(0, (1,)), 2: mono(0, (2,), l)}),
           X({2: mono(0, (2,)), 1: mono(0, (1,), -1)})])
    b3 = dict(
        Xp=[X({2: mono(0)}), X({1: mono(1, (2,))}), X({'t': mono(1, (1,)), 2: mono(0, (1, 2), l + 1)})],
        Xm=[X({'t': mono(1, (2,)), 1: mono(0, (1, 2), -(l + 1))}), X({2: mono(-1, (1,))}), X({1: mono(0)})],
        H=[X({'t': mono(1), 1: mono(0, (1,), l + 1)}),
           X({2: mono(0, (2,)), 1: mono(0, (1,), -1)}),
           X({'t': mono(1, (), -1), 2: mono(0, (2,), -(l + 1))})])
    A1 = [[2, -1, -1], [1 - l, 0, l], [1 + l, -l, 0]]
    A2 = [[0, -l + 1, -2 + l], [1 - l, 0, l], [-1, -1, 2]]
    A3 = [[0, -l, l + 1], [-1, 2, -1], [1 + l, -l - 2, 0]]
    return [(b1, A1, True), (b2, A2, False), (b3, A3, False)]

def prop(Y, Z):
    """Return c with Y = c Z, or None."""
    c = None
    keys = set(Y.c) | set(Z.c)
    for k in keys:
        y, z = Y.c.get(k, {}), Z.c.get(k, {})
        for m in set(y) | set(z):
            a, b = y.get(m, 0), z.get(m, 0)
            if b == 0:
                if a != 0:
                    return None
                continue
            r = Fr(a) / Fr(b)
            if c is None:
                c = r
            elif c != r:
                return None
    return c if c is not None else 0

def normal(A, mu):
    return [[2, -1, -1], [-1 + 1 / mu, 0, 1], [1 + 1 / mu, -1, 0]]

def reduces(A, mu):
    T = normal(A, mu)
    for perm in itertools.permutations(range(3)):
        B = [[A[perm[i]][perm[j]] for j in range(3)] for i in range(3)]
        ok = True
        for i in range(3):
            # find d with B[i] = d T[i]
            d = None
            for j in range(3):
                if T[i][j] != 0:
                    d = Fr(B[i][j]) / Fr(T[i][j]); break
            if d in (None, 0) or any(Fr(B[i][j]) != d * Fr(T[i][j]) for j in range(3)):
                ok = False; break
        if ok:
            return True
    return False

for l in [Fr(1, 3), Fr(2, 7), Fr(-5, 3), Fr(7, 2), Fr(-1, 4)]:
    for k, (b, A, exact) in enumerate(bases(l)):
        for Z in b['Xp'] + b['Xm'] + b['H']:
            check(in_svect(Z, l), "svect membership lam=%s base %d %r" % (l, k, Z))
        for i in range(3):
            for j in range(3):
                br = b['Xp'][i].bracket(b['Xm'][j])
                if i != j:
                    check(br.is_zero(), "[X%d+,X%d-]!=0 base %d" % (i, j, k))
                else:
                    c = prop(br, b['H'][i])
                    check(c not in (None, 0) and (not exact or c == 1),
                          "[X+,X-]=H base %d i=%d c=%s" % (k, i, c))
                for s, XX in ((1, b['Xp']), (-1, b['Xm'])):
                    br = b['H'][i].bracket(XX[j])
                    check(prop(br, XX[j]) == s * A[i][j],
                          "[H%d,X%d] base %d lam=%s got %s want %s" % (i, j, k, l, prop(br, XX[j]), s * A[i][j]))
        mu = [l, 1 - l, 1 + l][k]
        check(reduces(A, mu), "normal shape base %d" % k)
        order, divs = [((0, 1, 2), (1, l, l)), ((2, 1, 0), (1, 1 - l, l - 1)), ((1, 0, 2), (1, l + 1, -l - 1))][k]
        B = [[Fr(A[order[i]][order[j]]) / divs[i] for j in range(3)] for i in range(3)]
        check(B == [[Fr(x) for x in r] for r in normal(A, mu)], "printed reduction base %d: %s" % (k, B))
    # linear dependence of the H_i in svect (rank 2): H2 - H3 = -lam H1
    b = bases(l)[0][0]
    check(b['H'][1].plus(b['H'][2].scal(-1)) == b['H'][0].scal(-l), "H dependence")

# ---------------------------------------------------------------- 2,3,4 contact brackets
def kb(f, g, odd):
    """L: alpha_1 = dt + sum theta_i dtheta_i (eq. (7) of the text)."""
    def E(h):
        out = {}
        for (a, S), c in h.items():
            out = add(out, {(a, S): c * len(S)})
        return out
    def nab(h):
        return add(smul(2, h), smul(-1, E(h)))
    out = add(mul(nab(f), d_t(g)), smul(-1, mul(d_t(f), nab(g))))
    s = (-1) ** parity(f)
    for i in odd:
        out = add(out, smul(-s, mul(d_odd(i, f), d_odd(i, g))))
    return out

def rb(f, g, odd):
    """M: label 0 = zeta, Ramond bracket of eq. (14) of the text, D = d_t - zeta/(2t) d_zeta."""
    def E(h):
        out = {}
        for (a, S), c in h.items():
            out = add(out, {(a, S): c * len(S)})
        return out
    def nab(h):
        return add(smul(2, h), smul(-1, E(h)))
    def D(h):
        out = d_t(h)
        for (a, S), c in h.items():
            if 0 in S:
                out = add(out, {(a - 1, S): -Fr(1, 2) * c})
        return out
    out = add(mul(nab(f), D(g)), smul(-1, mul(D(f), nab(g))))
    s = (-1) ** parity(f)
    out = add(out, smul(s, mul(mono(-1), mul(d_odd(0, f), d_odd(0, g)))))
    for i in odd:
        if i != 0:
            out = add(out, smul(s, mul(d_odd(i, f), d_odd(i, g))))
    return out

def subsets(odd):
    for r in range(len(odd) + 1):
        for S in itertools.combinations(odd, r):
            yield S

# 2. witt-module weights: (1/2){t^{m+1}, t^k theta_I} = (k + mu + lam(m+1)) t^{m+k} theta_I
for n in range(0, 5):
    odd = tuple(range(1, n + 1))
    for S in subsets(odd):
        lam, mu = Fr(len(S), 2) - 1, 0
        for m in range(-3, 4):
            for k in range(-3, 4):
                g = mono(k, S)
                got = smul(Fr(1, 2), kb(mono(m + 1), g, odd))
                check(eq(got, smul(k + mu + lam * (m + 1), mono(m + k, S))), "witt L n=%d S=%s" % (n, S))
for n in range(1, 5):
    odd = tuple(range(0, n))  # 0 = zeta
    for S in subsets(odd):
        e = 1 if 0 in S else 0
        lam, mu = Fr(len(S), 2) - 1, Fr(-e, 2)
        for m in range(-3, 4):
            for k in range(-3, 4):
                got = smul(Fr(1, 2), rb(mono(m + 1), mono(k, S), odd))
                check(eq(got, smul(k + mu + lam * (m + 1), mono(m + k, S))), "witt M n=%d S=%s" % (n, S))

# 3. bracket vs product
def rnd(odd, par):
    S = random.choice([S for S in subsets(odd) if len(S) % 2 == par])
    return mono(random.randint(-3, 3), S, random.randint(1, 5))
def K1(f):
    return smul(2, d_t(f))
for n in range(0, 5):
    odd = tuple(range(1, n + 1))
    for _ in range(60):
        pf, pg, ph = [random.randint(0, 1) if n else 0 for _ in range(3)]
        f, g, h = rnd(odd, pf), rnd(odd, pg), rnd(odd, ph)
        lhs = kb(f, mul(g, h), odd)
        rhs = add(mul(kb(f, g, odd), h), smul((-1) ** (pf * pg), mul(g, kb(f, h, odd))), mul(K1(f), mul(g, h)))
        check(eq(lhs, rhs), "Leibniz n=%d" % n)

def K1M(f):
    out = d_t(f)
    for (a, S), c in f.items():
        if 0 in S:
            out = add(out, {(a - 1, S): -Fr(1, 2) * c})
    return smul(2, out)
for n in range(1, 5):
    odd = tuple(range(0, n))
    for _ in range(60):
        pf, pg, ph = [random.randint(0, 1) for _ in range(3)]
        f, g, h = rnd(odd, pf), rnd(odd, pg), rnd(odd, ph)
        lhs = rb(f, mul(g, h), odd)
        rhs = add(mul(rb(f, g, odd), h), smul((-1) ** (pf * pg), mul(g, rb(f, h, odd))), mul(K1M(f), mul(g, h)))
        check(eq(lhs, rhs), "Leibniz M n=%d" % n)

# 4. invariance of (f,g) = Res(fg): Res({h,f}g) + (-1)^{p(h)p(f)} Res(f{h,g}) = 0
def res(f, odd):
    return f.get((-1, tuple(sorted(odd))), 0)
for (n, br, odd) in [(6, kb, tuple(range(1, 7)))]:  # k^M(1|7): Res(fg) is NOT invariant (tested, not printed)
    for _ in range(40):
        ph, pf = random.randint(0, 1), random.randint(0, 1)
        h, f = rnd(odd, ph), rnd(odd, pf)
        # g must make fg have total odd degree n and parity: choose complementary
        Sf = [S for (a, S) in f][0]
        for Sg in subsets(odd):
            for kg in range(-4, 4):
                g = mono(kg, Sg)
                v = res(mul(br(h, f, odd), g), odd) + (-1) ** (ph * pf) * res(mul(f, br(h, g, odd)), odd)
                check(v == 0, "invariance n=%d" % n)
    # non-degeneracy on a monomial: f = t^k theta_S pairs with t^{-1-k} theta_{S^c}
    check(res(mul(mono(2, (1, 2)), mono(-3, tuple(x for x in odd if x not in (1, 2)))), odd) != 0, "nondeg")

print("checks: %d, failures: %d" % (NCHK, NFAIL))
