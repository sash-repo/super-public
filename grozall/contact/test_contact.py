"""Calibration of contact.py.  Nothing downstream is trusted before this passes."""

import sys, random
from fractions import Fraction as Fr
from contact import *
from contact import Q, HALF

random.seed(11)
FAIL = []
NCHECK = 0


def chk(name, cond):
    global NCHECK
    NCHECK += 1
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s" % name)
        FAIL.append(name)


def rand_poly(maxdeg=3, terms=5):
    out = {}
    for _ in range(terms):
        a = random.randint(0, maxdeg)
        b = random.randint(0, maxdeg)
        c = random.randint(0, maxdeg)
        out = padd(out, pmono(a, b, c, Fr(random.randint(-4, 4))))
    return out


def rand_mod(k, maxdeg=3, terms=5):
    return from_components({i: rand_poly(maxdeg, terms) for i in range(k + 1)})


def vf_apply(f, h):
    return Kf(f, h)


def vf_bracket_on(f, g, h):
    """[K_f,K_g](h)."""
    return padd(Kf(f, Kf(g, h)), pscal(-1, Kf(g, Kf(f, h))))


print("== the vector fields and the bracket ==")
tests = [rand_poly() for _ in range(6)] + [pmono(1, 0, 0), pmono(0, 1, 1), pmono(0, 0, 3)]
ok = True
for f in tests[:5]:
    for g in tests[:5]:
        for h in tests[:3]:
            lhs = vf_bracket_on(f, g, h)
            rhs = Kf(kbracket(f, g), h)
            if padd(lhs, pscal(-1, rhs)):
                ok = False
chk("[K_f,K_g] = K_{{f,g}}  (75 triples)", ok)

# alpha = dt + (p dq - q dp)/2 ;  L_{K_f} alpha = f_t alpha.
# L_X alpha = d(iota_X alpha) + iota_X d alpha  with  iota_X alpha = A + (pC - qB)/2
# and  d alpha = dp ^ dq,  iota_X(dp^dq) = B dq - C dp.  So
#   L_X alpha = d(A + (pC-qB)/2) + B dq - C dp
# and this must equal  f_t (dt + (p dq - q dp)/2).
HF = Fr(1, 2)
ok = True
for f in tests:
    A, B, C = Kf_coeffs(f)
    g = padd(A, pscal(HF, padd(pmul(P, C), pscal(-1, pmul(Q, B)))))   # alpha(K_f)
    if padd(g, pscal(-1, f)):
        ok = False
    # dt-component
    if padd(DT(g), pscal(-1, DT(f))):
        ok = False
    # dp-component:  g_p - C = -(q/2) f_t
    if padd(DP(g), pscal(-1, C), pscal(HF, pmul(Q, DT(f)))):
        ok = False
    # dq-component:  g_q + B = (p/2) f_t
    if padd(DQ(g), B, pscal(-HF, pmul(P, DT(f)))):
        ok = False
chk("alpha(K_f)=f and L_{K_f}alpha = f_t alpha", ok)

# the frame
ok = True
for h in tests[:4]:
    if padd(e1(e2(h)), pscal(-1, e2(e1(h))), pscal(-1, reeb(h))):
        ok = False
chk("[e_1,e_2] = R = d_t", ok)

ok = True
for f in [pconst(1), pmono(0, 1, 0), pmono(0, 0, 1)]:  # k_{-2}+k_{-1}
    for X in (e1, e2, reeb):
        for h in tests[:4]:
            if padd(Kf(f, X(h)), pscal(-1, X(Kf(f, h)))):
                ok = False
chk("the frame e_1,e_2,R commutes with k_{-2}+k_{-1}", ok)

# [K_f, e_a] = sum C_a^b e_b
ok = True
for f in tests:
    (C11, C12), (C21, C22) = Cmat(f)
    for a, ea, (Ca1, Ca2) in ((1, e1, (C11, C12)), (2, e2, (C21, C22))):
        for h in tests[:3]:
            lhs = padd(Kf(f, ea(h)), pscal(-1, ea(Kf(f, h))))
            rhs = padd(pmul(Ca1, e1(h)), pmul(Ca2, e2(h)))
            if padd(lhs, pscal(-1, rhs)):
                ok = False
chk("[K_f,e_a] = C_a^b e_b  (no Reeb component)", ok)

ok = all(not padd(Cmat(f)[0][0], Cmat(f)[1][1], DT(f)) for f in tests)
chk("tr C = -f_t   (so Lambda^2 D^* = L)", ok)

print("== the modules ==")
for k in (0, 1, 2, 3):
    for lam in (Fr(0), Fr(1), Fr(-1), Fr(2, 3)):
        M = Module(k, lam)
        ok = True
        for f in tests[:4]:
            for g in tests[:4]:
                for F in [rand_mod(k) for _ in range(2)]:
                    lhs = madd(M.act(f, M.act(g, F)), mscal(-1, M.act(g, M.act(f, F))))
                    rhs = M.act(kbracket(f, g), F)
                    if madd(lhs, mscal(-1, rhs)):
                        ok = False
        chk("T(%d,%s) is a k(3)-module" % (k, lam), ok)

# the adjoint module is F_{-1}
M = Module(0, -1)
ok = True
for f in tests[:5]:
    for g in tests[:5]:
        F = from_components({0: g})
        if madd(M.act(f, F), mscal(-1, from_components({0: kbracket(f, g)}))):
            ok = False
chk("F_{-1} = the adjoint module", ok)

# Vol = F_2 : the divergence of K_f w.r.t. dt^dp^dq is 2 f_t?
ok = True
for f in tests:
    A, B, C = Kf_coeffs(f)
    div = padd(DT(A), DP(B), DQ(C))
    if padd(div, pscal(-2, DT(f))):
        ok = False
chk("div K_f = 2 f_t, i.e. Vol = F_2 = alpha ^ d alpha", ok)

print("== the grading ==")
# z = K_{2t-pq} is the grading element
z = pmono(1, 0, 0, 2)
ok = True
for i in range(-2, 5):
    for f in k_piece(i):
        if padd(kbracket(z, f), pscal(-i, f)):
            ok = False
chk("K_{2t} is the grading element, deg t=2, deg p=deg q=1", ok)

chk("dim k_{-2},k_{-1},k_0,k_1,k_2 = 1,2,4,6,9",
    [len(k_piece(i)) for i in range(-2, 3)] == [1, 2, 4, 6, 9])


def span_bracket(As, Bs):
    """The span of {A,B} as a set of polynomials, returned as a row list."""
    rows = []
    for a in As:
        for b in Bs:
            rows.append(kbracket(a, b))
    return rows


def dim_span(polys):
    keys = sorted(set(k for p in polys for k in p))
    if not keys:
        return 0
    idx = {k: i for i, k in enumerate(keys)}
    mat = []
    for p in polys:
        row = [Fr(0)] * len(keys)
        for k, v in p.items():
            row[idx[k]] = v
        mat.append(row)
    # plain fraction-free elimination
    r = 0
    ncols = len(keys)
    for c in range(ncols):
        piv = None
        for i in range(r, len(mat)):
            if mat[i][c]:
                piv = i
                break
        if piv is None:
            continue
        mat[r], mat[piv] = mat[piv], mat[r]
        pv = mat[r][c]
        for i in range(len(mat)):
            if i != r and mat[i][c]:
                f = mat[i][c] / pv
                mat[i] = [x - f * y for x, y in zip(mat[i], mat[r])]
        r += 1
    return r


ok = True
gen = {i: k_piece(i) for i in range(-2, 2)}
for d in range(2, 7):
    prod = span_bracket(gen[1], gen[d - 1])
    if dim_span(prod) != len(k_piece(d)):
        ok = False
        print("     degree %d: got %d of %d" % (d, dim_span(prod), len(k_piece(d))))
    gen[d] = k_piece(d)
chk("k_1 generates k_i for 2<=i<=6, so k_{<=1} generates k(3)", ok)

print("\n%d checks, %d failed" % (NCHECK, len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
