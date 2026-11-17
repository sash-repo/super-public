"""The space of contact elements of a surface, and what kGz does to the fields it carries.

The question (H. Andersen): does a segmented object in R^3, together with the
geometry derived from it (position, surface normal, curvature), carry a contact structure
canonically -- or is a contact structure an assumption imposed from outside?

The canonical contact 3-manifold in that picture is the space PT^*Sigma of tangent line
elements of the boundary SURFACE Sigma (dim 2*2-1 = 3), not of the ambient space
(dim 2*3-1 = 5).  Its projection pi: PT^*Sigma -> Sigma is a LEGENDRIAN fibration: the
fibers are 1-dimensional and tangent to the contact distribution D.

In the engine's coordinates (t,p,q) with alpha = dt + (1/2)(p dq - q dp) such a fibration
is at hand.  The vertical field is e_1 (any line field inside D is integrable), and its
invariants are

        X = q,      Y = t - p q / 2,        pi(t,p,q) = (X, Y),

in terms of which  alpha = dY + p dX  -- the Darboux shape of PT^*Sigma with p the slope
of the line element.  Then

        d pi (e_1) = 0,        d pi (e_2) = -d_X + p d_Y  =: v,

so a symmetric 2-tensor field A on Sigma pulls back to the field

        (pi^* A)(w,w) := A(d pi (w), d pi (w)),        w in D,

which is an element of T(2,0) whose only non-zero component is the one along (u^2)^2,
namely A(v,v) = a - 2 b p + c p^2 with A = a dX^2 + 2b dX dY + c dY^2 evaluated at pi.

Every field the imaging pipeline produces is of this kind: it is a tensor field on the
surface, and the surface is the base.  So the question "what does kGz do to the data" is
the question "what does kGz do to pullbacks", and this file answers it.

Everything is exact rational arithmetic.  Run:  python3 legend.py
"""
import os
import sys
from fractions import Fraction as Fr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contact import (padd, pmul, pscal, pmono, pconst, pzero, pdiff,
                     e1, e2, reeb, from_components, component, madd, mscal,
                     Module, Kf)
from gen import apply_term, rand_field
from affine import pencil, ops_at

OK = [0, 0]


def check(name, cond):
    OK[0] += 1
    OK[1] += 0 if cond else 1
    print("   %-66s %s" % (name, "ok" if cond else "FAILED"))


# --------------------------------------------------------------- the fibration
# X = q, Y = t - pq/2 as polynomials in (t,p,q)
X = pmono(0, 0, 1)
Y = padd(pmono(1, 0, 0), pscal(Fr(-1, 2), pmono(0, 1, 1)))
P = pmono(0, 1, 0)


def eq(p1, p2):
    return padd(p1, pscal(-1, p2)) == {}


print("1) pi(t,p,q) = (X,Y) = (q, t-pq/2) is a Legendrian fibration:")
check("e_1(X) = 0 and e_1(Y) = 0   (the fibers are the e_1-lines)",
      e1(X) == {} and e1(Y) == {})
check("d pi(e_2) = -d_X + p d_Y   (e_2(X) = -1, e_2(Y) = p)",
      eq(e2(X), pconst(-1)) and eq(e2(Y), P))
check("R(X) = 0, R(Y) = 1", reeb(X) == {} and eq(reeb(Y), pconst(1)))
# alpha = dY + p dX : both sides agree on the frame (e_1,e_2,R), on which
# alpha = (0,0,1) and dY + p dX = (e_a(Y) + p e_a(X), R(Y) + p R(X)).
check("alpha = dY + p dX  (equal on the frame e_1,e_2,R)",
      padd(e1(Y), pmul(P, e1(X))) == {}
      and padd(e2(Y), pmul(P, e2(X))) == {}
      and eq(padd(reeb(Y), pmul(P, reeb(X))), pconst(1)))

# ---------------------------------------------------------------- Sigma-side algebra
# A polynomial on Sigma is a dict {(i,j): coef} for X^i Y^j.


def s_mono(i, j, c=1):
    c = Fr(c)
    return {(i, j): c} if c else {}


def s_add(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            n = out.get(k, Fr(0)) + v
            if n:
                out[k] = n
            elif k in out:
                del out[k]
    return out


def s_scal(c, p):
    c = Fr(c)
    return {k: c * v for k, v in p.items()} if c else {}


def s_mul(p1, p2):
    out = {}
    for (i1, j1), v1 in p1.items():
        for (i2, j2), v2 in p2.items():
            k = (i1 + i2, j1 + j2)
            n = out.get(k, Fr(0)) + v1 * v2
            if n:
                out[k] = n
            elif k in out:
                del out[k]
    return out


def s_diff(p, which):
    out = {}
    for (i, j), v in p.items():
        if which == 0 and i:
            k = (i - 1, j)
            n = out.get(k, Fr(0)) + v * i
        elif which == 1 and j:
            k = (i, j - 1)
            n = out.get(k, Fr(0)) + v * j
        else:
            continue
        if n:
            out[k] = n
        elif k in out:
            del out[k]
    return out


def pull(p):
    """The pullback pi^* of a polynomial on Sigma: substitute X = q, Y = t - pq/2."""
    out = pzero()
    for (i, j), v in p.items():
        term = pconst(v)
        for _ in range(i):
            term = pmul(term, X)
        for _ in range(j):
            term = pmul(term, Y)
        out = padd(out, term)
    return out


def s_rand(seed, deg=2, nmon=6):
    import random
    rnd = random.Random(seed)
    keys = [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]
    if len(keys) > nmon:
        keys = rnd.sample(keys, nmon)
    return {k: Fr(rnd.randint(1, 9)) for k in keys}


def pullback_tensor(A):
    """A = (a,b,c) on Sigma  |-->  pi^*A in T(2,0):  component 0 = a - 2bp + cp^2."""
    a, b, c = A
    f0 = padd(pull(a),
              pscal(-2, pmul(P, pull(b))),
              pmul(pmul(P, P), pull(c)))
    return from_components({0: f0})


# ------------------------------------------------- the pullback class is natural
# A vector field w = w^X d_X + w^Y d_Y on Sigma lifts to the contact field K_f with
# f = pi^*(w^Y) + p pi^*(w^X)  (because alpha = dY + p dX and f = alpha(K_f)).
print()
print("2) the lift of a vector field on Sigma, and naturality of the pullback:")


def lift(w):
    wX, wY = w
    return padd(pull(wY), pmul(P, pull(wX)))


def lie_sigma(w, A):
    """(L_w A)_{ij} = w^k d_k A_ij + A_kj d_i w^k + A_ik d_j w^k, with A = (a,2b,c)."""
    wX, wY = w
    a, b, c = A
    G = [[a, b], [b, c]]
    W = [wX, wY]
    out = [[{}, {}], [{}, {}]]
    for i in range(2):
        for j in range(2):
            acc = s_add(*[s_mul(W[k], s_diff(G[i][j], k)) for k in range(2)])
            acc = s_add(acc, *[s_mul(G[k][j], s_diff(W[k], i)) for k in range(2)])
            acc = s_add(acc, *[s_mul(G[i][k], s_diff(W[k], j)) for k in range(2)])
            out[i][j] = acc
    return (out[0][0], out[0][1], out[1][1])


M20 = Module(2, 0)
bad = 0
tested = 0
for sw in range(4):
    w = (s_rand(300 + sw, deg=2), s_rand(400 + sw, deg=2))
    f = lift(w)
    for sa in range(3):
        A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
        lhs = M20.act(f, pullback_tensor(A))
        rhs = pullback_tensor(lie_sigma(w, A))
        tested += 1
        if madd(lhs, mscal(-1, rhs)):
            bad += 1
check("L_{lift w}(pi^*A) = pi^*(L_w A) on %d pairs (w,A), weight lambda = 0" % tested,
      bad == 0)
# the fibration is preserved by the lift, i.e. K_f(X), K_f(Y) are pullbacks
badf = 0
for sw in range(4):
    w = (s_rand(300 + sw, deg=2), s_rand(400 + sw, deg=2))
    f = lift(w)
    for coord, wc in ((X, w[0]), (Y, w[1])):
        if padd(Kf(f, coord), pscal(-1, pull(wc))):
            badf += 1
check("K_{lift w} projects: K_f(X) = pi^*w^X and K_f(Y) = pi^*w^Y", badf == 0)

# ------------------------------------------------------------------ the operators
print()
print("3) the invariant pairings T(2,lam) (x) T(2,mu) -> F_nu of the census:")

terms2, V02, A0, A1, A2 = pencil(2, 2, 0, 2, nfield=1)
kgz = ops_at(terms2, V02, A0, A1, A2, Fr(0), Fr(0))
check("W=2, lambda = mu = 0: the space of operators is 1-dimensional (kGz)",
      len(kgz) == 1)
KGZ = kgz[0]

terms0, V00, B0, B1, B2 = pencil(2, 2, 0, 0, nfield=1)
c0 = ops_at(terms0, V00, B0, B1, B2, Fr(0), Fr(0))
print("   W=0, lambda = mu = 0: %d order-zero contractions" % len(c0))


def ev(vec, terms, F, G):
    acc = {}
    for cf, t in zip(vec, terms):
        if cf:
            acc = madd(acc, mscal(cf, apply_term(t, F, G)))
    return acc


print()
print("4) kGz on the fields the pipeline produces (both arguments pullbacks):")
pairs = 0
nonzero = 0
for sa in range(4):
    A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
    for sb in range(4):
        B = (s_rand(130 + sb), s_rand(170 + sb), s_rand(210 + sb))
        F, G = pullback_tensor(A), pullback_tensor(B)
        pairs += 1
        if ev(KGZ, terms2, F, G):
            nonzero += 1
check("kGz(pi^*A, pi^*B) = 0 on all %d pairs" % pairs, nonzero == 0)

z0 = 0
for sa in range(4):
    A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
    for sb in range(4):
        B = (s_rand(130 + sb), s_rand(170 + sb), s_rand(210 + sb))
        F, G = pullback_tensor(A), pullback_tensor(B)
        for op in c0:
            if ev(op, terms0, F, G):
                z0 += 1
check("every order-zero contraction vanishes on pullbacks as well", z0 == 0)

# the vanishing is not a triviality of kGz: with one generic argument it is non-zero
print()
print("5) control: the vanishing is a property of the PAIR, not of kGz:")
nz = 0
trials = 0
for sa in range(3):
    A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
    F = pullback_tensor(A)
    for sg in range(3):
        G = rand_field(2, 3, seed=700 + sg)
        trials += 1
        if ev(KGZ, terms2, F, G):
            nz += 1
check("kGz(pi^*A, generic) is non-zero in %d of %d trials" % (nz, trials), nz == trials)

# and the reason: both arguments are rank one with the SAME kernel line, the vertical
print()
print("6) the reason is the common kernel, not the fibration:")
nzz = 0
tr = 0
for i in range(6):
    F = from_components({0: {k[:3]: v for k, v in rand_field(0, 3, seed=800 + i).items()}})
    for j in range(6):
        G = from_components({0: {k[:3]: v
                                 for k, v in rand_field(0, 3, seed=900 + j).items()}})
        tr += 1
        if ev(KGZ, terms2, F, G):
            nzz += 1
check("kGz(F,G) = 0 whenever both have only the (u^2)^2-component (%d pairs)" % tr,
      nzz == 0)
# whereas two rank-one forms with DIFFERENT kernels do pair non-trivially
nzd = 0
trd = 0
for i in range(4):
    F = from_components({0: {k[:3]: v for k, v in rand_field(0, 3, seed=800 + i).items()}})
    for j in range(4):
        G = from_components({2: {k[:3]: v for k, v in rand_field(0, 3, seed=900 + j).items()}})
        trd += 1
        if ev(KGZ, terms2, F, G):
            nzd += 1
check("two rank-one forms with different kernels: non-zero in %d of %d" % (nzd, trd),
      nzd == trd)

# ------------------------------------------------------------------ what a lift costs
print()
print("7) what it takes to leave the degenerate locus:")
# add to pi^*A an arbitrary vertical part: the (u^1)^2- and u^1u^2-components.
# One extra function is enough to make kGz non-zero; that function is not supplied
# by the surface data.
nzv = 0
trv = 0
for sa in range(3):
    A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
    F = pullback_tensor(A)
    for sb in range(3):
        B = (s_rand(130 + sb), s_rand(170 + sb), s_rand(210 + sb))
        G = pullback_tensor(B)
        # the "vertical completion": give G the vertical component 1 (the fiber form)
        G2 = madd(G, from_components({2: pconst(1)}))
        trv += 1
        if ev(KGZ, terms2, F, G2):
            nzv += 1
check("kGz(pi^*A, pi^*B + (u^1)^2) non-zero in %d of %d" % (nzv, trv), nzv == trv)

# ------------------------------------------------ the other configurations of the census
print()
print("8) the same for every other pairing the census leaves alive:")
for k3, expect_zero in ((2, True), (4, False)):
    tt, VV, C0, C1, C2 = pencil(2, 2, k3, 0, nfield=1)
    ops = ops_at(tt, VV, C0, C1, C2, Fr(0), Fr(0))
    nzc = 0
    trc = 0
    for sa in range(3):
        A = (s_rand(10 + sa), s_rand(50 + sa), s_rand(90 + sa))
        for sb in range(3):
            B = (s_rand(130 + sb), s_rand(170 + sb), s_rand(210 + sb))
            F, G = pullback_tensor(A), pullback_tensor(B)
            for op in ops:
                trc += 1
                if ev(op, tt, F, G):
                    nzc += 1
    if expect_zero:
        check("W=0, k_3=%d: all %d contractions vanish on pullbacks (%d values)"
              % (k3, len(ops), trc), nzc == 0)
    else:
        # k_3 = 4 is the symmetric product: it contracts nothing, so it survives --
        # and creates nothing, being the tensor product of the two arguments.
        check("W=0, k_3=4: the symmetric product survives (%d of %d values non-zero)"
              % (nzc, trc), nzc == trc)

# and on the densities: two FUNCTIONS on the surface, at lambda = mu = 0
td, Vd, D0, D1, D2 = pencil(0, 0, 0, 2, nfield=1)
opd = ops_at(td, Vd, D0, D1, D2, Fr(0), Fr(0))
nzd2 = 0
trd2 = 0
for sa in range(3):
    F = from_components({0: pull(s_rand(10 + sa))})
    for sb in range(3):
        G = from_components({0: pull(s_rand(130 + sb))})
        for op in opd:
            trd2 += 1
            if ev(op, td, F, G):
                nzd2 += 1
check("the W=2 family on densities vanishes on pullbacks of functions (%d values)"
      % trd2, nzd2 == 0)

print()
print("checks: %d run, %d failed" % (OK[0], OK[1]))
