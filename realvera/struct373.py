"""Which structure does each real form of a stringy Lie superalgebra preserve?

This is the engine for the question which structure each real form preserves.

For every row of Table (219) (real forms) it does three things, in exact
arithmetic over Q(i, sqrt 2) [Laurent polynomials in z = e^{ia/2}, see lau.py]:

 (1) AUDIT.  It checks the two conditions a row must satisfy:
       (C1)  phi(A) A = 1_n              (phi is an involution: a real form)
       (C2)  A G(phi~(x)) A^t = phi~(x)' G(x)          (contact case), or
             det A = const * phi~(x)'                  (svect case), or
             det A invertible                          (vect case),
     the latter being Table (231).

 (2) REAL FRAME.  It exhibits an explicit frame theta = C xi of the real odd
     module: C is required to satisfy phi(C) A C^{-1} = 1_n, which says that
     in the indeterminates theta the real structure is the standard one
     (conjugate the coefficients).

 (3) THE INVARIANT FORM.  The contact form alpha = dx + xi^t G d xi becomes,
     in the indeterminates theta,   alpha = dx + theta^t Gt d theta   with
     Gt = (C^{-1})^t G C^{-1}, and h*alpha is phi-real for h = -i/x (the
     normalization that turns dx into i x da, i.e. gives the form  da + ...).
     The engine prints Gh := h * Gt and checks that every entry of it is
     phi-real, i.e. that  da + theta^t Gh d theta  is a real contact form.
     Comparing Gh with the three forms of the Exercise
         alpha_{n,k},  alpha^l_{n,k},  alpha_a
     is then a matter of reading the diagonal.

Run:  python3 struct373.py
"""

import sys
from sympy import I, S, sqrt, exp, pi, Rational, expand, simplify, conjugate

from lau import (L, ZERO, ONE, const, zpow, add, sub, neg, mul, smul, is_zero, eq,
                 inv, star, subst, ddx, pretty, periodic, x, xinv, z, cosn, sinn,
                 tau, tauM, J2, Jn, mzero, mid, mdiag, mblock, mmul, mt, mstar,
                 msubst, msmul, mmulf, madd, msub, meq, mdet, minv, mpretty)

# --------------------------------------------------------------- the three sigmas
# phi(f) = star(f) evaluated at z -> lam z^s ;  phi~(x) = phi(x).

SIGMA = {
    "id":  (S(1), +1),   # x -> x
    "inv": (S(1), -1),   # x -> x^{-1}
    "oin": (I, -1),      # x -> -x^{-1}
}


def phi(f, sname):
    lam, s = SIGMA[sname]
    return subst(star(f), lam, s)


def mphi(M, sname):
    return [[phi(c, sname) for c in row] for row in M]


def phitilde(sname):
    return phi(x, sname)


def dphitilde(sname):
    return ddx(phitilde(sname))


def is_real(f, sname):
    return eq(phi(f, sname), f)


# ------------------------------------------------------------------ the algebras

def G_of(kind, n, arg):
    """The Gram matrix of the contact form in the indeterminate `arg` (= x or phi~(x)).

    k^L(n):  1_n ;   k^M(n): diag(1_{n-1}, arg)."""
    if kind == "kL":
        return mid(n)
    if kind == "kM":
        return mdiag(*([ONE] * (n - 1) + [arg]))
    raise ValueError(kind)


# --------------------------------------------------------------------- the audit

def audit(name, kind, n, sname, A, lam=0, verbose=True):
    """Check (C1) and (C2).  Returns (ok1, ok2)."""
    P = phitilde(sname)
    dP = dphitilde(sname)

    # (C1) phi(A) A = 1
    ok1 = meq(mmul(mphi(A, sname), A), mid(n))

    # (C2)
    if kind in ("kL", "kM"):
        lhs = mmul(A, G_of(kind, n, P), mt(A))
        rhs = mmulf(dP, G_of(kind, n, x))
        ok2 = meq(lhs, rhs)
        cond = "A G(phi~) A^t = phi~' G(x)"
    elif kind == "svect":
        d = mdet(A)
        # det A = const * phi~' * (phi~/x)^lam ; here lam = 0 unless stated
        want = dP
        if lam:
            want = mul(want, powL(mul(P, inv(x)), lam))
        q = mul(d, inv(want))
        ok2 = (len(q) == 1 and 0 in q)
        cond = "det A / (phi~' (phi~/x)^lam) = const"
    else:  # vect
        d = mdet(A)
        ok2 = (len(d) == 1)
        cond = "det A invertible"
    if verbose:
        print("  %-26s C1 %s   C2 %s   [%s]" % (name, "ok" if ok1 else "FAIL",
                                                "ok" if ok2 else "FAIL", cond))
        if kind == "svect":
            print("       det A = %s ,  phi~' = %s" % (pretty(mdet(A)), pretty(dP)))
    return ok1, ok2


def powL(f, k):
    r = ONE
    for _ in range(abs(k)):
        r = mul(r, f)
    return r if k >= 0 else inv(r)


# ------------------------------------------------------- frames and the real form

def hnorm(sname):
    """The factor h making h*alpha phi-real: phi(h) phi~' = h.

    For sigma = id one has phi~' = 1 and h = 1: the form stays written with dx,
    and x = t is itself a real coordinate.  For sigma = inv and oin one has
    phi~' = -+ x^{-2} and h = -i/x, which turns dx into i x da, i.e. writes the
    form as da + ... : the real coordinate is the angle a."""
    if sname == "id":
        return ONE
    return smul(-I, xinv)


def realform(name, kind, n, sname, A, C, verbose=True):
    """Given the frame theta = C xi, print Gh and check its reality."""
    okC = meq(mmul(mphi(C, sname), A, minv(C)), mid(n))
    D = minv(C)
    Gt = mmul(mt(D), G_of(kind, n, x), D)
    h = hnorm(sname)
    Gh = mmulf(h, Gt)
    okreal = all(is_real(Gh[i][j], sname) for i in range(n) for j in range(n))
    per = all(periodic(c) for row in C for c in row)
    if verbose:
        print("  %-26s frame %s   Gh real %s   C periodic %s"
              % (name, "ok" if okC else "FAIL", "ok" if okreal else "FAIL",
                 "yes" if per else "no"))
        print("       Gh = %s" % mpretty(Gh).replace("\n", "\n            "))
    return okC, okreal, Gh


# ------------------------------------------------------------- frame constructors

E4 = exp(I * pi / 4)          # e^{i pi/4}


def frame_id(entries):
    """sigma = id.  entries: list of '+', '-', 'tau' (the last eats 2 slots)."""
    blocks = []
    for e in entries:
        if e == "+":
            blocks.append([[ONE]])
        elif e == "-":
            blocks.append([[const(I)]])
        elif e == "tau":
            # theta_1 = (1+x) xi_1 + i(1-x) xi_2 ,  theta_2 = i(1-x) xi_1 - (1+x) xi_2
            p = add(ONE, x)
            q = smul(I, sub(ONE, x))
            blocks.append([[p, q], [q, neg(p)]])
        else:
            raise ValueError(e)
    return mblock(*blocks)


def frame_inv(entries):
    """sigma = inv.  entries: '+' (A_jj = i x^{-1}), '-' (A_jj = -i x^{-1}),
    'M' (A_jj = i, the k^M-twisted slot), 'tauM' (a tau^M block, 2 slots)."""
    cc = smul(E4, zpow(-1))          # c = e^{i pi/4} z^{-1},  c^2 = i x^{-1}
    blocks = []
    for e in entries:
        if e == "+":
            blocks.append([[cc]])
        elif e == "-":
            blocks.append([[smul(I, cc)]])
        elif e == "M":
            blocks.append([[const(E4)]])
        elif e == "tauM":
            # in the rotating frame v_+ = (cos(a/2), sin(a/2)), v_- = (-sin, cos)
            # tau^M acts as diag(1,-1); so take theta_1 = c (v_+ . xi),
            # theta_2 = i c (v_- . xi).
            c2, s2 = cosn(1), sinn(1)
            blocks.append([[mul(cc, c2), mul(cc, s2)],
                           [mul(smul(I, cc), neg(s2)), mul(smul(I, cc), c2)]])
        else:
            raise ValueError(e)
    return mblock(*blocks)


def frame_oin(entries):
    """sigma = oin.  entries: 'J' (an x^{-1} J_2 block, 2 slots), 'i' (A_jj = i)."""
    blocks = []
    for e in entries:
        if e == "J":
            # real elements: u xi_1 + x^{-1} phi(u) xi_2 ; take u = 1 and u = i
            blocks.append([[ONE, xinv], [const(I), smul(-I, xinv)]])
        elif e == "i":
            blocks.append([[const(E4)]])
        else:
            raise ValueError(e)
    return mblock(*blocks)


# ------------------------------------------------------------------ the row list

def A_id(entries):
    """A = diag(1_k, tau_l, -1_m) written as a list of '+','-','tau'."""
    blocks = []
    for e in entries:
        if e == "+":
            blocks.append([[ONE]])
        elif e == "-":
            blocks.append([[smul(-1, ONE)]])
        elif e == "tau":
            blocks.append(tau())
        else:
            raise ValueError(e)
    return mblock(*blocks)


def A_inv(entries):
    """A = i x^{-1} diag(...) written as '+','-','M','tauM'."""
    blocks = []
    ix = smul(I, xinv)
    for e in entries:
        if e == "+":
            blocks.append([[ix]])
        elif e == "-":
            blocks.append([[neg(ix)]])
        elif e == "M":
            blocks.append([[const(I)]])          # i x^{-1} * x
        elif e == "tauM":
            blocks.append(mmulf(ix, tauM()))
        else:
            raise ValueError(e)
    return mblock(*blocks)


def A_oin(entries):
    blocks = []
    for e in entries:
        if e == "J":
            blocks.append(mmulf(xinv, J2))
        elif e == "i":
            blocks.append([[const(I)]])
        else:
            raise ValueError(e)
    return mblock(*blocks)


def size(entries, two=("tau", "tauM", "J")):
    return sum(2 if e in two else 1 for e in entries)


ROWS = [
    # (name, kind, sigma, entries, frame-entries)
    # ---- sigma = id : phi~(x) = x -------------------------------------------
    ("kL(4;4,0)   k^L, A=1_4",      "kL", "id",  ["+"] * 4,               None),
    ("kL(4;2,0)   k^L, A=(1,1,-1,-1)", "kL", "id", ["+", "+", "-", "-"],  None),
    ("kL(4;0,4)   k^L, A=tau_2",    "kL", "id",  ["tau", "tau"],          None),
    ("kL(4;1,2)   k^L, A=(1,tau,-1)", "kL", "id", ["+", "tau", "-"],      None),
    ("kM(4;.,1)   k^M, A=(1,1,1,-1)", "kM", "id", ["+", "+", "+", "-"],   None),
    ("kM(4;.,3)   k^M, A=(1,tau,-1)", "kM", "id", ["+", "tau", "-"],      None),
    ("kM(3;.,1)   k^M, A=(1,1,-1)", "kM", "id",  ["+", "+", "-"],         None),
    # ---- sigma = inv : phi~(x) = x^{-1} -------------------------------------
    ("kL(4;2)     k^L, A=ix^{-1}(1,1,-1,-1)", "kL", "inv", ["+", "+", "-", "-"], None),
    ("kL(4;0)     k^L, A=ix^{-1}(-1_4)", "kL", "inv", ["-"] * 4,          None),
    ("kL(4;0)'    k^L, A=ix^{-1}(tau^M,1_2)", "kL", "inv", ["tauM", "+", "+"], None),
    ("kM(4;1)     k^M, A=ix^{-1}(1,-1,-1,x)", "kM", "inv", ["+", "-", "-", "M"], None),
    ("kM(3;1)     k^M, A=ix^{-1}(1,-1,x)", "kM", "inv", ["+", "-", "M"],  None),
    ("kL(3;1)     k^L, A=ix^{-1}(1,-1,-1)", "kL", "inv", ["+", "-", "-"], None),
    # ---- sigma = oin : phi~(x) = -x^{-1} ------------------------------------
    ("ukL(4)      k^L, A=x^{-1}J_4", "kL", "oin", ["J", "J"],             None),
    ("ukL(2)      k^L, A=x^{-1}J_2", "kL", "oin", ["J"],                  None),
    ("ukM(5)      k^M, A=(x^{-1}J_4,i)", "kM", "oin", ["J", "J", "i"],    None),
    ("ukM(3)      k^M, A=(x^{-1}J_2,i)", "kM", "oin", ["J", "i"],         None),
]

MAKE_A = {"id": A_id, "inv": A_inv, "oin": A_oin}
MAKE_C = {"id": frame_id, "inv": frame_inv, "oin": frame_oin}


def solve_frame(A, sname, N=6, verbose=False):
    """Solve  u = A^t phi(u)  for the real odd elements, and return a frame C.

    An odd element of R(n) is u_1 xi_1 + ... + u_n xi_n with u_j in R(0), i.e.
    with u_j an EVEN Laurent polynomial in z; it is phi-real iff u = A^t phi(u).
    This is an R-linear condition, and we solve it exactly: the unknowns are the
    real and the imaginary part of every coefficient in the window |e| <= N.

    Out of the solution space we pick n elements whose matrix has a monomial
    determinant, i.e. is invertible on the whole circle; that matrix is the
    frame C of `realform`."""
    from sympy import symbols, linsolve, Matrix, im, re as sre
    n = len(A)
    exps = [e for e in range(-N, N + 1) if e % 2 == 0]
    P = symbols("p0:%d" % (n * len(exps)), real=True)
    Q = symbols("q0:%d" % (n * len(exps)), real=True)
    u = []
    for j in range(n):
        f = {}
        for t, e in enumerate(exps):
            f[e] = P[j * len(exps) + t] + I * Q[j * len(exps) + t]
        u.append(f)
    # rhs = A^t phi(u)
    At = mt(A)
    rhs = []
    for j in range(n):
        s = ZERO
        for k in range(n):
            s = add(s, mul(At[j][k], phi(u[k], sname)))
        rhs.append(s)
    eqs = []
    for j in range(n):
        d = sub(u[j], rhs[j])
        for e, c in d.items():
            c = expand(c)
            eqs.append(sre(c))
            eqs.append(im(c))
    unknowns = list(P) + list(Q)
    sol = linsolve(eqs, unknowns)
    basis = []
    for point in sol:
        # the solution space is parametrized by the free unknowns
        free = sorted({s for expr in point for s in expr.free_symbols},
                      key=lambda s: s.name)
        for fs in free:
            subsmap = {g: (1 if g == fs else 0) for g in free}
            vec = [expr.subs(subsmap) for expr in point]
            val = dict(zip(unknowns, vec))
            row = []
            for j in range(n):
                f = {}
                for t, e in enumerate(exps):
                    c = expand(val[P[j * len(exps) + t]] + I * val[Q[j * len(exps) + t]])
                    if c != 0:
                        f[e] = c
                row.append(f)
            basis.append(row)
    if verbose:
        print("    solution space: %d real generators" % len(basis))
    # look for n of them with a monomial determinant
    from itertools import combinations
    for comb in combinations(range(len(basis)), n):
        C = [basis[i] for i in comb]
        d = mdet(C)
        if len(d) == 1:
            return C
    # otherwise try small integer combinations of the first few
    return None


def svect_check():
    """The rows svect^L_inv(n) and svect^L_oin(n) of Table 12.

    Table 13 asks det A = const * x^{-2} for BOTH of them, whereas Table 12
    prints A = diag(1_{n-1}, x^2), of determinant x^{+2}.  Here we check
      (a) that the printed entry indeed fails,
      (b) that diag(1_{n-1}, x^{-2}) passes, and so does x^{-1} 1_2, which is
          what the table of explicit formulas of the text for svect^L(2) gives,
      (c) that the two of them are equivalent (so nothing is lost or gained):
          B A_1 phi(B)^{-1} = A_2 with det B = 1, B periodic.
    """
    print("=" * 78)
    print("The svect^L rows of Table 12")
    print("=" * 78)
    bad = 0
    n = 2
    variants = [
        ("A = diag(1, x^2)   [as printed]", mdiag(ONE, zpow(4)), False),
        ("A = diag(1, x^{-2})", mdiag(ONE, zpow(-4)), True),
        ("A = x^{-1} 1_2     [the text's explicit table]", mdiag(xinv, xinv), "inv"),
    ]
    for sname in ("inv", "oin"):
        for label, A, want in variants:
            ok1, ok2 = audit("%s  %s" % (sname, label), "svect", n, sname, A,
                             verbose=False)
            w = want if isinstance(want, bool) else (sname == want)
            good = (ok1 and ok2) == w
            bad += (not good)
            print("  %-4s %-42s C1 %s  C2 %s   (expected %s) %s"
                  % (sname, label, "ok" if ok1 else "no", "ok" if ok2 else "no",
                     "pass" if w else "fail", "ok" if good else "FAIL"))

    # (c) the two admissible shapes are equivalent
    c2, s2 = cosn(1), sinn(1)
    zz, zi = zpow(1), zpow(-1)
    B = [[mul(zz, c2), neg(mul(zz, s2))],
         [mul(zi, s2), mul(zi, c2)]]
    A1 = mdiag(xinv, xinv)
    A2 = mdiag(ONE, zpow(-4))
    got = mmul(B, A1, minv(mphi(B, "inv")))
    r = meq(got, A2) and eq(mdet(B), ONE) and all(periodic(c) for row in B for c in row)
    bad += (not r)
    print("  equivalence  B (x^{-1}1_2) phi(B)^{-1} = diag(1,x^{-2}),  det B = %s : %s"
          % (pretty(mdet(B)), "ok" if r else "FAIL"))
    print()
    return bad


def R2(n=1):
    """The rotation R(n a / 2)."""
    c, s = cosn(n), sinn(n)
    return [[c, neg(s)], [s, c]]


def refine():
    """Periodic frames: which real forms have a CONSTANT invariant form.

    The frames of `frame_inv` are anti-periodic (they are multiplied by
    c = e^{i pi/4} z^{-1}), so they do not consist of odd indeterminates of the
    real supercircle F_inv(n) itself.  A periodic frame is got from them by a
    rotation R(a/2) inside a definite block; whether this can be done for the
    WHOLE frame is what distinguishes the shapes of the answer:

      both eigenbundles orientable    -> constant Gh   -> alpha_{n,k}
      both non-orientable             -> rotating Gh   -> alpha_a
    """
    print("=" * 78)
    print("Periodic frames: constant answer (alpha_{n,k}) or rotating one (alpha_a)")
    print("=" * 78)
    bad = 0
    cases = [
        # name, kind, entries, the rotation blocks to apply, expected shape
        ("k^L(2), A=ix^{-1}(1,-1)  [signature (1,1): p,q both odd]",
         "kL", ["+", "-"], [(0, "R")], "rotating"),
        ("k^L(4), A=ix^{-1}(1,1,-1,-1)  [(2,2): p,q both even]",
         "kL", ["+", "+", "-", "-"], [(0, "R"), (2, "R")], "constant"),
        ("k^L(4), A=ix^{-1}(tau^M,1,1)  [(1,3), the tau^M row]",
         "kL", ["tauM", "+", "+"], [(2, "R")], "constant"),
    ]
    for name, kind, ent, rots, want in cases:
        n = size(ent)
        A = MAKE_A["inv"](ent)
        C = MAKE_C["inv"](ent)
        # apply the rotations R(a/2) to the indicated pairs of frame vectors
        Rot = mid(n)
        for (o, _) in rots:
            r = R2()
            for i in range(2):
                for j in range(2):
                    Rot[o + i][o + j] = r[i][j]
        C = mmul(Rot, C)
        okC = meq(mmul(mphi(C, "inv"), A, minv(C)), mid(n))
        per = all(periodic(c) for row in C for c in row)
        D = minv(C)
        Gh = mmulf(hnorm("inv"), mmul(mt(D), G_of(kind, n, x), D))
        constant = all(all(e == 0 for e in Gh[i][j]) or set(Gh[i][j]) == {0}
                       for i in range(n) for j in range(n))
        got = "constant" if constant else "rotating"
        ok = okC and per and got == want
        bad += (not ok)
        print("  %-52s frame ok %s  periodic %s  -> %s  %s"
              % (name, "y" if okC else "N", "y" if per else "N", got,
                 "ok" if ok else "FAIL"))
        print("       Gh = %s" % mpretty(Gh).replace("\n", "\n            "))
    print()
    return bad



def census():
    """For sigma = inv: the two classes of every signature (the two components
    of SO(k,n-k)), which is what Table 12 lists only for one value of k.

    For each n and each k with 1 <= k <= n-k the plain row
        A = i x^{-1} diag(1_k, -1_{n-k})
    and the twisted one
        A = i x^{-1} diag(tau^M(a), 1_{k-1}, -1_{n-k-1})
    satisfy the same conditions and have one and the same signature, and differ
    exactly in the pair (eps_+, eps_-) of the orientabilities of the two
    eigen-subbundles of the invariant form: they are the two connected
    components of SO(k, n-k).  For k = 0 the group SO(n) is connected and there
    is no twisted partner."""
    print("=" * 78)
    print("sigma = inv: the two classes of each signature")
    print("=" * 78)
    bad = 0
    for n in (4, 5, 6):
        for k in range(0, n - k_bound(n) + 1) if False else range(0, n // 2 + 1):
            rows = [["+"] * k + ["-"] * (n - k)]
            if k >= 1:
                rows.append(["tauM"] + ["+"] * (k - 1) + ["-"] * (n - k - 1))
            for ent in rows:
                A = MAKE_A["inv"](ent)
                C = MAKE_C["inv"](ent)
                ok1, ok2 = audit(str(ent), "kL", n, "inv", A, verbose=False)
                okC, okr, Gh = realform(str(ent), "kL", n, "inv", A, C, verbose=False)
                per = [all(periodic(c) for c in row) for row in C]
                diag = [pretty(Gh[i][i]) for i in range(n)]
                p = sum(1 for d in diag if d.startswith("(1)"))
                ep = em = 1
                for i in range(n):
                    s_i = 1 if per[i] else -1
                    if diag[i].startswith("(1)"):
                        ep *= s_i
                    else:
                        em *= s_i
                ok = ok1 and ok2 and okC and okr
                bad += (not ok)
                print("  n=%d k=%d %-30s signature (%d,%d)  (eps_+,eps_-) = (%+d,%+d)  %s"
                      % (n, k, "".join(e[0] for e in ent), p, n - p, ep, em,
                         "ok" if ok else "FAIL"))
    print()
    return bad


def k_bound(n):
    return n // 2


def lemmas():
    """The three congruences the reading of the answer rests on."""
    print("=" * 78)
    print("The congruences used to put the answers in normal form")
    print("=" * 78)
    ok = True

    # (L1)  over F_id :  diag(x, -x)  ~  diag(1, -1)
    u = smul(Rational(1, 2), add(x, ONE))
    v = smul(Rational(1, 2), sub(x, ONE))
    B = [[u, v], [v, u]]
    lhs = mmul(mt(B), mdiag(ONE, smul(-1, ONE)), B)
    r1 = meq(lhs, mdiag(x, neg(x))) and len(mdet(B)) == 1 \
        and all(is_real(c, "id") for row in B for c in row)
    print("  (L1) diag(1,-1) ~ diag(x,-x) over F_id,  det B = %s : %s"
          % (pretty(mdet(B)), "ok" if r1 else "FAIL"))
    ok &= r1

    # (L2)  over F_inv :  R(a/2) diag(1,-1) R(a/2)^t = tau^M(a)
    c2, s2 = cosn(1), sinn(1)
    R = [[c2, neg(s2)], [s2, c2]]
    lhs = mmul(R, mdiag(ONE, smul(-1, ONE)), mt(R))
    r2 = meq(lhs, tauM())
    print("  (L2) R(a/2) diag(1,-1) R(a/2)^t = tau^M(a) : %s" % ("ok" if r2 else "FAIL"))
    ok &= r2

    # (L3)  a tau-block of the sigma = id family gives x^{-1}/4 * 1_2 ~ x * 1_2
    D = mdiag(x, x)                      # theta -> x^{-1} theta
    lhs = mmul(mt(D), mdiag(smul(Rational(1, 4), xinv), smul(Rational(1, 4), xinv)), D)
    r3 = meq(lhs, mdiag(smul(Rational(1, 4), x), smul(Rational(1, 4), x)))
    print("  (L3) (x^{-1}/4) 1_2 ~ (x/4) 1_2 over F_id : %s" % ("ok" if r3 else "FAIL"))
    ok &= r3
    print()
    return 0 if ok else 1


def main():
    print("=" * 78)
    print("Audit of Table 12 (real forms) against Table 13 (conditions on A)")
    print("=" * 78)
    bad = 0
    for name, kind, sname, ent, _ in ROWS:
        n = size(ent)
        A = MAKE_A[sname](ent)
        ok1, ok2 = audit(name, kind, n, sname, A)
        bad += (not ok1) + (not ok2)
    print()
    print("=" * 78)
    print("The real frame and the invariant form  Gh  (alpha = da + theta^t Gh dtheta)")
    print("=" * 78)
    for name, kind, sname, ent, _ in ROWS:
        n = size(ent)
        A = MAKE_A[sname](ent)
        C = MAKE_C[sname](ent)
        okC, okr, Gh = realform(name, kind, n, sname, A, C)
        bad += (not okC) + (not okr)
    print()
    bad += svect_check()
    bad += refine()
    bad += census()
    bad += lemmas()
    print("failures: %d" % bad)
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
