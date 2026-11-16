#!/usr/bin/env python3
"""The uniform formula for the GRC-brackets, all k at once (Theorem 7.2.22 of the text).

The claim proved in the text and checked here.  Write the bracket of order
k = N/2 as an operator on the tensor product,

    Bhat_N = sum_{i+j=N} lam_ij D_1^i D_2^j,   phi = Delta^* o Bhat_N,

where D_a is the invariant derivative D_theta of eq. (54) in the a-th pair of
variables.  Put  n := ceil(N/2),  eps := N mod 2,

    s_i := floor((i+eps)/2),      P_s(mu) := prod_{0<=l<s} (mu+n-1-l),

    lam_ij = (-1)^{s_j} P_{s_j}(mu1) P_{s_i}(mu2) / ( floor(j/2)! floor(i/2)! ).

Then:
  (A) phi is osp(1|2)-invariant, in the sign convention  phi o cQ = (-1)^N Q o phi
      -- the standard one for a morphism of parity N;
  (B) invariance is EQUIVALENT to the two-term recursion

          (-1)^q alpha(p+1) lam_{p+1,q} + beta(q+1) lam_{p,q+1} = 0,
          alpha(i) = r if i = 2r, -(mu1+r) if i = 2r+1,
          beta(j)  = s if j = 2s, -(mu2+s) if j = 2s+1,

      the theta-part of the Q_+-condition being an identity in the lam's;
  (C) the closed form above solves that recursion;
  (D) in the convention of the text -- Q(B(f,g)) = B(Qf,g)+(-1)^{p(f)}B(f,Qg) --
      the bracket is  beta^{2k} o Delta^* o Bhat_{2k}  with  beta := 1-2 theta D_theta,
      which for k integer is the printed (133) and for k in N-1/2 the printed
      (137);
  (E) the printed (134) of [Gi,GTh] -- the one the text calls not invariant --
      is Delta^* o Bhat_N itself, i.e. it IS the invariant operator in the other
      sign convention, the two differing by beta.

Run:  ../venv/bin/python dproof.py [NMAX]
"""
import sys

import sympy as sp

import dform as M
from check_grc_assoc import add, mul, dth, thmul, dt, dtn, Qminus, Qplus

mu1, mu2 = M.mu1, M.mu2
ZERO, ONE = sp.Integer(0), sp.Integer(1)


# --------------------------------------------------------------- the closed form
def Ppoch(s, mu, n):
    r = ONE
    for l in range(s):
        r *= (mu + n - 1 - l)
    return sp.expand(r)


def lam_closed(N):
    n, eps = (N + 1) // 2, N % 2
    out = {}
    for i in range(N + 1):
        j = N - i
        si, sj = (i + eps) // 2, (j + eps) // 2
        v = ((-1) ** sj * Ppoch(sj, mu1, n) * Ppoch(si, mu2, n)
             / (sp.factorial(j // 2) * sp.factorial(i // 2)))
        out[(i, j)] = sp.expand(sp.together(v))
    return out


def alpha(i):
    return sp.Integer(i // 2) if i % 2 == 0 else -(mu1 + (i - 1) // 2)


def beta_(j):
    return sp.Integer(j // 2) if j % 2 == 0 else -(mu2 + (j - 1) // 2)


# --------------------------------------------------------------- (C) recursion
def check_recursion(N):
    lam = lam_closed(N)
    bad = []
    for p in range(N):
        q = N - 1 - p
        e = ((-1) ** q * alpha(p + 1) * lam[(p + 1, q)]
             + beta_(q + 1) * lam[(p, q + 1)])
        if sp.simplify(sp.expand(e)) != 0:
            bad.append((p, q, sp.factor(e)))
    return bad


# --------------------------------------------------------------- (A) invariance
def check_invariance_4var(N, A=None):
    lam = lam_closed(N)
    return dcross_check_4var(N, lam, {}, (-1) ** N, A)


def dcross_check_4var(N, lam, rho, sigma, A=None):
    A = A if A is not None else N + 3
    nu = mu1 + mu2 + N
    bad = []
    for u in M.monomials(A):
        for sign, Q1 in ((-1, lambda w: M.Qm1(w)), (1, lambda w: M.Qp1(w, nu))):
            lhs = M.apply_B(M.cQ(u, sign, (mu1, mu2)), N, lam, rho)
            rhs = M.smul(sigma, Q1(M.apply_B(u, N, lam, rho)))
            res = dict(lhs)
            M.add1(res, rhs, -1)
            res = {k: sp.simplify(v) for k, v in res.items()}
            res = {k: v for k, v in res.items() if v != 0}
            if res:
                bad.append((sign, u, res))
    return bad


# ------------------------------------------- the 1-variable engine of the text
def D1v(u):
    out = dth(u)
    return add(out, thmul(dt(u)), 1)


def smul1(c, u):
    return {k: sp.expand(c * v) for k, v in u.items() if sp.expand(c * v) != 0}


def uniform_bracket(N, f, g, pf, twist=True):
    """beta^N o Delta^* o Bhat_N in the 1-variable engine; beta = 1-2 theta D."""
    lam = lam_closed(N)
    out = {}
    for (i, j), c in lam.items():
        if c == 0:
            continue
        u, v = f, g
        for _ in range(i):
            u = D1v(u)
        for _ in range(j):
            v = D1v(v)
        add(out, smul1((-1) ** (j * pf) * c, mul(u, v)))
    out = {k: sp.expand(v) for k, v in out.items() if sp.expand(v) != 0}
    if twist and N % 2:                                # beta = 1 - 2 theta D
        w = dict(out)
        add(w, thmul(D1v(out)), -2)
        out = {k: sp.expand(v) for k, v in w.items() if sp.expand(v) != 0}
    return out


def check_invariance_1var(N, twist=True, A=None):
    """Invariance in the convention of the text."""
    A = A if A is not None else N + 3
    nu = mu1 + mu2 + N
    bad = []
    for Q in (Qminus, Qplus):
        for a in range(A + 1):
            for e in (0, 1):
                for b in range(A + 1):
                    for e2 in (0, 1):
                        f, g, pf = {(a, e): ONE}, {(b, e2): ONE}, e
                        sg = (-1) ** pf
                        res = dict(Q(uniform_bracket(N, f, g, pf, twist), nu))
                        add(res, uniform_bracket(N, Q(f, mu1), g,
                                                 (pf + 1) % 2, twist), -1)
                        add(res, uniform_bracket(N, f, Q(g, mu2), pf, twist), -sg)
                        res = {k: sp.simplify(v) for k, v in res.items()}
                        res = {k: v for k, v in res.items() if v != 0}
                        if res:
                            bad.append((Q.__name__, (a, e), (b, e2), res))
    return bad


# ----------------------------------------------- the printed formulas (GCRe/GCRoAll)
def binom(x, k):
    r = ONE
    for i in range(k):
        r *= (x - i)
    return sp.expand(r / sp.factorial(k))


def gamma_(a, m):
    if a < 0 or a > m:
        return ZERO
    return (-1) ** (a + 1) * binom(mu1 + m, m - a) * binom(mu2 + m, a)


def printed_bracket(N, f, g, pf):
    """(133) for N = 2k even, (137) for N = 2m+1 odd."""
    sgn = (-1) ** pf
    out = {}
    if N % 2 == 0:
        k = N // 2
        for a in range(k):
            add(out, smul1(sgn * gamma_(a, k - 1),
                           mul(D1v(dtn(f, a)), D1v(dtn(g, k - 1 - a)))))
        for a in range(k + 1):
            gt = (-1) ** (a + 1) * binom(mu1 + k - 1, k - a) * binom(mu2 + k - 1, a)
            add(out, smul1(-gt, mul(dtn(f, a), dtn(g, k - a))))
    else:
        m = (N - 1) // 2
        C = [sp.expand(gamma_(a, m) * (mu1 + a)) for a in range(m + 1)] + [ZERO]
        E = [ZERO] + [sp.expand(-gamma_(a, m) * (mu2 + m - a)) for a in range(m + 1)]
        for a in range(m + 1):
            add(out, smul1(sgn * C[a], mul(dtn(f, a), D1v(dtn(g, m - a)))))
            add(out, smul1(E[a + 1], mul(D1v(dtn(f, a)), dtn(g, m - a))))
        th = {}
        for a in range(m + 2):
            c = -2 * (C[a] + E[a])
            if sp.expand(c) != 0:
                add(th, smul1(sp.expand(c), mul(dtn(f, a), dtn(g, m + 1 - a))))
        for a in range(m + 1):
            c = 2 * (mu1 + mu2 + m) * gamma_(a, m)
            if sp.expand(c) != 0:
                add(th, smul1(sp.expand(c), mul(D1v(dtn(f, a)), D1v(dtn(g, m - a)))))
        add(out, thmul(th))
    return {k_: sp.expand(v) for k_, v in out.items() if sp.expand(v) != 0}


def printed_GCRo(N, f, g, pf):
    """The (134) of [Gi,GTh] -- the first line of (137) alone."""
    assert N % 2 == 1
    m = (N - 1) // 2
    sgn = (-1) ** pf
    out = {}
    for a in range(m + 1):
        gk = gamma_(a, m)
        add(out, smul1(sgn * gk * (mu1 + a), mul(dtn(f, a), D1v(dtn(g, m - a)))))
        add(out, smul1(-gk * (mu2 + m - a), mul(D1v(dtn(f, a)), dtn(g, m - a))))
    return {k_: sp.expand(v) for k_, v in out.items() if sp.expand(v) != 0}


def proportional(N, B1, B2, A=None):
    """Is B1 = c B2 for one constant c?  Returns c, or None."""
    A = A if A is not None else N + 3
    c = None
    for a in range(A + 1):
        for e in (0, 1):
            for b in range(A + 1):
                for e2 in (0, 1):
                    f, g, pf = {(a, e): ONE}, {(b, e2): ONE}, e
                    u, v = B1(N, f, g, pf), B2(N, f, g, pf)
                    keys = set(u) | set(v)
                    for kk in keys:
                        x, y = sp.expand(u.get(kk, ZERO)), sp.expand(v.get(kk, ZERO))
                        if y == 0:
                            if x != 0:
                                return None
                            continue
                        r = sp.cancel(sp.together(x / y))
                        if c is None:
                            c = r
                        elif sp.simplify(r - c) != 0:
                            return None
    return c


if __name__ == "__main__":
    NMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print("(C) the closed form solves the recursion:")
    for N in range(1, 4 * NMAX + 1):
        bad = check_recursion(N)
        if bad:
            print("    N=%d  FAILS %s" % (N, bad[:2]))
            break
    else:
        print("    N = 1 ... %d  all ok" % (4 * NMAX))

    print("(A) invariance in the 4-variable model, sigma = (-1)^N:")
    for N in range(NMAX + 1):
        bad = check_invariance_4var(N)
        print("    N=%d  %s" % (N, "ok" if not bad else "FAILS %s" % bad[:1]))

    print("(D) beta^N o Delta^* o Bhat_N  vs the printed formulas:")
    for N in range(NMAX + 1):
        c = proportional(N, uniform_bracket, printed_bracket)
        print("    N=%d  %s" % (N, "printed = %s x uniform" % sp.factor(1 / c)
                                if c is not None else "NOT proportional"))

    print("(E) Delta^* o Bhat_N (no twist)  vs the printed (134) of [Gi,GTh]:")
    for N in range(1, NMAX + 1, 2):
        c = proportional(N, lambda n_, f, g, pf: uniform_bracket(n_, f, g, pf,
                                                                 twist=False),
                         printed_GCRo)
        print("    N=%d  %s" % (N, "(134) = %s x uniform" % sp.factor(1 / c)
                                if c is not None else "NOT proportional"))
