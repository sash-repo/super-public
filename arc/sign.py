#!/usr/bin/env python3
"""Which sign convention: (a) Phi.cQ = Q.Phi  or  (b) Phi.cQ = (-1)^N Q.Phi ?

For an ODD invariant operator the Sign Rule offers (b), the other natural
convention is (a), and the two answers
differ by the parity automorphism kappa: theta |-> -theta of the target.

This engine settles it by the only test that is not itself a convention: what
happens when an odd operator is used the way brackets are used -- plugged into
ONE slot of another operator.  Nothing here is put in by hand: everything is
computed in the polynomial model

    C[t_1,th_1, ..., t_n,th_n]   =   SM_{mu_1} (x) ... (x) SM_{mu_n},

where the Koszul signs are produced by d_{th_i} passing th_1...th_{i-1}, and
the osp(1|2)-action is cQ_eps = sum_i Q_eps^{(i)}.

    D_i     = d_{th_i} + th_i d_{t_i}
    Q_-^(i) = d_{th_i} - th_i d_{t_i},   Q_+^(i) = t_i Q_-^(i) - mu_i th_i
    iota*   = the merge t_i,t_{i+1} |-> t, th_i,th_{i+1} |-> th  (the order-0 bracket)

    Phi_N   = iota* o sum_{i+j=N} lam_ij D_1^i D_2^j          (Theorem 7.2.22)

Tests, all with SYMBOLIC weights, so a pass is a proof for all weights at once:

  T1  Phi_N satisfies (b) and, for N odd, not (a); kappa.Phi_N the other way
      round.
  T2  Phi_N (x) id, the operator applied in one slot of a three-fold tensor
      product: in the sense (b) it is again invariant, in the sense (a) it is
      invariant in NEITHER sense -- and the defect is printed.
  T3  the composite Phi_N o (Phi_{N'} (x) id): under (b) it is invariant of
      parity N+N', under (a) it is not invariant.
  T4  the naive commutator X*Phi := X.Phi - Phi.X is not an action of osp(1|2):
      the bracket identity fails on the odd generators by 2(X.Phi.Y + Y.Phi.X).

Run:  ../venv/bin/python sign.py [NMAX]
"""
import sys
from math import factorial

import sympy as sp

ZERO, ONE = sp.Integer(0), sp.Integer(1)


# --------------------------------------------------------------- the algebra
# A vector is a dict {key: coefficient}, key = (a_1,e_1, ..., a_n,e_n).

def clean(d):
    return {k: v for k, v in d.items() if sp.expand(v) != 0}


def add(dst, src, c=1):
    for k, v in src.items():
        dst[k] = sp.expand(dst.get(k, ZERO) + c * v)
        if dst[k] == 0:
            del dst[k]
    return dst


def smul(c, u):
    return clean({k: sp.expand(c * v) for k, v in u.items()})


def sub(u, v):
    out = dict(u)
    add(out, v, -1)
    return clean(out)


def _sign(key, i):
    """Koszul sign of an odd operator in slot i passing slots 1..i-1."""
    return (-1) ** sum(key[2 * l + 1] for l in range(i))


def dt(u, i):
    out = {}
    for k, c in u.items():
        a = k[2 * i]
        if a:
            kk = list(k)
            kk[2 * i] = a - 1
            kk = tuple(kk)
            out[kk] = out.get(kk, ZERO) + a * c
    return clean(out)


def dth(u, i):
    out = {}
    for k, c in u.items():
        if k[2 * i + 1]:
            kk = list(k)
            kk[2 * i + 1] = 0
            kk = tuple(kk)
            out[kk] = out.get(kk, ZERO) + _sign(k, i) * c
    return clean(out)


def thmul(u, i):
    out = {}
    for k, c in u.items():
        if k[2 * i + 1] == 0:
            kk = list(k)
            kk[2 * i + 1] = 1
            kk = tuple(kk)
            out[kk] = out.get(kk, ZERO) + _sign(k, i) * c
    return clean(out)


def tmul(u, i):
    out = {}
    for k, c in u.items():
        kk = list(k)
        kk[2 * i] += 1
        kk = tuple(kk)
        out[kk] = out.get(kk, ZERO) + c
    return clean(out)


def Dop(u, i):
    out = dth(u, i)
    add(out, thmul(dt(u, i), i))
    return clean(out)


def Qm(u, i):
    out = dth(u, i)
    add(out, thmul(dt(u, i), i), -1)
    return clean(out)


def Qp(u, i, mu):
    out = tmul(Qm(u, i), i)
    add(out, thmul(u, i), -mu)
    return clean(out)


def cQ(u, eps, mus, signs=None):
    """The osp(1|2)-action on the tensor product: sum_i s_i Q_eps^{(i)}.

    The signs are all +1 for the tensor product itself; they are what a
    parity shift of the factors changes, see the Pi-twisted action in T5.
    """
    out = {}
    for i in range(len(mus)):
        s = 1 if signs is None else signs[i]
        add(out, Qm(u, i) if eps < 0 else Qp(u, i, mus[i]), s)
    return clean(out)


def merge(u, i):
    """iota* in the slots i, i+1: the order-0 bracket, an even algebra map."""
    out = {}
    for k, c in u.items():
        if k[2 * i + 1] and k[2 * i + 3]:
            continue                       # theta^2 = 0
        kk = (k[:2 * i]
              + (k[2 * i] + k[2 * i + 2], k[2 * i + 1] + k[2 * i + 3])
              + k[2 * i + 4:])
        out[kk] = out.get(kk, ZERO) + c
    return clean(out)


def kappa(u, i):
    """The parity automorphism th_i |-> -th_i of the i-th factor."""
    return clean({k: (-1) ** k[2 * i + 1] * c for k, c in u.items()})


def monomials(n, A):
    keys = [()]
    for _ in range(n):
        keys = [k + (a, e) for k in keys for a in range(A + 1) for e in (0, 1)]
    return [{k: ONE} for k in keys]


# --------------------------------------------------------------- the operator
# Theorem 7.2.18: lam_ij = (-1)^{s_j} <mu_1>_{s_j} <mu_2>_{s_i} /
#                          (floor(j/2)! floor(i/2)!),  i+j = N.

def bracket(mu, n, s):
    out = ONE
    for l in range(s):
        out *= (mu + n - 1 - l)
    return sp.expand(out)


def lam_closed(N, mu1, mu2):
    n = (N + 1) // 2                       # ceil(k), k = N/2
    eps = N % 2
    lam = {}
    for i in range(N + 1):
        j = N - i
        si, sj = (i + eps) // 2, (j + eps) // 2
        lam[(i, j)] = sp.expand(
            (-1) ** sj * bracket(mu1, n, sj) * bracket(mu2, n, si)
            / (factorial(j // 2) * factorial(i // 2)))
    return lam


def Bhat(u, N, lam, i0=0):
    """sum_{i+j=N} lam_ij D_{i0}^i D_{i0+1}^j, acting in the slots i0, i0+1."""
    out = {}
    for i in range(N + 1):
        j = N - i
        c = lam.get((i, j), ZERO)
        if c == 0:
            continue
        w = u
        for _ in range(j):
            w = Dop(w, i0 + 1)
        for _ in range(i):
            w = Dop(w, i0)
        add(out, smul(c, w))
    return clean(out)


def Phi(u, N, lam, i0=0, twist=False):
    """The bracket of order N/2 in the slots i0, i0+1; twist = insert kappa."""
    out = merge(Bhat(u, N, lam, i0), i0)
    return kappa(out, i0) if twist else out


# --------------------------------------------------------------- the tests

def intertwines(op, mus_in, mus_out, eps_list, A, n_in):
    """Return the set of signs s with  op(cQ u) = s * cQ(op u)  for all u."""
    good = {1, -1}
    for u in monomials(n_in, A):
        for eps in eps_list:
            lhs = op(cQ(u, eps, mus_in))
            rhs = cQ(op(u), eps, mus_out)
            for s in list(good):
                if sub(lhs, smul(s, rhs)):
                    good.discard(s)
        if not good:
            break
    return good


def defect(op, mus_in, mus_out, eps, A, n_in, s):
    """The first non-vanishing  op(cQ u) - s cQ(op u), for a report."""
    for u in monomials(n_in, A):
        d = sub(op(cQ(u, eps, mus_in)), smul(s, cQ(op(u), eps, mus_out)))
        if d:
            return u, d
    return None, {}


def show(name, good):
    tag = {frozenset({1, -1}): "both (the operator is 0?)",
           frozenset({1}): "(a) only", frozenset({-1}): "(b) only",
           frozenset(): "NEITHER"}[frozenset(good)]
    print("    %-46s %s" % (name, tag))
    return tag


def main(NMAX):
    mu1, mu2, mu3 = sp.symbols("mu1 mu2 mu3")

    print("=" * 74)
    print("T1.  Phi_N itself, in the two-factor model")
    print("     (a): Phi.cQ = +Q.Phi     (b): Phi.cQ = (-1)^N Q.Phi")
    print("=" * 74)
    for N in range(NMAX + 1):
        lam = lam_closed(N, mu1, mu2)
        nu = mu1 + mu2 + N
        A = N + 2
        for twist in (False, True):
            op = lambda u, N=N, lam=lam, tw=twist: Phi(u, N, lam, 0, tw)
            good = intertwines(op, (mu1, mu2), (nu,), (-1, 1), A, 2)
            # translate: sign s=+1 is (a); s=(-1)^N is (b)
            names = set()
            if 1 in good:
                names.add("(a)")
            if (-1) ** N in good:
                names.add("(b)")
            print("    N=%d  %-14s : %s" % (
                N, "kappa.Phi_N" if twist else "Phi_N",
                ", ".join(sorted(names)) if names else "NEITHER"))

    print()
    print("=" * 74)
    print("T2.  Phi_N (x) id  on  SM_{mu1} (x) SM_{mu2} (x) SM_{mu3}")
    print("     Is the operator still invariant when applied in ONE slot?")
    print("     conv. (b) = eq. (170) without kappa^N, conv. (a) = with it")
    print("=" * 74)
    for N in range(min(NMAX, 3) + 1):
        lam = lam_closed(N, mu1, mu2)
        nu = mu1 + mu2 + N
        A = N + 2
        for conv in ("b", "a"):
            tw = (conv == "a") and bool(N % 2)      # kappa^N of eq. (170)
            op = lambda u, N=N, lam=lam, tw=tw: Phi(u, N, lam, 0, tw)
            good = intertwines(op, (mu1, mu2, mu3), (nu, mu3), (-1, 1), A, 3)
            want = 1 if conv == "a" else (-1) ** N
            ok = "invariant" if want in good else "NOT invariant"
            if not good:
                ok += " in either sense"
            print("    N=%d  conv. (%s):  Phi_N (x) id  is  %s" % (N, conv, ok))
            if not good and N % 2:
                u, d = defect(op, (mu1, mu2, mu3), (nu, mu3), 1, A, 3, 1)
                if d:
                    ks = sorted(d)[:2]
                    print("          defect at u = %s :  %s" % (
                        list(u)[0], {k: sp.factor(d[k]) for k in ks}))

    print()
    print("=" * 74)
    print("T3.  the composite  Phi_N o (Phi_{N'} (x) id)")
    print("=" * 74)
    for Np in range(min(NMAX, 2) + 1):
        for N in range(min(NMAX, 2) + 1):
            lamp = lam_closed(Np, mu1, mu2)
            nup = mu1 + mu2 + Np
            lam = lam_closed(N, nup, mu3)
            nu = nup + mu3 + N
            A = N + Np + 2
            for conv in ("b", "a"):
                tw = bool(N % 2) if conv == "a" else False
                twp = bool(Np % 2) if conv == "a" else False

                def op(u, N=N, Np=Np, lam=lam, lamp=lamp, tw=tw, twp=twp):
                    return Phi(Phi(u, Np, lamp, 0, twp), N, lam, 0, tw)
                good = intertwines(op, (mu1, mu2, mu3), (nu,), (-1, 1), A, 3)
                want = 1 if conv == "a" else (-1) ** (N + Np)
                ok = ("invariant of parity %d" % ((N + Np) % 2)
                      if want in good else "NOT invariant")
                if not good:
                    ok += " in either sense"
                print("    N'=%d N=%d  conv. (%s) : %s" % (Np, N, conv, ok))

    print()
    print("=" * 74)
    print("T6.  item 2) of the Discussion: are the two triple brackets")
    print("     [[f,g]_k,h]_k  and  [f,[g,h]_k]_k  proportional?")
    print("     (the claim there -- no associative product on the odd part --")
    print("      was computed in convention (a) and has to be re-checked)")
    print("=" * 74)
    for N in (1, 2, 3):
        A = N + 2
        for conv in ("b", "a"):
            tw = (conv == "a") and bool(N % 2)
            lam12 = lam_closed(N, mu1, mu2)
            nup = mu1 + mu2 + N
            lam_out1 = lam_closed(N, nup, mu3)
            lam23 = lam_closed(N, mu2, mu3)
            nupp = mu2 + mu3 + N
            lam_out2 = lam_closed(N, mu1, nupp)

            def left(u):
                return Phi(Phi(u, N, lam12, 0, tw), N, lam_out1, 0, tw)

            def right(u):
                return Phi(Phi(u, N, lam23, 1, tw), N, lam_out2, 0, tw)

            c, ok = None, True
            for u in monomials(3, A):
                L, R = left(u), right(u)
                keys = set(L) | set(R)
                for k in keys:
                    a, b = L.get(k, ZERO), R.get(k, ZERO)
                    if c is None:
                        if sp.expand(b) != 0:
                            c = sp.cancel(a / b)
                            continue
                        if sp.expand(a) != 0:
                            ok = False        # R = 0 but L =/= 0
                            break
                    else:
                        if sp.expand(sp.together(a - c * b)) != 0:
                            ok = False
                            break
                if not ok:
                    break
            if ok and c is not None:
                print("    N=%d  conv. (%s) : PROPORTIONAL, ratio %s"
                      % (N, conv, sp.factor(c)))
            else:
                print("    N=%d  conv. (%s) : not proportional" % (N, conv))

    print()
    print("=" * 74)
    print("T5.  the Gargoubi-Ovsienko product on Pi(SM_{.-1/2}), eq. (139)")
    print("     On Pi V the action is X.pi(v) = (-1)^{p(X)} pi(Xv), so on")
    print("     Pi V_1 (x) Pi V_2 the odd part of osp(1|2) acts by -Q^(1)+Q^(2)")
    print("     and m(pi u, pi v) := (-1)^{p(u)} pi(Phi(u,v)) is EVEN.")
    print("     Is m a morphism of osp(1|2)-modules?")
    print("=" * 74)
    for N in (1, 3):
        lam = lam_closed(N, mu1, mu2)
        nu = mu1 + mu2 + N
        A = N + 2
        for conv in ("b", "a"):
            tw = (conv == "a")

            def m(u, N=N, lam=lam, tw=tw):
                return Phi(kappa(u, 0), N, lam, 0, tw)
            bad = None
            for u in monomials(2, A):
                for eps in (-1, 1):
                    lhs = m(cQ(u, eps, (mu1, mu2), (-1, 1)))
                    rhs = smul(-1, cQ(m(u), eps, (nu,)))
                    d = sub(lhs, rhs)
                    if d:
                        bad = (u, d)
                        break
                if bad:
                    break
            if bad:
                u, d = bad
                k = sorted(d)[0]
                print("    N=%d  conv. (%s) : NOT a morphism (at u = %s the "
                      "coefficient of %s is %s)"
                      % (N, conv, list(u)[0], k, sp.factor(d[k])))
            else:
                print("    N=%d  conv. (%s) : m is an invariant even "
                      "multiplication" % (N, conv))

    print()
    print("=" * 74)
    print("T4.  the naive commutator is not an action of osp(1|2)")
    print("     [X,Y].Phi - (X.(Y.Phi) + Y.(X.Phi)) = -2(X.Phi.Y + Y.Phi.X)")
    print("=" * 74)
    # For X, Y odd, the bracket identity for the map X |-> (X * -) reads
    #     [X,Y]*Phi = X*(Y*Phi) + Y*(X*Phi),
    # and the two sides differ by  2( Phi.{X,Y} - (X.Phi.Y + Y.Phi.X) ).
    # So the map is an action if and only if that expression vanishes.
    for N in (0, 1, 2, 3):
        lam = lam_closed(N, mu1, mu2)
        nu = mu1 + mu2 + N
        A = N + 2
        bad = None
        for u in monomials(2, A):
            anti = add(cQ(cQ(u, 1, (mu1, mu2)), -1, (mu1, mu2)),
                       cQ(cQ(u, -1, (mu1, mu2)), 1, (mu1, mu2)))
            lhs = Phi(anti, N, lam, 0, False)
            rhs = add(cQ(Phi(cQ(u, 1, (mu1, mu2)), N, lam, 0, False),
                         -1, (nu,)),
                      cQ(Phi(cQ(u, -1, (mu1, mu2)), N, lam, 0, False),
                         1, (nu,)))
            d = sub(lhs, rhs)
            if d:
                bad = (u, d)
                break
        if bad:
            u, d = bad
            k = sorted(d)[0]
            print("    N=%d : NOT an action -- already at u = %s the defect is "
                  "nonzero (coefficient of %s: %s)"
                  % (N, list(u)[0], k, sp.factor(d[k])))
        else:
            print("    N=%d : the identity holds (as it must for N even)" % N)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
