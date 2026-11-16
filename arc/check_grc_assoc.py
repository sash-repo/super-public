#!/usr/bin/env python3
"""Associativity of the products built from the Gieres GRC-brackets of
the text, eqs. (133), (134) -- Discussion 2) and eq. (139).

Everything is computed from the definitions printed in the text:

    D_theta = d_theta - theta d_t                          (item 6.1.3a of the text)
    gamma_{a,k} = (-1)^{a+1} C(mu1+[k], [k]-a) C(mu2+[k], a)

    k in N   (even):  [f,g]_k = sum_{a<=k-1} (-1)^{p(f)} gamma_{a,k-1}
                                     D(f^{(a)}) D(g^{(k-1-a)})
                               - sum_{a<=k} gamma_{a,k} f^{(a)} g^{(k-a)}
    k in N-1/2 (odd): [f,g]_k = sum_{a<=[k]} gamma_{a,k} ( (-1)^{p(f)}(mu1+a)
                                     f^{(a)} D(g^{([k]-a)})
                                   - (mu2+[k]-a) D(f^{(a)}) g^{([k]-a)} )

    [-,-]_k : SM_{mu1} (x) SM_{mu2} --> SM_{mu1+mu2+2k}

PART 1 (validation of the engine AND of the printed formulas) checks that these
operators really are osp(1|2)-invariant.  The osp(1|2)-action on SM_mu is
derived from the text's own eq. (69),
    phi(xt,tht) = phi(x,th) (Cx+D+delta th)^k,
by differentiating the two odd one-parameter subgroups of SOSp(1|2) given by
the C-points of the text:

    beta = eps:  gamma = -eps,  xt = x - eps th, tht = th + eps, j = 1
                 ==>  Q_- = d_th - th d_x = D_theta                (lowering)
    alpha = eps: delta =  eps,  xt = x - eps x th, tht = th + eps x, j = 1+eps th
                 ==>  Q_+ = x d_th - x th d_x - mu th              (raising)

Invariance of a bilinear B: SM_{mu1} (x) SM_{mu2} -> SM_{mu3} reads
    Q_{mu3}(B(f,g)) = B(Q f, g) + (-1)^{p(f)} B(f, Q g).

PART 2 asks the well-posed form of Discussion 2): does a normalisation
lam_k(mu1,mu2), lam_0 = 1, exist making

    f * g = sum_k lam_k(mu1,mu2) [f,g]_k

associative?  Each total order is an inhomogeneous linear system in the four
unknowns lam_N(mu1,mu2), lam_N(mu1+mu2,mu3), lam_N(mu2,mu3), lam_N(mu1,mu2+mu3),
so every order is decided exactly.  Three variants are tested:
  (a) the sum over the EVEN brackets only        (on SM_.),
  (b) the sum over the ODD brackets only         (eq. (139), on Pi(SM_{.-1/2})),
  (c) the sum over ALL brackets                  (on SM_. (+) Pi(SM_{.-1/2})).

No dependencies:  python3 check_grc_assoc.py
"""
from fractions import Fraction as F

Z = F(0)

# --------------------------------------------------------------- C[t,theta]
# element = dict {(a, e): coeff},  a >= 0, e in {0,1}  <->  coeff t^a theta^e


def clean(d):
    return {k: v for k, v in d.items() if v}


def add(dst, src, c=1):
    for k, v in src.items():
        dst[k] = dst.get(k, Z) + c * v
        if dst[k] == 0:
            del dst[k]
    return dst


def mul(u, v):
    out = {}
    for (a1, e1), c1 in u.items():
        for (a2, e2), c2 in v.items():
            if e1 and e2:
                continue                       # theta^2 = 0
            out[(a1 + a2, e1 + e2)] = out.get((a1 + a2, e1 + e2), Z) + c1 * c2
    return clean(out)


def smul(c, u):
    return clean({k: c * v for k, v in u.items()})


def dt(u):
    return clean({(a - 1, e): a * c for (a, e), c in u.items() if a})


def dth(u):                                    # left derivative d/d theta
    return clean({(a, 0): c for (a, e), c in u.items() if e})


def thmul(u):                                  # theta . u
    return clean({(a, 1): c for (a, e), c in u.items() if e == 0})


def tmul(u):
    return clean({(a + 1, e): c for (a, e), c in u.items()})


def Dth(u):                                    # D_theta = d_theta - theta d_t
    out = dth(u)
    return add(out, thmul(dt(u)), -1)


def dtn(u, n):
    for _ in range(n):
        u = dt(u)
    return u


# ------------------------------------------------------------- the brackets
def binom(x, n):
    r = F(1)
    for i in range(n):
        r *= (x - i)
    for i in range(1, n + 1):
        r /= i
    return r


def gamma(a, kf, mu1, mu2):
    return (-1) ** (a + 1) * binom(mu1 + kf, kf - a) * binom(mu2 + kf, a)


def bracket(k2, mu1, mu2, f, g, pf):
    """[f,g]_{k2/2}; k2 = 2k >= 0 integer, pf = parity of f."""
    sgn = -1 if pf % 2 else 1
    out = {}
    if k2 % 2 == 0:                                            # (133)
        k = k2 // 2
        for a in range(0, k):
            c = sgn * gamma(a, k - 1, mu1, mu2)
            add(out, mul(Dth(dtn(f, a)), Dth(dtn(g, k - 1 - a))), c)
        for a in range(0, k + 1):
            c = -gamma(a, k, mu1, mu2)
            add(out, mul(dtn(f, a), dtn(g, k - a)), c)
    else:                                                      # (134)
        m = (k2 - 1) // 2
        for a in range(0, m + 1):
            gk = gamma(a, m, mu1, mu2)
            add(out, mul(dtn(f, a), Dth(dtn(g, m - a))), sgn * gk * (mu1 + a))
            add(out, mul(Dth(dtn(f, a)), dtn(g, m - a)), -gk * (mu2 + m - a))
    return clean(out)


def par(k2, pf, pg):
    return (pf + pg + k2) % 2


# ---------------------------------------------------------- osp(1|2) action
def Qminus(u, mu):
    return Dth(u)


def Qplus(u, mu):
    out = tmul(dth(u))
    add(out, tmul(thmul(dt(u))), -1)
    add(out, thmul(u), -mu)
    return clean(out)


def monos(A):
    return [({(a, e): F(1)}, e) for a in range(A + 1) for e in (0, 1)]


def check_invariance(k2max, weights, A=6, verbose=True):
    bad = []
    for k2 in range(0, k2max + 1):
        for mu1 in weights:
            for mu2 in weights:
                mu3 = mu1 + mu2 + k2          # = mu1 + mu2 + 2k
                for Q in (Qminus, Qplus):
                    for f, pf in monos(A):
                        for g, pg in monos(A):
                            lhs = Q(bracket(k2, mu1, mu2, f, g, pf), mu3)
                            r = bracket(k2, mu1, mu2, Q(f, mu1), g, (pf + 1) % 2)
                            r2 = bracket(k2, mu1, mu2, f, Q(g, mu2), pf)
                            rhs = dict(r)
                            add(rhs, r2, -1 if pf % 2 else 1)
                            d = dict(lhs)
                            add(d, rhs, -1)
                            if d:
                                bad.append((k2, mu1, mu2, Q.__name__))
                                break
                        else:
                            continue
                        break
        if verbose:
            hits = [b for b in bad if b[0] == k2]
            print(f"    2k = {k2}:  "
                  f"{'INVARIANT' if not hits else 'NOT invariant: ' + str(hits[:2])}")
    return bad


# --------------------------------------------------- linear-algebra machinery
class Echelon:
    def __init__(self, n):
        self.n = n
        self.piv = {}
        self.bad = False

    def add(self, row, rhs):
        r = list(row) + [rhs]
        for c in sorted(self.piv):
            if r[c]:
                f0, pr = r[c], self.piv[c]
                r = [x - f0 * y for x, y in zip(r, pr)]
        for c in range(self.n):
            if r[c]:
                pv = r[c]
                r = [x / pv for x in r]
                for c2 in list(self.piv):
                    pr = self.piv[c2]
                    if pr[c]:
                        f0 = pr[c]
                        self.piv[c2] = [x - f0 * y for x, y in zip(pr, r)]
                self.piv[c] = r
                return
        if r[self.n]:
            self.bad = True

    def solution(self):
        if self.bad:
            return None
        part = [Z] * self.n
        for c, r in self.piv.items():
            part[c] = r[self.n]
        free = [c for c in range(self.n) if c not in self.piv]
        basis = []
        for fc in free:
            v = [Z] * self.n
            v[fc] = F(1)
            for c, r in self.piv.items():
                v[c] = -r[fc]
            basis.append(v)
        return part, basis


def solve_order(N, step, mu1, mu2, mu3, lam, A=None):
    """Associativity of  f*g = sum_p lam_p(.,.) [f,g]_{p*step/2}  in total
    order N (in units of step/2 in the D_theta-grading).  lam_p known for p < N.
    Unknowns: lam_N at the four pairs below."""
    pairs = [(mu1, mu2), (mu1 + mu2, mu3), (mu2, mu3), (mu1, mu2 + mu3)]
    E = Echelon(4)
    if A is None:
        A = N * step + 2
    tests = monos(A)
    for f, pf in tests:
        for g, pg in tests:
            for h, ph in tests:
                lin = [{}, {}, {}, {}]
                const = {}
                for p in range(N + 1):
                    q = N - p
                    kp, kq = p * step, q * step
                    u = bracket(kp, mu1, mu2, f, g, pf)
                    pu = par(kp, pf, pg)
                    v = bracket(kq, mu1 + mu2 + kp, mu3, u, h, pu)
                    u2 = bracket(kp, mu2, mu3, g, h, pg)
                    v2 = bracket(kq, mu1, mu2 + mu3 + kp, f, u2, pf)
                    if p == N:                       # lam_N(mu1,mu2) . lam_0
                        add(lin[0], v)
                        add(lin[2], v2, -1)
                    elif q == N:                     # lam_0 . lam_N(mu1+mu2,mu3)
                        add(lin[1], v)
                        add(lin[3], v2, -1)
                    else:
                        c0 = lam[p](mu1, mu2) * lam[q](mu1 + mu2 + kp, mu3)
                        add(const, v, c0)
                        c0 = lam[p](mu2, mu3) * lam[q](mu1, mu2 + mu3 + kp)
                        add(const, v2, -c0)
                for m in set(const) | {m for d in lin for m in d}:
                    row = [d.get(m, Z) for d in lin]
                    if any(row) or const.get(m):
                        E.add(row, -const.get(m, Z))
    return pairs, E


TWNAME = ["1", "(-1)^p(f)", "(-1)^p(g)", "(-1)^p(f)p(g)"]


def twist(tw, pf, pg):
    if tw == 0:
        return 1
    if tw == 1:
        return -1 if pf % 2 else 1
    if tw == 2:
        return -1 if pg % 2 else 1
    return -1 if (pf % 2 and pg % 2) else 1


def rising(n):
    """lam_n(u,v) = (u)(u+1)...(u+n-1)/n! = binom(u+n-1, n)"""
    def f(u, v):
        r = F(1)
        for i in range(n):
            r *= (u + i)
        for i in range(1, n + 1):
            r /= i
        return r
    return f


def report(name, step, orders, triples, W, lam0=None):
    print("=" * 76)
    print(name)
    print("=" * 76)
    lam = {0: (lambda u, v: F(1))}
    if lam0:
        lam.update(lam0)
    else:
        lam.update({n: rising(n) for n in range(1, 8)})
    for N in orders:
        print(f"\n  --- total order {N*step}/2 in the D_theta-grading "
              f"(bracket index k = {F(N*step,2)}) ---")
        for (a, b, c) in triples:
            mu1, mu2, mu3 = W(a), W(b), W(c)
            pairs, E = solve_order(N, step, mu1, mu2, mu3, lam)
            sol = E.solution()
            if sol is None:
                print(f"    mu=({a},{b},{c})/P:  INCONSISTENT -- "
                      f"no associative product at this order")
                continue
            part, ns = sol
            print(f"    mu=({a},{b},{c})/P:  consistent, solution space of "
                  f"dim {len(ns)}")
            print(f"       particular solution  lam_{N} at "
                  f"{[(str(u),str(v)) for u,v in pairs]}:")
            print(f"         {[str(x) for x in part]}")
            if ns:
                for b_ in ns[:2]:
                    print(f"         + free direction {[str(x) for x in b_]}")
            pred = [rising(N * step // 2)(u, v) for (u, v) in pairs] \
                if (N * step) % 2 == 0 else None
            if pred is not None:
                print(f"       binom(mu1+n-1,n) would be "
                      f"{[str(x) for x in pred]}")


# ==========================================================================
if __name__ == "__main__":
    P = 31
    W = lambda a: F(a, P)
    triples = [(1, 2, 3), (1, 1, 1), (2, 3, 5), (3, 4, 5)]

    print("#" * 76)
    print("# PART 1.  Are the printed brackets (133), (134) osp(1|2)-invariant?")
    print("#" * 76)
    print("  target weight mu1+mu2+2k, D_theta = d_theta - theta d_t, "
          "action from (69):")
    bad = check_invariance(5, [W(1), W(2), W(5), F(0)], A=6)
    print(f"\n  ==> {'ALL INVARIANT' if not bad else 'failures: ' + str(bad)}")

    print()
    report("PART 2a.  EVEN product on SM_. :  f*g = sum_{k in N} lam_k [f,g]_k",
           2, [1, 2, 3], triples, W)

    print()
    print("=" * 76)
    print("PART 2b.  ODD product, eq. (139):  f*g = sum_{k in N-1/2} lam_k [f,g]_k")
    print("=" * 76)
    print("  There is no unit among the odd brackets, so the lowest total order")
    print("  of  (f*g)*h = f*(g*h)  is  2m+1  with  m = min{k : lam_k =/= 0},")
    print("  and it says that the single bracket [.,.]_{m} must satisfy")
    print("     L [[f,g]_m,h]_m = R [f,[g,h]_m]_m   with L,R =/= 0,")
    print("  possibly after a Pi-twist, i.e. after inserting one of the signs")
    print("  s(f,g) in {1, (-1)^p(f), (-1)^p(g), (-1)^p(f)p(g)} into each bracket.")
    for k2 in (1, 3, 5):
        print(f"\n  --- lowest surviving bracket k = {F(k2,2)} ---")
        for tw in range(4):
            ok_all = True
            detail = []
            for (a, b, c) in triples:
                mu1, mu2, mu3 = W(a), W(b), W(c)
                E = Echelon(2)
                for f, pf in monos(k2 + 2):
                    for g, pg in monos(k2 + 2):
                        for h, ph in monos(k2 + 2):
                            u = bracket(k2, mu1, mu2, f, g, pf)
                            pu = par(k2, pf, pg)
                            v = bracket(k2, mu1 + mu2 + k2, mu3, u, h, pu)
                            u2 = bracket(k2, mu2, mu3, g, h, pg)
                            v2 = bracket(k2, mu1, mu2 + mu3 + k2, f, u2, pf)
                            s1 = twist(tw, pf, pg) * twist(tw, pu, ph)
                            s2 = twist(tw, pg, ph) * twist(tw, pf,
                                                           par(k2, pg, ph))
                            for m in set(v) | set(v2):
                                E.add([s1 * v.get(m, Z), -s2 * v2.get(m, Z)], Z)
                sol = E.solution()
                nz = sol is not None and any(all(x != 0 for x in bvec)
                                             for bvec in sol[1])
                detail.append(f"{a},{b},{c}:{'yes' if nz else 'NO'}")
                ok_all = ok_all and nz
            print(f"    twist {TWNAME[tw]:>16s}: "
                  f"{'associative bracket' if ok_all else 'NOT proportional'}"
                  f"   [{'  '.join(detail)}]")

    print()
    report("PART 2c.  FULL product on SM_. (+) Pi(SM_{.-1/2}) : all k in (1/2)N",
           1, [1, 2], triples, W, lam0={n: (lambda u, v: F(0))
                                        for n in range(1, 8)})
