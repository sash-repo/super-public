#!/usr/bin/env python3
"""Theorem 7.3 of the paper (the pe(2) subset m(1) case).

The weights F = F alpha_0^lambda vol^nu of the paper are the
F_{a,b} = F alpha_0^a vol^b of [BLS1, (38)]:  a = lambda, b = nu.

(1) The Bol operator Bbar_k of [BLS1, Thm 5.1] is invariant  F_{a,b} -> F_{a+1-k, b-1+k/2}
    iff a = (k-1)/2 (b free).                       -- unary check, all 8 generators
(2) Hence  J^k_l = Bbar_k (x) 1  and  J^k_r  are invariant
        F_{(k-1)/2, nu_1} (x) F_{(k-1)/2, nu_2}  ->  F_{0, nu_1+nu_2-1+k/2},
    i.e. mu = 0 and omega = nu_1+nu_2+k/2-1.
(3) Control: the weights F_{(k-1)/2-nu_1, nu_1} (x) F_{(k-1)/2-nu_2, nu_2} -> F_{mu,-mu},
    mu = (k-2)/2 - nu_1 - nu_2, admit no such operator for generic nu.
"""
from sympy import symbols, S, Matrix, expand, Rational, factor
from super12 import add, smul, mul, mono, parity
from m1 import M, D_1, D_xi, D_q, L, PE, TESTM

ALL = tuple(PE)
MS = [(a, e, f) for a in range(0, 6) for e in (0, 1) for f in (0, 1)]


def Bbar(k):
    """Bbar_k = D_q(-D_xi)^{k-1} + (k-1)(-D_xi)^{k-2} d_tau ,  [BLS1, Thm 5.1]. Odd."""
    def r(u):
        x = u
        for _ in range(k - 1):
            x = smul(-1, D_xi(x))
        out = D_q(x)
        if k >= 2:
            y = u
            for _ in range(k - 2):
                y = smul(-1, D_xi(y))
            out = add(out, smul(k - 1, D_1(y)))
        return out
    return r


def check_unary(k, a, b):
    """Is Bbar_k : F_{a,b} -> F_{a+1-k, b-1+k/2} invariant?  Returns the equations."""
    B = Bbar(k)
    At, Bt = a + 1 - k, b - 1 + S(k) / 2
    bad = set()
    for nm in ALL:
        f = PE[nm]
        pf = (parity(f) + 1) % 2          # parity of the vector field M_f
        s = -1 if pf else 1               # Bbar_k is odd
        Ls, Lt = L(f, a, b), L(f, At, Bt)
        for m in MS:
            u = mono(*m)
            res = add(Lt(B(u)), smul(-s, B(Ls(u))))
            for v in res.values():
                e = expand(v)
                if e:
                    bad.add(factor(e))
    return bad


def check_binary(k, A1, B1, A2, B2, At, Bt, side='l'):
    """Is J^k_side invariant  F_{A1,B1} (x) F_{A2,B2} -> F_{At,Bt} ?"""
    B = Bbar(k)
    bad = set()
    for nm in ALL:
        f = PE[nm]
        pf = (parity(f) + 1) % 2
        sB = -1 if pf else 1              # J is odd
        Lf1, Lf2, Lft = L(f, A1, B1), L(f, A2, B2), L(f, At, Bt)
        for m1 in MS:
            u = mono(*m1)
            pu = (m1[1] + m1[2]) % 2
            Lu = Lf1(u)
            for m2 in MS:
                v = mono(*m2)
                Lv = Lf2(v)

                def J(x, y, px):
                    if side == 'l':                      # Bbar_k(x) . y
                        return mul(B(x), y)
                    return smul(-1 if px else 1, mul(x, B(y)))   # (-1)^p(x) x . Bbar_k(y)
                s2 = -1 if (pf and pu) else 1
                res = add(Lft(J(u, v, pu)),
                          smul(-sB, add(J(Lu, v, (pu + pf) % 2), smul(s2, J(u, Lv, pu)))))
                for w in res.values():
                    e = expand(w)
                    if e:
                        bad.add(factor(e))
    return bad


if __name__ == '__main__':
    a, b = symbols('a b')
    n1, n2, a2 = symbols('nu_1 nu_2 a_2')

    print("=" * 78)
    print("(1) UNARY:  Bbar_k : F_{a,b} -> F_{a+1-k, b-1+k/2}   (all 8 generators)")
    print("=" * 78)
    for k in (1, 2, 3, 4):
        bad = check_unary(k, a, b)
        print(f"  k={k}: obstructions {sorted({str(e) for e in bad})}")
        # at a = (k-1)/2
        bad0 = check_unary(k, Rational(k - 1, 2), b)
        print(f"       at a=(k-1)/2={Rational(k-1,2)}, b free: "
              f"{'INVARIANT' if not bad0 else sorted({str(e) for e in bad0})}")

    print()
    print("=" * 78)
    print("(2) BINARY at the corrected weights:")
    print("    F_{(k-1)/2,nu_1} (x) F_{(k-1)/2,nu_2} -> F_{0, nu_1+nu_2-1+k/2}")
    print("=" * 78)
    for k in (1, 2, 3, 4):
        A = Rational(k - 1, 2)
        At, Bt = S(0), n1 + n2 - 1 + S(k) / 2
        for side in ('l', 'r'):
            bad = check_binary(k, A, n1, A, n2, At, Bt, side)
            print(f"  k={k} J^k_{side}: "
                  f"{'INVARIANT' if not bad else sorted({str(e) for e in bad})[:3]}")

    print()
    print("=" * 78)
    print("(2a) only one argument constrained:  J^k_l needs a_1=(k-1)/2, a_2 FREE")
    print("=" * 78)
    for k in (1, 2, 3):
        A = Rational(k - 1, 2)
        At, Bt = A + a2 + 1 - k, n1 + n2 - 1 + S(k) / 2
        bad = check_binary(k, A, n1, a2, n2, At, Bt, 'l')
        print(f"  k={k} J^k_l with a_2 symbolic -> F_{{{At},{Bt}}}: "
              f"{'INVARIANT' if not bad else sorted({str(e) for e in bad})[:3]}")

    print()
    print("=" * 78)
    print("(3) CONTROL: other weights, which must carry no invariant operator:")
    print("    F_{(k-1)/2-nu_1, nu_1} (x) F_{(k-1)/2-nu_2, nu_2} -> F_{mu,-mu},")
    print("    mu = (k-2)/2 - nu_1 - nu_2")
    print("=" * 78)
    for k in (1, 2, 3):
        A1, A2 = Rational(k - 1, 2) - n1, Rational(k - 1, 2) - n2
        mu = Rational(k - 2, 2) - n1 - n2
        for side in ('l', 'r'):
            bad = check_binary(k, A1, n1, A2, n2, mu, -mu, side)
            print(f"  k={k} J^k_{side}: "
                  f"{'INVARIANT' if not bad else 'NOT invariant, e.g. ' + sorted({str(e) for e in bad})[0]}")
        # and: is the source locus alone right?  Bbar_k on F_{(k-1)/2-nu, nu}
        bad = check_unary(k, Rational(k - 1, 2) - n1, n1)
        print(f"       source locus lambda=(k-1)/2-nu: Bbar_k "
              f"{'invariant' if not bad else 'NOT invariant (needs nu=0)'}")
