#!/usr/bin/env python3
"""
The two expansion formulas of Subsec. 2.5 of the text, eqs. (37) and (38)-(39)
(the variants of an earlier draft are printed as controls):

  (A) eq. (37):  D^j o f  for D = d/dx      -- is the (-1)^k there?
  (B) the Manin--Radul display for Delta = d_theta + theta d_x.

(A) is checked on formal (pseudo-)differential symbols, (B) by composing the
actual operators on C[x,theta].  Exact symbolic algebra throughout (sympy).
"""
import sympy as sp

x = sp.Symbol('x')


# =====================================================================  (A)
# A symbol is a dict {i: coefficient}, meaning sum_i coeff_i * D^i, i in Z.
def sym_mul(P, Q, lo=-6):
    """Product of two symbols, using  D^i o g = sum_{k>=0} C(i,k) g^{(k)} D^{i-k}."""
    out = {}
    for i, a in P.items():
        for j, b in Q.items():
            k = 0
            while i - k >= lo - abs(j):
                c = sp.binomial(i, k)
                if c != 0:
                    term = sp.expand(a * sp.diff(b, x, k) * c)
                    if term != 0:
                        e = i + j - k
                        if e >= lo:
                            out[e] = sp.expand(out.get(e, 0) + term)
                k += 1
                if k > 12:
                    break
    return {e: c for e, c in out.items() if sp.simplify(c) != 0}


def show(P):
    return " + ".join("(%s)D^%d" % (sp.simplify(c), e)
                      for e, c in sorted(P.items(), reverse=True)) or "0"


def check_A():
    print("=" * 70)
    print("(A)  eq. (37):  D^j o f  --  is the factor (-1)^k right?")
    f = sp.Function('f')(x)
    for j in (1, 2, 3):
        # the true composition D^j o f, computed by iterating D o (-)
        cur = {0: f}
        for _ in range(j):
            cur = sym_mul({1: sp.Integer(1)}, cur)
        printed = {}      # as printed: sum_k (-1)^k C(j,k) f^{(j-k)} D^k
        nosign = {}       # without the sign
        for k in range(0, j + 1):
            c = sp.binomial(j, k)
            printed[k] = sp.expand(printed.get(k, 0) + (-1)**k * c * sp.diff(f, x, j - k))
            nosign[k] = sp.expand(nosign.get(k, 0) + c * sp.diff(f, x, j - k))
        printed = {e: c for e, c in printed.items() if c != 0}
        nosign = {e: c for e, c in nosign.items() if c != 0}
        print("\n  j = %d" % j)
        print("    true  D^%d o f      = %s" % (j, show(cur)))
        print("    earlier draft, with (-1)^k = %s   --> %s"
              % (show(printed),
                 "AGREES" if show(printed) == show(cur) else "DISAGREES"))
        print("    without the (-1)^k      = %s   --> %s"
              % (show(nosign),
                 "AGREES" if show(nosign) == show(cur) else "DISAGREES"))

    print("\n  the negative powers:  is  D^{-1} o f  given by the earlier-draft formula?")
    print("    the earlier-draft sum runs over k in Z_+ with k <= j, which is EMPTY for j<0;")
    print("    the formula that works for every j is  D^j o f = sum_{0<=k} C(j,k) f^{(k)} D^{j-k}:")
    g = sp.Function('g')(x)
    lhs = sym_mul({-1: sp.Integer(1)}, {0: g})
    print("      D^{-1} o g =", show(lhs))
    idp = sym_mul({1: sp.Integer(1)}, lhs)
    # the series was truncated at D^{-6}, so only the terms above that count
    kept = {e: c for e, c in idp.items() if e > -6}
    print("      D o (D^{-1} o g) =", show(kept), " (mod D^{-6}; must be g)")
    print("      --> %s" % ("OK" if sp.simplify(kept.get(0, 0) - g) == 0
                            and all(sp.simplify(c) == 0 for e, c in kept.items() if e != 0)
                            else "FAIL"))


# =====================================================================  (B)
# Operators on C[x,theta]:  f = a(x) + theta*b(x)  is the pair (a,b).
def Delta(f):
    a, b = f
    return (b, sp.diff(a, x))


def mult_by(u_par, alpha):
    """Multiplication by an even (u_par=0) or odd (u_par=1) element."""
    if u_par == 0:                      # u = alpha(x)
        return lambda f: (sp.expand(alpha * f[0]), sp.expand(alpha * f[1]))
    return lambda f: (sp.Integer(0), sp.expand(alpha * f[0]))   # u = theta*alpha


def Delta_pow(k):
    def g(f):
        for _ in range(k):
            f = Delta(f)
        return f
    return g


def superbinom(j, k):
    """Manin--Radul:  0 if k>j or (j,k) = (0,1) mod 2;  C([j/2],[k/2]) else."""
    if k > j:
        return sp.Integer(0)
    if j % 2 == 0 and k % 2 == 1:
        return sp.Integer(0)
    return sp.binomial(j // 2, k // 2)


def sdelta_i(u_par, alpha, i):
    """u^{[i]} = Delta^i(u), returned as (parity, coefficient) of the result."""
    f = (alpha, sp.Integer(0)) if u_par == 0 else (sp.Integer(0), alpha)
    for _ in range(i):
        f = Delta(f)
    # f is (a,b); it is homogeneous since Delta is odd and u was homogeneous
    par = (u_par + i) % 2
    return par, (f[0] if par == 0 else f[1])


def check_B():
    print("\n" + "=" * 70)
    print("(B)  the Manin--Radul display:  Delta^j o a = sum_k [j;k] (-1)^{p(a)k} a^{[j-k]} Delta^k")
    A0, B0 = sp.symbols('A0:6'), sp.symbols('B0:6')
    gen = (sum(A0[i] * x**i for i in range(6)), sum(B0[i] * x**i for i in range(6)))
    alpha = sp.Function('alpha')(x)
    ok_all = True
    for u_par in (0, 1):
        for j in range(1, 5):
            lhs_op = lambda f, j=j, u_par=u_par: Delta_pow(j)(mult_by(u_par, alpha)(f))
            def rhs_op(f, j=j, u_par=u_par):
                tot = (sp.Integer(0), sp.Integer(0))
                for k in range(0, j + 1):
                    c = superbinom(j, k)
                    if c == 0:
                        continue
                    par, coef = sdelta_i(u_par, alpha, j - k)
                    sign = (-1) ** (u_par * k)
                    piece = mult_by(par, coef)(Delta_pow(k)(f))
                    tot = (sp.expand(tot[0] + sign * c * piece[0]),
                           sp.expand(tot[1] + sign * c * piece[1]))
                return tot
            L, R = lhs_op(gen), rhs_op(gen)
            ok = (sp.simplify(L[0] - R[0]) == 0 and sp.simplify(L[1] - R[1]) == 0)
            ok_all &= ok
            print("   p(a)=%d, j=%d :  %s" % (u_par, j, "OK" if ok else "MISMATCH"))
    print("   --> the MR formula, with [j;k] = C([j/2],[k/2]) unless k>j or"
          " (j,k)=(0,1) mod 2,\n       and the sign (-1)^{p(a)k}: %s"
          % ("verified" if ok_all else "FAILS"))

    print("\n   the branches as in an earlier draft:")
    print("     '0 for k>j and for (j,k) != (0,1) mod 2;  [j/2 ; k/2] for k<=j"
          " or (j,k) = (0,1) mod 2'")
    print("     -- the two conditions overlap and contradict each other, e.g.:")
    for (j, k) in ((2, 1), (4, 1), (4, 3), (2, 3), (1, 3)):
        first = (k > j) and not (j % 2 == 0 and k % 2 == 1)     # "and", as printed
        second = (k <= j) or (j % 2 == 0 and k % 2 == 1)        # "or",  as printed
        if first and not second:
            val = "0"
        elif second and not first:
            val = "C(%d,%d) = %s" % (j // 2, k // 2, sp.binomial(j // 2, k // 2))
        elif first and second:
            val = "BOTH branches apply (0 and C(%d,%d)=%s)" % (
                j // 2, k // 2, sp.binomial(j // 2, k // 2))
        else:
            val = "NEITHER branch applies (undefined)"
        print("       (j,k)=(%d,%d): earlier draft -> %-19s   correct value %s"
              % (j, k, val, superbinom(j, k)))


if __name__ == '__main__':
    check_A()
    check_B()
