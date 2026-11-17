"""
Statement 2.8.3 and Remark 5.9, headings 1) and 2): which members of the family
K_{theta_1} ... K_{theta_n} K_1^m (-1 <= m <= 4) give a 2-cocycle
        c(f, g) = eps^{p(f)} Res( f . K_th1 .. K_thn K_1^m (g) )
that is super anti-symmetric (by Lemma 5.7 this is the anti-self-adjointness (75)).

Realization (i) of both series (formula (119)): k^L(1|n) and k^M(1|n) are spanned by the
same contact fields K_f with the same contact form alpha_n = dx - sum theta_i d theta_i,
only the ring of the generating functions differs:

        R^L(1|n) = C[x,x^{-1}] (x) Lambda(theta_1..theta_n),
        R^M(1|n) = C[x,x^{-1}] (x) Lambda(theta_1..theta_{n-1}) (+) zeta * (...)
                   with zeta = x^{1/2} theta_n,

so in R^M a monomial carries a half-integer power of x exactly when theta_n occurs in it,
and the residue is that of formula (153):

        Res^L h = coefficient of x^{-1}   theta_1...theta_n ,
        Res^M h = coefficient of x^{-1/2} theta_1...theta_n .

In this file the odd summand of the contact bracket carries the sign opposite to that of
Lemma 5.8, heading 1) (the bracket is then that of the contact form of the opposite sign,
under which the roles of theta d_x + d_theta and theta d_x - d_theta are interchanged, see
kdops.py part 2b), and the prefactor is eps = 1.  Expected output for the series M: "n=1: M works for m in [2]",
and no m for n = 2, 3 (there the Ramond operators of Table (155) are not of this shape).  The
series L is printed for comparison: with the contact fields K_theta (not the D_theta of
formula (70)) it gives "n=0: [3, 1]", "n=1: [2]", "n=2: []", "n=3: [0]".

A function is a dict {(a, S): c}: a a Fraction (the power of x), S a strictly increasing
tuple of odd indices, c a Fraction.  Exact rational arithmetic.
Run:  python3 selfadj.py   (about 3 minutes;  python3 selfadj.py L  or  M  for one series).
"""

from fractions import Fraction as Q
from itertools import combinations
import sys


# ------------------------------------------------------------------ functions

def _sort(S):
    S = list(S)
    sgn = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn


def mono(a, S=(), c=1):
    srt, sgn = _sort(tuple(S))
    if sgn == 0:
        return {}
    c = Q(c) * sgn
    return {(Q(a), srt): c} if c else {}


def add(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, Q(0)) + v
    return {k: v for k, v in out.items() if v}


def smul(c, f):
    c = Q(c)
    return {k: c * v for k, v in f.items() if c * v}


def mul(f, g):
    """Product in the supercommutative ring; odd generators anticommute."""
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            S, sgn = _sort(S1 + S2)
            if sgn == 0:
                continue
            k = (a1 + a2, S)
            out[k] = out.get(k, Q(0)) + c1 * c2 * sgn
    return {k: v for k, v in out.items() if v}


def parity(f):
    ps = {len(S) % 2 for (_, S) in f}
    if len(ps) > 1:
        raise ValueError("inhomogeneous parity")
    return ps.pop() if ps else 0


def d_x(f):
    out = {}
    for (a, S), c in f.items():
        if a:
            out[(a - 1, S)] = out.get((a - 1, S), Q(0)) + a * c
    return {k: v for k, v in out.items() if v}


def int_x(f):
    """The antiderivative in x; K_1^{-1} = (1/2) int_x.  Undefined on x^{-1}."""
    out = {}
    for (a, S), c in f.items():
        if a == -1:
            raise ZeroDivisionError("log: x^{-1} in the range of int_x")
        out[(a + 1, S)] = out.get((a + 1, S), Q(0)) + c / (a + 1)
    return {k: v for k, v in out.items() if v}


def d_odd(i, f):
    """Left derivative d/d theta_i."""
    out = {}
    for (a, S), c in f.items():
        if i not in S:
            continue
        j = S.index(i)
        sgn = (-1) ** j
        T = S[:j] + S[j + 1:]
        out[(a, T)] = out.get((a, T), Q(0)) + sgn * c
    return {k: v for k, v in out.items() if v}


def E_op(f, n):
    """The odd Euler operator sum theta_i d/d theta_i."""
    return {k: Q(len(k[1])) * v for k, v in f.items() if len(k[1]) and v}


# ------------------------------------------------------- the contact field K_f

def K(f, n, kap=1):
    """Return the operator h |-> K_f(h), formulas (20) and (21).

    kap is the normalisation of the odd (Hamiltonian) part relative to the even
    one.  It is a genuine free parameter: the homomorphism [K_f,K_g]=K_{{f,g}}
    holds for every kap (rescaling the theta's), so it must be scanned, not
    guessed."""
    pf = parity(f)
    E_f = E_op(f, n)
    nab = add(smul(2, f), smul(-1, E_f))          # (2 - E)(f)
    fx = d_x(f)
    dtf = [d_odd(i, f) for i in range(1, n + 1)]

    def op(h):
        out = mul(nab, d_x(h))                     # (2-E)(f) . d_x h
        out = add(out, mul(fx, E_op(h, n)))        # f_x . E(h)
        # - H_f(h),  H_f = -(-1)^{p(f)} sum d_{theta_i}f . d_{theta_i}
        s = {}
        for i in range(1, n + 1):
            s = add(s, mul(dtf[i - 1], d_odd(i, h)))
        return add(out, smul(kap * ((-1) ** pf), s))

    return op


def bracket(f, g, n, kap=1):
    """The contact bracket: [K_f, K_g] = K_{{f,g}}."""
    pf, pg = parity(f), parity(g)
    nf = add(smul(2, f), smul(-1, E_op(f, n)))
    ng = add(smul(2, g), smul(-1, E_op(g, n)))
    out = add(mul(nf, d_x(g)), smul(-1, mul(d_x(f), ng)))
    s = {}
    for i in range(1, n + 1):
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    # the sign of the odd term is the one for which [K_f, K_g] = K_{{f,g}}
    # with the operators K of this file.
    return add(out, smul(kap * ((-1) ** pf), s))


# ---------------------------------------------------------------- the operator

def St(n, m, kind, kap=1):
    """K_{theta_1} ... K_{theta_n} K_1^m  as an operator on functions.

    K_1 = 2 d_x, so K_1^m = 2^m d_x^m  (m may be negative).

    In the M (Ramond) case the last factor is K_zeta, zeta = x^{1/2} theta_n,
    and NOT K_{theta_n}: the field K_{theta_n} does not belong to k^M(1|n) --
    theta_n = x^{-1/2} zeta is not in R^M -- and with K_{theta_n} the residue
    below vanishes identically.  This one substitution is the whole Ramond
    twist."""
    Ks = [K(mono(0, (i,)), n, kap) for i in range(1, n + 1)]
    if kind == 'M':
        Ks[n - 1] = K(mono(Q(1, 2), (n,)), n, kap)

    def op(h):
        out = h
        if m >= 0:
            for _ in range(m):
                out = smul(2, d_x(out))
        else:
            for _ in range(-m):
                out = smul(Q(1, 2), int_x(out))
        for i in range(n - 1, -1, -1):             # apply K_{theta_n} first
            out = Ks[i](out)
        return out

    return op


def residue(h, n, kind):
    a = Q(-1) if kind == 'L' else Q(-1, 2)
    return h.get((a, tuple(range(1, n + 1))), Q(0))


# ------------------------------------------------------------------ the basis

def basis(n, kind, amin=-3, amax=3):
    """Monomials of R^L / R^M with x-exponents in the given window."""
    out = []
    idx = list(range(1, n + 1))
    for r in range(n + 1):
        for S in combinations(idx, r):
            for a2 in range(2 * amin, 2 * amax + 1):
                a = Q(a2, 2)
                if kind == 'L':
                    if a.denominator != 1:
                        continue
                else:
                    half = (n in S)
                    if (a.denominator == 2) != half:
                        continue
                out.append(mono(a, S))
    return out


# ------------------------------------------------------------------- the tests

def cocycle(n, m, kind, kap=1, eps=1):
    """c(f,g) = eps^{p(f)} Res( f . K_th1..K_thn K_1^m (g) ).

    eps = -1 goes with the sign convention for H_f of Lemma 5.8; eps = +1 with
    the opposite one, which this file uses."""
    L = St(n, m, kind, kap)

    def c(f, g):
        try:
            return (eps ** parity(f)) * residue(mul(f, L(g)), n, kind)
        except ZeroDivisionError:
            return None
    return c


def check(n, m, kind, amin=-3, amax=3, verbose=False, kap=1, eps=1):
    """Return (skew_failures, cocycle_failures, nonzero) for K_th..K_th K_1^m."""
    c = cocycle(n, m, kind, kap, eps)
    B = basis(n, kind, amin, amax)
    skew = coc = 0
    nz = 0
    for f in B:
        for g in B:
            v = c(f, g)
            w = c(g, f)
            if v is None or w is None:
                continue
            if v:
                nz += 1
            if v != -((-1) ** (parity(f) * parity(g))) * w:
                skew += 1
                if verbose and skew < 4:
                    print("   skew fails:", f, g, v, w)
    # The 2-cocycle identity in its invariance form (insensitive to the
    # parity of the cochain):
    #     c(f,{g,h}) = c({f,g},h) + (-1)^{p(f)p(g)} c(g,{f,h}).
    Bs = basis(n, kind, max(amin, -2), min(amax, 2))
    for f in Bs:
        for g in Bs:
            for h in Bs:
                pf, pg = parity(f), parity(g)
                lhs = c(f, bracket(g, h, n, kap))
                r1 = c(bracket(f, g, n, kap), h)
                r2 = c(g, bracket(f, h, n, kap))
                if lhs is None or r1 is None or r2 is None:
                    continue
                t = lhs - r1 - ((-1) ** (pf * pg)) * r2
                if t:
                    coc += 1
                    if verbose and coc < 4:
                        print("   cocycle fails:", f, g, h, t)
    return skew, coc, nz


def scan(kind, ns, ms):
    print(f"=== {kind}:  c(f,g) = (-1)^p(f) Res( f . K_th1..K_thn K_1^m (g) )")
    print(f"    residue = coeff of x^{{{'-1' if kind=='L' else '-1/2'}}}"
          f" theta_1..theta_n\n")
    for n in ns:
        hits = []
        for m in ms:
            try:
                skew, coc, nz = check(n, m, kind)
            except Exception as e:
                print(f"  n={n} m={m:+d}:  ERROR {e}")
                continue
            ok = (skew == 0 and coc == 0 and nz > 0)
            flag = "  <== 2-cocycle, skew, non-zero" if ok else ""
            print(f"  n={n} m={m:+d}:  skew-failures={skew:5d}  "
                  f"cocycle-failures={coc:6d}  non-zero values={nz:5d}{flag}")
            if ok:
                hits.append(m)
        print(f"  --> n={n}: {kind} works for m in {hits}\n")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("L", "both"):
        scan('L', [0, 1, 2, 3], [4, 3, 2, 1, 0, -1])
    if which in ("M", "both"):
        scan('M', [1, 2, 3], [4, 3, 2, 1, 0, -1])
