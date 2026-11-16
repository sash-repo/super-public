#!/usr/bin/env python3
"""
multdeg.py -- the multiplicities in SM_{mu_1} (x) SM_{mu_2} for ALL weights,
including the degenerate ones.  Cf. Section 7 and Problem 10.2.1 of the text.

Everything is exact (fractions.Fraction); nothing is reused from the other
engines of this directory: the modules, the two odd generators and the whole
linear system are built from scratch out of the formulas of Subsection 7.2 of
the text,

    SM_mu = C[t, theta],   Q_- = d_theta - theta d_t,   Q_+ = t d_theta - t theta d_t - mu theta,

i.e., on the basis  t^a  (level 2a, even),  t^a theta  (level 2a+1, odd):

    Q_-(t^a)        = -a t^{a-1} theta          Q_-(t^a theta)        = t^a
    Q_+(t^a)        = -(a+mu) t^a theta         Q_+(t^a theta)        = t^{a+1}

and  H = {Q_-,Q_+}  acts on the level n by -(mu+n).

WHAT IS COMPUTED.  For d := nu - mu_1 - mu_2 in Z_{>=0},

    dim Hom_{osp(1|2)} (SM_{mu_1} (x) SM_{mu_2},  SM_nu).

A morphism is graded: it maps the level n of the tensor product to the level
n-d of SM_nu, so its parity is forced to be d mod 2 (the level determines the
parity in both modules), which is why only dimensions are printed: the
superdimension is (dim|0) for d even and (0|dim) for d odd.

The unknowns are the components phi_n: T_n -> (SM_nu)_{n-d} for d <= n <= N;
the equations are phi(Q_+- u) = (-1)^d Q_+- phi(u) for every basis vector u of
every level.  Since Q_-: T_n ->> T_{n-1} is onto and Q_-: (SM_nu)_j -> _{j-1}
is bijective for j >= 1, a morphism is determined by phi_d, so dim <= d+1 and
the answer stabilizes in N; the stabilization is verified, not assumed.
"""

import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace  # noqa: E402


# ---------------------------------------------------------------- one factor

def sm_basis(n):
    """Basis of the level n of SM_mu: [(a,e)] with 2a+e = n."""
    return [(n // 2, n % 2)]


def q_minus(a, e):
    """Q_- on the basis vector t^a theta^e; returns list of (coef,(a,e))."""
    if e == 0:
        return [] if a == 0 else [(F(-a), (a - 1, 1))]
    return [(F(1), (a, 0))]


def q_plus(a, e, mu):
    if e == 0:
        c = -(F(a) + mu)
        return [] if c == 0 else [(c, (a, 1))]
    return [(F(1), (a + 1, 0))]


# ------------------------------------------------------------ tensor product

def t_basis(n):
    """Basis of the level n of SM_{mu_1} (x) SM_{mu_2}: (a1,e1,a2,e2)."""
    out = []
    for e1 in (0, 1):
        for e2 in (0, 1):
            rest = n - e1 - e2
            if rest < 0 or rest % 2:
                continue
            for a1 in range(rest // 2 + 1):
                a2 = rest // 2 - a1
                out.append((a1, e1, a2, e2))
    return sorted(out)


def t_apply(op, v, mu1, mu2):
    """op in {'-','+'}; v = (a1,e1,a2,e2); returns list of (coef, vector)."""
    a1, e1, a2, e2 = v
    out = []
    first = q_minus(a1, e1) if op == '-' else q_plus(a1, e1, mu1)
    for c, (b1, f1) in first:
        out.append((c, (b1, f1, a2, e2)))
    sign = F(-1) ** e1                      # (-1)^{p(x)}, p(x) = e1
    second = q_minus(a2, e2) if op == '-' else q_plus(a2, e2, mu2)
    for c, (b2, f2) in second:
        out.append((sign * c, (a1, e1, b2, f2)))
    return out


# ---------------------------------------------------------- the target SM_nu

def w_minus(j):
    """Q_-(w_j) = coef * w_{j-1}, w_{2a}=t^a, w_{2a+1}=t^a theta."""
    a, e = j // 2, j % 2
    if e == 0:
        return F(0) if a == 0 else F(-a)
    return F(1)


def w_plus(j, nu):
    """Q_+(w_j) = coef * w_{j+1}."""
    a, e = j // 2, j % 2
    if e == 0:
        return -(F(a) + nu)
    return F(1)


# ------------------------------------------------------------- the Hom space

def hom_dim(mu1, mu2, d, extra=10):
    """dim Hom(SM_{mu1} (x) SM_{mu2}, SM_{mu1+mu2+d}) with the cutoff d+extra."""
    if d < 0:
        return 0
    nu = mu1 + mu2 + d
    N = d + extra
    sign = F(-1) ** (d % 2)                 # (-1)^{p(phi)}

    idx, bases = {}, {}
    col = 0
    for n in range(d, N + 1):
        bases[n] = t_basis(n)
        for i, v in enumerate(bases[n]):
            idx[(n, v)] = col
            col += 1
    ncols = col

    rows = []
    for n in range(d, N + 1):
        for i, v in enumerate(bases[n]):
            here = idx[(n, v)]
            # (A)  phi(Q_- v) = (-1)^d Q_- phi(v)
            row = [F(0)] * ncols
            for c, w in t_apply('-', v, mu1, mu2):
                if n - 1 >= d:
                    row[idx[(n - 1, w)]] += c
            row[here] -= sign * w_minus(n - d)
            if any(row):
                rows.append(row)
            # (B)  phi(Q_+ v) = (-1)^d Q_+ phi(v)   (dropped at the cutoff)
            if n + 1 <= N:
                row = [F(0)] * ncols
                for c, w in t_apply('+', v, mu1, mu2):
                    row[idx[(n + 1, w)]] += c
                row[here] -= sign * w_plus(n - d, nu)
                if any(row):
                    rows.append(row)
    if not rows:
        return ncols
    return len(nullspace(rows))


def hom_dim_stable(mu1, mu2, d):
    """Same, with the stabilization in the cutoff verified."""
    a = hom_dim(mu1, mu2, d, extra=8)
    b = hom_dim(mu1, mu2, d, extra=14)
    if a != b:
        c = hom_dim(mu1, mu2, d, extra=20)
        if b != c:
            raise RuntimeError(f"not stable: {mu1},{mu2},{d}: {a},{b},{c}")
        return c
    return b


# ------------------------------------------------------------------ the rule

def predicted(mu1, mu2, d):
    """The rule conjectured from the data; 1 generically, 2 in the window."""
    if d < 0:
        return 0
    deg1 = (mu1 == int(mu1) and mu1 <= 0)
    deg2 = (mu2 == int(mu2) and mu2 <= 0)
    if not (deg1 and deg2):
        return 1
    m1, m2 = -int(mu1), -int(mu2)
    lo = 2 * max(m1, m2) + 1
    hi = 2 * (m1 + m2) + 1
    return 2 if lo <= d <= hi else 1


# ---------------------------------------------------------------------- runs

def part1_reproduce(out):
    """Part I: reproduce the nine exceptional pairs of the table (154)."""
    print("=" * 78, file=out)
    print("PART I.  The 169 pairs of eq. (154): (mu1,mu2) in (Z/2 cap [-3,3])^2,",
          file=out)
    print("         the shifts d = nu-mu1-mu2 = 0..6.  Printed: the failing pairs.",
          file=out)
    print("=" * 78, file=out)
    halves = [F(k, 2) for k in range(-6, 7)]
    bad = []
    for mu1 in halves:
        for mu2 in halves:
            dims = [hom_dim_stable(mu1, mu2, d) for d in range(7)]
            if any(x != 1 for x in dims):
                bad.append((mu1, mu2, dims))
    for mu1, mu2, dims in bad:
        print(f"  ({mu1}, {mu2}): " + ", ".join(str(x) for x in dims), file=out)
    print(f"  -> {len(bad)} pairs out of 169 fail; "
          f"the text says nine (for the orders k <= 3, i.e., d <= 6).", file=out)
    return bad


def part2_window(out):
    """Part II: the whole degenerate quadrant, far beyond d = 6."""
    print("=" * 78, file=out)
    print("PART II.  mu_i = -m_i, 0 <= m_1 <= m_2 <= 4, d = 0 .. 2(m1+m2)+4.", file=out)
    print("          Printed: the set of d with dim > 1, and the predicted window",
          file=out)
    print("          [2max(m1,m2)+1, 2(m1+m2)+1].", file=out)
    print("=" * 78, file=out)
    ok = True
    for m1 in range(5):
        for m2 in range(m1, 5):
            mu1, mu2 = F(-m1), F(-m2)
            dmax = 2 * (m1 + m2) + 4
            dims = {d: hom_dim_stable(mu1, mu2, d) for d in range(dmax + 1)}
            big = [d for d in range(dmax + 1) if dims[d] > 1]
            vals = sorted(set(dims[d] for d in big))
            lo, hi = 2 * max(m1, m2) + 1, 2 * (m1 + m2) + 1
            pred = list(range(lo, hi + 1))
            good = (big == pred) and (vals in ([], [2]))
            ok = ok and good
            print(f"  (m1,m2)=({m1},{m2}): dims {[dims[d] for d in range(dmax+1)]}",
                  file=out)
            print(f"      d with dim>1: {big};  predicted [{lo},{hi}] -> {pred};"
                  f"  values {vals}  {'OK' if good else '*** MISMATCH ***'}", file=out)
    print(f"  -> {'all agree' if ok else 'MISMATCH'}", file=out)
    return ok


def part3_nondegenerate(out):
    """Part III: mu1+mu2 in Z_{<=0} but not both mu_i in Z_{<=0}."""
    print("=" * 78, file=out)
    print("PART III.  The other half of the exceptional set of Corollary 7.2.6:", file=out)
    print("           mu1+mu2 in Z_{<=0}, but NOT both mu_i in Z_{<=0}.", file=out)
    print("           Claim: multiplicity-free all the same.", file=out)
    print("=" * 78, file=out)
    cases = []
    # half-integers, thirds, and a few random-looking rationals
    for num in range(-7, 8, 2):                       # mu1 in Z + 1/2
        mu1 = F(num, 2)
        for s in range(0, -5, -1):                    # mu1 + mu2 = s in Z_{<=0}
            cases.append((mu1, F(s) - mu1, s))
    for mu1 in (F(1, 3), F(-2, 3), F(5, 3), F(7, 4), F(-11, 4), F(2), F(3), F(-1, 5)):
        for s in range(0, -4, -1):
            cases.append((mu1, F(s) - mu1, s))
    ok = True
    for mu1, mu2, s in cases:
        both = all(x == int(x) and x <= 0 for x in (mu1, mu2))
        if both:
            continue
        dims = [hom_dim_stable(mu1, mu2, d) for d in range(9)]
        good = all(x == 1 for x in dims)
        ok = ok and good
        if not good:
            print(f"  *** ({mu1},{mu2}), sum {s}: {dims}", file=out)
    print(f"  -> {len(cases)} pairs tested for d <= 8: "
          f"{'every one multiplicity-free' if ok else 'A FAILURE, see above'}", file=out)
    return ok


def part4_generic(out):
    """Part IV: a sanity sweep of the generic range (Corollary 7.2.6 itself)."""
    print("=" * 78, file=out)
    print("PART IV.  Sanity: mu1+mu2 NOT in Z_{<=0} -- Corollary 7.2.6 says dim = 1.",
          file=out)
    print("=" * 78, file=out)
    ok = True
    for mu1 in (F(1, 2), F(3, 2), F(1), F(2), F(1, 3), F(-5, 2), F(-3)):
        for mu2 in (F(1, 2), F(1), F(7, 3), F(-1, 2), F(-2)):
            if (mu1 + mu2) == int(mu1 + mu2) and mu1 + mu2 <= 0:
                continue
            dims = [hom_dim_stable(mu1, mu2, d) for d in range(8)]
            good = all(x == 1 for x in dims)
            ok = ok and good
            if not good:
                print(f"  *** ({mu1},{mu2}): {dims}", file=out)
    print(f"  -> {'agrees with Corollary 7.2.6' if ok else 'CONTRADICTS the Corollary'}",
          file=out)
    return ok


def part5_rule(out):
    """Part V: the rule checked against the computation on a large random-ish set."""
    print("=" * 78, file=out)
    print("PART V.  The rule  dim = 2 iff both mu_i in Z_{<=0} and", file=out)
    print("         2max(m1,m2)+1 <= d <= 2(m1+m2)+1  (m_i = -mu_i),  else 1,", file=out)
    print("         checked against the computation on every pair below.", file=out)
    print("=" * 78, file=out)
    tested = bad = 0
    pool = [F(k, 2) for k in range(-8, 5)] + [F(1, 3), F(-4, 3), F(7, 5)]
    for mu1 in pool:
        for mu2 in pool:
            if mu2 < mu1:
                continue
            for d in range(0, 10):
                got = hom_dim_stable(mu1, mu2, d)
                exp = predicted(mu1, mu2, d)
                tested += 1
                if got != exp:
                    bad += 1
                    print(f"  *** ({mu1},{mu2}), d={d}: computed {got}, rule {exp}",
                          file=out)
    print(f"  -> {tested} values checked, {bad} disagreements", file=out)
    return bad == 0


def part6_clebsch(out):
    """Part VI: the window is the Clebsch-Gordan range, shifted by one."""
    print("=" * 78, file=out)
    print("PART VI.  Reading of the window: for mu_i = -m_i, the exceptional", file=out)
    print("          nu = mu1+mu2+d run over  |m1-m2|+1, ..., m1+m2+1,  i.e., over", file=out)
    print("          k+1 for the k with F_k a constituent of F_{m1} (x) F_{m2},", file=out)
    print("          where F_m is the finite-dimensional submodule of SM_{-m},", file=out)
    print("          of sdim (m+1|m).  Their number is 2min(m1,m2)+1.", file=out)
    print("=" * 78, file=out)
    ok = True
    for m1 in range(5):
        for m2 in range(m1, 5):
            mu1, mu2 = F(-m1), F(-m2)
            dmax = 2 * (m1 + m2) + 4
            nus = [mu1 + mu2 + d for d in range(dmax + 1)
                   if hom_dim_stable(mu1, mu2, d) > 1]
            cg = [F(k + 1) for k in range(abs(m1 - m2), m1 + m2 + 1)]
            good = nus == cg
            ok = ok and good
            print(f"  (m1,m2)=({m1},{m2}): exceptional nu {[str(x) for x in nus]}"
                  f"   Clebsch-Gordan+1 {[str(x) for x in cg]}"
                  f"   count {len(nus)} = 2min+1 = {2*min(m1,m2)+1}"
                  f"  {'OK' if good else '*** MISMATCH ***'}", file=out)
    print(f"  -> {'the two lists coincide in every case' if ok else 'MISMATCH'}",
          file=out)
    return ok


def main():
    with open("runs-multdeg.txt", "w") as out:
        print("multdeg.py -- multiplicities in SM_{mu1} (x) SM_{mu2}, all weights",
              file=out)
        print("exact rational arithmetic; every dimension verified stable in the cutoff\n",
              file=out)
        bad = part1_reproduce(out)
        r2 = part2_window(out)
        r3 = part3_nondegenerate(out)
        r4 = part4_generic(out)
        r5 = part5_rule(out)
        r6 = part6_clebsch(out)
        print("=" * 78, file=out)
        print(f"SUMMARY: part I {len(bad)} failing pairs; part II {r2}; part III {r3}; "
              f"part IV {r4}; part V {r5}; part VI {r6}", file=out)
        print("=" * 78, file=out)


if __name__ == "__main__":
    main()
