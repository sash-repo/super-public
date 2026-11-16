"""Does ignoring the Sign Rule change the Shapovalov determinant?

The footnote to condition C) of Subsec. "When is Sh_mu defined" says: since
Sh_mu is defined up to a scalar factor, and the Sign Rule only affects the sign
of Sh_mu, one may ignore the Sign Rule.  This engine settles, by exact
computation over Q(i)[h, c] resp. Q(i)[h, q, c], exactly how much of that is
true, and where ignoring the Sign Rule does bite.

Two notions of anti-automorphism of a Lie superalgebra g:

  (S)  sigma([x, y]) = (-1)^{p(x)p(y)} [sigma(y), sigma(x)]   "with the Sign Rule"
  (N)  s([x, y])     =                 [s(y), s(x)]           "ignoring it"

and their extensions to U(g), on a monomial u = x_1 ... x_k:

  (S)  sigma(u) = (-1)^{sum_{a<b} p_a p_b} sigma(x_k) ... sigma(x_1)
  (N)  s(u)     =                                s(x_k) ... s(x_1)

The Shapovalov form is (u v | w v) = the coefficient of v in <anti>(u) w v, and
Sh_mu = det of its Gram matrix on a weight subspace.

What is checked, for vir, ns(1) and ns(2) -- the three algebras whose
determinants the paper prints:

  1. the map s of the paper is a (N)-anti-automorphism (on every pair of
     generators, central terms included), and is NOT an (S)-one;
  2. sigma := i^{-p} s IS an (S)-anti-automorphism -- so the two notions are
     carried into each other by that twist, and nothing else is needed;
  3. s is involutive, s^2 = id, while sigma is SUPER-involutive,
     sigma^2 = (-1)^p id.  This is the trouble: condition C) asks for an
     involutive anti-automorphism, and which maps qualify depends on the
     convention;
  4. the two Gram matrices differ by a diagonal factor, hence the two
     determinants by a scalar -- computed here, not assumed.  The scalar is a
     4th root of unity, not always a sign;
  5. the symmetry type of the form: (S) gives a supersymmetric form,
     (N) a plain symmetric one.

No claim below is asserted; each is printed with the range over which it was
verified, and a FAIL is loud.
"""

import sys

import sympy

import verma
from verma import NS1, NS2, Vir, Module

I = sympy.I

# --------------------------------------------------------------------------
#  1-3.  the two notions on g itself
# --------------------------------------------------------------------------


def has(alg, key):
    return key in alg._gens


def bracket_vec(alg, k1, k2):
    """[g_{k1}, g_{k2}] as (dict{key: coeff}, central coeff)."""
    g1, g2 = alg.gen(k1), alg.gen(k2)
    terms, cen = alg.bracket(g1, g2)
    out = {}
    for c, k in terms:
        out[k] = sympy.expand(out.get(k, 0) + c)
    return {k: v for k, v in out.items() if v != 0}, sympy.expand(cen)


def apply_map(alg, vec, cen, twist):
    """apply x -> twist(p(x)) * s(x) to (dict, central part).

    The central element is even and fixed by s, so it only picks up twist(0).
    """
    out = {}
    for k, c in vec.items():
        p = alg.gen(k).parity
        sk = alg.sigma(k)
        out[sk] = sympy.expand(out.get(sk, 0) + twist(p) * c)
    return ({k: v for k, v in out.items() if v != 0},
            sympy.expand(twist(0) * cen))


def eq_vec(a, b):
    (v1, c1), (v2, c2) = a, b
    if sympy.expand(c1 - c2) != 0:
        return False
    keys = set(v1) | set(v2)
    return all(sympy.expand(v1.get(k, 0) - v2.get(k, 0)) == 0 for k in keys)


def check_anti(alg, keys, twist, sign_rule, label, margin):
    """Is x -> twist(p(x)) s(x) an anti-automorphism in the given convention?

    Only pairs whose bracket stays inside the truncated generator set are
    tested: the algebras of verma.py are cut off at nmax, and a term dropped by
    that cut-off would read as a violation.
    """
    bad = 0
    tested = 0
    for k1 in keys:
        for k2 in keys:
            g1, g2 = alg.gen(k1), alg.gen(k2)
            if abs(g1.mode) > margin or abs(g2.mode) > margin:
                continue
            lhs = apply_map(alg, *bracket_vec(alg, k1, k2), twist=twist)
            # [f(y), f(x)], with f = twist . s
            sk1, sk2 = alg.sigma(k1), alg.sigma(k2)
            if not (has(alg, sk1) and has(alg, sk2)):
                continue
            rv, rc = bracket_vec(alg, sk2, sk1)
            fac = twist(g1.parity) * twist(g2.parity)
            if sign_rule:
                fac *= (-1) ** (g1.parity * g2.parity)
            rhs = ({k: sympy.expand(fac * c) for k, c in rv.items()},
                   sympy.expand(fac * rc))
            tested += 1
            if not eq_vec(lhs, rhs):
                bad += 1
    return tested, bad, label


def check_involutive(alg, keys, twist):
    """s^2 = id?  and with which sign on odd elements?"""
    plain = supr = 0
    for k in keys:
        g = alg.gen(k)
        sk = alg.sigma(k)
        if not has(alg, sk):
            continue
        # (twist . s)^2 (x) = twist(p) twist(p) s^2(x)
        c = twist(g.parity) ** 2
        if alg.sigma(sk) != k:
            continue
        if sympy.expand(c - 1) == 0:
            plain += 1
        if sympy.expand(c - (-1) ** g.parity) == 0:
            supr += 1
    return plain, supr


# --------------------------------------------------------------------------
#  4-5.  the two Gram matrices
# --------------------------------------------------------------------------


def npar(mod, mono):
    """number of odd factors of a PBW monomial."""
    return sum(mod.alg.gen(k).parity for k in mono)


def sr_factor(P):
    """sigma(u) / s(u) for a monomial with P odd factors.

    sigma extends with (-1)^{sum_{a<b} p_a p_b} = (-1)^{binom(P,2)}, and each
    generator carries its own twist i^{-p}, i.e. i^{-P} in all.
    """
    return (-1) ** (P * (P - 1) // 2) * I ** (-P)


def grams(alg, level, charge=None):
    mod = Module(alg)
    b, Gn = mod.gram(level, charge)
    if not b:
        return None
    D = [sr_factor(npar(mod, u)) for u in b]
    Gs = sympy.zeros(len(b), len(b))
    for i in range(len(b)):
        for j in range(len(b)):
            Gs[i, j] = sympy.expand(D[i] * Gn[i, j])
    return b, Gn, Gs, D, mod


def symmetry(mod, b, G, supers):
    """is G symmetric (supers=False) / supersymmetric (supers=True)?"""
    bad = 0
    for i in range(len(b)):
        for j in range(len(b)):
            s = (-1) ** (npar(mod, b[i]) * npar(mod, b[j])) if supers else 1
            if sympy.expand(G[i, j] - s * G[j, i]) != 0:
                bad += 1
    return bad


def run(name, alg, levels, charges=None, margin=3):
    print("=" * 74)
    print(name)
    print("=" * 74)
    keys = [g.key for g in alg._gens.values()]

    # --- 1-2: the two conventions on g -----------------------------------
    t_id = lambda p: sympy.Integer(1)
    t_i = lambda p: I ** (-p)
    for twist, sr, lab in ((t_id, False, "s  as (N)-anti-automorphism"),
                           (t_id, True, "s  as (S)-anti-automorphism"),
                           (t_i, True, "i^-p s as (S)-anti-automorphism"),
                           (t_i, False, "i^-p s as (N)-anti-automorphism")):
        n, bad, _ = check_anti(alg, keys, twist, sr, lab, margin)
        verdict = "holds" if bad == 0 else "FAILS on %d" % bad
        print("  %-34s %-12s (%d generator pairs)" % (lab, verdict, n))

    # --- 3: involutive vs super-involutive -------------------------------
    p1, s1 = check_involutive(alg, keys, t_id)
    p2, s2 = check_involutive(alg, keys, t_i)
    nodd = sum(1 for g in alg._gens.values() if g.parity == 1)
    print("  s:       involutive on %d gens, super-involutive on %d  (%d odd)"
          % (p1, s1, nodd))
    print("  i^-p s:  involutive on %d gens, super-involutive on %d"
          % (p2, s2))

    # --- 4-5: the determinants -------------------------------------------
    for lev in levels:
        for chg in (charges if charges is not None else [None]):
            got = grams(alg, lev, chg)
            if got is None:
                continue
            b, Gn, Gs, D, mod = got
            dn = sympy.expand(sympy.factor(Gn.det()))
            ds = sympy.expand(sympy.factor(Gs.det()))
            ratio = sympy.simplify(ds / dn) if dn != 0 else None
            pred = sympy.expand(sympy.prod(D))
            ok = "ok" if (ratio is None or
                          sympy.simplify(ratio - pred) == 0) else "MISMATCH"
            sym_n = symmetry(mod, b, Gn, False)
            sym_s = symmetry(mod, b, Gs, False)
            pars = sorted({npar(mod, u) % 2 for u in b})
            scalar = len(set(D)) == 1
            tag = "level %s" % lev + ("" if chg is None else ", charge %s" % chg)
            print("  %-22s dim %2d  det ratio %-10s (predicted %-10s %s)"
                  % (tag, len(b), ratio, pred, ok))
            print("  %-22s   symmetric: (N) %s, (S) %s;  parities %s;  "
                  "D scalar: %s"
                  % ("", "yes" if sym_n == 0 else "NO (%d)" % sym_n,
                     "yes" if sym_s == 0 else "NO (%d)" % sym_s,
                     pars, "yes" if scalar else "NO"))
            if dn == 0:
                print("  %-22s   determinant identically 0" % "")


def main():
    run("vir", Vir(6), [1, 2, 3, 4])
    run("ns(1)", NS1(6), [sympy.Rational(1, 2), 1, sympy.Rational(3, 2), 2,
                          sympy.Rational(5, 2)])
    run("ns(2)", NS2(5, sympy.Symbol("c")), [sympy.Rational(1, 2), 1, sympy.Rational(3, 2), 2],
        charges=[-1, 0, 1], margin=2)


if __name__ == "__main__":
    main()
