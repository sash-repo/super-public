"""The zeros of the Shapovalov determinant as the critical exponents.

The bridge asked about: a critical point of a two-dimensional system with
dynamical exponent 1 carries an action of vir (or of a superconformal algebra),
the space of states is a sum of highest-weight modules, and a module is
reducible exactly where Sh vanishes.  Whence:

  * the admissible conformal weights -- the critical exponents -- ARE the zeros
    of the Shapovalov determinant;
  * the level of the singular vector is the order of the differential equation
    the correlators satisfy;
  * the multiplicity of a zero counts the singular vectors, i.e., the equations.

What is checked here, all in exact rational arithmetic:

  A) vir.  With c = c(p,q) = 1 - 6(p-q)^2/(pq), the zeros of phi_{r,s} of
     eq. (14) as printed in Subsec. 3.2 are exactly the entries
         h_{r,s} = ((rq-sp)^2 - (p-q)^2)/(4pq)
     of the table of the minimal models.  Verified as an identity in p and q,
     and over a grid of coprime pairs.

  B) The level of the singular vector at h_{r,s} is rs: the determinant of the
     Shapovalov form, built from the commutation relations from scratch,
     vanishes at (h_{r,s}, c(p,q)) for level n iff n >= rs; and the order of
     vanishing is the one (14) prescribes.

  C) The exponent tables of the first universality classes come out.

  D) ns(1).  With c = 1 - 8/(m(m+2)) IN THE NORMALIZATION OF (15) -- the
     central charge of the standard normalization is 3c/2, see the Statement of
     Subsec. 3.4 -- the zeros of phi_{k,s} of eq. (15) are the entries
         h_{r,s} = (((m+2)r - ms)^2 - 4)/(8m(m+2)) + (1-(-1)^{r-s})/32
     of the table of the superminimal models, the sector being the
     Neveu-Schwarz one for r-s even and the Ramond one for r-s odd.
     Verified as an identity in m.

  E) The factor h - c/24 of item 3) of the Statement of Subsec. 3.4 bounds the
     Ramond sector from below: h_{r,s} >= c/24 for every entry, with equality
     nowhere in the interior -- the ground state of the Ramond sector.

  F) The trap the whole bridge turns on: substituting the central charge of the
     standard (Virasoro) normalization into (15) gives a false formula
     already at level 3/2.
"""

import sys
from fractions import Fraction
from math import gcd

import sympy

from verma import Vir, NS1, Module, h, c

LOG = open("run-crit.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


FAIL = []


def check(name, ok, detail=""):
    say("   %-58s %s%s" % (name, "ok" if ok else "FAIL", ("   " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
#  A) vir: the printed phi_{k,s} and the table of the minimal models
# ---------------------------------------------------------------------------

def phi_vir(k, s, hh, cc):
    """The printed phi_{k,s} of Subsec. 3.2: the monic quadratic in h whose
       roots are the two printed expressions."""
    if k == s:
        return hh + sympy.Rational(1, 24) * (k ** 2 - 1) * (cc - 1)
    base = sympy.Rational(1, 48) * ((13 - cc) * (k ** 2 + s ** 2) - 24 * k * s + 2 * cc - 2)
    dev = sympy.Rational(1, 48) * sympy.sqrt(cc ** 2 - 26 * cc + 25) * (k ** 2 - s ** 2)
    return sympy.expand((hh - base - dev) * (hh - base + dev))


def c_min(p, q):
    """the central charge of the minimal model (p, q)."""
    return 1 - sympy.Rational(6, 1) * (p - q) ** 2 / (p * q)


def h_min(r, s, p, q):
    """the entry (r, s) of the table of the minimal model (p, q)."""
    return (sympy.Rational(1, 1) * ((r * q - s * p) ** 2 - (p - q) ** 2)) / (4 * p * q)


def run_vir_table():
    say("")
    say("=" * 72)
    say("A) vir: the zeros of the printed phi_{r,s} vs the minimal models")
    say("=" * 72)

    # --- as an identity in p, q ------------------------------------------
    P, Q = sympy.symbols("p q", positive=True)
    cc = 1 - 6 * (P - Q) ** 2 / (P * Q)
    # sqrt(c^2-26c+25) = sqrt((c-1)(c-25)) = 6|p^2-q^2|/(pq); take p < q below,
    # so the radicand is a square and the sign is fixed.
    rad = sympy.simplify(cc ** 2 - 26 * cc + 25)
    want = (6 * (Q ** 2 - P ** 2) / (P * Q)) ** 2
    check("the radicand (c-1)(c-25) is a perfect square in p, q",
          sympy.simplify(rad - want) == 0)

    say("")
    say("   the identity phi_{r,s}(h_{r,s}, c(p,q)) = 0 in p and q:")
    for (r, s) in [(1, 1), (1, 2), (2, 1), (1, 3), (2, 3), (3, 2), (2, 5), (3, 4)]:
        hh = ((r * Q - s * P) ** 2 - (P - Q) ** 2) / (4 * P * Q)
        # the root with the sign that the perfect square fixes
        base = sympy.Rational(1, 48) * ((13 - cc) * (r ** 2 + s ** 2) - 24 * r * s + 2 * cc - 2)
        dev = sympy.Rational(1, 48) * (6 * (Q ** 2 - P ** 2) / (P * Q)) * (r ** 2 - s ** 2)
        val = sympy.simplify((hh - base - dev) * (hh - base + dev))
        check("  (r,s) = (%d,%d)" % (r, s), val == 0)

    # --- over a grid ------------------------------------------------------
    say("")
    say("   over the grid of coprime pairs 2 <= p < q <= 13, every table entry:")
    n_pairs = n_entries = 0
    bad = []
    for p in range(2, 14):
        for q in range(p + 1, 14):
            if gcd(p, q) != 1:
                continue
            n_pairs += 1
            cc0 = c_min(p, q)
            for r in range(1, p):
                for s in range(1, q):
                    n_entries += 1
                    hh0 = h_min(r, s, p, q)
                    v = sympy.nsimplify(sympy.expand(phi_vir(r, s, hh0, cc0)))
                    if sympy.simplify(v) != 0:
                        bad.append((p, q, r, s, v))
    check("%d pairs, %d table entries, every phi_{r,s}(h_{r,s}, c) = 0"
          % (n_pairs, n_entries), not bad, "" if not bad else str(bad[:3]))


# ---------------------------------------------------------------------------
#  B) the level of the singular vector is rs
# ---------------------------------------------------------------------------

def K(n, tab={}):
    """the classical partition function, called K in the paper."""
    if n in tab:
        return tab[n]
    t = [1] + [0] * n
    for k in range(1, n + 1):
        for m in range(k, n + 1):
            t[m] += t[m - k]
    tab[n] = t[n]
    return t[n]


def run_vir_levels(NMAX=4):
    say("")
    say("=" * 72)
    say("B) the singular vector at h_{r,s} sits at level rs")
    say("=" * 72)
    alg = Vir(NMAX + 1)
    M = Module(alg)

    dets = {}
    for n in range(1, NMAX + 1):
        b, G = M.gram(n)
        dets[n] = sympy.factor(G.det())
        say("   level %d: dim %d,  det = %s" % (n, len(b), dets[n]))

    # The table has the symmetry h_{r,s} = h_{p-r,q-s}, so one weight is carried
    # by two cells and the first singular vector sits at the smaller of the two
    # levels: the level is min(rs, (p-r)(q-s)).
    say("")
    n_cells = 0
    bad = []
    for (p, q) in [(3, 4), (4, 5), (5, 6), (2, 5)]:
        cc0 = c_min(p, q)
        for r in range(1, p):
            for s in range(1, q):
                lev = min(r * s, (p - r) * (q - s))
                if lev > NMAX:
                    continue
                n_cells += 1
                hh0 = h_min(r, s, p, q)
                for n in range(1, NMAX + 1):
                    van = (sympy.simplify(dets[n].subs({h: hh0, c: cc0})) == 0)
                    if van != (n >= lev):
                        bad.append((p, q, r, s, n, van, lev))
    check("%d cells of (3,4), (4,5), (5,6), (2,5): det_n = 0 iff "
          "n >= min(rs, (p-r)(q-s))" % n_cells, not bad,
          "" if not bad else str(bad[:3]))

    # the order of vanishing at the Ising point
    say("")
    say("   the order of vanishing at c = 1/2 (the Ising model):")
    cc0 = c_min(3, 4)
    eps = sympy.symbols("eps")
    for (r, s) in [(1, 1), (1, 2), (1, 3)]:
        hh0 = h_min(r, s, 3, 4)
        for n in range(1, NMAX + 1):
            d = sympy.simplify(dets[n].subs({c: cc0}))
            if sympy.simplify(d.subs({h: hh0})) != 0:
                continue
            k = 0
            f = sympy.expand(d.subs({h: hh0 + eps}))
            while sympy.simplify(f.subs({eps: 0})) == 0:
                f = sympy.simplify(sympy.cancel(f / eps))
                k += 1
            # what (14) prescribes: sum over i <= n of K(n-i) times the
            # number of unordered pairs {k,l} with kl = i and h_{k,l} = h_{r,s}
            pred = 0
            for i in range(1, n + 1):
                mult = 0
                for a in range(1, i + 1):
                    if i % a == 0 and a <= i // a:
                        bb = i // a
                        if sympy.simplify(phi_vir(a, bb, hh0, cc0)) == 0:
                            mult += 1
                pred += K(n - i) * mult
            check("  h_{%d,%d} = %s at level %d: order %d" % (r, s, hh0, n, k),
                  k == pred, "eq. (14) predicts %d" % pred)


# ---------------------------------------------------------------------------
#  C) the exponent tables
# ---------------------------------------------------------------------------

NAMED = [(3, 4, "Ising"), (4, 5, "tricritical Ising"), (5, 6, "3-state Potts"),
         (2, 5, "Yang--Lee")]


def run_tables():
    say("")
    say("=" * 72)
    say("C) the exponents the zeros of (14) yield")
    say("=" * 72)
    for (p, q, name) in NAMED:
        cc0 = c_min(p, q)
        ws = sorted({h_min(r, s, p, q) for r in range(1, p) for s in range(1, q)},
                    key=lambda x: sympy.Rational(x))
        say("   (p,q) = (%d,%d)  c = %s   %-18s  h in {%s}"
            % (p, q, cc0, name, ", ".join(str(w) for w in ws)))
    known = {(3, 4): {sympy.Integer(0), sympy.Rational(1, 16), sympy.Rational(1, 2)},
             (4, 5): {sympy.Integer(0), sympy.Rational(1, 10), sympy.Rational(3, 5),
                      sympy.Rational(3, 2), sympy.Rational(7, 16), sympy.Rational(3, 80)},
             (2, 5): {sympy.Integer(0), sympy.Rational(-1, 5)}}
    for (p, q), want in known.items():
        got = {h_min(r, s, p, q) for r in range(1, p) for s in range(1, q)}
        check("the table of (%d,%d) is the known one" % (p, q), got == want,
              "" if got == want else "got %s" % sorted(got, key=sympy.Rational))


# ---------------------------------------------------------------------------
#  D), E), F)  ns(1): the superminimal models
# ---------------------------------------------------------------------------

def phi_ns(k, s, hh, cc):
    """The printed phi_{k,s} of Subsec. 3.3; cc is the central charge in the
       normalization of (15), the standard one being 3cc/2."""
    if k == s:
        return hh + sympy.Rational(1, 16) * (k ** 2 - 1) * (cc - 1)
    base = sympy.Rational(1, 32) * ((5 - cc) * (k ** 2 + s ** 2) - 8 * k * s + 2 * cc - 2)
    dev = sympy.Rational(1, 32) * sympy.sqrt(cc ** 2 - 10 * cc + 9) * (k ** 2 - s ** 2)
    return sympy.expand((hh - base - dev) * (hh - base + dev))


def phi_ns_R(k, s, hh, cc):
    """The printed phi^R_{k,s} of eq. (24): the phi_{k,s} of (15) with its
       constant term 2c-2 replaced by 2c."""
    base = sympy.Rational(1, 32) * ((5 - cc) * (k ** 2 + s ** 2) - 8 * k * s + 2 * cc)
    dev = sympy.Rational(1, 32) * sympy.sqrt(cc ** 2 - 10 * cc + 9) * (k ** 2 - s ** 2)
    return sympy.expand((hh - base - dev) * (hh - base + dev))


def run_ns_table():
    say("")
    say("=" * 72)
    say("D) ns(1): the zeros of the printed phi_{k,s} vs the superminimal models")
    say("   (c is the one of (15); the standard normalization is 3c/2)")
    say("=" * 72)
    m = sympy.symbols("m", positive=True)
    cc = 1 - 8 / (m * (m + 2))
    rad = sympy.simplify(cc ** 2 - 10 * cc + 9)
    want = (8 * (m + 1) / (m * (m + 2))) ** 2
    check("the radicand (c-1)(c-9) is a perfect square in m",
          sympy.simplify(rad - want) == 0,
          "= (8(m+1)/(m(m+2)))^2")

    say("")
    say("   the identity phi_{r,s}(h_{r,s}, c(m)) = 0 in m:")
    say("   an entry with r-s even is a Neveu-Schwarz one and is a zero of (15);")
    say("   an entry with r-s odd is a Ramond one and is a zero of (24).")
    for (r, s) in [(1, 1), (1, 3), (3, 1), (2, 2), (1, 5), (2, 4), (3, 3), (4, 2),
                   (1, 2), (2, 1), (2, 3), (3, 2), (1, 4), (3, 4), (2, 5)]:
        hh = (((m + 2) * r - m * s) ** 2 - 4) / (8 * m * (m + 2))
        odd = bool((r - s) % 2)
        tail = 2 * cc if odd else 2 * cc - 2
        if odd:
            hh = hh + sympy.Rational(1, 16)
        base = sympy.Rational(1, 32) * ((5 - cc) * (r ** 2 + s ** 2) - 8 * r * s + tail)
        dev = sympy.Rational(1, 32) * (8 * (m + 1) / (m * (m + 2))) * (r ** 2 - s ** 2)
        val = sympy.simplify((hh - base - dev) * (hh - base + dev))
        check("  (r,s) = (%d,%d)  %s" % (r, s, "R " if odd else "NS"), val == 0)

    # the tricritical Ising point, m = 3
    say("")
    say("   m = 3 (the tricritical Ising model): c = %s, i.e. 3c/2 = %s"
        % (1 - sympy.Rational(8, 15), sympy.Rational(3, 2) * (1 - sympy.Rational(8, 15))))
    ns, ram = set(), set()
    for r in range(1, 3):
        for s in range(1, 5):
            v = (sympy.Rational(((3 + 2) * r - 3 * s) ** 2 - 4, 8 * 3 * 5))
            if (r - s) % 2:
                ram.add(v + sympy.Rational(1, 16))
            else:
                ns.add(v)
    say("      NS: %s      R: %s" % (sorted(ns, key=sympy.Rational),
                                     sorted(ram, key=sympy.Rational)))
    check("the NS weights are {0, 1/10}", ns == {sympy.Integer(0), sympy.Rational(1, 10)})
    check("the R weights are {3/80, 7/16}",
          ram == {sympy.Rational(3, 80), sympy.Rational(7, 16)})
    # The vir-content of the ns-module of weight h is {h, h+1/2}, except for the
    # vacuum, where G_{-1/2}v = 0 and the partner is G_{-3/2}v, of weight 3/2.
    both = set(ns) | ram | {x + sympy.Rational(1, 2) for x in ns if x != 0} \
        | ({sympy.Rational(3, 2)} if 0 in ns else set())
    check("the ns weights, their vir-partners and R give the vir table of (4,5)",
          both == {h_min(r, s, 4, 5) for r in range(1, 4) for s in range(1, 5)},
          "" if both == {h_min(r, s, 4, 5) for r in range(1, 4) for s in range(1, 5)}
          else str(sorted(both, key=sympy.Rational)))

    # E) the Ramond ground state
    say("")
    say("=" * 72)
    say("E) the factor h - c/24 of item 3): the Ramond sector is bounded below")
    say("=" * 72)
    worst = None
    for mm in range(3, 20):
        cc0 = 1 - sympy.Rational(8, mm * (mm + 2))
        for r in range(1, mm):
            for s in range(1, mm + 2):
                if (r - s) % 2 == 0:
                    continue
                hh0 = sympy.Rational(((mm + 2) * r - mm * s) ** 2 - 4,
                                     8 * mm * (mm + 2)) + sympy.Rational(1, 16)
                d = hh0 - cc0 / 24
                if worst is None or d < worst[0]:
                    worst = (d, mm, r, s)
    check("h >= c/24 for every Ramond entry, 3 <= m <= 19",
          worst[0] >= 0, "the smallest margin is %s, at m=%d, (r,s)=(%d,%d)"
          % (worst[0], worst[1], worst[2], worst[3]))

    # F) the trap
    say("")
    say("=" * 72)
    say("F) the same formula fed the central charge of the standard normalization")
    say("=" * 72)
    alg = NS1(4)
    M = Module(alg)
    b, G = M.gram(sympy.Rational(3, 2))
    det = sympy.factor(G.det())
    say("   ns(1), level 3/2: dim %d,  det = %s" % (len(b), det))
    # (18) at level 3/2 is psi_1^{K_1(2)} psi_3^{K_1(0)} = phi_{1,1} phi_{1,3}
    cs = sympy.Rational(2, 3) * c          # the c of (15)
    good = sympy.expand(phi_ns(1, 1, h, cs) * phi_ns(1, 3, h, cs))
    bad = sympy.expand(phi_ns(1, 1, h, c) * phi_ns(1, 3, h, c))
    r_good = sympy.simplify(sympy.cancel(det / good))
    r_bad = sympy.simplify(sympy.cancel(det / bad))
    check("with c of (15), i.e. 2/3 of the standard one: det/(18) is a scalar",
          r_good.is_number, "= %s" % r_good)
    check("with the standard c substituted: det/(18) is NOT a scalar",
          not r_bad.is_number, "= %s" % sympy.simplify(r_bad))


# ---------------------------------------------------------------------------

def main():
    say("the zeros of the Shapovalov determinant as the critical exponents")
    run_vir_table()
    run_vir_levels()
    run_tables()
    run_ns_table()
    say("")
    say("=" * 72)
    if FAIL:
        say("FAILURES: %d" % len(FAIL))
        for f in FAIL:
            say("   %s" % f)
    else:
        say("all checks passed")
    say("=" * 72)


if __name__ == "__main__":
    main()
