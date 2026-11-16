"""Check the Kac determinant of vir and of ns(1) as printed in Subsecs. 3.2--3.3.

The text prints
    Sh_n(h, c) = prod_{1<=i<=n} (psi_n(h, c))^{K(n-i)},   psi_n = prod_{s|n} phi_{s, n/s},
i.e., with the FIXED index n inside the product.  The classical formula has the
RUNNING index i there.  Both readings are computed here and compared with the
determinant of the Shapovalov form, which is computed from scratch.
"""

import sys
import sympy
from verma import Vir, NS1, Module, h, c

LOG = open("run-%s.log" % (sys.argv[1] if len(sys.argv) > 1 else "vir"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


# ---------------------------------------------------------------- vir
def phi_vir(k, s):
    """the quadratic polynomial in h with the two roots printed in Subsec. 3.2."""
    if k == s:
        return h + sympy.Rational(1, 24) * (k ** 2 - 1) * (c - 1)
    r1 = sympy.Rational(1, 48) * ((13 - c) * (k ** 2 + s ** 2)
                                  + sympy.sqrt(c ** 2 - 26 * c + 25) * (k ** 2 - s ** 2)
                                  - 24 * k * s + 2 * c - 2)
    r2 = sympy.Rational(1, 48) * ((13 - c) * (k ** 2 + s ** 2)
                                  - sympy.sqrt(c ** 2 - 26 * c + 25) * (k ** 2 - s ** 2)
                                  - 24 * k * s + 2 * c - 2)
    return sympy.expand((h - r1) * (h - r2))


def psi_vir(n, unordered=True):
    """psi_n = prod over the divisors s of n.

    ``unordered``: take one factor per unordered pair {s, n/s}.  This is forced
    by the fact that phi_{k,s} = phi_{s,k} already carries BOTH roots h_{k,s}
    and h_{s,k}; running s over all the divisors squares every factor with
    s != n/s.
    """
    out = sympy.Integer(1)
    for s in range(1, n + 1):
        if n % s == 0 and (not unordered or s <= n // s):
            out *= phi_vir(s, n // s)
    return sympy.expand(sympy.simplify(out))


def partitions(n):
    """K(n): the number of partitions of n into positive integers."""
    tab = [1] + [0] * n
    for k in range(1, n + 1):
        for m in range(k, n + 1):
            tab[m] += tab[m - k]
    return tab[n]


def run_vir(NMAX=5):
    say("=" * 72)
    say("Virasoro: the determinant of the Shapovalov form vs. the two readings")
    say("=" * 72)
    alg = Vir(2 * NMAX + 4)
    M = Module(alg)
    for n in range(1, NMAX + 1):
        b, G = M.gram(n)
        det = sympy.factor(sympy.expand(G.det()))
        pr_running = sympy.Integer(1)   # prod_i psi_i^{K(n-i)}   (corrected)
        pr_fixed = sympy.Integer(1)     # prod_i psi_n^{K(n-i)}   (as printed)
        for i in range(1, n + 1):
            pr_running *= psi_vir(i) ** partitions(n - i)
            pr_fixed *= psi_vir(n, unordered=False) ** partitions(n - i)
        pr_running = sympy.factor(sympy.expand(sympy.simplify(pr_running)))
        pr_fixed = sympy.factor(sympy.expand(sympy.simplify(pr_fixed)))
        ok_run = sympy.simplify(sympy.cancel(det / pr_running)).free_symbols == set()
        ok_fix = (sympy.simplify(sympy.cancel(det / pr_fixed)).free_symbols == set()
                  if pr_fixed != 0 else False)
        say("level %d: dim = %d" % (n, len(b)))
        say("   det          = %s" % det)
        say("   prod psi_i   = %s" % pr_running)
        say("   det/prod_i   = %s   -> %s"
            % (sympy.simplify(sympy.cancel(det / pr_running)),
               "AGREES (up to a scalar)" if ok_run else "DIFFERS"))
        say("   det/prod_n   = %s   -> %s"
            % (sympy.simplify(sympy.cancel(det / pr_fixed)),
               "agrees" if ok_fix else "DIFFERS"))


# ---------------------------------------------------------------- ns(1)
def phi_ns(k, s, pref=sympy.Rational(1, 32), tail=None):
    """The printed phi_{k,s} of Subsec. 3.3 has pref = 1/8 and tail = c/2 - 1/2.

    The corrected one -- the exact analogue of the Virasoro phi_{k,s} of
    Subsec. 3.2 -- has pref = 1/32 and tail = 2c - 2.  Here c is the central
    charge of ns(1) normalized so that the Virasoro one is 3c/2 (which is what
    the printed phi_{k,k} = h + (k^2-1)(c-1)/16 already presupposes).
    """
    if tail is None:
        tail = 2 * c - 2
    if k == s:
        return h + sympy.Rational(1, 16) * (k ** 2 - 1) * (c - 1)
    base = pref * ((5 - c) * (k ** 2 + s ** 2) - 8 * k * s + tail)
    dev = pref * sympy.sqrt(c ** 2 - 10 * c + 9) * (k ** 2 - s ** 2)
    return sympy.expand((h - base - dev) * (h - base + dev))


def psi_ns(j, printed=False):
    """psi_j = prod over the unordered pairs {k,s} with ks = j and k+s even."""
    out = sympy.Integer(1)
    for k in range(1, j + 1):
        if j % k == 0 and k <= j // k and (k + j // k) % 2 == 0:
            if printed:
                out *= phi_ns(k, j // k, sympy.Rational(1, 8),
                              sympy.Rational(1, 2) * c - sympy.Rational(1, 2))
            else:
                out *= phi_ns(k, j // k)
    return sympy.expand(sympy.simplify(out))


def K1(n):
    """partitions of n in which every odd summand occurs at most once,
       n running over the half-integers doubled: here n is an integer."""
    tab = [1] + [0] * n
    # even parts: unrestricted; odd parts: at most once
    for k in range(2, n + 1, 2):
        for m in range(k, n + 1):
            tab[m] += tab[m - k]
    for k in range(1, n + 1, 2):
        for m in range(n, k - 1, -1):
            tab[m] += tab[m - k]
    return tab[n]


def run_ns1(NMAX=4):
    say("")
    say("=" * 72)
    say("ns(1): the determinant of the Shapovalov form, levels 1/2 .. %s" % NMAX)
    say("   (the engine's central charge is the Virasoro one; the section's c")
    say("    is the ns-one, the two being related by c_vir = 3c/2)")
    say("=" * 72)
    alg = NS1(2 * NMAX + 6)
    M = Module(alg)
    lev = sympy.Rational(1, 2)
    while lev <= NMAX:
        b, G = M.gram(lev)
        det = sympy.factor(sympy.expand(G.det()))
        det = sympy.factor(det.subs(c, sympy.Rational(3, 2) * c))
        say("")
        say("level %s: dim = %d" % (lev, len(b)))
        say("   det = %s" % det)
        two_n = int(2 * lev)
        pr = sympy.Integer(1)
        pr_pr = sympy.Integer(1)
        for j in range(1, two_n + 1):
            pr *= psi_ns(j) ** K1(two_n - j)
            pr_pr *= psi_ns(j, printed=True) ** K1(two_n - j)
        pr = sympy.factor(sympy.expand(sympy.simplify(pr)))
        pr_pr = sympy.factor(sympy.expand(sympy.simplify(pr_pr)))
        r = sympy.simplify(sympy.cancel(det / pr))
        say("   prod_{1<=j<=2n} psi_j^{K_1(2n-j)} = %s" % pr)
        say("   det / that = %s  -> %s"
            % (r, "AGREES (up to a scalar)" if r.free_symbols == set() else "DIFFERS"))
        rp = sympy.simplify(sympy.cancel(det / pr_pr)) if pr_pr != 0 else None
        say("   with the printed phi: det / that = %s  -> %s"
            % (rp, "agrees" if (rp is not None and rp.free_symbols == set())
               else "DIFFERS"))
        lev += sympy.Rational(1, 2)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "vir"
    if what == "vir":
        run_vir(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        run_ns1(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
