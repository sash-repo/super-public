"""The quadratic Casimir of po(0|2k) for k = 2,3,4, and the invariant form on
the root lattice.  These settle the shift rho of [GL] Theorem 2.
"""
import itertools
import sympy
from poi import Po, Verma, parity

LOG = open("run-cas.log", "w")
CHECKS = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def check(name, ok, note=""):
    CHECKS.append((name, ok))
    say(f"  [{'OK  ' if ok else 'FAIL'}] {name}{('  -- ' + note) if note else ''}")


def dual_pairs(g):
    """The Casimir as a list (a, b, coef): B is a signed permutation matrix."""
    full = (1 << g.N) - 1
    out = []
    for b in g.basis:
        bar = full ^ b
        s = g.form(b, bar)          # B(b, bar b) = +-1 is the (b, bar) entry of G
        assert s, "the form is degenerate on a monomial"
        out.append((bar, b, sympy.Rational(1, s)))
    return out


# --- calibrate the shortcut against the honest matrix inverse, for k = 2
g = Po(2)
G = sympy.Matrix([[g.form(a, b) for b in g.basis] for a in g.basis])
Gi = G.inv()
sparse = {(g.basis.index(a), g.basis.index(b)): c for a, b, c in dual_pairs(g)}
same = all(Gi[i, j] == sparse.get((i, j), 0) for i in range(g.dim) for j in range(g.dim))
check("the signed-permutation shortcut for B^{-1} equals the honest inverse (k=2)", same)

say("\n=== HC(C) for po(0|2k), triangular decomposition of [GS] 2.6.3 ===")
for n in (2, 3, 4):
    g = Po(n)
    V = Verma(g)
    pairs = dual_pairs(g)
    memo = {}
    hc = sympy.Integer(0)
    for a, b, c in pairs:
        hc += c * V.act_word([a, b], {(): sympy.Integer(1)}, memo).get((), 0)
    hc = sympy.expand(hc)
    P = sympy.expand(sum(V.lam_of(J) * V.lam_of(tuple(i for i in range(n) if i not in J))
                         for k in range(n + 1) for J in itertools.combinations(range(n), k)))
    # HC(C) = c0 * (P + linear).  Read off c0 from a quadratic term.
    # read the overall normalization off the term l_empty * l_I
    top = V.hsym[()] * V.hsym[tuple(range(n))]
    c0 = sympy.Poly(hc, *sorted(hc.free_symbols, key=str)).coeff_monomial(top) / 2
    lin = sympy.expand(hc / c0 - P)
    say(f"  k={n}:  HC(C) = {sympy.factor(hc)}")
    say(f"          normalized: HC(C)/({c0}) - sum_J l_J l_{{J*}} = {lin}")
    target = -2 ** (n - 1) * V.hsym[tuple(i for i in range(n) if i != n - 1)]
    check(f"k={n}: the shift of HC(C) is -2^(k-1) l_{{I minus k}}",
          sympy.expand(lin - target) == 0,
          f"predicted {target}")

say("\n=== the invariant form on the root lattice ===")
for n in (2, 3, 4):
    g = Po(n)
    V = Verma(g)
    # eps_i corresponds under B to h_{I minus i}: check it, then pair them
    hmasks = {}
    for k in range(n + 1):
        for J in itertools.combinations(range(n), k):
            hmasks[J] = g.h_elt(J)

    def pair(i, j):
        si, mi = hmasks[tuple(x for x in range(n) if x != i)]
        sj, mj = hmasks[tuple(x for x in range(n) if x != j)]
        return si * sj * g.form(mi, mj)

    M = [[pair(i, j) for j in range(n)] for i in range(n)]
    say(f"  k={n}: Gram matrix of (eps_i, eps_j) = {M}")
    if n >= 3:
        check(f"k={n}: the form vanishes identically on the root lattice",
              all(all(x == 0 for x in row) for row in M),
              "so (beta,beta) and (rho,beta) of [GL] Thm 2 are 0 for every beta in Q")
    else:
        check("k=2: the form does NOT vanish on the root lattice",
              any(any(x for x in row) for row in M))

say("\n" + "=" * 70)
say(f"{sum(1 for _, o in CHECKS if o)}/{len(CHECKS)} checks pass")
for nm, o in CHECKS:
    if not o:
        say("  FAILED:", nm)
LOG.close()
