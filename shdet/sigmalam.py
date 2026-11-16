"""All anti-involutions of the centrally extended svect^L_lambda(1|2).

Nothing is guessed: the root grading of Subsec. (68) fixes the shape of
sigma (a generator can only go to the sum of the generators of the opposite
root), the defining equations

     sigma([x, y]) = [sigma(y), sigma(x)]      (the Sign Rule ignored,
                                                which is the paper's choice)

are generated for every pair of basis elements in a window of modes, and the
resulting polynomial system is solved.  The solver is the naive one that keeps
substituting an unknown as soon as some equation determines it rationally;
what is left at the end is printed, so nothing is silently dropped.

The output is the anti-involution used in the Shapovalov computation of
``shsvect.py``, together with the value sigma(z) = zeta * z on the centre.
"""

import itertools
import sys

import sympy

from svectlam import (NAMES, PARITY, ROOT, LAM, KAP, br, root)

MODES = list(range(-3, 4))

# ---------------------------------------------------------------------------
#  the shape of sigma, forced by the roots
# ---------------------------------------------------------------------------
#  root(name, m) = (m, b, q); sigma sends it to the space of root (-m, -b, -q).
#  (m, 0, 0)  <- LL_{-m}, T0_{-m}      (m, 0, 2) <- Tp_{-m}   (m,0,-2) <- Tm_{-m}
#  (m, 1, 1)  <- S1_{-m}               (m,-1,-1) <- P1_{-m}
#  (m, 1,-1)  <- S2_{-m}               (m,-1, 1) <- P2_{-m}

TARGETS = {
    "LL": ("LL", "T0"),
    "T0": ("LL", "T0"),
    "Tp": ("Tm",),
    "Tm": ("Tp",),
    "S1": ("P1",),
    "S2": ("P2",),
    "P1": ("S1",),
    "P2": ("S2",),
}

UNK = {}


def u(name, m, tgt):
    k = (name, m, tgt)
    if k not in UNK:
        UNK[k] = sympy.Symbol("%s_%s_%s" % (name, str(m).replace("-", "m"), tgt))
    return UNK[k]


ZETA = sympy.Symbol("zeta")


def sigma(key):
    """sigma(x) as {(name, m): coeff}."""
    name, m = key
    return {(t, -m): u(name, m, t) for t in TARGETS[name]}


def sigma_of(terms, cen):
    """sigma applied to an element (terms, central part)."""
    out, oc = {}, ZETA * cen
    for k, v in terms.items():
        for k2, v2 in sigma(k).items():
            out[k2] = out.get(k2, 0) + v * v2
    return out, oc


def bracket_of(A, B):
    """[A, B] for A, B given as {(name,m): coeff}; returns (terms, central)."""
    out, cen = {}, sympy.Integer(0)
    for ka, va in A.items():
        for kb, vb in B.items():
            t, c = br(ka, kb)
            for k2, v2 in t.items():
                out[k2] = out.get(k2, 0) + va * vb * v2
            cen += va * vb * c
    return out, cen


def in_window(key):
    return key[1] in MODES


def equations():
    eqs = []
    gens = [(n, m) for n in NAMES for m in MODES]
    for x in gens:
        for y in gens:
            t, c = br(x, y)
            if any(not in_window(k) for k in t):
                continue
            lhs_t, lhs_c = sigma_of(t, c)
            rhs_t, rhs_c = bracket_of(sigma(y), sigma(x))
            keys = set(lhs_t) | set(rhs_t)
            for k in keys:
                e = sympy.expand(lhs_t.get(k, 0) - rhs_t.get(k, 0))
                if e != 0:
                    eqs.append(e)
            e = sympy.expand(lhs_c - rhs_c)
            if e != 0:
                eqs.append(e)
    return eqs


def involutive():
    """sigma^2 = id."""
    eqs = []
    for n in NAMES:
        for m in MODES:
            if -m not in MODES:
                continue
            comp = {}
            for k, v in sigma((n, m)).items():
                for k2, v2 in sigma(k).items():
                    comp[k2] = comp.get(k2, 0) + v * v2
            for k, v in comp.items():
                tgt = 1 if k == (n, m) else 0
                e = sympy.expand(v - tgt)
                if e != 0:
                    eqs.append(e)
    eqs.append(sympy.expand(ZETA ** 2 - 1))
    return eqs


# ---------------------------------------------------------------------------
#  a naive propagating solver
# ---------------------------------------------------------------------------

def solve_system(eqs, unknowns, log):
    sub = {}
    eqs = [sympy.expand(e) for e in eqs]
    changed = True
    while changed:
        changed = False
        eqs = [sympy.expand(e.subs(sub)) for e in eqs]
        eqs = [e for e in eqs if e != 0]
        for e in eqs:
            free = [s for s in unknowns if s in e.free_symbols and s not in sub]
            if len(free) != 1:
                continue
            s = free[0]
            p = sympy.Poly(e, s)
            if p.degree() != 1:
                continue
            a, b = p.all_coeffs()
            if sympy.simplify(a) == 0:
                continue
            val = sympy.cancel(-b / a)
            sub[s] = sympy.simplify(val)
            changed = True
            break
    eqs = [sympy.simplify(sympy.expand(e.subs(sub))) for e in eqs]
    eqs = [e for e in eqs if e != 0]
    return sub, eqs


if __name__ == "__main__":
    log = open("run-sigmalam.log", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")
        log.flush()

    say("=== the shape of sigma forced by the roots ===")
    for n in NAMES:
        say("   sigma(%s_m) in Span(%s)" % (n, ", ".join(t + "_{-m}" for t in TARGETS[n])))

    eqs = equations() + involutive()
    say("\n%d equations in %d unknowns" % (len(eqs), len(UNK) + 1))

    #  The group of automorphisms of the algebra preserving the torus acts on
    #  the set of anti-involutions: t |-> a t rescales LL_m by a^m, and the
    #  torus of the sl(2) rescales Tp against Tm.  Two normalizations kill that
    #  action; nothing else is assumed, and the solver reports what is left.
    NORM = {u("LL", 1, "LL"): sympy.Integer(1),
            u("Tp", 0, "Tm"): sympy.Integer(1),
            u("S1", 0, "P1"): sympy.Integer(1)}
    say("normalization (killing the automorphisms t -> a t and the sl(2) torus):")
    for k, v in NORM.items():
        say("   %s := %s" % (k, v))
    eqs = [sympy.expand(e.subs(NORM)) for e in eqs]
    eqs = [e for e in eqs if e != 0]

    unknowns = [s for s in list(UNK.values()) + [ZETA] if s not in NORM]
    sub, rest = solve_system(eqs, unknowns, log)
    sub.update(NORM)
    say("determined: %d unknowns; %d equations left" % (len(sub), len(rest)))
    if rest:
        say("  residual equations (first 10):")
        for e in rest[:10]:
            say("   ", e)

    free = [s for s in unknowns if s not in sub]
    say("\nfree parameters: %s" % (free if free else "none"))

    say("\n=== the solution ===")
    say("   sigma(z) = %s z" % sub.get(ZETA, ZETA))
    for n in NAMES:
        row = []
        for m in (-2, -1, 0, 1, 2):
            for t in TARGETS[n]:
                v = sub.get(u(n, m, t), u(n, m, t))
                row.append("sigma(%s_%d) = %s %s_%d" % (n, m, sympy.simplify(v), t, -m))
        say("   " + ";  ".join(row))
    log.close()
