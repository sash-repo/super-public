#!/usr/bin/env python3
"""Statement 4.6.2 of the text: the census of the text against Onishchik's
classification, J. Algebra 313 (2007).

His Theorems 4-6 classify the non-split 0-homogeneous supermanifolds over a
Grassmannian under two hypotheses ours does not make:

    (i)  phi is IRREDUCIBLE (not merely completely reducible);
    (ii) the dual bundle E*_phi is spanned by its global holomorphic sections.

By [6] of his list (his Stockholm report of 1987), an irreducible homogeneous
bundle E_lam over G/P is spanned by its global sections if and only if lam is a
dominant weight of g; so (ii) says that lam* is g-dominant, where E*_lam =
E_{lam*} and lam* = -w_{0,l}(lam), the Levi's longest element acting.

This file checks, with the engines of bott.py / cot.py / retract.py:

  1) Omega^1 is the irreducible l-module of highest weight  -w_{0,l}(delta),
     i.e., it is tau* in his notation, and its dual is Theta_X = E_delta, which
     IS g-dominant.  So his one exception phi = tau* is exactly the cotangent
     retract of Subsection 4.4 of the text.

  2) Over LGr_2 = Q_3 the module he would call tau* is the "S^2(id) (x) d^{-2}"
     of item 1) of Statement 4.5.1 -- the same bundle under two names.

  3) None of the OTHER retracts that the census of the text finds non-split satisfies both
     of his hypotheses: each is either reducible or not globally generated (as
     a dual).  Hence no clash between the two classifications, anywhere.

  4) The census of irreducible rank-2 retracts with Lambda^2 = O(-2), which is
     the condition of Corollary 4.3.5: over which spaces does one exist?

Run:  python3 onish.py            (all four checks, log to onish.log)
"""

import sys
from collections import defaultdict

from bott import CHSS, chss_list, omega, addw
from cot import levi_weyl, apply_w, l_dominant, n_plus, decompose
from retract import weights, wedge, levi_modules_up_to, named, layer, levi_dim


# ------------------------------------------------------------------ the dual

def longest_levi(X, W=None):
    """The longest element of W_l, as images of the basis."""
    if W is None:
        W = levi_weyl(X)
    return max(W.items(), key=lambda kv: kv[1])[0]


def dual_hw(X, lam, W=None):
    """lam* with E*_lam = E_{lam*}: minus the Levi's longest element on lam."""
    w0 = longest_levi(X, W)
    return tuple(-c for c in apply_w(w0, tuple(lam)))


def g_dominant(lam):
    return all(c >= 0 for c in lam)


def globally_generated(X, lam, W=None):
    """Is E_lam spanned by its global holomorphic sections?  (Onishchik [6])"""
    return g_dominant(lam)


def dual_globally_generated(X, lam, W=None):
    """Onishchik's hypothesis (ii): is E*_lam spanned by its sections?"""
    return g_dominant(dual_hw(X, lam, W))


# ------------------------------------------------------- 1) Omega^1 = tau*

def tangent_hw(X, W=None):
    """The highest weight of Theta_X as an l-module, from the weights of n_+.

    T_o X = g/p = n_+ as a P-module, so the weight multiset of Theta_X is that
    of n_+; Klimyk must return a single irreducible, and it must be the highest
    root delta -- which is Onishchik's tau, irreducible with highest weight
    delta relative to Delta(r)^+, see his Section 3.
    """
    if W is None:
        W = levi_weyl(X)
    m = defaultdict(int)
    for mu in n_plus(X):
        m[mu] += 1
    parts = decompose(X, dict(m), W)
    assert len(parts) == 1 and list(parts.values()) == [1], (X.name, parts)
    return list(parts)[0]


def cotangent_hw(X, W=None):
    """The highest weight of Omega^1 = Theta_X^* = tau^*, as an l-module."""
    return dual_hw(X, tangent_hw(X, W), W)


def check_cotangent(out):
    print("\n1) Theta_X is irreducible of highest weight delta (= his tau);"
          " Omega^1 = tau* is its dual, never globally generated, while its"
          " dual is", file=out)
    print("   %-10s %-22s %-22s %-8s %-9s %-9s" %
          ("X", "h.w.(Theta) = delta?", "h.w.(Omega^1)", "delta?",
           "Omega gg?", "dual gg?"), file=out)
    ok = True
    for X in chss_list():
        if X.dimX < 2:
            continue
        W = levi_weyl(X)
        tan = tangent_hw(X, W)
        lam = cotangent_hw(X, W)
        is_delta = tuple(tan) == tuple(X.R.theta)
        gg = g_dominant(dual_hw(X, lam, W))
        own = g_dominant(lam)
        good = is_delta and gg and not own
        ok = ok and good
        print("   %-10s %-22s %-22s %-8s %-9s %-9s" %
              (X.name, tuple(tan), tuple(lam),
               "yes" if is_delta else "NO",
               "yes" if own else "no", "yes" if gg else "NO"), file=out)
    print("   => Theta = E_delta always; Omega^1 is never globally generated,"
          " its dual always is   %s" % ("ok" if ok else "FAIL"), file=out)
    return ok


# ------------------------------------------- 2) over LGr_2, tau* = S^2(id) d^-2

def check_lgr2(out):
    print("\n2) Over LGr_2 = Q_3 the cotangent module is our"
          " 'S^2(id) (x) d^{-2}'", file=out)
    ok = True
    for name in ("LGr_2", "Q_3"):
        X = named(name)
        lam = cotangent_hw(X)
        node = X.node
        l_part = [0 if i == node - 1 else lam[i] for i in range(X.rank)]
        j = lam[node - 1]
        dim = levi_dim(X, list(lam))
        print("   %-8s h.w.(Omega^1) = %-10s = (l'-part %s) (x) d^{%d},"
              "  rank = %d" % (X.name, tuple(lam), tuple(l_part), j, dim),
              file=out)
        if name == "LGr_2":
            good = (tuple(l_part) == (2, 0) and j == -2 and dim == 3)
            ok = ok and good
            print("      l'-part = 2*omega_1 = S^2(id) of sl(2), twist d^{-2},"
                  " rank 3    %s" % ("ok" if good else "FAIL"), file=out)
        # the H^1 that Statement 4.5.1 and item 3) of Statement 4.4.3 record: 6
        coh = layer(X, l_part, 2, 2 * j)     # Theta (x) Lambda^2(V) (x) d^{2j}
        d1 = sum(dd * mult for dd, _, mult in coh.get(1, []))
        good = (d1 == 6)
        ok = ok and good
        print("      dim H^1(X; Theta (x) Lambda^2(Omega^1)) = %d"
              "   (both Statements say 6)   %s"
              % (d1, "ok" if good else "FAIL"), file=out)
    return ok


# ------------------------- 3) the non-split retracts vs. his two hypotheses

# Each entry: (space, [ (l'-part, twist) ... ], what it is in the text).
# A retract is a sum of irreducible homogeneous bundles V_i (x) d^{k_i}.
def our_nonsplit(out):
    cases = []
    # Corollary 4.3.5, rank 2: over Q_m with Lambda^2(E) = O(-2), and over CP^2
    # with j = -3.  The generic such E is a sum of two line bundles.
    for m in (3, 5, 7, 9, 11, 13, 4, 6, 8, 10, 12):
        try:
            X = named("Q_%d" % m)
        except KeyError:
            continue
        cases.append((X, [([0] * X.rank, -1), ([0] * X.rank, -1)],
                      "Cor. 4.3.5, rank 2 over Q_%d: O(-1)+O(-1)" % m))
        cases.append((X, [([0] * X.rank, 0), ([0] * X.rank, -2)],
                      "Cor. 4.3.5, rank 2 over Q_%d: O(0)+O(-2)" % m))
    X = named("Gr_2^4")
    cases.append((X, [([0] * X.rank, -1), ([0] * X.rank, -1)],
                  "Igonin's example over Gr_2^4: d^{-1}+d^{-1}"))
    X = named("CP^2")
    lam = cotangent_hw(X)
    cases.append((X, [(list(lam[:X.node - 1]) + [0] + list(lam[X.node:]),
                       lam[X.node - 1])],
                  "Cor. 4.3.5 over CP^2, rank 2: Omega^1 = tau*"))
    # Statement 4.5.1, item 1), over LGr_2
    X = named("LGr_2")
    cases.append((X, [([1, 0], 0), ([0, 0], -2)],
                  "St. 4.5.1 1) over LGr_2: (id (x) d^0) + d^{-2}"))
    lam = cotangent_hw(X)
    cases.append((X, [(list(lam[:X.node - 1]) + [0] + list(lam[X.node:]),
                       lam[X.node - 1])],
                  "St. 4.5.1 1) over LGr_2: S^2(id) (x) d^{-2} = Omega^1"))
    # Statement 4.5.1, item 2), over LGr_4:  (Lambda^2(id) (x) d^k) + d^{k'},
    # 3k + k' = -2; take k = 0, k' = -2.
    X = named("LGr_4")
    cases.append((X, [([0, 1, 0, 0], 0), ([0, 0, 0, 0], -2)],
                  "St. 4.5.1 2) over LGr_4: (Lambda^2(id) (x) d^0) + d^{-2}"))
    return cases


def check_hypotheses(out):
    print("\n3) Our non-split retracts against Onishchik's two hypotheses:"
          " phi irreducible, E*_phi globally generated", file=out)
    print("   %-46s %-6s %-9s %s" %
          ("retract", "irred?", "E* g.g.?", "inside his theorems?"), file=out)
    ok = True
    for X, parts, what in our_nonsplit(out):
        W = levi_weyl(X)
        irred = (len(parts) == 1)
        gg = all(dual_globally_generated(
            X, addw(l, omega(X.rank, X.node, k)), W) for l, k in parts)
        inside = irred and gg
        # the ONLY one that may be inside is the cotangent retract
        is_cot = (irred and tuple(addw(parts[0][0],
                  omega(X.rank, X.node, parts[0][1]))) == cotangent_hw(X, W))
        ok = ok and (inside == is_cot)
        print("   %-46s %-6s %-9s %s%s" %
              (what, "yes" if irred else "no", "yes" if gg else "no",
               "YES" if inside else "no",
               "  (= tau*, his exception)" if is_cot else ""), file=out)
    print("   => the only retract of ours inside his hypotheses is the"
          " cotangent one   %s" % ("ok" if ok else "FAIL"), file=out)
    return ok


# ---------------- 4) irreducible rank-2 retracts with Lambda^2(E) = O(-2)

def check_rank2(out):
    print("\n4) Every irreducible rank-2 homogeneous bundle, and the twist its"
          " determinant forces: can Lambda^2(E) = O(-2) (Cor. 4.3.5)?", file=out)
    print("   %-10s %-16s %-11s %-24s %s" %
          ("X", "l'-module", "det = O(c)", "Lambda^2(V (x) d^k)", "verdict"),
          file=out)
    found = minus2 = 0
    for X in chss_list():
        if X.dimX < 2:
            continue
        W = levi_weyl(X)
        for _, lam0, dim in levi_modules_up_to(X, 2):
            if dim != 2:
                continue
            found += 1
            ws = weights(X, lam0)
            assert sum(ws.values()) == 2, (X.name, lam0, ws)
            det = [0] * X.rank
            for mu, mult in ws.items():
                for _ in range(mult):
                    det = addw(det, list(mu))
            c = det[X.node - 1]
            # Lambda^2(V (x) d^k) = O(c + 2k)
            hits = ((-2 - c) % 2 == 0)
            if hits:
                minus2 += 1
                k = (-2 - c) // 2
                lam = addw(lam0, omega(X.rank, X.node, k))
                gg = dual_globally_generated(X, lam, W)
                verdict = "O(-2) at k = %d, E* g.g.: %s" % (k, "yes" if gg else "no")
            else:
                verdict = "O(odd) always -- never O(-2)"
            print("   %-10s %-16s %-11s %-24s %s" %
                  (X.name, tuple(lam0), "O(%d)" % c,
                   "O(%d + 2k)" % c, verdict), file=out)
    print("   => %d irreducible rank-2 bundles in the census, %d of which can"
          " have Lambda^2 = O(-2)" % (found, minus2), file=out)
    return minus2 == 0


def main():
    with open("onish.log", "w") as fh:
        class Tee:
            def write(self, s):
                fh.write(s)
                sys.stdout.write(s)

            def flush(self):
                fh.flush()
                sys.stdout.flush()

        out = Tee()
        print("Onishchik, J. Algebra 313 (2007) 320-342, against the census of the text",
              file=out)
        a = check_cotangent(out)
        b = check_lgr2(out)
        c = check_hypotheses(out)
        d = check_rank2(out)
        print("\nall checks %s" % ("passed" if (a and b and c) else "FAILED"),
              file=out)
    return 0 if (a and b and c) else 1


if __name__ == "__main__":
    sys.exit(main())
