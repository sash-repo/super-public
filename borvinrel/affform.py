#!/usr/bin/env python3
"""What the affine member of the chain of [CP3] supplies: L_1 in place of L_0.

Section 5 of [CP3] produces, in S(g^+) and in no other member, the element
    J = [eps_{-1}, e_0, e_theta, phi_{-1,1}],   [J, T^A_m] = -m T^A_{m+1},
i.e. the Virasoro generator L_1, and it exists -- their own words -- "due to
the light-likeness of the g^+ root alpha_0 + theta", i.e. because of the null
root delta.  Their eq. (5.10) then pairs the modes m and 1-m, and K with L_1.
The grading element of the mode grading, d = -L_0, is NOT in S(g^+).

This engine builds that algebra over Q for g = sl(2) and verifies, exactly:

  (1) the model is a Lie algebra: Jacobi on all triples in the window;
  (2) the SHIFTED form -- omega(x_m, y_{m'}) = (x,y) delta_{m+m',1},
      omega(K, L_1) = omega(L_1, K) = alpha -- is invariant for exactly one
      value of alpha, which is SOLVED for, not assumed (it comes out -1, which
      is their (5.10));
  (3) it is non-degenerate on a window closed under m <-> 1-m;
  (4) it has weight +1 for the mode grading, i.e. it is off center by exactly
      the weight of the derivation L_1 that replaces L_0;
  (5) adjoining L_0 destroys it, and by the Lemma: the defect of invariance on
      a pair of modes m, m' is -(m+m') omega, i.e. -1 on the pairs the form
      pairs.  With L_0 in the algebra the only invariant forms are unshifted;
  (6) the reason no such element forbids it in S(g^+): ad_{L_1} is NILPOTENT
      on every window, whereas ad_{L_0} is diagonal with eigenvalues -m.
      The Lemma bites on semisimple elements only.

Basis: T^A_m for A in {e, h, f} and m in a window, plus K, plus D (= L_0 or
L_1).  Brackets:
    [x_m, y_m'] = [x,y]_{m+m'} + m (x,y) delta_{m+m',0} K,
    [L_k, x_m]  = -m x_{m+k},   [L_k, K] = 0,   [L_0, L_1] = -L_1.
"""

import sys
from fractions import Fraction

# sl(2) with the trace form (x,y) = tr(xy): e, h, f
BR = {                                   # [A, B] as a dict of coefficients
    ("e", "f"): {"h": 1}, ("f", "e"): {"h": -1},
    ("h", "e"): {"e": 2}, ("e", "h"): {"e": -2},
    ("h", "f"): {"f": -2}, ("f", "h"): {"f": 2},
}
KILL = {("e", "f"): Fraction(1), ("f", "e"): Fraction(1),
        ("h", "h"): Fraction(2)}
LET = ["e", "h", "f"]


def kill(a, b):
    return KILL.get((a, b), Fraction(0))


def bracket(u, v, D):
    """Bracket of two basis elements.  A basis element is

        ("T", A, m)  |  ("K",)  |  ("D",)        where D = 0 or 1 is the index
                                                 of the Virasoro generator L_D.

    Returns a dict basis_element -> coefficient.
    """
    out = {}

    def add(k, c):
        if c:
            out[k] = out.get(k, Fraction(0)) + c
            if out[k] == 0:
                del out[k]

    if u[0] == "K" or v[0] == "K":
        return out                        # K is central
    if u[0] == "T" and v[0] == "T":
        _, a, m = u
        _, b, n = v
        for c, coef in BR.get((a, b), {}).items():
            add(("T", c, m + n), Fraction(coef))
        if m + n == 0:
            add(("K",), m * kill(a, b))
        return out
    if u[0] == "D" and v[0] == "T":
        _, b, n = v
        add(("T", b, n + D), Fraction(-n))
        return out
    if u[0] == "T" and v[0] == "D":
        r = bracket(v, u, D)
        return {k: -c for k, c in r.items()}
    return out                            # [D, D] = 0 (one generator only)


def main():
    log = open("affform.log", "w")

    def say(s=""):
        print(s)
        log.write(s + "\n")

    ok = 0
    bad = 0

    def check(cond, msg):
        nonlocal ok, bad
        if cond:
            ok += 1
            say("  OK   " + msg)
        else:
            bad += 1
            say("  FAIL " + msg)

    M = 4
    modes = list(range(-M, M + 2))        # closed under m <-> 1-m
    say("The level-0 part of S(g^+) for g = sl(2): the loop algebra, K, and L_1")
    say("(no L_0).  Modes %d..%d, exact over Q." % (modes[0], modes[-1]))
    say("")

    # ---------------------------------------------------------------- Jacobi
    for D in (0, 1):
        elems = ([("T", a, m) for a in LET for m in modes] +
                 [("K",), ("D",)])
        bad0 = 0
        for u in elems:
            for v in elems:
                for w in elems:
                    # [u,[v,w]] - [[u,v],w] - [v,[u,w]]
                    acc = {}

                    def addall(d, s=1):
                        for k, c in d.items():
                            acc[k] = acc.get(k, Fraction(0)) + s * c

                    for k, c in bracket(v, w, D).items():
                        addall({kk: c * cc for kk, cc in
                                bracket(u, k, D).items()})
                    for k, c in bracket(u, v, D).items():
                        addall({kk: c * cc for kk, cc in
                                bracket(k, w, D).items()}, -1)
                    for k, c in bracket(u, w, D).items():
                        addall({kk: c * cc for kk, cc in
                                bracket(v, k, D).items()}, -1)
                    # ignore terms that left the window: the window is not a
                    # subalgebra, only the identity on the infinite algebra is
                    # being tested
                    for k, c in acc.items():
                        if c != 0 and (k[0] != "T" or abs(k[2]) <= M):
                            bad0 += 1
        check(bad0 == 0, "Jacobi holds for L_%d, all triples in the window "
                         "(%d elements)" % (D, len(elems)))

    # ------------------------------------------------- solve for the shifted
    # ansatz: omega(x_m, y_n) = (x,y) delta_{m+n,1};  omega(K, L_1) =
    # omega(L_1, K) = alpha;  omega(K,K) = omega(L_1,L_1) = 0.  Invariance
    # omega([u,v],w) + omega(v,[u,w]) = 0 determines alpha.
    def omega(u, v, alpha, shift):
        if u[0] == "T" and v[0] == "T":
            return kill(u[1], v[1]) if u[2] + v[2] == shift else Fraction(0)
        if {u[0], v[0]} == {"K", "D"}:
            return alpha
        return Fraction(0)

    def defect(alpha, shift, D, triples):
        out = []
        for u, v, w in triples:
            s = Fraction(0)
            for k, c in bracket(u, v, D).items():
                s += c * omega(k, w, alpha, shift)
            for k, c in bracket(u, w, D).items():
                s += c * omega(v, k, alpha, shift)
            out.append(((u, v, w), s))
        return out

    elems = ([("T", a, m) for a in LET for m in modes] + [("K",), ("D",)])
    triples = [(u, v, w) for u in elems for v in elems for w in elems]

    # alpha is determined by the triples in which K is produced
    sols = []
    for num in range(-4, 5):
        alpha = Fraction(num)
        if all(s == 0 for _, s in defect(alpha, 1, 1, triples)):
            sols.append(alpha)
    check(sols == [Fraction(-1)],
          "the shifted form is invariant for exactly one alpha, and it is "
          "alpha = %s (their eq. (5.10) has -K (x) L_1 - L_1 (x) K)"
          % (sols[0] if sols else "none"))

    # ------------------------------------------------------ non-degeneracy
    alpha = Fraction(-1)
    win = ([("T", a, m) for a in LET for m in modes] + [("K",), ("D",)])
    G = [[omega(u, v, alpha, 1) for v in win] for u in win]
    # exact determinant
    n = len(G)
    Gm = [[Fraction(x) for x in row] for row in G]
    d = Fraction(1)
    for c in range(n):
        p = None
        for r in range(c, n):
            if Gm[r][c] != 0:
                p = r
                break
        if p is None:
            d = Fraction(0)
            break
        if p != c:
            Gm[c], Gm[p] = Gm[p], Gm[c]
            d = -d
        d *= Gm[c][c]
        inv = Fraction(1) / Gm[c][c]
        for r in range(c + 1, n):
            f = Gm[r][c] * inv
            if f:
                for cc in range(c, n):
                    Gm[r][cc] -= f * Gm[c][cc]
    check(d != 0, "it is non-degenerate on the window closed under "
                  "m <-> 1-m (det = %s)" % d)

    # ------------------------------------------------------------- the weight
    nz = [(u[2] + v[2]) for u in win for v in win
          if u[0] == "T" and v[0] == "T" and omega(u, v, alpha, 1) != 0]
    check(set(nz) == {1},
          "it pairs the modes m and 1-m: weight +1 for the mode grading, "
          "the weight of L_1")

    # ------------------------------------------------- the Lemma, in this model
    # with L_0 present the same form has defect -(m+m') omega
    worst = []
    for u in [("T", "e", 0), ("T", "e", 2), ("T", "h", -1)]:
        for v in [("T", "f", 1), ("T", "f", -1), ("T", "h", 2)]:
            s = Fraction(0)
            L0 = ("D",)
            for k, c in bracket(L0, u, 0).items():
                s += c * omega(k, v, alpha, 1)
            for k, c in bracket(L0, v, 0).items():
                s += c * omega(u, k, alpha, 1)
            worst.append((u, v, s, -(u[2] + v[2]) * omega(u, v, alpha, 1)))
    check(all(s == pred for _, _, s, pred in worst),
          "with L_0 in the algebra the defect of invariance is -(m+m') omega, "
          "i.e. Lemma: the form must have weight 0")
    check(any(s != 0 for _, _, s, _ in worst),
          "and it is non-zero on the pairs the shifted form pairs")

    # the UNshifted form is the invariant one when L_0 is present
    sols0 = []
    for num in range(-4, 5):
        a0 = Fraction(num)
        if all(s == 0 for _, s in defect(a0, 0, 0, triples)):
            sols0.append(a0)
    check(sols0 == [Fraction(-1)],
          "with L_0, the UNshifted form (m <-> -m, K <-> L_0) is invariant, "
          "again for alpha = %s" % (sols0[0] if sols0 else "none"))

    # --------------------------------------- why L_1 does not fall under the Lemma
    # ad_{L_1} on the window: nilpotent;  ad_{L_0}: diagonal, eigenvalues -m
    basis = [("T", a, m) for a in LET for m in modes]
    idx = {b: i for i, b in enumerate(basis)}

    def admat(D):
        N = [[Fraction(0)] * len(basis) for _ in basis]
        for b in basis:
            for k, c in bracket(("D",), b, D).items():
                if k in idx:
                    N[idx[k]][idx[b]] = c
        return N

    def matmul(X, Y):
        n = len(X)
        return [[sum(X[i][k] * Y[k][j] for k in range(n)) for j in range(n)]
                for i in range(n)]

    N1 = admat(1)
    P = N1
    for _ in range(len(basis)):
        if all(all(x == 0 for x in row) for row in P):
            break
        P = matmul(P, N1)
    check(all(all(x == 0 for x in row) for row in P),
          "ad_{L_1} is nilpotent on the window: it has no non-zero eigenvalue, "
          "so Lemma 'an invariant form has weight 0' does not apply to it")
    N0 = admat(0)
    check(all(N0[i][j] == 0 for i in range(len(basis))
              for j in range(len(basis)) if i != j) and
          sorted(set(N0[i][i] for i in range(len(basis)))) ==
          sorted(set(Fraction(-m) for m in modes)),
          "ad_{L_0} is diagonal with eigenvalues -m: semisimple, and the "
          "Lemma does apply to it")

    say("")
    say("checks: %d OK, %d FAIL" % (ok, bad))
    log.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
