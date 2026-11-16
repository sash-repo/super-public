"""The maximal toral subalgebra of k_0 = Kee z + osp(m|2n), and the weights of the frame.

The symmetrization of the contact form must reach the odd indeterminates too,
otherwise the weights with respect to a basis of the maximal toral subalgebra cannot
be written down.  Checked here:

  1) t = < z, K_{p_i q_i}, K_{xi_j eta_j} > is a maximal toral subalgebra of k_0, of
     dimension 1 + n + [m/2]: its elements commute, act semisimply, and their
     centralizer in k_0 is t itself (which is what makes it maximal);

  2) in the SYMMETRIC shape every frame operator is a WEIGHT VECTOR of t, with

         e_{2i-1}: -delta_i,  e_{2i}: +delta_i,  D_{xi_j}: -eps_j,  D_{eta_j}: +eps_j,
         D_theta: 0,  R: 0,   and z-weight -W (minus the weighted order);

  3) in the ORTHONORMAL shape they are not: the same toral element rotates D_1 into D_2.
     So the weight decomposition -- of the frame, of the ansatz, of a density -- exists
     in the one shape and not in the other;

  4) an invariant operator is a vector of t-weight 0, so the ansatz of the census may be
     replaced by its zero-weight part; the file prints by how much that shrinks it.

Exact rational arithmetic.
"""

import os, sys
from fractions import Fraction as Fr

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import rank  # noqa: E402

from kmix import KMix, add, scal, mul, parity, hyperbolic_gram  # noqa: E402
from genmix import term_basis, frame_words  # noqa: E402
from test_khyp import rand_f  # noqa: E402

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------------ the torus
def torus_gens(alg):
    """The generating functions of the candidate maximal toral subalgebra."""
    out = [("z", alg.evar(0, 2))]
    for i in range(1, alg.n + 1):
        out.append(("p%dq%d" % (i, i),
                    mul(alg.evar(alg.ip(i)), alg.evar(alg.iq(i)))))
    for j in range(1, alg.m // 2 + 1):
        out.append(("2xi%deta%d" % (j, j),
                    scal(2, mul(alg.ovar(2 * j - 1), alg.ovar(2 * j)))))
    return out


def k0_basis(alg):
    return [f for f in alg.k_piece(0) if f]


def ad_matrix(alg, f, basis):
    """The matrix of ad_{K_f} on a list of generating functions (f even)."""
    keys = sorted({k for g in basis for k in alg.kbracket(f, g)} |
                  {k for g in basis for k in g})
    idx = {k: i for i, k in enumerate(keys)}
    cols = []
    for g in basis:
        col = [Fr(0)] * len(keys)
        for k, v in alg.kbracket(f, g).items():
            col[idx[k]] = v
        cols.append(col)
    return [[cols[j][i] for j in range(len(basis))] for i in range(len(keys))], keys


def frame_ops(alg):
    """(name, operator, weighted order) for every frame operator."""
    out = []
    for a in range(1, 2 * alg.n + 1):
        out.append(("e%d" % a, (lambda a: lambda h: alg.e(a, h))(a), 1, 0))
    for b in range(1, alg.m + 1):
        out.append(("D%d" % b, (lambda b: lambda h: alg.D(b, h))(b), 1, 1))
    out.append(("R", alg.R, 2, 0))
    return out


def bracket_with_frame(alg, f, op, pop, h):
    """[K_f, X] h for a frame operator X of parity pop (f even here)."""
    return add(alg.Kf(f, op(h)), scal(-1, op(alg.Kf(f, h))))


def frame_weight(alg, f, op, pop, tests):
    """The scalar c with [K_f,X] = c X, or None if X is not a weight vector."""
    c = None
    for h in tests:
        lhs = bracket_with_frame(alg, f, op, pop, h)
        rhs = op(h)
        if not lhs and not rhs:
            continue
        if not rhs:
            return None
        # solve lhs = c rhs
        k = sorted(rhs)[0]
        cc = lhs.get(k, Fr(0)) / rhs[k]
        if lhs != scal(cc, rhs):
            return None
        if c is None:
            c = cc
        elif c != cc:
            return None
    return c if c is not None else Fr(0)


def report(n, m, gram, label, Wmax=4):
    alg = KMix(n, m, gram=gram)
    print("k(%d|%d), %s shape:" % (2 * n + 1, m, label))
    tor = torus_gens(alg)
    basis = k0_basis(alg)
    tests = [t for t in (rand_f(alg, 300 + i, 3, 8) for i in range(4)) if t]

    # 1) commuting, semisimple, self-centralizing
    ok = all(alg.kbracket(f, g) == {} for _, f in tor for _, g in tor)
    check("the %d generators commute" % len(tor), ok)
    # semisimplicity: ad is diagonalizable over Qee since the k_0-basis of monomials
    # already diagonalizes it -- check that every basis monomial is an eigenvector
    ok = True
    for _, f in tor:
        for g in basis:
            br = alg.kbracket(f, g)
            if br and set(br) != set(g):
                ok = False
    if label == "symmetric":
        check("every monomial of k_0 is a common eigenvector (t is toral)", ok)
    else:
        print(("  ok   " if not ok or alg.m < 2 else "  --   ") +
              "the monomial basis of k_0 is%s a weight basis"
              % ("" if ok else " NOT"))
    # centralizer
    cent = [g for g in basis if all(alg.kbracket(f, g) == {} for _, f in tor)]
    span = rank([[v for v in _vec(g, basis)] for g in cent]) if cent else 0
    check("the centralizer of t in k_0 is t itself: dim = %d = 1 + %d + %d"
          % (len(cent), n, m // 2), len(cent) == 1 + n + m // 2)

    # 2)/3) the weights of the frame
    rows = []
    allweights = True
    for name, op, w, pop in frame_ops(alg):
        wt = [frame_weight(alg, f, op, pop, tests) for _, f in tor]
        if any(x is None for x in wt):
            allweights = False
            rows.append((name, None))
        else:
            rows.append((name, wt))
    if allweights:
        print("      weights (%s):" % ", ".join(nm for nm, _ in tor))
        for name, wt in rows:
            print("        %-4s %s" % (name, [str(x) for x in wt]))
    else:
        bad = [nm for nm, wt in rows if wt is None]
        print("      NOT weight vectors: %s" % ", ".join(bad))
    return alg, tor, allweights, rows


def _short(alg, f):
    """A one-line rendering of a function, for the printout."""
    if not f:
        return "0"
    out = []
    for (e, S), c in sorted(f.items()):
        nm = "".join("u_%d" % j for j in S) or "1"
        if any(e):
            nm = ("t^%d" % e[0] if e[0] else "") + nm
        out.append("%s %s" % (c, nm))
    return " + ".join(out)


def _vec(g, basis):
    return [Fr(1) if g == b else Fr(0) for b in basis]


# ------------------------------------------- 4) the ansatz splits by weight
def term_weight(alg, tor, term, tests):
    """The t-weight of a basis term of the ansatz: the sum of the weights of its
    frame letters, both slots.  Returns a tuple over the torus generators."""
    wts = {}
    for name, op, w, pop in frame_ops(alg):
        wts[name] = [frame_weight(alg, f, op, pop, tests) for _, f in tor]
    out = [Fr(0)] * len(tor)
    for (I, S, s) in term:
        for a, k in enumerate(I, start=1):
            for _ in range(k):
                out = [x + y for x, y in zip(out, wts["e%d" % a])]
        for b in S:
            out = [x + y for x, y in zip(out, wts["D%d" % b])]
        for _ in range(s):
            out = [x + y for x, y in zip(out, wts["R"])]
    return tuple(out)


def ansatz_shrink(alg, tor, W, tests):
    terms = term_basis(alg, W)
    zero = [t for t in terms if all(x == 0 for x in term_weight(alg, tor, t, tests)[1:])]
    return len(terms), zero


def thetas(alg, term):
    """How many times the last odd letter -- the theta present when m is odd -- occurs."""
    return sum(1 for (I, S, s) in term if alg.m in S) if alg.m % 2 else 0


def main():
    for (n, m) in [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (2, 2)]:
        alg, tor, allw, rows = report(n, m, hyperbolic_gram(m), "symmetric")
        check("every frame operator is a weight vector of t", allw)
        tests = [t for t in (rand_f(alg, 300 + i, 3, 8) for i in range(4)) if t]
        odd_ok, theta_ok = True, True
        for W in (1, 2, 3, 4, 5):
            tot, zer = ansatz_shrink(alg, tor, W, tests)
            print("      W = %d: ansatz %d terms, of t-weight 0: %d" % (W, tot, len(zer)))
            if W % 2:
                if m % 2 == 0 and zer:
                    odd_ok = False
                if m % 2 and any(thetas(alg, t) != 1 for t in zer):
                    theta_ok = False
        if m % 2 == 0:
            check("m even: NO term of t-weight 0 in odd weighted order", odd_ok)
        else:
            check("m odd: every t-weight 0 term of odd W uses theta exactly once",
                  theta_ok)
        print()

    # ---- what the toral element does to the odd coordinates in the two shapes, and
    # ---- the second weight of k(1|2) as its eigenvalue
    print("the toral element h = K_{2 xi eta} of k(1|2):")
    for shape, g in (("symmetric", hyperbolic_gram(2)), ("orthonormal", None)):
        alg = KMix(0, 2, gram=g)
        h = scal(2, mul(alg.ovar(1), alg.ovar(2)))
        b1, b2 = alg.kbracket(h, alg.ovar(1)), alg.kbracket(h, alg.ovar(2))
        print("      %-12s {h,u_1} = %s,  {h,u_2} = %s"
              % (shape, _short(alg, b1), _short(alg, b2)))
        diag = (b1 == alg.ovar(1) and b2 == scal(-1, alg.ovar(2)))
        if shape == "symmetric":
            check("symmetric: u_1, u_2 are eigenvectors of h, of weights +1 and -1",
                  diag)
        else:
            check("orthonormal: ad_h is a rotation (char. polynomial x^2+1, no "
                  "eigenvalue over Ree)",
                  b1 == scal(-1, alg.ovar(2)) and b2 == alg.ovar(1))
        from kmix import Dens
        M = Dens(alg, Fr(0), Fr(1))
        val = M.act(h, alg.const(1))
        check("%s: h acts on the density 1 of F_{lam,nu} by -2nu (got %s)"
              % (shape, _short(alg, val)), val == alg.const(-2))
    print()

    print("the same in the orthonormal shape:")
    for (n, m) in [(0, 2), (0, 3), (1, 2)]:
        alg, tor, allw, rows = report(n, m, None, "orthonormal")
        check("some frame operator is NOT a weight vector (m >= 2)", not allw)
        # what ad_h does instead: a rotation of the first two odd frame directions
        h = scal(2, mul(alg.ovar(1), alg.ovar(2)))
        tests = [t for t in (rand_f(alg, 700 + i, 3, 8) for i in range(5)) if t]
        ok = True
        for j, sgn, k in ((1, -1, 2), (2, 1, 1)):
            for t in tests:
                lhs = add(alg.Kf(h, alg.D(j, t)), scal(-1, alg.D(j, alg.Kf(h, t))))
                if lhs != scal(sgn, alg.D(k, t)):
                    ok = False
        check("[K_{2 theta_1 theta_2}, D_1] = -D_2 and [.., D_2] = D_1 "
              "(char. polynomial x^2+1)", ok)
        print()

    print("%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
