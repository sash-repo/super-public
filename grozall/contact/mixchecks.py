"""Direct checks over k(2n+1|m), no solving:

  1) the number of the weights: the space of multipliers rho with
         L_{K_f} F = K_f(F) + rho(f) F
     a module structure, rho ranging over the frame operators of weighted order 2.
     It is 1-dimensional except for k(1|2), where it is 2-dimensional (the second
     multiplier being D_1 D_2), which is the k(1|n) statement extended to
     every (n,m);
  2) the volume:  Vol = Pi^m F_{n+1-m/2}, read off the divergence of K_f;
  3) the bracket family
         {F,G}_{lam,mu} = varpi^{ab}(e_a F)(e_b G) - (1/2) sum_i (D_i F)(D_i G)
                          + mu (RF) G - lam F (RG)
     is invariant F_lam (x) F_mu -> F_{lam+mu+1}, and at lam = mu = -1 it is the
     contact bracket -- for every (n,m) tried.
"""

import os, sys
from fractions import Fraction as Fr
from itertools import combinations

_HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
from exactlin import nullspace, rank  # noqa: E402

from kmix import KMix, Dens, add, scal, mul, parity  # noqa: E402
from genmix import rand_f  # noqa: E402

FAIL = []


def check(name, cond):
    print(("  ok   " if cond else "  FAIL ") + name)
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------------- 1) the weights
def order2_ops(alg):
    """The frame operators of weighted order 2, as (name, function)."""
    out = [("R", alg.R)]
    for a in range(1, 2 * alg.n + 1):
        for b in range(a, 2 * alg.n + 1):
            out.append(("e%de%d" % (a, b),
                        (lambda a, b: lambda h: alg.e(a, alg.e(b, h)))(a, b)))
    for i in range(1, alg.m + 1):
        for j in range(i, alg.m + 1):
            out.append(("D%dD%d" % (i, j),
                        (lambda i, j: lambda h: alg.D(i, alg.D(j, h)))(i, j)))
    # only EVEN multipliers: rho(f) F must have the parity of f F, so an odd rho
    # (e.g. e_a D_i) does not give a module structure and is excluded.
    return out


def weight_space(alg, seeds=(3, 11, 29, 47)):
    """A basis of the space of multipliers, in the basis order2_ops(alg)."""
    ops = order2_ops(alg)
    fs = []
    for i in alg.k_piece(0) + alg.k_piece(1) + alg.k_piece(2):
        fs.append(i)
    mat, keys = [], set()
    cols = []
    for f in fs:
        for g in fs:
            if not f or not g:
                continue
            s = (-1) ** (parity(f) * parity(g))
            br = alg.kbracket(f, g)
            col = []
            for _, op in ops:
                d = op(br)
                d = add(d, scal(-1, alg.Kf(f, op(g))), scal(s, alg.Kf(g, op(f))))
                col.append(d)
            cols.append(col)
    keys = set()
    for col in cols:
        for d in col:
            keys |= set(d)
    keys = sorted(keys)
    rows = []
    for ci, col in enumerate(cols):
        block = {}
        for j, d in enumerate(col):
            for k, v in d.items():
                block.setdefault(k, [Fr(0)] * len(ops))[j] = v
        for k in sorted(block):
            rows.append(block[k])
    if not rows:
        return [[Fr(1) if i == j else Fr(0) for i in range(len(ops))]
                for j in range(len(ops))], ops
    return nullspace(rows), ops


def op_rank(alg, ns, ops, tests):
    """The dimension of the SPAN as operators: the generators are dependent (D_iD_i is
    a multiple of R), so counting solution vectors would overcount."""
    vals, keys = [], set()
    for vec in ns:
        d = {}
        for i, h in enumerate(tests):
            acc = {}
            for c, (_, op) in zip(vec, ops):
                if c:
                    acc = add(acc, scal(c, op(h)))
            for k, v in acc.items():
                d[(i,) + (k,)] = v
        vals.append(d)
        keys |= set(d)
    keys = sorted(keys, key=repr)
    rows = [[d.get(k, Fr(0)) for k in keys] for d in vals]
    rows = [r for r in rows if any(r)]
    return rank(rows) if rows else 0


# --------------------------------------------------------------- 2) the volume
def divergence(alg, f):
    """div K_f with respect to the coordinate volume."""
    A, B, C, Th = alg.Kf_coeffs(f)
    out = alg.dt(A)
    for i in range(1, alg.n + 1):
        out = add(out, alg.de(alg.ip(i), B[i - 1]), alg.de(alg.iq(i), C[i - 1]))
    # the super divergence: an odd direction contributes with the sign (-1)^{p(f)+1}
    p = parity(f) if f else 0
    for j in range(1, alg.m + 1):
        out = add(out, scal(-(-1) ** p, alg.do(j, Th[j - 1])))
    return out


# -------------------------------------------------------------- 3) the bracket
def bracket_op(alg, lam, mu, F, G):
    out = {}
    for a in range(1, 2 * alg.n + 1):
        for b in range(1, 2 * alg.n + 1):
            v = alg.varpi_up(a, b)
            if v:
                out = add(out, scal(v, mul(alg.e(a, F), alg.e(b, G))))
    # the operator is  -(1/2) sum_{a,b} (G^{-1})^{ab} D_a (x) D_b, so writing it as a
    # product of values costs the Koszul sign of moving the second D past F
    sgn = -Fr(1, 2) * (-1) ** (parity(F) if F else 0)
    for a in range(1, alg.m + 1):
        for b in range(1, alg.m + 1):
            g = alg.gram_up(a, b)
            if g:
                out = add(out, scal(sgn * g, mul(alg.D(a, F), alg.D(b, G))))
    out = add(out, scal(mu, mul(alg.R(F), G)), scal(-lam, mul(F, alg.R(G))))
    return out


def main(shape="orthonormal"):
    from kmix import hyperbolic_gram
    for (n, m) in [(1, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (2, 1)]:
        gram = hyperbolic_gram(m) if shape == "symmetric" else None
        alg = KMix(n, m, gram=gram)
        print("k(%d|%d), %s shape:" % (2 * n + 1, m, shape))

        # 1) the weights
        ns, ops = weight_space(alg)
        tests = [rand_f(alg, 900 + i, 4, 12, par=p) for p in (0, 1) for i in range(3)]
        tests = [t for t in tests if t]
        got = op_rank(alg, ns, ops, tests)
        want = 2 if (n, m) == (0, 2) else 1
        check("the space of weights is %d-dimensional (got %d)" % (want, got),
              got == want)
        if (n, m) == (0, 2):
            names = [nm for nm, _ in ops]
            second = [nm for v in ns for nm, c in zip(names, v) if c]
            check("the second multiplier involves D_1D_2",
                  any("D1D2" in s for s in second))
            for v in ns:
                terms = ["%s*%s" % (c, nm) for nm, c in zip(names, v) if c]
                print("        multiplier: " + " + ".join(terms))

        # 2) the volume
        lam = Fr(n + 1) - Fr(m, 2)
        fs = [rand_f(alg, 700 + i, 4, 10, par=p) for p in (0, 1) for i in range(2)]
        fs = [f for f in fs if f]
        check("div K_f = (%s) f_t, i.e. Vol = Pi^%d F_{%s}" % (lam, m, lam),
              all(divergence(alg, f) == scal(lam, alg.dt(f)) for f in fs))

        # 3) the bracket family
        gens = alg.k_piece(0) + alg.k_piece(1) + alg.k_piece(2)
        Fs = [rand_f(alg, 11 + i, 3, 9, par=p) for p in (0, 1) for i in range(2)]
        Gs = [rand_f(alg, 55 + i, 3, 9, par=p) for p in (0, 1) for i in range(2)]
        Fs, Gs = [f for f in Fs if f], [g for g in Gs if g]
        ok = True
        for (lm, mu) in [(Fr(0), Fr(0)), (Fr(2, 3), Fr(-1, 5)), (Fr(-1), Fr(-1)),
                         (Fr(-3, 2), Fr(5, 4))]:
            M1, M2 = Dens(alg, lm), Dens(alg, mu)
            M3 = Dens(alg, lm + mu + 1)
            for f in gens:
                pf = parity(f) if f else 0
                for F in Fs:
                    sF = (-1) ** (pf * parity(F))
                    for G in Gs:
                        lhs = M3.act(f, bracket_op(alg, lm, mu, F, G))
                        r = bracket_op(alg, lm, mu, M1.act(f, F), G)
                        r = add(r, scal(sF, bracket_op(alg, lm, mu, F, M2.act(f, G))))
                        if lhs != r:
                            ok = False
        check("the bracket family is invariant, four pairs of weights", ok)
        ok = all(bracket_op(alg, Fr(-1), Fr(-1), F, G) == alg.kbracket(F, G)
                 for F in Fs for G in Gs)
        check("at lambda = mu = -1 it is the contact bracket", ok)
        print()

    print("%d checks failed" % len(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
