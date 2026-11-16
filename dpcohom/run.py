"""run.py -- the checks of record for the non-restricted half.

  1) dpco.py's dual complex against A. Lebedev's XC^*-cochains of [BGL, 4.1],
     entry by entry, on random cochains, in degrees 1 and 2;
  2) the resolution V_* = U(g) (x) C_*: d^2 = 0 and exactness;
  3) the cohomology of the divided power complex against the cohomology of the
     anti-symmetric one (= the cochains of g with the squaring FORGOTTEN),
     showing that the two theories differ, i.e. that the squaring relation
     x*x = s(x) in U(g) is not cosmetic;
  4) sample answers: H^n(g;K) and H^n(g;g) for the sample algebras.
"""

import random
import sys

from dpco import (SLA2, Env, CE, Resolution, cohomology_dims, Module)
from lebedev import compare
from smoke import (ex_lambda, ex_poly, ex_gl11, ex_heis, check_d2, exactness)

LOG = open("run.log", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def forget(g):
    """g with the squaring forgotten: the underlying Lie algebra, all even.
    Its divided power complex is Lambda(g^*) (x) M, i.e. the anti-symmetric
    cochains AC^*(g;M) of [BGL, Sec. 4]."""
    d = g.dim
    br = {}
    for i in range(d):
        for j in range(i + 1, d):
            if g.br[i][j]:
                br[(i, j)] = g.br[i][j]
    return SLA2(d, 0, br, [], g.name + " (squaring forgotten)")


def ex_sl11():
    """sl(1|1) for p=2 inside gl(1|1): e = E11+E22 (central), f0 = E12,
    f1 = E21, [f0,f1] = e, s(f_i) = 0.  This is the odd Heisenberg."""
    return ex_heis()


def ex_q2():
    """a (1|2) example with a non-zero squaring on one letter:
    g_0 = <e>, g_1 = <f0,f1>, s(f0) = e, s(f1) = 0, [f0,f1] = 0, e central."""
    return SLA2(1, 2, {}, [1, 0], "(1|2), s(f0)=e")


def ex_p2():
    """g_0 = <e0,e1>, g_1 = <f>, s(f) = e0, [e1,f] = f, [e0,e1] = e0.
    A solvable example with a non-trivial even action."""
    br = {(0, 1): 1, (1, 2): 1 << 2}
    return SLA2(2, 1, br, [1], "(2|1) solvable, s(f)=e0")


def main():
    rng = random.Random(20260817)
    samples = [ex_lambda(), ex_poly(), ex_gl11(), ex_heis(), ex_q2(), ex_p2()]

    say("=" * 74)
    say("1) Lebedev's XC^* codifferential vs the dual of V_* = U(g) (x) C_*")
    say("=" * 74)
    tot1 = tot2 = 0
    for g in samples:
        pr = g.check_axioms()
        if pr:
            say("  !! axioms fail for", g.name, pr[:3])
            continue
        for M in [g.trivial_module(), g.adjoint_module()]:
            b1, b2 = compare(g, M, 25, rng)
            tot1 += b1
            tot2 += b2
            say("  %-34s M = %-2s  d^1: %s   d^2: %s"
                % (g.name, M.name,
                   "agree" if b1 == 0 else "%d MISMATCHES" % b1,
                   "agree" if b2 == 0 else "%d MISMATCHES" % b2))
    say("  total mismatches over 25 random cochains per case:", tot1 + tot2)

    say("")
    say("=" * 74)
    say("2) the resolution: d^2 = 0 and exactness")
    say("=" * 74)
    for g in samples:
        env = Env(g)
        res = Resolution(g, env)
        assoc = env.check_associativity(maxdeg=2 if g.dim > 3 else 3)
        b2 = check_d2(res, 5 if g.dim <= 3 else 4)
        D = 6 if g.dim <= 2 else 5
        badx, sizes = exactness(res, D)
        say("  %-34s U(g) assoc: %s  d^2=0: %s  exact to degree %d: %s"
            % (g.name, "OK" if assoc == 0 else "FAIL",
               "OK" if b2 == 0 else "FAIL", D - 2,
               "OK" if not badx else "FAIL (%d)" % len(badx)))

    say("")
    say("=" * 74)
    say("3) divided power (= Ext over U(g)) vs anti-symmetric (= Ext over the")
    say("   enveloping algebra of the underlying Lie algebra)")
    say("=" * 74)
    for g in samples:
        res = Resolution(g)
        dp = cohomology_dims(res, g.trivial_module(), 4)
        gf = forget(g)
        resf = Resolution(gf)
        an = cohomology_dims(resf, gf.trivial_module(), 4)
        say("  %-34s dpH^n(g;K) = %s" % (g.name, dp))
        say("  %-34s  AH^n(g;K) = %s%s"
            % ("", an, "   <- differ" if an != dp else "   (equal here)"))

    say("")
    say("=" * 74)
    say("4) sample answers")
    say("=" * 74)
    for g in samples:
        res = Resolution(g)
        say("  %-34s H^n(g;K) n=0..4: %s   H^n(g;g) n=0..2: %s"
            % (g.name, cohomology_dims(res, g.trivial_module(), 4),
               cohomology_dims(res, g.adjoint_module(), 2)))
    LOG.close()


if __name__ == "__main__":
    main()
