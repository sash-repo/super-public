#!/usr/bin/env python3
"""
A. Lebedev's description of the localization of a supercommutative superalgebra
at the complement of a prime ideal.  Exact check of every step of it.

The models and the algebra machinery are those of `localiz.py`:

    MODEL U = Q[x]/(x^3) (x) Lambda[xi_1,xi_2,xi_3],  p = (x, xi_1, xi_2, xi_3)
              -- here S = A \\ p consists of UNITS, so every candidate relation
                 can be compared against GENUINE equality of fractions;
    MODEL T = Q[x,y]/(xy) (x) Lambda[xi_1,xi_2,xi_3], p = (x, xi_1, xi_2, xi_3)
              -- here S contains the non-unit y and the zero divisor x, so the
                 auxiliary factor t of the equivalence relation is not vacuous.

Notation.  s_c := sigma(s) = s_ev - s_od (Lebedev's "conjugate"; our sbar of localiz.py).
S_0 := S \\cap A_ev, and ~_0 is the commutative-style equivalence on A x S_0:

    (r,s) ~_0 (r',s')  <=>  exists t in S_0 such that t(r s' - r' s) = 0.

Lebedev's relation on A x S:

    (r,s) ~ (r',s')  <=>  (r s_c, s_ev^2) ~_0 (r' s'_c, s'^2_ev)
                     <=>  exists t in S_0 such that t(r s_c s'^2_ev - r' s'_c s^2_ev) = 0,

and his two requirements

    (1)  (r,s) ~ (ru, su)  for every u in S;
    (2)  for s,s' in S_0:  (r,s) ~ (r',s')  <=>  (r,s) ~_0 (r',s').

What is checked here, step by step:

 A. S = A \\ p is closed under conjugation, and S_0 = S \\cap A_ev is
    multiplicatively closed;  s s_c = s_c s = s_ev^2 lies in S_0 whenever s in S.
 B. "t in S may be replaced by t in S_0": if t z = 0 for a t in S, then
    t_ev^2 z = 0 and t_ev^2 in S_0.  (This is what makes (2) an "iff".)
 C. Lebedev's relation satisfies (1) -- and does so with t = 1: the bracket vanishes
    IDENTICALLY.
 D. Lebedev's relation satisfies (2), in both directions, constructively.
 E. It is an equivalence relation; transitivity is checked in the constructive
    form used in its proof (t t' s'^2_ev is a witness for the composite).
 F. Uniqueness: the step "(1) forces (r,s) ~ (r s_c, s s_c) = (r s_c, s_ev^2)",
    which is (1) with u = s_c, and it is legitimate by A.
 G. In model U, where the fractions genuinely exist, Lebedev's relation holds if and
    only if r s^{-1} = r' s'^{-1}.  Likewise on the left with s_c r.
 H. The bijection A_S <-> A_{S_0}, (r,s) |-> (r s_c, s_ev^2), is well defined,
    injective on classes, onto, and is the IDENTITY on classes of pairs from
    A x S_0; transporting + and * along it gives the true sum and product.
 I. The naive cross-multiplication  r s' = r' s  violates requirement (1), which
    is why it is not the relation looked for (witness printed).

Run:  python3 localiz2.py
"""

import random
import sys

from localiz import (U, T, one, zero, add, sub, neg, mul, smul, sigma, ev, od,
                     eq, show, const, inv_U, rnd, rnd_unit, rnd_S_T, in_S_T,
                     gen_xi, polgen)

NSAMP = 300


# ---------------------------------------------------------------- helpers ---

def conj(a):
    """Lebedev's a_c = a_ev - a_od."""
    return sigma(a)


def sq_ev(R, s):
    """(s_ev)^2."""
    e = ev(s)
    return mul(R, e, e)


def in_S_U(a):
    """Model U: A/p = Q, so a is outside p iff its constant term is non-zero."""
    return const(a) != 0


IN_S = {id(U): in_S_U, id(T): in_S_T}


def prod(R, *xs):
    r = one()
    for x in xs:
        r = mul(R, r, x)
    return r


class Res:
    def __init__(self):
        self.d = {}
        self.order = []

    def bump(self, key, ok):
        if key not in self.d:
            self.d[key] = [0, 0]
            self.order.append(key)
        self.d[key][0] += 1
        if not ok:
            self.d[key][1] += 1

    def report(self, title):
        print("=" * 78)
        print(title)
        print("=" * 78)
        bad = 0
        for k in self.order:
            n, f = self.d[k]
            bad += f
            flag = "ok  " if f == 0 else "FAIL"
            print("  [%s] %-62s %4d/%d" % (flag, k, n - f, n))
        return bad


# =========================================================== A, B: the set ===

def check_set(rng, R, n=NSAMP):
    """A: S closed under conjugation; S_0 multiplicative; s s_c = s_ev^2 in S_0.
       B: a t in S may be replaced by t_ev^2 in S_0."""
    res = Res()
    inS = IN_S[id(R)]
    hits = [0]
    sample = (lambda: rnd_unit(rng)) if R is U else (lambda: rnd_S_T(rng))
    for _ in range(n):
        s, u = sample(), sample()
        res.bump("s in S  =>  s_c in S", inS(conj(s)))
        res.bump("s in S  =>  s_ev in S", inS(ev(s)))
        e, f = sq_ev(R, s), sq_ev(R, u)
        res.bump("s s_c = s_c s = s_ev^2",
                 eq(mul(R, s, conj(s)), e) and eq(mul(R, conj(s), s), e))
        res.bump("s_ev^2 in S_0 (even and in S)", inS(e) and not od(e))
        res.bump("S_0 multiplicatively closed", inS(mul(R, e, f)) and not od(mul(R, e, f)))
        res.bump("even elements are central", eq(mul(R, e, s), mul(R, s, e)))

        # B: t in S kills z  =>  t_ev^2 in S_0 kills z.  The implication is
        # vacuous unless t z = 0 actually happens, so the hits are counted.
        t = sample()
        z = rnd(R, rng)
        if not mul(R, t, z):
            hits[0] += 1
            res.bump("t z = 0  =>  t_ev^2 z = 0 (t in S)  [non-vacuous only]",
                     not mul(R, sq_ev(R, t), z))
    # explicit non-vacuous instances of B, one per model
    if R is T:
        # t = y + xi_1 is in S, is NOT even, and kills the non-zero z = x xi_1
        t = add(polgen('y', 1), gen_xi(0))
        z = mul(T, polgen('x', 1), gen_xi(0))
        res.bump("model T: t = y+xi_1 in S is non-homogeneous, t z = 0 for z = x xi_1 != 0",
                 in_S_T(t) and bool(od(t)) and bool(z) and not mul(T, t, z))
        res.bump("model T: and then t_ev^2 = y^2 in S_0 kills the same z",
                 in_S_T(sq_ev(T, t)) and not od(sq_ev(T, t)) and not mul(T, sq_ev(T, t), z))
    print("  [note] model %s: %d of %d samples had t z = 0 (the rest are vacuous)"
          % (R.name.split(":")[0], hits[0], n))
    return res


# ================================================ the two relations, as maps ==

def bracket(R, r, s, r2, s2):
    """Lebedev's bracket:  r s_c s2_ev^2 - r2 s2_c s_ev^2."""
    return sub(prod(R, r, conj(s), sq_ev(R, s2)),
               prod(R, r2, conj(s2), sq_ev(R, s)))


def bracket0(R, r, s, r2, s2):
    """the commutative bracket of ~_0 (s, s2 even):  r s2 - r2 s."""
    return sub(mul(R, r, s2), mul(R, r2, s))


def rel_U(r, s, r2, s2):
    """model U: Lebedev's relation, decided exactly (t is a unit, so t drops out)."""
    return not bracket(U, r, s, r2, s2)


# ==================================================== C, D, E: the two axioms ==

def check_axioms(rng, R, n=NSAMP):
    res = Res()
    sample = (lambda: rnd_unit(rng)) if R is U else (lambda: rnd_S_T(rng))
    for _ in range(n):
        s, s2, s3, u = sample(), sample(), sample(), sample()
        r, r2, r3 = rnd(R, rng), rnd(R, rng), rnd(R, rng)

        # C: axiom (1), with t = 1 -- the bracket must vanish identically
        res.bump("(1) (r,s) ~ (ru,su): bracket vanishes identically",
                 not bracket(R, r, s, mul(R, r, u), mul(R, s, u)))

        # ... and the same on the left, (r,s) ~ (ur, us)
        res.bump("(1-left) (r,s) ~ (ur,us) for even u",
                 not bracket(R, r, s, mul(R, ev(u), r), mul(R, ev(u), s)))

        # D: axiom (2).  Take even denominators.
        e, e2 = ev(s), ev(s2)
        if in_S[R](e) and in_S[R](e2):
            b_al = bracket(R, r, e, r2, e2)          # = e e2 (r e2 - r2 e)
            b_00 = bracket0(R, r, e, r2, e2)
            res.bump("(2) for s,s' in S_0: Lebedev bracket = s s' (commutative bracket)",
                     eq(b_al, prod(R, e, e2, b_00)))
            # forward: t kills the commutative bracket => t kills Lebedev's
            t = sq_ev(R, sample())
            res.bump("(2=>) t(r s'-r' s)=0  =>  t(Lebedev bracket)=0",
                     (not mul(R, t, b_00)) <= (not mul(R, t, b_al)))
            # backward: t kills Lebedev's => t s s' in S_0 kills the commutative one
            tss = prod(R, t, e, e2)
            res.bump("(2<=) t(Lebedev)=0  =>  t s s' in S_0 kills (r s'-r' s)",
                     (not mul(R, t, b_al)) <= (not mul(R, tss, b_00)))

        # E: reflexivity, symmetry, and the constructive transitivity step
        res.bump("reflexive", not bracket(R, r, s, r, s))
        res.bump("symmetric", eq(bracket(R, r, s, r2, s2),
                                 neg(bracket(R, r2, s2, r, s))))
        b1 = bracket(R, r, s, r2, s2)
        b2 = bracket(R, r2, s2, r3, s3)
        e2s = sq_ev(R, s2)
        t1, t2 = sq_ev(R, sample()), sq_ev(R, sample())
        # the composite witness of the proof:  t1 t2 s2_ev^2
        lhs = prod(R, t1, t2, e2s, bracket(R, r, s, r3, s3))
        rhs = add(prod(R, t2, sq_ev(R, s3), mul(R, t1, b1)),
                  prod(R, t1, sq_ev(R, s), mul(R, t2, b2)))
        res.bump("transitivity: t1 t2 s'^2_ev (bracket_13) = "
                 "s''^2 t1 b_12 + s^2 t2 b_23", eq(lhs, rhs))
    return res


in_S = {U: in_S_U, T: in_S_T}


# ============================================ F, G: uniqueness and the truth ==

def check_unique_and_truth(rng, n=NSAMP):
    """Model U only: the fractions really exist there."""
    res = Res()
    for _ in range(n):
        s, s2, u = rnd_unit(rng), rnd_unit(rng), rnd_unit(rng)
        r, r2 = rnd(U, rng), rnd(U, rng)
        sc = conj(s)

        # F: the uniqueness step -- (1) with u = s_c
        res.bump("F: s_c in S, so (1) applies with u = s_c", in_S_U(sc))
        res.bump("F: (r,s) ~ (r s_c, s s_c) and s s_c = s_ev^2",
                 rel_U(r, s, mul(U, r, sc), mul(U, s, sc))
                 and eq(mul(U, s, sc), sq_ev(U, s)))

        # G: Lebedev's relation <=> genuine equality of right fractions
        truth = eq(mul(U, r, inv_U(s)), mul(U, r2, inv_U(s2)))
        res.bump("G: Lebedev relation <=> r s^{-1} = r' s'^{-1}", rel_U(r, s, r2, s2) == truth)

        # and on the left, with s_c r
        truth_L = eq(mul(U, inv_U(s), r), mul(U, inv_U(s2), r2))
        left = not sub(prod(U, sq_ev(U, s2), sc, r),
                       prod(U, sq_ev(U, s), conj(s2), r2))
        res.bump("G-left: Lebedev relation on (s,r) <=> s^{-1} r = s'^{-1} r'", left == truth_L)

        # the class of (r,s) and of (ru,su) is the same fraction
        res.bump("G: (r,s) and (ru,su) are the same element of A_S",
                 eq(mul(U, r, inv_U(s)), mul(U, mul(U, r, u), inv_U(mul(U, s, u)))))
    return res


# ================================================ H: the bijection A_S = A_S0 ==

def check_bijection(rng, n=NSAMP):
    """Model U: (r,s) |-> (r s_c, s_ev^2) is a bijection of classes, identical on
       A x S_0, and transports + and * to the true operations."""
    res = Res()
    for _ in range(n):
        s, s2 = rnd_unit(rng), rnd_unit(rng)
        r, r2 = rnd(U, rng), rnd(U, rng)
        a, e = mul(U, r, conj(s)), sq_ev(U, s)
        a2, e2 = mul(U, r2, conj(s2)), sq_ev(U, s2)

        res.bump("H: image lands in A x S_0", in_S_U(e) and not od(e))
        res.bump("H: the map does not change the fraction",
                 eq(mul(U, r, inv_U(s)), mul(U, a, inv_U(e))))
        # identity on A x S_0 (up to ~_0): (r,e) |-> (r e, e^2)
        ee = ev(s)
        if in_S_U(ee):
            res.bump("H: identical on classes of A x S_0",
                     eq(mul(U, r, inv_U(ee)),
                        mul(U, mul(U, r, ee), inv_U(mul(U, ee, ee)))))
        # onto: every (a,e) with e in S_0 is the image of itself-ish
        res.bump("H: onto (a,e) with e even is the image of (a,e)",
                 eq(mul(U, a, inv_U(e)),
                    mul(U, mul(U, a, conj(e)), inv_U(sq_ev(U, e)))))
        # + and * transported from A_{S_0}
        res.bump("H: sum transported = true sum",
                 eq(add(mul(U, r, inv_U(s)), mul(U, r2, inv_U(s2))),
                    mul(U, add(mul(U, a, e2), mul(U, a2, e)), inv_U(mul(U, e, e2)))))
        res.bump("H: product transported = true product",
                 eq(mul(U, mul(U, r, inv_U(s)), mul(U, r2, inv_U(s2))),
                    mul(U, mul(U, a, a2), inv_U(mul(U, e, e2)))))
        # no signs anywhere: the transported operations are the commutative ones
        res.bump("H: A_{S_0} is (super)commutative on the nose",
                 eq(mul(U, mul(U, a, e2), inv_U(mul(U, e, e2))),
                    mul(U, mul(U, e2, a), inv_U(mul(U, e2, e)))))
    return res


# ================================== I: why the naive cross-multiplication fails ==

def check_naive(rng, n=NSAMP):
    res = Res()
    bad = 0
    witness = None
    for _ in range(n):
        s, u = rnd_unit(rng), rnd_unit(rng)
        r = rnd(U, rng)
        naive = not sub(mul(U, r, mul(U, s, u)), mul(U, mul(U, r, u), s))
        res.bump("I: Lebedev's relation does hold for (r,s) vs (ru,su)",
                 rel_U(r, s, mul(U, r, u), mul(U, s, u)))
        if not naive:
            bad += 1
            if witness is None:
                witness = (r, s, u)
    print()
    print("  the naive cross-multiplication r_1 s_2 = r_2 s_1 fails requirement (1)")
    print("  on %d of %d samples." % (bad, n))
    if witness:
        r, s, u = witness
        print("  a minimal witness of the same kind:")
        r, s, u = one(), add(one(), gen_xi(0)), add(one(), gen_xi(1))
        print("    r = %s,  s = %s,  u = %s" % (show(r), show(s), show(u)))
        print("    r s^{-1} = (ru)(su)^{-1} = %s" % show(mul(U, r, inv_U(s))))
        print("    r(su) - (ru)s = %s   (non-zero)"
              % show(sub(mul(U, r, mul(U, s, u)), mul(U, mul(U, r, u), s))))
        print("    Lebedev bracket    = %s   (zero)"
              % show(bracket(U, r, s, mul(U, r, u), mul(U, s, u))))
    return res


# ====================================================================== main ==

def main():
    rng = random.Random(20260906)
    total = 0
    print("localization at the complement of a prime ideal: A. Lebedev's answer")
    print("model U = Q[x]/(x^3) (x) Lambda(3);   model T = Q[x,y]/(xy) (x) Lambda(3)")
    print()
    total += check_set(rng, U).report("A, B -- the multiplicative system (model U)")
    total += check_set(rng, T).report("A, B -- the multiplicative system (model T)")
    total += check_axioms(rng, U).report("C, D, E -- requirements (1), (2), equivalence (model U)")
    total += check_axioms(rng, T).report("C, D, E -- requirements (1), (2), equivalence (model T)")
    total += check_unique_and_truth(rng).report("F, G -- uniqueness step, and the relation is the truth")
    total += check_bijection(rng).report("H -- A_S = A_{S_0} as algebras")
    total += check_naive(rng).report("I -- the naive cross-multiplication")
    print()
    print("=" * 78)
    print("TOTAL FAILURES: %d" % total)
    print("=" * 78)
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
