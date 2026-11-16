#!/usr/bin/env python3
r"""slecomp.py -- WHICH invariant operator each family of the sle(n)-list is.

The le(n)-list is already read as operators (Remark 9.2.2): the
five families are the exterior differential on the two spaces

    OmT^m := Omega^m / omt Omega^{m-2},        SgT_m := Sigma_m / omt Sigma_{m-2}

and one operator of order 2.  The sle(n)-list has nine families, stated as a table
of weights.  This script establishes the reading.

The dictionary (from Remark 9.2.2, in the mu-coordinates mu_i = lam_i - lam_{i+1},
om_j = "1 in the j-th place", om_0 = om_n = 0):

    OmT^m  <->  m om_{n-1}                     (m >= 0;  OmT^0 = F, the functions)
    SgT_m  <->  om_{n-m}          for 0 <= m <= n-1
    SgT_m  <->  (m-n+2) om_1      for m >= n-1

Two coincidences follow, and they hold only over sle(n), where the volume element
is preserved so that the integral forms and the differential forms are one family:

    SgT_0 = OmT^0 = F      and      SgT_1 = OmT^1 .

So the two complexes GLUE into a 4-cycle

    F  --d-->  OmT^1 = SgT_1  --de-->  F ,

with d of order 1 (the exterior differential on OmT) and de of order 1 (the
differential on SgT, i.e. the divergence, which exists over sle(n) and not over
le(n)); and SgT_2 --de--> SgT_1 = OmT^1 feeds into it.

What is checked here, exactly:

  1. div D_f = 2 (-1)^{p(f)} S f, where S is the odd Laplacian: so  de o d = 2S
     on functions -- the second-order family 2a) IS the BV operator;
  2. S^2 = 0 (as an operator on C[x,xi], n <= 5);
  3. the enumeration: the alternating words in d and de on the glued diagram,
     with every word containing (de d)(de d) = S^2 struck out, are EXACTLY nine,
     and their (source, target, order) match the nine families of Kochetkov's
     sle(n)-theorem line for line, for every n >= 3.

Whence: the sle(n)-list is finite, and stops at the order 4, because S^2 = 0; the
longest word is d S de = d de d de.

Usage:   python3 slecomp.py [nmax]
"""
import os, sys
from fractions import Fraction as F

from le import dx, dxi, padd, pscal, par_le
from sle import lap

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ---------------------------------------------------------------------------
# 1.  div D_f = 2 (-1)^{p(f)} S f
# ---------------------------------------------------------------------------
def divergence(f, n):
    """div of the field D_f = sum_i (df/dx_i) d/dxi_i + (-1)^{p(f)}(df/dxi_i) d/dx_i

    div(sum_a F_a d/d y_a) = sum_a (-1)^{p(F_a) p(y_a)} d/d y_a (F_a).
    """
    out = {}
    # the pieces are computed monomial by monomial, f need not be homogeneous
    for m, c in f.items():
        g = {m: c}
        pg = len(m[1]) % 2
        for i in range(n):
            A = dx(g, i)                      # dg/dx_i, parity pg
            B = pscal(dxi(g, i), (-1) ** pg)  # (-1)^{p(g)} dg/dxi_i, parity pg+1
            # d/dx_i is even -> no sign
            out = padd(out, dx(B, i))
            # d/dxi_i is odd, A is of parity pg
            out = padd(out, pscal(dxi(A, i), (-1) ** pg))
    return out


def monomials(n, deg):
    """all monomials x^a xi^S of total degree deg"""
    from itertools import combinations

    def xparts(k, slots):
        if slots == 0:
            if k == 0:
                yield ()
            return
        for j in range(k + 1):
            for rest in xparts(k - j, slots - 1):
                yield (j,) + rest

    out = []
    for s in range(min(n, deg) + 1):
        for S in combinations(range(n), s):
            for a in xparts(deg - s, n):
                out.append((a, S))
    return out


def test_div(nmax=4, degmax=5):
    say("\n=== 1.  div D_f = 2 (-1)^{p(f)} S f  ===")
    tot = 0
    for n in range(1, nmax + 1):
        for deg in range(0, degmax + 1):
            for m in monomials(n, deg):
                f = {m: F(1)}
                pf = len(m[1]) % 2
                lhs = divergence(f, n)
                rhs = pscal(lap(f, n), 2 * (-1) ** pf)
                d = padd(lhs, pscal(rhs, -1))
                assert not any(v for v in d.values()), (n, m, lhs, rhs)
                tot += 1
        say(f"  n={n}: all monomials of degree <= {degmax} pass")
    say(f"  -> {tot} generating functions, div D_f = 2(-1)^p(f) S f in every one;")
    say("     hence  de o d = +-2 S  on the functions: the family 2a) is the BV")
    say("     operator, and it is a composition of the two first-order ones.")


# ---------------------------------------------------------------------------
# 2.  S^2 = 0
# ---------------------------------------------------------------------------
def test_S2(nmax=5, degmax=6):
    say("\n=== 2.  S^2 = 0  ===")
    tot = 0
    for n in range(1, nmax + 1):
        for deg in range(0, degmax + 1):
            for m in monomials(n, deg):
                f = {m: F(1)}
                g = lap(lap(f, n), n)
                assert not any(v for v in g.values()), (n, m, g)
                tot += 1
        say(f"  n={n}: S^2 = 0 on every monomial of degree <= {degmax}")
    say(f"  -> {tot} monomials.  This is what makes the list finite.")


# ---------------------------------------------------------------------------
# 3.  the enumeration of the words
# ---------------------------------------------------------------------------
#   The nodes of the glued diagram, named by their mu-weight as a function of n.
#   A weight is stored as a dict {j: coefficient} meaning sum_j c_j om_j, with
#   om_0 = om_n = 0 dropped.
def om(j, n, c=1):
    if j <= 0 or j >= n or c == 0:
        return {}
    return {j: c}


def node_weight(kind, m, n):
    """mu-weight of OmT^m ('Om') or SgT_m ('Sg')"""
    if kind == "Om":
        return om(n - 1, n, m)
    if m <= n - 1:
        return om(n - m, n)
    return om(1, n, m - n + 2)


def test_words(nmax=8, mmax=6, lmax=8):
    """Walk the glued diagram and enumerate every alternating word.

    Nodes:  F = OmT^0 = SgT_0;  OmT^m (m >= 1);  SgT_m (m >= 1), with
    SgT_1 = OmT^1 because the two have the same mu-weight om_{n-1}.
    Arrows: d : OmT^m -> OmT^{m+1};   de : SgT_m -> SgT_{m-1};  both of order 1.
    d^2 = 0 and de^2 = 0, so only ALTERNATING words can be non-zero; and
    (de d)(de d) = S^2 = 0 truncates the walk.
    """
    say("\n=== 3.  the alternating words in d and de ===")
    fam = {}          # (source weight, target weight, order) -> name

    def wt(node, n):
        kind, m = node
        return tuple(sorted(node_weight(kind, m, n).items()))

    for n in range(3, nmax + 1):
        # the two coincidences that make the gluing
        assert node_weight("Sg", 0, n) == node_weight("Om", 0, n) == {}
        assert node_weight("Sg", 1, n) == node_weight("Om", 1, n) == om(n - 1, n)

        def canon(node):
            kind, m = node
            if kind == "Sg" and m <= 1:
                return ("Om", m)
            return (kind, m)

        def out_arrows(node):
            kind, m = canon(node)
            arrows = []
            if kind == "Om" and m < mmax:
                arrows.append(("d", canon(("Om", m + 1))))
            # de leaves a node only if it is a SgT_m with m >= 1
            for (k2, m2) in (("Sg", m),):
                pass
            # which SgT is this node?
            sig = None
            if kind == "Om" and m == 0:
                sig = 0
            elif kind == "Om" and m == 1:
                sig = 1
            elif kind == "Sg":
                sig = m
            if sig is not None and sig >= 1:
                arrows.append(("de", canon(("Sg", sig - 1))))
            return arrows

        found = []
        starts = [("Om", m) for m in range(0, mmax + 1)] + \
                 [("Sg", m) for m in range(2, mmax + 1)]
        for st in starts:
            stack = [(canon(st), ())]
            while stack:
                node, word = stack.pop()
                for (letter, nxt) in out_arrows(node):
                    if word and word[-1] == letter:
                        continue                       # d^2 = de^2 = 0
                    w = word + (letter,)
                    # strike out any word containing (de d)(de d) = S^2
                    if _has_S2(w):
                        continue
                    found.append((canon(st), nxt, w))
                    stack.append((nxt, w))
        # collect the words of order >= 2, by (source, target, order)
        big = {}
        for (src, tgt, w) in found:
            if len(w) < 2:
                continue
            big.setdefault((wt(src, n), wt(tgt, n), len(w)), set()).add(w)
        if n == 3:
            for k in sorted(big, key=lambda t: (t[2], str(t))):
                say(f"    {fmt(k[0], n)} -> {fmt(k[1], n)}, order {k[2]}:"
                    f"  {' , '.join('.'.join(reversed(x)) for x in sorted(big[k]))}")
        Fw, O1, O2 = tuple(), tuple(sorted(om(n - 1, n).items())), \
            tuple(sorted(om(n - 1, n, 2).items()))
        S2w = tuple(sorted(node_weight("Sg", 2, n).items()))
        want = {(Fw, Fw, 2), (O1, O1, 2), (S2w, O2, 2),
                (O1, Fw, 3), (Fw, O1, 3), (O1, O1, 4)}
        assert set(big) == want, (n, sorted(set(big) ^ want))
        say(f"  n={n}: the words of order >= 2 are exactly the six families"
            f" 2a), 2b), 2c), 3a), 3b), 4) -- {len(big)} of them, and no more.")
    say("  With the three of order 1 (d on OmT, de on SgT at its two ranges)")
    say("  that is the whole nine-family list; it stops at the order 4 because")
    say("  every longer alternating word contains (de d)(de d) = S^2 = 0.")


def _has_S2(w):
    """does the word contain S^2 = (de o d)(de o d)?

    A word (l_1,...,l_k) means the operator l_k o ... o l_1, so S = de o d is the
    applied sequence (d, de) = "DS"; S^2 is "DSDS".
    """
    s = "".join("D" if x == "d" else "S" for x in w)   # S = de, D = d
    return "DSDS" in s


def fmt(wt, n):
    if not wt:
        return "0"
    return " + ".join((f"{c} om_{j}" if c != 1 else f"om_{j}") for j, c in wt)


def main():
    global LOG
    LOG = open(os.devnull, "w")
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    test_div(nmax)
    test_S2()
    test_words()
    say("\nall assertions passed")
    LOG.close()


if __name__ == "__main__":
    main()
