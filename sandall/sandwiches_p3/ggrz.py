#!/usr/bin/env python3
r"""
ggrz.py --- the sandwich subalgebra of [GGRZ], cf. Remark remGGRZ of the text:

    "In [GGRZ], they write that their sandwich subalgebra is 3-dim.
     This contradicts (fs), right?"

[GGRZ] = Grishkov, Guzzo jr., Rasskazova, Zusmanovich, On simple 15-dimensional
Lie algebras in characteristic 2, J. Algebra 593 (2022) 295-318 =
arXiv:2104.01400.  Their algebra is the member L(0, 0) of a 2-parameter family
of 15-dimensional simple Lie algebras over a field of characteristic 2 -- by
their Theorem 1 the whole family is one algebra -- and it is the smallest
member of Skryabin's series [Sk, pp. 691-692].  Their Proposition 2(i): the
sandwich subalgebra of L is 3-dimensional abelian, spanned by c_2, c_4, c_5.

This script builds L(0, 0) from their own multiplication table (3), p. 4, and
computes, over F_2:

  * the Lie algebra axioms and simplicity (i.e., that the table is transcribed
    right and defines the algebra they say it does);
  * the number of x with (ad_x)^2 = 0, the dimension of Der L, and the
    sandwich subalgebra in THEIR definition -- (ad_x)^2 = 0 together with
    [[L, x], [L, x]] = 0;
  * the same for the basis of Skry in the library of [CSS] accompanying the
    classification of the simple 15-dimensional Lie algebras over F_2, which is
    the row "Skry" of the table of sandwich subalgebras of dimension 15;
  * the calibration of the identity (I) of oddsand.py at p = 2: for an EVEN x
    with (ad_x)^2 = 0, the condition [[L, x], [L, x]] = 0 is a genuine extra
    requirement -- the same degeneration that makes it one for an odd element
    in any characteristic.

Output: runs/ggrz.log
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from css import LieF2, rref, all_elements, load                        # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "runs", "ggrz.log")

NAMES = ["b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
         "c1", "c2", "c3", "c4", "c5", "d"]
IDX = {n: i for i, n in enumerate(NAMES)}

# Table (3) of [GGRZ] with beta = delta = 0, upper triangle, row by row.
# The column order is b2, b3, ..., b9, c1, ..., c5, d.
TABLE = {
    "b1": dict(zip(NAMES[1:], ["b3", "b1", "0", "b6", "b4", "0", "b9", "b7",
                               "d", "c3", "c1", "b2", "b5", "0"])),
    "b2": dict(zip(NAMES[2:], ["b2", "b6", "0", "b5", "b9", "0", "b8",
                               "c3", "0", "c2", "0", "0", "0"])),
    "b3": dict(zip(NAMES[3:], ["b4", "b5", "0", "b7", "b8", "0",
                               "c1", "c2", "0", "0", "0", "0"])),
    "b4": dict(zip(NAMES[4:], ["0", "0", "d", "c3", "c1",
                               "b3", "c4", "b2", "b5", "0", "b1"])),
    "b5": dict(zip(NAMES[5:], ["0", "c3", "0", "c2",
                               "c4", "0", "0", "0", "0", "b2"])),
    "b6": dict(zip(NAMES[6:], ["c1", "c2", "0",
                               "0", "0", "c4", "0", "0", "b3"])),
    "b7": dict(zip(NAMES[7:], ["0", "0",
                               "b6", "c5", "b5", "b8", "c2", "b4"])),
    "b8": dict(zip(NAMES[8:], ["0",
                               "c5", "0", "0", "0", "0", "b5"])),
    "b9": dict(zip(NAMES[9:], ["0", "0", "c5", "0", "0", "b6"])),
    "c1": dict(zip(NAMES[10:], ["0", "b8", "c2", "0", "b7"])),
    "c2": dict(zip(NAMES[11:], ["0", "0", "0", "b8"])),
    "c3": dict(zip(NAMES[12:], ["0", "0", "b9"])),
    "c4": dict(zip(NAMES[13:], ["0", "0"])),
    "c5": dict(zip(NAMES[14:], ["c4"])),
}


def ggrz_algebra():
    n = len(NAMES)
    tab = [[0] * n for _ in range(n)]
    for a, row in TABLE.items():
        for b, val in row.items():
            v = 0 if val == "0" else (1 << IDX[val])
            i, j = IDX[a], IDX[b]
            tab[i][j] = v
            tab[j][i] = v                       # p = 2: the bracket is symmetric
    return LieF2(tab, "L(0,0) of [GGRZ]")


def profile(say, L):
    ok, msg = L.check_jacobi()
    say("   Lie algebra axioms: %s" % msg)
    assert ok
    say("   dim %d, simple: %s, dim [L, L] = %d, dim Der L = %d"
        % (L.n, L.is_simple(), len(L.derived()), L.dim_der()))
    n2 = L.num_2nilps()
    S = L.sandwiches()
    B = rref(S)
    say("   elements with (ad_x)^2 = 0: %d" % n2)
    say("   of them satisfying [[L,x],[L,x]] = 0: %d, spanning dim %d"
        % (len(S), len(B)))
    sub = L.subalgebra_generated(S)
    say("   the sandwich subalgebra: dim %d, abelian: %s"
        % (len(sub), not any(L.mul(a, b) for a in sub for b in sub)))
    return n2, len(B), len(sub)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    say("ggrz: the sandwich subalgebra of the 15-dimensional Skryabin algebra,")
    say("      p = 2, from the multiplication table (3) of arXiv:2104.01400")
    say("")
    L = ggrz_algebra()
    say("1) L(0,0) built from their table (3):")
    n2a, spana, suba = profile(say, L)
    names = [NAMES[i] for i in range(L.n)
             if (1 << i) in L.sandwiches()]
    say("   the sandwiches among the basis elements: %s" % ", ".join(names))

    say("")
    say("2) the basis Skry.txt of the [CSS] library:")
    data = load(os.path.join(HERE, "css-bases", "dim15", "Skry.txt"))
    K = LieF2.from_adjoint_basis(data, "Skry [CSS]")
    n2b, spanb, subb = profile(say, K)

    say("")
    say("3) the two agree on every invariant computed: %s"
        % ("yes" if (n2a, spana, suba) == (n2b, spanb, subb) else "NO"))

    say("")
    say("4) p = 2 calibration: the 2-nilpotent elements that are NOT sandwiches")
    say("   number %d - %d = %d, so at p = 2 the condition [[L,x],[L,x]] = 0"
        % (n2a, len(L.sandwiches()), n2a - len(L.sandwiches())))
    say("   does not follow from (ad_x)^2 = 0 even for an EVEN x -- which is the")
    say("   same degeneration as for an odd x in any characteristic, both being")
    say("   the vanishing of the factor 1 + (-1)^{p(x)}.")

    say("")
    say("   and the identity (I), ad_x ad_y ad_x (z) = [[x,y],[x,z]], holds here")
    bad = 0
    for u in range(1, 1 << L.n):
        if any(L.ad2(u)):
            continue
        cols = L.ad(u)
        for j in range(L.n):
            xj = cols[j]                      # [u, e_j]
            for k in range(L.n):
                lhs = L.mul(u, L.mul(1 << j, cols[k]))
                rhs = L.mul(xj, cols[k])
                if lhs != rhs:
                    bad += 1
    say("   for every one of the %d elements with (ad_x)^2 = 0: %s"
        % (n2a, "OK" if not bad else "FAILED %d times" % bad))
    assert not bad

    say("")
    say("5) the rest of the table in Sec. 6 of [GGRZ], from the [CSS] library:")
    say("   (their S = the dimension of the sandwich subalgebra)")
    for name, theirs in (("W(1;[4])'", ("W(4)", 10)),
                         ("P(2;[3,1])", ("P(3,1)", 5)),
                         ("P(2;[2,2])", ("P(2,2)", 3)),
                         ("P(3;[2,1,1])", ("P(2,1,1)", 1)),
                         ("Skry", ("--", 3))):
        data = load(os.path.join(HERE, "css-bases", "dim15", "%s.txt" % name))
        M = LieF2.from_adjoint_basis(data, name)
        d = len(rref(M.sandwiches()))
        say("   %-14s = %-9s : ours %2d, theirs %2d  %s"
            % (name, theirs[0], d, theirs[1], "agree" if d == theirs[1] else "DISAGREE"))
        assert d == theirs[1]
    say("   so the 15-dimensional member of the four Cartan series at p = 2 is")
    say("   vect(1;(4))^(1) = W(1;[4])', whose sandwich subalgebra is of dim 10,")
    say("   not 3: the algebra of [GGRZ] is another one.")

    say("")
    say("NB on the convention: the counts above are over the NON-ZERO elements;")
    say("Table C.3 of [CSS] and the table of the text count the zero element too,")
    say("whence 80 and 8 there against 79 and 7 here.")

    say("")
    say("done.")
    log.close()


if __name__ == "__main__":
    main()
