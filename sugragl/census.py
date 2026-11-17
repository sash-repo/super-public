#!/usr/bin/env python3
"""
census.py -- which models M~(Par) there are, and which of them are "the most
symmetric".

A model is a parity string Par with four positions of one parity (the spacetime
sl(4)) and N of the other, together with the marking read off it by the block
rule.  Two strings give the SAME model when they differ by

  (F) the global parity flip 0 <-> 1.  This changes nothing at all: a matrix
      unit E_ij has parity p_i + p_j, which is invariant under p -> p + 1, and
      so is the parity sum(c_i p_i) of a weight vector, since sum c_i = 0 for
      every weight of gl.  (This is the Table's own remark, and the engine
      confirms it: run.py on Par and on its flip returns identical answers.)
  (R) reversal of Par together with reversal of the blocks.  Conjugating by the
      antidiagonal permutation sends deg to -deg, and X -> -X^{st} sends it back,
      so the reversed string gives an isomorphic graded model.

"Most symmetric" is then two conditions, both of them the paper's own:

  (S1) every block has size 2, so that g_0 is a product of copies of sl(2) and
       there are as many of them as possible -- which is what the text asks for
       ("the models whose stationary subgroup is ... a product of several copies
       of SL(2): otherwise the equations will be of order > 2");
  (S2) the model is isomorphic to its own reversal, i.e. it admits the extra
       automorphism w_0 o (-st) -- the block word is a palindrome (up to (F)).

Under (S1) the datum is just the pair of block slots carrying the spacetime, so
the census is a count of 2-subsets of {1,...,r}, r = (N+4)/2, up to reversal.
"""

from itertools import combinations

from models import blocks_of

TABULATED = ["0000", "00100", "00001", "001100", "100001", "000011",
             "00111100", "00001111", "111100001111"]


def word(r, pair, spacetime="0"):
    """The Par string of the all-size-2 model whose spacetime blocks are the
    two slots in `pair` (1-based) out of r."""
    other = "1" if spacetime == "0" else "0"
    return "".join((spacetime if i + 1 in pair else other) * 2
                   for i in range(r))


def canonical(par):
    """A canonical name for the model of Par, under (F) and (R)."""
    flip = par.translate(str.maketrans("01", "10"))
    return min(par, par[::-1], flip, flip[::-1])


def census(N):
    r = (N + 4) // 2
    seen, out = set(), []
    for pair in combinations(range(1, r + 1), 2):
        par = word(r, pair)
        c = canonical(par)
        if c in seen:
            continue
        seen.add(c)
        rev = tuple(sorted(r + 1 - i for i in pair))
        out.append((pair, par, rev == pair))
    return out


if __name__ == "__main__":
    tab = {canonical(p) for p in TABULATED}
    print("All-size-2 models, by N.  * = self-dual (S2);  T = in the Table.\n")
    print(f"{'N':>2} {'slots':<8} {'Par':<14} {'blocks':<16} {'S2':>3} {'T':>2}")
    todo = []
    for N in (0, 2, 4, 6, 8):
        for (pair, par, sym) in census(N):
            bl = "|".join(f"{s}{'o' if p else 'e'}" for s, p in blocks_of(par))
            intab = canonical(par) in tab
            print(f"{N:>2} {str(pair):<8} ({par})".ljust(28)
                  + f"{bl:<16} {'*' if sym else '':>3} {'T' if intab else '':>2}")
            if sym and not intab:
                todo.append((N, par))
    print()
    for N in (0, 2, 4, 6, 8):
        c = census(N)
        ns = sum(1 for _, _, s in c if s)
        nt = sum(1 for _, p, _ in c if canonical(p) in tab)
        print(f"  N = {N}: {len(c)} models, {ns} self-dual, {nt} in the Table")
    print("\nthe most symmetric models NOT yet computed:")
    for N, par in todo:
        print(f"  N = {N}:  Par = ({par})")
    # the odd-N and mixed-block models, for the record
    print("\n(the Table's other columns have a block of size 1, hence fewer "
          "sl(2) factors:")
    for p in TABULATED:
        b = blocks_of(p)
        if any(s == 1 for s, _ in b):
            print(f"   ({p}): "
                  + "|".join(f"{s}{'o' if q else 'e'}" for s, q in b))
    print(" so they are not 'most symmetric' in the sense of (S1).)")
