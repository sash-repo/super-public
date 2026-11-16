"""hI.py -- the structure constants of h_I(3;1), the smallest Lie algebra we
know with a (2,4)-structure and no 2-structure.

h_I(2n+1;1) = Span{H_f | f in O(2n+1;1)} for the non-alternating symplectic
form omega_I = (dx)^(2) + sum_i du_i du~_i; see
[BLLS, Subsec. 2.6.1].
For n = 1: dim = 2^3 - 1 = 7, the Z/2-grading is by the x-degree, dim g_+ = 3,
dim g_- = 4, there is no 2-structure and there is a (2,4)-structure.

The vectorial engine is nalt2.py + hamcheck.py; this file turns its output
into a plain table of structure constants over F_2.
"""

import os
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from nalt2 import Alg, bracket, act                      # noqa: E402
from hamcheck import inv_gram, hamiltonian               # noqa: E402
from nalt2 import gram                                   # noqa: E402


def omega_std(alg):
    """(dx)^(2) + du_1 du~_1 + ... ; index 0 is x, then the pairs."""
    m = alg.m
    om = {('b', 0): alg.one}
    n = (m - 1) // 2
    for i in range(1, n + 1):
        om[('a', i, n + i)] = alg.one
    return om


def flat(alg, D):
    """a vector field as an int: coefficient of d/du_k in block k"""
    v = 0
    for k in range(alg.m):
        v |= D[k] << (k * alg.dim)
    return v


def unflat(alg, v):
    mask = (1 << alg.dim) - 1
    return [(v >> (k * alg.dim)) & mask for k in range(alg.m)]


def xdeg_mono(alg, mono):
    """the degree in x = u_0 of the monomial with index mono"""
    return 1 if (alg.mons[mono][0] % 2) else 0


def build(m=3):
    """returns (dim, br, grading, two_map, four_map, names)"""
    alg = Alg(m, [1] * m)
    om = omega_std(alg)
    Ginv = inv_gram(alg, gram(alg, om))

    # H_f for f a monomial; H_1 = 0
    gens = []
    names = []
    for t in range(alg.dim):
        f = 1 << t
        D = hamiltonian(alg, Ginv, f)
        if any(D):
            gens.append((t, D))

    # a basis of Span{H_f} among the H_monomial, with tracked combinations so
    # that "express" below can write any field in it
    piv = {}
    basis2 = []
    for (t, D) in gens:
        v = flat(alg, D)
        w, comb = v, 0
        added = False
        while w:
            h = w.bit_length() - 1
            if h in piv:
                w ^= piv[h][0]
                comb ^= piv[h][1]
            else:
                idx = len(basis2)
                piv[h] = (w, comb ^ (1 << idx))
                basis2.append((t, D, v))
                added = True
                break
        if not added:
            pass
    dim = len(basis2)

    def express(v):
        w, comb = v, 0
        while w:
            h = w.bit_length() - 1
            if h not in piv:
                raise ValueError("outside Span{H_f}")
            w ^= piv[h][0]
            comb ^= piv[h][1]
        return comb

    br = [[0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(i + 1, dim):
            b = bracket(alg, basis2[i][1], basis2[j][1])
            v = express(flat(alg, b))
            br[i][j] = v
            br[j][i] = v

    # the grading: with deg x = odd and deg u_i = deg u~_i = even, the field
    # H_f has the degree of f, i.e. the parity of its x-degree
    grading = []
    for (t, D, v) in basis2:
        grading.append(xdeg_mono(alg, t))

    def sq(D):
        return [act(alg, D, D[k]) for k in range(alg.m)]

    two_map, four_map = {}, {}
    for i in range(dim):
        D = basis2[i][1]
        D2 = sq(D)
        D4 = sq(sq(D))
        if grading[i] == 0:
            two_map[i] = express(flat(alg, D2))
        else:
            four_map[i] = express(flat(alg, D4))
    names = [monomial_name(alg, t) for (t, D, v) in basis2]
    return dim, br, grading, two_map, four_map, names


def monomial_name(alg, t):
    ex = alg.mons[t]
    s = ""
    lbl = ["x"] + ["u%d" % i for i in range(1, (alg.m + 1) // 2)] + \
          ["v%d" % i for i in range(1, (alg.m + 1) // 2)]
    for k, e in enumerate(ex):
        if e:
            s += lbl[k] + ("^%d" % e if e > 1 else "")
    return "H_" + (s if s else "1")


if __name__ == "__main__":
    dim, br, grading, two_map, four_map, names = build(3)
    print("dim h_I(3;1) =", dim)
    print("basis:", names)
    print("grading (0 = g_+, 1 = g_-):", grading)
    print("dim g_+ =", grading.count(0), " dim g_- =", grading.count(1))
    print("two_map:", two_map)
    print("four_map:", four_map)
