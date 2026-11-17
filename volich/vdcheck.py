#!/usr/bin/env python3
"""Independent, deliberately generous check of the one point on which the
answer of Subsecs. 3.7--3.9 turns: does a + [d,a] preserve M = k[Z] for every even
derivation a of S, as Iyer's Proposition 4.2.1(1) asserts?

Nothing is truncated cleverly here: M is spanned by ALL words in the z_i up to
a length given on the command line, and every claim is reported with the
witness.  If this disagrees with vdmeta.py, this file is the one to trust.
"""

import sys
from fractions import Fraction

from iyer import (mono, add, scal, sub, is_zero, mul_S, z_word_n, words, Span,
                  basis)
from vdmeta import (VF, gen_x, gen_y, sbracket, vf_add, show_vf, der_basis,
                    s_basis)


def m_span_upto(n, length):
    sp = Span.__new__(Span)
    sp.rows = {}
    for L in range(length + 1):
        for w in words(n, L):
            sp.insert(z_word_n(w, n))
    return sp


def max_xdeg(a):
    return max((sum(k[0]) for k in a), default=0)


def check(n, xdeg=1, wordlen=3, spanlen=9):
    d = VF(n)
    for k in range(n):
        d.a[k] = gen_y(n, k)

    sp = m_span_upto(n, spanlen)
    testers = [(w, z_word_n(w, n)) for L in range(wordlen + 1)
               for w in words(n, L)]

    print(f"n={n}: M spanned by all words of length <= {spanlen} "
          f"({len(sp.rows)} independent elements); testing a+[d,a] on words of "
          f"length <= {wordlen}, a of x-degree <= {xdeg}")

    bad = []
    worst = 0
    for name, a in der_basis(n, xdeg, parity=0):
        D = vf_add(a, sbracket(d, a, n))
        A = D.op()
        for w, zm in testers:
            img = A(zm)
            worst = max(worst, max_xdeg(img))
            if sp.reduce(img):
                bad.append((name, w, img))
                break
    print(f"   highest x-degree reached by any image: {worst} "
          f"(the span must cover it: words of length <= {spanlen} reach "
          f"x-degree {spanlen})")
    if not bad:
        print(f"   OK: all {len(der_basis(n, xdeg, parity=0))} of the "
              f"a + [d,a] preserve M")
    else:
        print(f"   {len(bad)} of {len(der_basis(n, xdeg, parity=0))} FAIL:")
        for name, w, img in bad[:4]:
            a = dict(der_basis(n, xdeg, parity=0))[name]
            D = vf_add(a, sbracket(d, a, n))
            print(f"      a = {name}")
            print(f"      a+[d,a] = {show_vf(D, n)}")
            print(f"      at the word z{tuple(k+1 for k in w)}: image leaves M")
            print(f"          image      = {img}")
            print(f"          remainder  = {sp.reduce(img)}")
    return bad


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    spanlen = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    check(n, spanlen=spanlen)
