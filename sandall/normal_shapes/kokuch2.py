#!/usr/bin/env python3
"""Two follow-ups to kokuch.py.

(1) The corner [KoKCh] excludes: n = 3, all heights 1, om = the standard shape.
    Their Theorem 4.1 asserts that [P,P] is simple of dim 2^|m| - 2 for n > 3,
    or n = 2,3 with m != 1.  For n = 3, m = (1,1,1) we found [P,P] of dim 6
    NOT simple; here we exhibit its proper ideals, so that the exclusion is
    documented and not just a negative from a black box.

(2) A positive control inside the range their Theorem 4.1 does cover:
    n = 4, u = 1 -- [P,P] should be simple of dim 2^4 - 2 = 14.
Output: kokuch2.log.
"""
from nalt2 import *
from run1 import show_form
from hamcheck import inv_gram, hamiltonian

LOG = None


def out(s):
    print(s)
    if LOG:
        LOG.write(s + "\n"); LOG.flush()


def ham_span(alg, om):
    Ginv = inv_gram(alg, gram(alg, om))
    return reduce_basis(alg, [hamiltonian(alg, Ginv, 1 << t)
                              for t in range(alg.dim)])


def canon(alg, vs):
    """canonical (fully reduced) row echelon form of a span, as a tuple."""
    piv = {}
    for D in vs:
        w = 0
        for i in range(alg.m):
            w |= D[i] << (i * alg.dim)
        while w:
            hb = w.bit_length() - 1
            if hb in piv:
                w ^= piv[hb]
            else:
                piv[hb] = w
                break
    keys = sorted(piv, reverse=True)
    for a in keys:                       # back-substitute
        for b in keys:
            if b > a and (piv[b] >> a) & 1:
                piv[b] ^= piv[a]
    return tuple(sorted(piv.values()))


def ideal_closure(alg, basis, seeds):
    cur = reduce_basis(alg, seeds)
    while True:
        new = reduce_basis(alg, cur + [bracket(alg, b, w)
                                       for b in basis for w in cur])
        if len(new) == len(cur):
            return cur
        cur = new


def show_field(alg, D):
    parts = []
    for i, f in enumerate(D):
        if f:
            parts.append("(%s)d_%d" % (alg.show(f), i + 1))
    return " + ".join(parts) if parts else "0"


def main():
    global LOG
    LOG = open(__file__.replace("kokuch2.py", "kokuch2.log"), "w")

    # ---------------------------------------------------------------- (1)
    alg = Alg(3)
    om = {('b', 0): alg.one, ('a', 1, 2): alg.one}
    P = ham_span(alg, om)
    D1 = reduce_basis(alg, derived(alg, P))
    out("n = 3, u = 1, omega = %s" % show_form(alg, om))
    out("dim P = %d, dim [P,P] = %d  (2^|m| - 1 = %d, 2^|m| - 2 = %d)"
        % (len(P), len(D1), alg.dim - 1, alg.dim - 2))
    ideals = []
    for bits in range(1, 1 << len(D1)):
        v = [0] * alg.m
        for i in range(len(D1)):
            if (bits >> i) & 1:
                for k in range(alg.m):
                    v[k] ^= D1[i][k]
        I = ideal_closure(alg, D1, [v])
        if len(I) < len(D1):
            ideals.append((len(I), canon(alg, I)))
    uniq = sorted(set(ideals))
    out("proper non-zero ideals of [P,P] over F_2: %d (as subspaces: %d)"
        % (len(ideals), len(uniq)))
    for d, I in uniq:
        flds = [[(w >> (i * alg.dim)) & ((1 << alg.dim) - 1)
                 for i in range(alg.m)] for w in I]
        out("  ideal of dim %d, abelian %s: %s"
            % (d, not reduce_basis(alg, derived(alg, flds)),
               "; ".join(show_field(alg, D) for D in flds)))
    out("[P,P] abelian? %s" % (not reduce_basis(alg, derived(alg, D1))))
    out("dims of the derived series of [P,P]: %s"
        % derived_series(alg, D1, steps=4))

    # ---------------------------------------------------------------- (2)
    out("")
    alg4 = Alg(4)
    om4 = {('b', 0): alg4.one, ('a', 1, 2): alg4.one, ('a', 0, 3): alg4.one}
    out("n = 4, u = 1, omega = %s : closed %s, nondeg %s, non-alternating %s"
        % (show_form(alg4, om4), not d2(alg4, om4), nondeg(alg4, om4),
           not alternating(om4)))
    P4 = ham_span(alg4, om4)
    D4 = reduce_basis(alg4, derived(alg4, P4))
    out("dim P = %d, dim [P,P] = %d  (2^|m| - 1 = %d, 2^|m| - 2 = %d)"
        % (len(P4), len(D4), alg4.dim - 1, alg4.dim - 2))
    bad = 0
    for bits in range(1, 1 << len(D4)):
        v = [0] * alg4.m
        for i in range(len(D4)):
            if (bits >> i) & 1:
                for k in range(alg4.m):
                    v[k] ^= D4[i][k]
        if len(ideal_closure(alg4, D4, [v])) < len(D4):
            bad += 1
    out("elements of [P,P] generating a proper ideal: %d (must be 0 for "
        "simplicity over F_2)" % bad)
    LOG.close()


if __name__ == "__main__":
    main()
