#!/usr/bin/env python3
"""What [KoKCh] = Kondrateva-Kuznetsov-Chebochko, arXiv:1812.11213, achieved:
a machine check of the three families they ANNOUNCE at the end of the paper
(their Theorems 5.3, 5.4, 5.5) in the only case our engine covers exhaustively,

        n = 3 indeterminates, all heights m_i = 1 (i.e. the shearing vector
        u = 1), ground field F_2.

Their forms, with the notation z_i = x_i^{(2^{m_i})}, dz_i = x_i^{(2^{m_i}-1)}dx_i,
which for m_i = 1 reads dz_i = x_i dx_i:

  om1 = dx_1dx_2 + (dx_3)^(2) + a x_1x_2 dx_1dx_2 + b x_1x_3 dx_1dx_3
                              + c x_2x_3 dx_2dx_3
  om2 = (dx_1)^(2) + (dx_2)^(2) + (dx_3)^(2) + <the same tail>
  om3 = dx_1dx_2 + (dx_2)^(2) + (dx_3)^(2) + <the same tail>

Their claims for m = (1,1,1):
  Th. 5.3  om1: if b != 0 or c != 0 (m_3 = 1 here), P(3,m,om1) is simple of
           dim 2^|m| - 1 = 7; if b = c = 0 and (m_1,m_2) != (1,1) -- excluded
           here -- the derived algebra is simple of dim 6.  For b = c = 0 and
           m = (1,1,1) their theorem says NOTHING; we compute it.
  Th. 5.4  om2: simple of dim 7 for (a,b,c) != (t,t,t).
  Th. 5.5  om3: if b != 0 or a != c, simple of dim 7; if a = c, b = 0 and
           (m_1,m_3) = (1,1), then -- m_2 = 1 as well here, so again their
           statement does not apply; we compute it.

What is checked here, for each of the 3 x 8 forms:
  1. closed, non-degenerate, non-alternating (so it IS a non-alternating
     Hamiltonian form in their sense);
  2. dim Span{H_f} and the derived series;
  3. simplicity over F_2 (brute force over all 2^7 - 1 elements);
  4. which of OUR two Aut-orbits (the census of run3.py) it lies
     in -- i.e. whether it is the standard shape or the second one.
Output goes to kokuch.log in this directory.
"""
import sys
from nalt2 import *
from run1 import show_form, key_of, form_of, stab_gens, diag_of
from hamcheck import inv_gram, hamiltonian

LOG = None


def out(s):
    print(s)
    if LOG:
        LOG.write(s + "\n"); LOG.flush()


def tail(alg, a, b, c):
    """a x_1x_2 dx_1dx_2 + b x_1x_3 dx_1dx_3 + c x_2x_3 dx_2dx_3."""
    om = {}
    x = [alg.gen(i) for i in range(3)]
    if a:
        addk(om, ('a', 0, 1), alg.mul(x[0], x[1]))
    if b:
        addk(om, ('a', 0, 2), alg.mul(x[0], x[2]))
    if c:
        addk(om, ('a', 1, 2), alg.mul(x[1], x[2]))
    return om


def base(alg, which):
    om = {}
    if which == 1:                      # dx_1dx_2 + (dx_3)^(2)
        addk(om, ('a', 0, 1), alg.one); addk(om, ('b', 2), alg.one)
    elif which == 2:                    # (dx_1)^(2)+(dx_2)^(2)+(dx_3)^(2)
        for i in range(3):
            addk(om, ('b', i), alg.one)
    else:                               # dx_1dx_2 + (dx_2)^(2) + (dx_3)^(2)
        addk(om, ('a', 0, 1), alg.one)
        addk(om, ('b', 1), alg.one); addk(om, ('b', 2), alg.one)
    return om


def ham_span(alg, om):
    """basis of Span{H_f | f in O}, and the list of the H_f themselves."""
    Ginv = inv_gram(alg, gram(alg, om))
    hs = [hamiltonian(alg, Ginv, 1 << t) for t in range(alg.dim)]
    return reduce_basis(alg, hs), hs


def orbit_of(alg, om0, b2, bvec):
    """the orbit of om0 under the stabilizer of the diagonal bvec, as a set of
    packed keys (this is the group run3.py used for the m = 3 census)."""
    gens = [linear_map(alg, phi, b2) for phi in stab_gens(alg, bvec)]
    seen = {key_of(alg, om0, b2)}
    front = [key_of(alg, om0, b2)]
    while front:
        nxt = []
        for k in front:
            for img in gens:
                k2 = apply_lin(img, k)
                if k2 not in seen:
                    seen.add(k2); nxt.append(k2)
        front = nxt
    return seen


def linear_autos(alg):
    """all linear automorphisms of O(m;1), as substitution lists."""
    m = alg.m
    outl = []
    for L in itertools.product(range(1 << m), repeat=m):
        rows = [[(l >> k) & 1 for k in range(m)] for l in L]
        if rank(rows, p=2) != m:
            continue
        phi = []
        for l in L:
            v = 0
            for i in range(m):
                if (l >> i) & 1:
                    v ^= alg.gen(i)
            phi.append(v)
        outl.append(phi)
    return outl


def normalize_diag(alg, om, lins, target):
    """a linearly equivalent form whose diagonal is `target`."""
    for phi in lins:
        om2 = pullback(alg, phi, om)
        if diag_of(alg, om2) == target:
            return om2
    return None


def linear_map(alg, phi, b2):
    img = []
    for s in range(len(b2) * alg.dim):
        k, t = divmod(s, alg.dim)
        img.append(key_of(alg, pullback(alg, phi, {b2[k]: 1 << t}), b2))
    return img


def apply_lin(img, key):
    r, s, k = 0, 0, key
    while k:
        if k & 1:
            r ^= img[s]
        k >>= 1; s += 1
    return r


def main():
    global LOG
    LOG = open(__file__.replace("kokuch.py", "kokuch.log"), "w")
    alg = Alg(3)                                   # O(3;1) over F_2
    b2 = g2_basis(3)
    out("O(m;u) with m = 3, u = (1,1,1); dim O = %d, p = 2\n" % alg.dim)

    # the two normal shapes of our own census, for the orbit test
    std = {('b', 0): alg.one, ('a', 1, 2): alg.one}          # (dx_1)^(2)+dx_2dx_3
    target = (1, 0, 0)
    lins = linear_autos(alg)
    out("standard shape omega_1 = %s" % show_form(alg, std))
    orb_std = orbit_of(alg, std, b2, target)
    out("its orbit under the stabilizer of the diagonal (1,0,0): %d forms\n"
        % len(orb_std))

    for which in (1, 2, 3):
        out("=" * 68)
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    om = dict(base(alg, which))
                    addform(om, tail(alg, a, b, c))
                    ok_cl = not d2(alg, om)
                    ok_nd = nondeg(alg, om)
                    ok_na = not alternating(om)
                    span, _ = ham_span(alg, om)
                    dims = derived_series(alg, span, steps=3)
                    simple = is_simple(alg, span)
                    dsimple = is_simple(alg, reduce_basis(alg, derived(alg, span)))
                    omn = normalize_diag(alg, om, lins, target)
                    if omn is None:
                        where = "?"
                    else:
                        where = ("standard" if key_of(alg, omn, b2) in orb_std
                                 else "second")
                    out("om%d (a,b,c)=(%d,%d,%d): closed %s nondeg %s nonalt %s | "
                        "dim P = %d, derived %s | simple %s, [P,P] simple %s | "
                        "orbit: %s"
                        % (which, a, b, c, ok_cl, ok_nd, ok_na,
                           dims[0], dims, simple, dsimple, where))
    LOG.close()


if __name__ == "__main__":
    main()
