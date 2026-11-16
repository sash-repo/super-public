r"""Theorem 1.9.6: two odd parameters hbar_1, hbar_2.

The flat model of flatmodel.py carries odd central parameters as the generators
('h', a): each squares to zero by the normal-ordering rule, two distinct ones
anticommute, and their product hbar_1 hbar_2 is a non-zero even central element.
So the two-parameter twist of a special almost periplectic M is the computation
of Theorem 1.9.4a with two connection forms

    alpha_a := d tau + (hbar_a / 2) sum (pi_i dq_i - q_i dpi_i),   a = 1, 2,

i.e. two operators D_a^+ of eq. (Dpm2), one per parameter, and D_a^- := [X^-, D_a^+].

Computed here: the closure of {D_1^+, D_1^-, D_2^+, D_2^-, X^+, X^-}, its filtration
by the number of hbar's, the hbar_1 hbar_2 layer, and independence of n.

Run (from this folder):  ../venv/bin/python twohbar.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
import sys
from fractions import Fraction


from exactlin import rank                                     # noqa: E402
from flatmodel import Model, Echelon, closure, d_operator     # noqa: E402

half = Fraction(1, 2)


def span(M, elts):
    ech = {0: Echelon(), 1: Echelon()}
    for e in elts:
        if e:
            ech[M.parity(e)].add(e)
    return ech


def sdim_of(M, elts):
    ech = span(M, elts)
    return (len(ech[0].rows), len(ech[1].rows))


def contains(M, ech, e):
    return not ech[M.parity(e)].reduce(e)[0]


def hcount(monomial):
    """Number of hbar generators in a normal-ordered monomial."""
    return sum(1 for g in monomial if g[0] == 'h')


def filt_dim(M, basis, k):
    r"""sdim of g^[k] := g \cap (monomials carrying at least k hbar's).

    The generators D_a^+ are NOT hbar-homogeneous (each is d minus an hbar-term),
    so selecting the basis elements whose monomials all carry >= k hbar's
    UNDERCOUNTS: a difference of two such generators can be hbar-divisible while
    neither of them is.  So g^[k] is computed as a kernel: it is what dies under
    the projection onto the monomials carrying FEWER than k hbar's, and
    dim g^[k] = dim g - rank(that projection).
    """
    out = []
    for par in (0, 1):
        elts = [e for e in basis if e and M.parity(e) == par]
        if not elts:
            out.append(0)
            continue
        mons = sorted({m for e in elts for m in e if hcount(m) < k},
                      key=lambda m: (len(m), m))
        if not mons:
            out.append(len(elts))
            continue
        rows = [[e.get(m, Fraction(0)) for m in mons] for e in elts]
        out.append(len(elts) - rank(rows))
    return tuple(out)


def filt(basis, k):
    """The basis elements all of whose monomials carry at least k hbar's."""
    return [e for e in basis if e and all(hcount(m) >= k for m in e)]


def build(n, params):
    """The periplectic flat model on C^{n|n} with `params` odd parameters.

    Returns (Model, X^+, X^-, [D_a^+], [D_a^-]).
    """
    P = Model([0] * n + [1] * n)
    q = list(range(n))
    pi = list(range(n, 2 * n))
    c = lambda a: P.gen('c', a)                                # noqa: E731
    b = lambda a: P.gen('b', a)                                # noqa: E731
    x = lambda a: P.gen('x', a)                                # noqa: E731
    t = lambda a: P.gen('t', a)                                # noqa: E731
    d = d_operator(P)

    Xp = P.add(*[P.mul(c(q[i]), c(pi[i])) for i in range(n)])
    Xm = P.add(*[P.mul(b(pi[i]), b(q[i])) for i in range(n)])

    Dps, Dms = [], []
    for a in range(params):
        hb = P.gen('h', a)
        Dp = P.add(d, P.smul(-half, P.mul(hb, P.add(
            *[P.add(P.mul(c(q[i]), x(pi[i])),
                    P.smul(-1, P.mul(c(pi[i]), x(q[i])))) for i in range(n)]))))
        Dm = P.br(Xm, Dp)      # D_a^- := [X^-, D_a^+], as in theory7.py
        Dps.append(Dp)
        Dms.append(Dm)
    return P, Xp, Xm, Dps, Dms


def report(n, params):
    P, Xp, Xm, Dps, Dms = build(n, params)
    gens = Dps + Dms + [Xp, Xm]
    basis, sd = closure(P, gens, limit=600)
    print('=== C^{%d|%d}, %d odd parameter(s) ===' % (n, n, params))
    print('  sdim of the closure          :', sd, ' (dim %d)' % sum(sd))

    # --- the hbar-adic filtration --------------------------------------
    dims = []
    for k in range(params + 2):
        sdk = filt_dim(P, basis, k) if k else sd
        dims.append(sdk)
        print('    g^[%d] (at least %d hbar)     : sdim %s' % (k, k, (sdk,)))
    for k in range(params + 1):
        gr = (dims[k][0] - dims[k + 1][0], dims[k][1] - dims[k + 1][1])
        print('    gr_%d = g^[%d]/g^[%d]           : sdim %s' % (k, k, k + 1, (gr,)))

    # --- is the hbar-part an ideal, and is it abelian? -------------------
    ideal = filt(basis, 1)   # a spanning set of g^[1] is not needed for the tests
    iech = span(P, ideal)
    ielts = iech[0].elements + iech[1].elements
    print('  sdim of the ideal (hbar_1, hbar_2) g :', filt_dim(P, basis, 1))
    print('    it is an ideal              :',
          all(contains(P, iech, P.br(u, v)) for u in ielts for v in basis))
    print('    it is abelian               :',
          all(not P.br(u, v) for u in ielts for v in ielts))

    # --- the top layer hbar_1 hbar_2 ------------------------------------
    if params == 2:
        top = filt(basis, 2)
        print('  the hbar_1 hbar_2 layer     : sdim', filt_dim(P, basis, 2),
              ' central :', all(not P.br(u, v)
                                for u in top for v in basis))

    # --- the quotient at hbar = 0 ---------------------------------------
    h1 = filt_dim(P, basis, 1)
    print('  the hbar = 0 quotient g/g^[1]: sdim', (sd[0] - h1[0], sd[1] - h1[1]))

    # --- the two one-parameter subalgebras and their sum -----------------
    if params == 2:
        for a in (0, 1):
            _, sda = closure(P, [Dps[a], Dms[a], Xp, Xm], limit=600)
            print('  <D_%d^pm, X^pm> alone        : sdim %s' % (a + 1, (sda,)))
        # do the two copies generate everything, or is there more?
        sub = []
        for a in (0, 1):
            ba, _ = closure(P, [Dps[a], Dms[a], Xp, Xm], limit=600)
            sub += ba
        print('  sum of the two copies        : sdim', sdim_of(P, sub))
        print('  ... equals the whole closure :',
              sdim_of(P, sub) == sd)

        # what the mixed brackets produce
        print('  [D_1^+, D_2^+] = 0           :', not P.br(Dps[0], Dps[1]))
        print('  [D_1^+, D_2^-] = 0           :', not P.br(Dps[0], Dms[1]))
        print('  [D_1^-, D_2^-] = 0           :', not P.br(Dms[0], Dms[1]))

        # is it generated by D_1^+, D_2^+ and sl(1|1), as for one parameter?
        _, sdg = closure(P, [Dps[0], Dps[1], Xp, Xm], limit=600)
        print('  <D_1^+, D_2^+, X^pm>         : sdim %s  = the whole : %s'
              % ((sdg,), sdg == sd))

        # h (x) Lambda(hbar_1, hbar_2) sits inside: how much of g is it?
        bh, sdh = closure(P, [d_operator(P), Xp, Xm], limit=600)
        hb1, hb2 = P.gen('h', 0), P.gen('h', 1)
        tensored = list(bh) + [P.mul(hb1, e) for e in bh] \
            + [P.mul(hb2, e) for e in bh] \
            + [P.mul(P.mul(hb1, hb2), e) for e in bh]
        gech = span(P, basis)
        print('  h := <d, X^pm>               : sdim', sdh)
        print('  h (x) Lambda(hbar_1,hbar_2)  : sdim', sdim_of(P, tensored))
        for name, part in (('h', bh),
                           ('hbar_1 h', [P.mul(hb1, e) for e in bh]),
                           ('hbar_2 h', [P.mul(hb2, e) for e in bh]),
                           ('hbar_1 hbar_2 h',
                            [P.mul(P.mul(hb1, hb2), e) for e in bh])):
            print('    %-16s inside g   : %s' % (
                name, all(contains(P, gech, e) for e in part)))

    # --- solvability / nilpotency --------------------------------------
    cur = list(basis)
    for step in range(1, 7):
        nxt = [P.br(u, v) for u in cur for v in cur]
        nxt = [e for e in nxt if e]
        sdn = sdim_of(P, nxt) if nxt else (0, 0)
        print('  derived g^(%d)                : sdim %s' % (step, (sdn,)))
        if sdn == (0, 0):
            print('  ==> solvable of derived length', step)
            break
        if sdn == sdim_of(P, cur):
            print('  ==> derived series stabilizes; not solvable')
            break
        cur, _ = closure(P, nxt, limit=600)

    cur = list(basis)
    for step in range(1, 9):
        nxt = [P.br(u, v) for u in basis for v in cur]
        nxt = [e for e in nxt if e]
        sdn = sdim_of(P, nxt) if nxt else (0, 0)
        if sdn == (0, 0):
            print('  ==> nilpotent of class', step)
            break
        if sdn == sdim_of(P, cur):
            print('  ==> lower central series stabilizes at sdim %s; not nilpotent'
                  % (sdn,))
            break
        cur, _ = closure(P, nxt, limit=600)

    # --- centre, exactly (kernel of ad), not just the basis elements ------
    bech = span(P, basis)
    elts = bech[0].elements + bech[1].elements
    cen = [0, 0]
    for par in (0, 1):
        vecs = [e for e in elts if P.parity(e) == par]
        if not vecs:
            continue
        # rows of the map u = sum a_i vecs[i]  |-->  ([u, v])_{v in elts}
        cols = []
        for v in elts:
            brs = [P.br(u, v) for u in vecs]
            mons = sorted({m for e2 in brs for m in e2}, key=lambda m: (len(m), m))
            cols.append((brs, mons))
        mat = []
        for i in range(len(vecs)):
            row = []
            for brs, mons in cols:
                row += [brs[i].get(m, Fraction(0)) for m in mons]
            mat.append(row)
        cen[par] = len(vecs) - rank(mat) if any(any(r) for r in mat) else len(vecs)
    print('  centre of g (exact)          : sdim', tuple(cen))

    # --- is g^[2] an ideal? ---------------------------------------------
    if params == 2:
        top = filt(basis, 2)
        tech = span(P, top)
        print('  g^[2] spanning set sdim %s ; an ideal : %s'
              % ((sdim_of(P, top),),
                 all(contains(P, tech, P.br(u, v))
                     for u in tech[0].elements + tech[1].elements
                     for v in basis)))
    print()
    return sd


if __name__ == '__main__':
    print(__doc__.split('Run:')[0].strip()[:0] or '', end='')
    NS = (1, 2, 3)
    one = [report(n, 1) for n in NS]
    two = [report(n, 2) for n in NS]
    print('one parameter, n =', NS, ':', one,
          ' independent of n :', len(set(one)) == 1)
    print('two parameters, n =', NS, ':', two,
          ' independent of n :', len(set(two)) == 1)
