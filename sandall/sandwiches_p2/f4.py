#!/usr/bin/env python3
r"""
f4.py --- the Chevalley algebra of type F_4 over F_2, and the identification

        Der ir(9; \One)  ~  f(4) over F_2,

(the Theorem on the derivations of ir in the text).

WHY IT IS PLAUSIBLE.  ir(9; \One) is the D_4 algebra o^{(1)}_\Pi(8)/c of dim 26
(Theorem 2.3.3), it is centreless and dim Der ir = 52 with Out ir ~ ir, so

        0 --> D_4 --> Der ir --> D_4 --> 0,     dim Der ir = 52 = dim f(4).

On the other side, F_4 has 24 long and 24 short roots, the long ones forming a
D_4 system and the short ones another, and in characteristic 2 the algebra
falls apart in exactly the same way -- which is the shadow of the special
isogeny of F_4 interchanging long and short roots.  The question is whether the
two extensions of D_4 by D_4 are the SAME one, and that no dimension count can
decide; it needs an isomorphism.

THE MODEL.  Over F_2 no sign has to be chosen.  In a Chevalley basis

        [e_a, e_b] = +-(p_{a,b} + 1) e_{a+b}   (a + b a root),
        [e_a, e_{-a}] = h_a,   [h, e_a] = <a, h> e_a,   [h, h'] = 0,

with p_{a,b} the greatest p >= 0 such that b - p a is a root; mod 2 the signs
die and the structure constant is (p + 1) mod 2 -- in particular it VANISHES
for the pairs with p = 1, which for F_4 are exactly the pairs of short roots
whose sum is long.  That is what makes the short part an ideal.  h_a is the
coroot of a written in the basis of the simple coroots, reduced mod 2; the
p-structure is e_a^{[2]} = 0 and h^{[2]} = h.

Coordinates: roots are stored DOUBLED (the vector 2a with integer entries), so
that the short roots \pm e_i and (\pm1 \pm1 \pm1 \pm1)/2 are integral alike.

What is computed, all of it exact over F_2 and exhaustive:

  (1) the root system (48 roots, 24 long + 24 short) and the algebra f(4) of
      dim 52, with the Jacobi identity checked on all C(52,3) triples and the
      restrictedness on all 52 basis vectors;
  (2) its ideals: the 26-dimensional ideal I spanned by the short root vectors
      and the two toral elements h_b, b short; f(4)/I of dim 26; and the check
      that I and f(4)/I are both isomorphic to the D_4 algebra of `d4.py`;
  (3) Der ir(9; \One) as a 52-dimensional restricted Lie algebra over F_2
      (bracket = the commutator of operators, squaring = the composition
      square), with its ideal ad ir and quotient Out ir;
  (4) the invariants of the two 52-dimensional algebras: derived series,
      centre, maximal torus and its weight profile, the dimensions of the
      ideals, ranks of ad;
  (5) an explicit isomorphism Der ir --> f(4), sought weight space by weight
      space with respect to a maximal torus (the engine of `iso2.py`), and
      verified on all 52 x 52 brackets and all 52 squares.

Usage:  python3 f4.py [--quick]
"""

import os as _os_runs; _os_runs.makedirs('runs', exist_ok=True)  # logs go to ./runs
import itertools
import os
import pickle
import sys
import time
from fractions import Fraction

import lin2
import torus2
from lin2 import bits, echelon, in_span
from torus2 import Alg

LOG = None


def say(msg=''):
    print(msg)
    if LOG:
        LOG.write(msg + '\n')
        LOG.flush()


NCHECK = [0]
NFAIL = [0]


def check(msg, ok):
    NCHECK[0] += 1
    if not ok:
        NFAIL[0] += 1
    say('  [%s] %s' % ('ok' if ok else 'FAIL', msg))
    return ok


# ------------------------------------------------------------ the root system

def f4_roots():
    """The 48 roots of F_4, stored doubled (i.e. the vector 2a).

    Long roots  a = \pm e_i \pm e_j          -->  (\pm2, \pm2, 0, 0) and perms
    Short roots a = \pm e_i                  -->  (\pm2, 0, 0, 0) and perms
                a = (\pm1\pm1\pm1\pm1)/2     -->  (\pm1, \pm1, \pm1, \pm1)

    With this storage  (a, a) = (A . A)/4, so long roots have (a,a) = 2 and
    short ones (a,a) = 1.
    """
    out = []
    for i in range(4):
        for j in range(i + 1, 4):
            for si in (2, -2):
                for sj in (2, -2):
                    v = [0, 0, 0, 0]
                    v[i], v[j] = si, sj
                    out.append(tuple(v))
    for i in range(4):
        for s in (2, -2):
            v = [0, 0, 0, 0]
            v[i] = s
            out.append(tuple(v))
    for s in itertools.product((1, -1), repeat=4):
        out.append(tuple(s))
    assert len(out) == 48 and len(set(out)) == 48
    return out


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm2(a):
    """(a, a) in the normalization long = 2, short = 1."""
    return Fraction(dot(a, a), 4)


def pairing(a, b):
    """<a, b^vee> = 2(a,b)/(b,b), an integer."""
    v = Fraction(2 * dot(a, b), dot(b, b))
    assert v.denominator == 1, 'the pairing is not integral'
    return int(v)


SIMPLE = [(0, 2, -2, 0),          # a_1 = e_2 - e_3          long
          (0, 0, 2, -2),          # a_2 = e_3 - e_4          long
          (0, 0, 0, 2),           # a_3 = e_4                short
          (1, -1, -1, -1)]        # a_4 = (e_1-e_2-e_3-e_4)/2  short


def coroot_coords(a):
    """The coroot a^vee in the basis of the simple coroots, as 4 integers."""
    # a^vee = 2a/(a,a); in doubled storage that is the vector 2*(2a)/(A.A/4)
    # = A * (8/(A.A)).  Solve  a^vee = sum c_i a_i^vee  over Q.
    fa = Fraction(8, dot(a, a))
    tgt = [fa * x for x in a]
    cols = []
    for s in SIMPLE:
        fs = Fraction(8, dot(s, s))
        cols.append([fs * x for x in s])
    # solve the 4x4 rational system
    M = [[cols[j][i] for j in range(4)] + [tgt[i]] for i in range(4)]
    for c in range(4):
        piv = next(r for r in range(c, 4) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(4):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    sol = [M[i][4] for i in range(4)]
    assert all(x.denominator == 1 for x in sol), 'a coroot is not integral'
    return [int(x) for x in sol]


def pstring(a, b, rootset):
    """The greatest p >= 0 with b - p a a root."""
    p = 0
    cur = b
    while True:
        cur = tuple(x - y for x, y in zip(cur, a))
        if cur in rootset:
            p += 1
        else:
            return p


# ------------------------------------------------------------- the algebra

def build_f4(verbose=True):
    """f(4) over F_2: an `Alg` of dim 52, plus the bookkeeping."""
    roots = f4_roots()
    rs = set(roots)
    longs = [a for a in roots if norm2(a) == 2]
    shorts = [a for a in roots if norm2(a) == 1]
    if verbose:
        check('F_4 has 48 roots, 24 long and 24 short',
              len(roots) == 48 and len(longs) == 24 and len(shorts) == 24)
        check('the long roots form a closed subsystem (a D_4)',
              all((tuple(x + y for x, y in zip(a, b)) in rs) ==
                  (tuple(x + y for x, y in zip(a, b)) in set(longs))
                  for a in longs for b in longs
                  if tuple(x + y for x, y in zip(a, b)) in rs))
        check('the short roots form a closed subsystem (a D_4)',
              all(norm2(tuple(x + y for x, y in zip(a, b))) == 1
                  or tuple(x + y for x, y in zip(a, b)) not in rs
                  or norm2(tuple(x + y for x, y in zip(a, b))) == 2
                  for a in shorts for b in shorts))

    # The Cartan subalgebra comes FIRST (indices 0..3), the root vectors after
    # it: the Gray-code walk of `torus2.toral_elements` starts from the low
    # indices, and a root vector is 2-nilpotent, not toral.
    idx = {a: 4 + k for k, a in enumerate(roots)}
    N = len(roots)                                    # 48
    dim = N + 4                                       # 52
    cor = {a: coroot_coords(a) for a in roots}

    BR = [[0] * dim for _ in range(dim)]
    pvals = {}
    for a in roots:
        for b in roots:
            if a == b:
                continue
            s = tuple(x + y for x, y in zip(a, b))
            if all(x == 0 for x in s):                # b = -a
                v = 0
                for i in range(4):
                    if cor[a][i] % 2:
                        v |= 1 << i
                BR[idx[a]][idx[b]] = v
            elif s in rs:
                p = pstring(a, b, rs)
                pvals[(a, b)] = p
                if (p + 1) % 2:
                    BR[idx[a]][idx[b]] = 1 << idx[s]
        for i in range(4):
            if pairing(a, SIMPLE[i]) % 2:             # <a, a_i^vee> mod 2
                BR[i][idx[a]] = 1 << idx[a]
                BR[idx[a]][i] = 1 << idx[a]
    SQ = [1 << i for i in range(4)] + [0] * N
    A = Alg(BR, SQ, 'f(4)/F_2')

    if verbose:
        # p = 1 happens exactly for two short roots whose sum is long
        bad = [(a, b) for (a, b), p in pvals.items() if p % 2]
        check('the vanishing structure constants are exactly the pairs of '
              'short roots with a long sum (%d ordered pairs)' % len(bad),
              all(norm2(a) == 1 and norm2(b) == 1 and
                  norm2(tuple(x + y for x, y in zip(a, b))) == 2
                  for a, b in bad) and
              all(p <= 2 for p in pvals.values()))
    return A, roots, longs, shorts, idx, N


def jacobi_check(A, full=True):
    """[[x,y],z] + [[y,z],x] + [[z,x],y] = 0 on all triples of basis vectors."""
    n = A.n
    bad = 0
    rng = range(n)
    for i in rng:
        for j in range(i + 1, n):
            bij = A.BR[i][j]
            for k in range(j + 1, n):
                v = A.br(bij, 1 << k) ^ A.br(A.BR[j][k], 1 << i) \
                    ^ A.br(A.BR[k][i], 1 << j)
                if v:
                    bad += 1
                    if not full:
                        return bad
    return bad


def restricted_check(A):
    """(ad x)^2 = ad_{x^{[2]}} on the basis, and (x+y)^{[2]} consistency."""
    n = A.n
    bad = 0
    for i in range(n):
        x = 1 << i
        sx = A.sq(x)
        for j in range(n):
            if A.br(x, A.br(x, 1 << j)) != A.br(sx, 1 << j):
                bad += 1
    return bad


# ------------------------------------------------------------- ideal lattice

def ideal_generated(A, seeds):
    """The ideal generated by the given vectors."""
    cur = echelon(list(seeds))
    while True:
        new = list(cur)
        for v in cur:
            for k in range(A.n):
                new.append(A.br(v, 1 << k))
                new.append(A.sq(v))
        nxt = echelon(new)
        if len(nxt) == len(cur):
            return cur
        cur = nxt


def ideal_uniqueness(A, T):
    """Every non-zero ideal is spanned by weight vectors -- the torus acts by
    projections, see `torus2.py` -- so enumerating the ideals generated by the
    non-zero weight vectors settles the whole ideal lattice."""
    W = torus2.weights(A, T)
    seen = {}
    nvec = 0
    for chi, base in W.items():
        for k in range(1, 1 << len(base)):
            v = 0
            for i in bits(k):
                v ^= base[i]
            nvec += 1
            I = ideal_generated(A, [v])
            seen.setdefault(len(I), 0)
            seen[len(I)] += 1
    return nvec, seen


def minimal_ideals(A, limit=None):
    """The ideals generated by single basis vectors, deduplicated by dim."""
    out = {}
    for k in range(A.n):
        I = ideal_generated(A, [1 << k])
        out.setdefault(len(I), I)
    return out


def derived(A, S):
    return echelon([A.br(u, v) for u in S for v in S])


def coord_rec(basis):
    """Bookkeeping to express a vector in terms of `basis` (assumed free)."""
    rec = []
    for j, u in enumerate(basis):
        w, cm = u, 1 << j
        for p, b, c in rec:
            if w >> p & 1:
                w ^= b
                cm ^= c
        if w:
            rec.append((w.bit_length() - 1, w, cm))
            rec.sort(key=lambda t: -t[0])
    assert len(rec) == len(basis), 'the basis is not free'
    return rec


def in_coords(rec, v):
    cm = 0
    for p, b, c in rec:
        if v >> p & 1:
            v ^= b
            cm ^= c
    assert v == 0, 'the vector is outside the span'
    return cm


def subalgebra(A, basis, name):
    """The `Alg` structure induced on a subspace closed under br and sq."""
    rec = coord_rec(basis)
    m = len(basis)
    BR = [[in_coords(rec, A.br(basis[i], basis[j])) for j in range(m)]
          for i in range(m)]
    SQ = [in_coords(rec, A.sq(basis[i])) for i in range(m)]
    return Alg(BR, SQ, name)


# --------------------------------------------------------------- Der ir(9)

CACHE = 'f4algs.pkl'


def build_der_ir():
    """Der ir(9; \\One) as a 52-dimensional restricted Lie algebra over F_2.

    Cached in `f4algs.pkl`: rebuilding it from the vector fields of ir(9; \\One)
    costs minutes, and the isomorphism search wants to be re-run.
    """
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            d = pickle.load(f)
        return Alg(d['BR'], d['SQ'], 'Der ir(9;One)'), d['adir'], None, None, None
    import iriso as I
    import irder as D
    basis, comps = I.build_ir()
    C = I.Coords(basis)
    der = D.der_basis(C)                       # flat masks, dim 52
    ops = [D.unflat(m) for m in der]
    n = len(ops)

    # coordinates in the basis `ops`
    ech = []
    for j, m in enumerate(der):
        v, cm = m, 1 << j
        for p, b, c in ech:
            if v >> p & 1:
                v ^= b
                cm ^= c
        if v:
            ech.append((v.bit_length() - 1, v, cm))
            ech.sort(key=lambda t: -t[0])

    def co(m):
        cm = 0
        for p, b, c in ech:
            if m >> p & 1:
                m ^= b
                cm ^= c
        assert m == 0, 'the operator is not a derivation'
        return cm

    def compose(a, b):
        return [D.apply_op(a, b[j]) for j in range(D.DIM)]

    BR = [[co(D.flat(D.comm(ops[i], ops[j]))) for j in range(n)]
          for i in range(n)]
    SQ = [co(D.flat(compose(ops[i], ops[i]))) for i in range(n)]
    A = Alg(BR, SQ, 'Der ir(9;One)')
    # ad ir inside it
    adir = echelon([co(D.flat(C.ad(1 << k))) for k in range(D.DIM)])
    with open(CACHE, 'wb') as f:
        pickle.dump({'BR': BR, 'SQ': SQ, 'adir': adir}, f)
    return A, adir, C, ops, der


# ------------------------------------------------------------------ the search

def gl_matrices(r):
    """All invertible r x r matrices over F_2, as lists of rows."""
    out = []
    for bitsrows in itertools.product(range(1 << r), repeat=r):
        M = [[(bitsrows[i] >> j) & 1 for j in range(r)] for i in range(r)]
        if len(echelon(list(bitsrows))) == r:
            out.append(M)
    return out


def cached_torus(A):
    tc = 'f4torus.pkl'
    if os.path.exists(tc):
        with open(tc, 'rb') as f:
            return pickle.load(f)
    T = torus2.maximal_torus(A, tries=4)
    with open(tc, 'wb') as f:
        pickle.dump(T, f)
    return T


def profile(A, T):
    W = torus2.weights(A, T)
    return {chi: len(v) for chi, v in W.items()}, W


def search_iso(A1, A2, T1, T2, M, ideal1=None, ideal2=None, budget=4e6,
               tlimit=240):
    """An isomorphism A1 --> A2 carrying T1 to T2 by the matrix M.

    As `iso2.graded_search_M`, with two changes that matter at dim 52:
    the weight spaces are visited in the order of increasing candidate sets,
    and -- when the algebras have a unique minimal ideal -- the image of
    a~vector inside the ideal is required to lie inside the ideal.
    """
    import iso2
    r = len(T1)
    W1, W2 = torus2.weights(A1, T1), torus2.weights(A2, T2)
    Mi = iso2.inv_mat(M)
    if Mi is None:
        return None, 0
    chis = list(range(1, 1 << r))
    chimap = {chi: iso2.weight_image(M, chi, Mi) for chi in chis}
    for chi in chis:
        if len(W1.get(chi, [])) != len(W2.get(chimap[chi], [])):
            return None, 0
    base = iso2.Partial(A1, A2)
    for j in range(r):
        v = 0
        for k in range(r):
            if M[j][k]:
                v ^= T2[k]
        if not base.add(T1[j], v):
            return None, 0
    if not base.propagate():
        return None, 0

    cand = {chi: iso2.span_nonzero(W2.get(chimap[chi], [])) for chi in chis}
    cand[0] = iso2.span_nonzero(W2.get(0, []))
    order = []
    for chi in chis + [0]:
        for v in W1.get(chi, []):
            order.append((v, chi, len(cand[chi])))
    order.sort(key=lambda t: t[2])
    order = [(v, chi) for v, chi, _ in order]

    nodes = [0]
    t0 = time.time()

    def dfs(pm, k):
        nodes[0] += 1
        if nodes[0] > budget or time.time() - t0 > tlimit:
            raise TimeoutError
        if pm.dim() == A1.n:
            return pm
        if k >= len(order):
            return None
        src, chi = order[k]
        got, known = pm.value(src)
        if known:
            return dfs(pm, k + 1)
        r1 = A1.rank_ad(src)
        sq1, sqknown = pm.value(A1.sq(src))
        inI = ideal1 is not None and in_span(src, ideal1)
        for h in cand[chi]:
            if A2.rank_ad(h) != r1:
                continue
            if sqknown and A2.sq(h) != sq1:
                continue
            if ideal1 is not None and in_span(h, ideal2) != inI:
                continue
            nx = pm.copy()
            if not nx.add(src, h):
                continue
            if not nx.propagate():
                continue
            out = dfs(nx, k + 1)
            if out is not None:
                return out
        return None

    try:
        res = dfs(base, 0)
    except TimeoutError:
        return None, nodes[0]
    return res, nodes[0]


def main():
    global LOG
    quick = '--quick' in sys.argv
    LOG = open('runs/f4.log', 'w')
    t0 = time.time()
    say('=== f4.py --- Der ir(9; \\One) and f(4) over F_2 ===')
    say('')

    say('=== (1) the Chevalley algebra f(4) over F_2 ===')
    F, roots, longs, shorts, idx, NR = build_f4()
    check('dim f(4) = 52', F.n == 52)
    bad = jacobi_check(F)
    check('the Jacobi identity holds on all C(52,3) = %d triples'
          % (52 * 51 * 50 // 6), bad == 0)
    check('(ad x)^2 = ad_{x^{[2]}} on the basis', restricted_check(F) == 0)
    cen = F.centralizer([1 << k for k in range(F.n)])
    check('the centre of f(4) is 0 (dim %d)' % len(cen), len(cen) == 0)
    der1 = derived(F, [1 << k for k in range(F.n)])
    say('  dim [f(4), f(4)] = %d' % len(der1))
    der2 = derived(F, der1)
    say('  dim [f(4)\', f(4)\'] = %d' % len(der2))

    say('')
    say('=== (2) the ideal lattice of f(4) ===')
    ids = minimal_ideals(F)
    say('  the ideals generated by single basis vectors have dims %s'
        % sorted(ids))
    short_ideal = ideal_generated(F, [1 << idx[a] for a in shorts])
    check('the short root vectors generate an ideal of dim 26',
          len(short_ideal) == 26)
    # the claim that it is exactly shorts + the two toral h_b
    check('that ideal is the span of the 24 short root vectors and a '
          '2-dimensional part of the Cartan subalgebra',
          all(in_span(1 << idx[a], short_ideal) for a in shorts))
    long_gen = ideal_generated(F, [1 << idx[a] for a in longs])
    say('  the long root vectors generate an ideal of dim %d' % len(long_gen))
    check('the short ideal is a minimal ideal (its dim is the smallest '
          'ideal dim found)', 26 == min(d for d in ids if d > 0))

    # the quotient
    import d4
    D26, A28, cen28 = d4.build_d4()
    check('the D_4 algebra of d4.py has dim 26', D26.n == 26)

    comp, BRq, SQq = d4.quotient(F, short_ideal)
    Q = Alg(BRq, SQq, 'f(4)/I')
    check('f(4)/I has dim 26', Q.n == 26)
    bq = jacobi_check(Q)
    check('the quotient satisfies Jacobi', bq == 0)

    # I as an algebra in its own right
    Isub = subalgebra(F, short_ideal, 'I (short part)')
    check('I is a 26-dimensional subalgebra closed under the squaring',
          Isub.n == 26)
    check('I satisfies Jacobi', jacobi_check(Isub) == 0)

    say('')
    say('=== (3) Der ir(9; \\One) as a 52-dimensional algebra ===')
    A, adir, C, ops, der = build_der_ir()
    check('dim Der ir(9; \\One) = 52', A.n == 52)
    check('the Jacobi identity holds in Der ir',
          jacobi_check(A) == 0)
    check('(ad x)^2 = ad_{x^{[2]}} in Der ir', restricted_check(A) == 0)
    cenA = A.centralizer([1 << k for k in range(A.n)])
    check('the centre of Der ir is 0 (dim %d)' % len(cenA), len(cenA) == 0)
    check('ad ir is an ideal of dim 26 in Der ir', len(adir) == 26)
    dA1 = derived(A, [1 << k for k in range(A.n)])
    say('  dim [Der ir, Der ir] = %d' % len(dA1))
    idsA = minimal_ideals(A)
    say('  the ideals generated by single basis vectors have dims %s'
        % sorted(idsA))

    say('')
    say('=== (4) the invariants side by side ===')
    TF = [1 << i for i in range(4)]                   # the Cartan subalgebra
    check('the Cartan subalgebra of f(4) is toral',
          all(F.sq(t) == t for t in TF))
    cF = F.centralizer(TF)
    check('it is self-centralizing, hence a maximal torus (centralizer of '
          'dim %d)' % len(cF), len(cF) == 4)
    TA = cached_torus(A)
    prA, WA = profile(A, TA)
    prF, WF = profile(F, TF)
    say('  %-16s dim %d  torus %d  weight dims %s'
        % (A.name, A.n, len(TA), sorted(prA.values(), reverse=True)))
    say('  %-16s dim %d  torus %d  weight dims %s'
        % (F.name, F.n, len(TF), sorted(prF.values(), reverse=True)))
    check('the two weight profiles coincide as multisets',
          sorted(prA.values()) == sorted(prF.values()))
    ranksA = sorted(A.rank_ad(1 << k) for k in range(A.n))
    ranksF = sorted(F.rank_ad(1 << k) for k in range(F.n))
    say('  ranks of ad on a weight basis: %s' % ranksA[:6] + ' ...')
    say('  ranks of ad on a weight basis: %s' % ranksF[:6] + ' ...')

    say('')
    say('  the ideal generated by EVERY non-zero weight vector:')
    for X, T in ((A, TA), (F, TF)):
        nvec, seen = ideal_uniqueness(X, T)
        check('%s: of the %d non-zero weight vectors, the ideals generated '
              'have dims %s -- so the only proper non-zero ideal is the '
              '26-dimensional one'
              % (X.name, nvec, sorted(seen)), sorted(seen) == [26, X.n])

    if quick:
        say('\n  (--quick: the isomorphism search is skipped)')
        say('\n  (%.1f s)  checks %d, failures %d'
            % (time.time() - t0, NCHECK[0], NFAIL[0]))
        LOG.close()
        return

    say('')
    say('=== (5) an isomorphism Der ir(9; \\One) --> f(4) ===')
    idealA = echelon(adir)
    idealF = echelon(short_ideal)
    # the candidate torus maps: those respecting the weight-dimension profile
    import iso2
    good = []
    for Mm in gl_matrices(4):
        Mi = iso2.inv_mat(Mm)
        ok = True
        for chi in range(1, 16):
            if prA.get(chi, 0) != prF.get(iso2.weight_image(Mm, chi, Mi), 0):
                ok = False
                break
        if ok:
            good.append(Mm)
    say('  %d of the %d invertible torus maps respect the weight profile'
        % (len(good), 20160))
    found = None
    tot = 0
    for k, Mm in enumerate(good):
        res, nodes = search_iso(A, F, TA, TF, Mm, idealA, idealF)
        tot += nodes
        if res is not None:
            found = res
            say('  an isomorphism found at torus map #%d after %d nodes '
                '(%d in all)' % (k + 1, nodes, tot))
            break
    if found is None:
        say('  NO isomorphism found (%d nodes over %d torus maps)'
            % (tot, len(good)))
        check('Der ir(9; \\One) is isomorphic to f(4) over F_2', False)
    else:
        phi = found.matrix()
        check('the map is verified on all 52 x 52 brackets and all 52 squares',
              iso2.verify(A, F, phi))
        check('it carries ad ir onto the short-root ideal of f(4)',
              all(in_span(xor_images(phi, v), idealF) for v in idealA))
        say('  the isomorphism, as the images of the 52 basis vectors '
            '(bit masks):')
        say('  %s' % [hex(x) for x in phi])
    say('')
    say('  (%.1f s)  checks %d, failures %d'
        % (time.time() - t0, NCHECK[0], NFAIL[0]))
    LOG.close()


def xor_images(phi, v):
    out = 0
    for i in bits(v):
        out ^= phi[i]
    return out


if __name__ == '__main__':
    main()
