"""
Checks for Section 27, "What the central extension does define" (Proposition 27.3,
Statement 27.5, table (66)).

Parts
  A  the flat model: Jacobi, and the irrelevance of the normalisation kappa
  B  the weight arithmetic: Res(f P g) is weight-0 exactly when ord P = 6-n
  C  the census of the invariant operators of order 6-n, and skewness
  D  the weight-0 2-cocycles, computed from scratch on a window
  E  the Schwarzian sheaf S = K^vee (x) Ber and the duality
         sdim H^0(S) = Pi^n sdim H^1(K)
  F  the numbers, genus by genus

Every check prints PASS/FAIL with the number of cases behind it.
"""

import sys
import random
from fractions import Fraction as Q
from itertools import combinations, product

import sympy as sp

import schw
import srs
from stringy import mono, mul

LOG = open('runschw.log', 'w')
FAILS = [0]
CHECKS = [0]


def out(s=''):
    print(s)
    LOG.write(s + '\n')
    LOG.flush()


def check(name, ok, detail=''):
    CHECKS[0] += 1
    if not ok:
        FAILS[0] += 1
    out('  [%s] %s%s' % ('PASS' if ok else 'FAIL', name,
                         ('   ' + detail) if detail else ''))


# ===================================================================== A
out('=' * 78)
out('A.  The flat model k^L(1|n): Jacobi, and the normalisation')
out('=' * 78)

random.seed(20260913)
for n in range(0, 7):
    keys = schw.monomials(n, range(-2, 3))
    random.shuffle(keys)
    ks = keys[:14]
    bad1 = schw.jacobi_fails(n, ks, Q(1))
    bad2 = schw.jacobi_fails(n, ks, Q(-2))
    check('n=%d: Jacobi on %d^3 triples, for kappa = 1 and for kappa = -2'
          % (n, len(ks)), bad1 == 0 and bad2 == 0,
          'failures %d, %d' % (bad1, bad2))
out('  (all kappa != 0 give isomorphic brackets: theta -> c theta takes kappa')
out('   to c^{-2} kappa, so the normalisation is a choice, not a datum.)')

# ===================================================================== B
out()
out('=' * 78)
out('B.  The weight arithmetic.  deg K_{t^a theta_S} = a-1+|S|/2 (doubled: ')
out('    w = 2a-2+|S|).  Res = the coefficient of t^{-1}theta_1...theta_n, of')
out('    weight n-4.  Since w(fg) = w(f)+w(g)+2 and every D lowers w by 1,')
out('    c(f,g) = Res(f P g) is weight-0 iff ord P = 6-n in D.')
out('=' * 78)

for n in range(0, 7):
    rng = range(-3, 5)
    keys = schw.monomials(n, rng)
    # the residue itself: weight of t^{-1}theta_1..theta_n
    wres = schw.weight((-1, schw.odd_labels(n)))
    check('n=%d: w(t^{-1}theta_1..theta_n) = n-4' % n, wres == n - 4,
          'w = %d' % wres)
    # w(fg) = w(f) + w(g) + 2
    bad = sum(1 for k1 in keys for k2 in keys
              if schw.weight((k1[0] + k2[0], k1[1] + k2[1]))
              != schw.weight(k1) + schw.weight(k2) + 2)
    check('n=%d: w(fg) = w(f)+w(g)+2 on %d pairs' % (n, len(keys) ** 2),
          bad == 0)
    for order in range(0, 9):
        for name, op in schw.invariant_ops(n, order):
            pairs = [(k1, k2) for k1 in keys for k2 in keys
                     if schw.res(mul(mono(*k1), schw.apply_op(op, mono(*k2), n)), n)]
            if not pairs:
                continue
            w0 = all(schw.weight(k1) + schw.weight(k2) == 0 for k1, k2 in pairs)
            check('n=%d: P=%s (order %d) pairs only weights summing to %d'
                  % (n, name, order, 0 if order == 6 - n else 6 - n - order),
                  (order == 6 - n) == w0,
                  '%d non-zero values; weight-0: %s' % (len(pairs), w0))

# ===================================================================== C
out()
out('=' * 78)
out('C.  The O(n)-invariant operators of order 6-n, and antisymmetry.')
out('    Invariant monomials in D_1..D_n, d_t: d_t^k (order 2k) and')
out('    D_1...D_n d_t^j (order n+2j).  Hence the census below.')
out('=' * 78)

KAPPAS = [Q(1), Q(-1), Q(1, 2), Q(-1, 2), Q(2), Q(-2)]
census = {}
for n in range(0, 8):
    order = 6 - n
    ops = schw.invariant_ops(n, order) if order >= 0 else []
    keys = schw.monomials(n, range(-3, 5))
    random.shuffle(keys)
    ks = keys[:18]
    surv = []
    for name, op in ops:
        skew = schw.is_skew(n, op, schw.monomials(n, range(-3, 5)))
        good = [k for k in KAPPAS if not schw.is_cocycle(n, op, ks, k)]
        out('    n=%d  order %2d  P = %-16s  antisymmetric: %-5s  cocycle for '
            'kappa in %s' % (n, order, name, not skew,
                             [str(k) for k in good] if not skew else '-'))
        if not skew and good:
            surv.append(name)
    census[n] = surv
    check('n=%d: exactly %d Schwarzian candidate(s)' % (n, 1 if n <= 4 else 0),
          len(surv) == (1 if n <= 4 else 0), 'survivors: %s' % (surv or 'none'))
out('    n>=7: the order 6-n is negative, so there is no operator at all.')

out()
out('    The kernel of the surviving operator -- the Lie superalgebra lf of')
out('    the infinitesimal linear fractional transformations:')
KERNELS = {0: (3, 0, 'sl(2)'), 1: (3, 2, 'osp(1|2)'), 2: (4, 4, 'osp(2|2)'),
           3: (6, 6, 'osp(3|2)'), 4: (8, 8, 'po(0|4) = Lambda(4)')}
for n in range(0, 5):
    name, op = [o for o in schw.invariant_ops(n, 6 - n)
                if not schw.is_skew(n, o[1], schw.monomials(n, range(-3, 5)))][0]
    keys = schw.monomials(n, range(-5, 7))
    ker = [k for k in keys if not schw.apply_op(op, mono(*k), n)]
    ev = sum(1 for k in ker if len(k[1]) % 2 == 0)
    od = len(ker) - ev
    has_t2 = any(k[0] == 2 for k in ker)
    e0, o0, nm = KERNELS[n]
    check('n=%d: sdim Ker P = (%d|%d) = %s' % (n, e0, o0, nm),
          (ev, od) == (e0, o0), 'got (%d|%d); contains t^2: %s' % (ev, od, has_t2))
    # closed under the bracket?
    kap = [k for k in KAPPAS
           if not schw.is_cocycle(n, op, schw.monomials(n, range(-2, 3)), k)][0]
    bad = 0
    for k1, k2 in product(ker, repeat=2):
        b = schw.bracket(mono(*k1), mono(*k2), n, kap)
        if any(k not in ker for k in b):
            bad += 1
    check('n=%d: Ker P is a subalgebra' % n, bad == 0, '%d bad brackets' % bad)
out('    For n <= 3 the kernel contains Span(1, t, t^2) = sl(2) and lf is the')
out('    Lie superalgebra of a group of genuinely fractional transformations;')
out('    for n = 4 it contains neither t nor t^2 -- it is po(0|4), which moves')
out('    the even coordinate by translations only.')

# ===================================================================== D
out()
out('=' * 78)
out('D.  The weight-0 2-cocycles from scratch (no ansatz), on the window')
out('    |a| <= B, modulo the weight-0 coboundaries.')
out('=' * 78)

RUNS = [(0, 5, None), (1, 4, None), (2, 3, None), (3, 2, None),
        (4, 2, None), (4, 2, [(-1, (1, 2, 3, 4))]), (5, 2, None), (6, 2, None)]
known = {0: 1, 1: 1, 2: 1, 3: 1, 5: 0, 6: 0}
for n, B, drop in RUNS:
    dim, dimcob, U, nB, basis = schw.cocycle_space(n, B=B, drop=drop,
                                                   verbose=False)
    h2 = dim - dimcob
    tag = "k'(1|4)" if drop else 'k(1|%d)' % n
    out('    %-9s B=%d  unknowns=%d  cocycles=%d  coboundaries=%d  H^2=%d'
        % (tag, B, nB, dim, dimcob, h2))
    if n in known and not drop:
        check('n=%d: H^2 on the window = %d' % (n, known[n]), h2 == known[n],
              'got %d' % h2)
out('    (a window can only over-count: fewer equations, more solutions.)')

# ===================================================================== E
out()
out('=' * 78)
out('E.  The Schwarzian sheaf S = Hom(K, Ber) on a split superstring.')
out('=' * 78)

g = srs.G


def rig_sets(n):
    """a few riggings: trivial, all-independent, and a mixed one."""
    out_ = [('trivial', [srs.TRIV] * n)]
    if n:
        out_.append(('independent', [srs.eta('n%d' % i) for i in range(n)]))
        out_.append(('mixed', [srs.TRIV if i % 2 else srs.eta('m%d' % i)
                               for i in range(n)]))
    return out_


for n in range(0, 9):
    for tag, rig in rig_sets(n):
        K = schw.contact_layers(n, rig)
        S = schw.schwarzian_layers(n, rig)
        h1K = schw.sdim_h1(K, g)
        h0S = schw.sdim_h0(S, g)
        # the duality: sdim H^0(S) = Pi^n sdim H^1(K)
        want = (h1K[1], h1K[0]) if n % 2 else h1K
        check('n=%d (%s): sdim H^0(S) = Pi^%d sdim H^1(K)' % (n, tag, n % 2),
              sp.simplify(h0S[0] - want[0]) == 0
              and sp.simplify(h0S[1] - want[1]) == 0,
              'H^0(S) = %s,  H^1(K) = %s' % (srs.sdim(h0S), srs.sdim(h1K)))
        # layer-wise: layer (q,S) of S is Serre-dual to layer (n-q,S^c) of K
        bad = 0
        for q in range(n + 1):
            for T in combinations(range(n), q):
                Tc = tuple(i for i in range(n) if i not in T)
                delta = srs.emul(*rig)
                b1 = srs.LB(4 - n + q,
                            srs.emul(*[rig[i] for i in T], delta))
                b2 = srs.LB(len(Tc) - 2, srs.emul(*[rig[i] for i in Tc]))
                if sp.simplify(srs.h0(b1, g) - srs.h1(b2, g)) != 0:
                    bad += 1
        check('n=%d (%s): layer-wise Serre duality on all %d layers'
              % (n, tag, 2 ** n), bad == 0, '%d bad' % bad)
    # the contact sheaf agrees with the paper's own engine
    if n:
        K = schw.contact_layers(n)
        check('n=%d: contact layers agree with srs.contact' % n,
              sorted(map(str, K)) == sorted(map(str, srs.contact(n).pieces)))

out()
out('    The base layer of S is L^{4-n} (x) delta, i.e. S = F_{2-n/2} (x) delta,')
out('    delta = det V; so the weight of the Schwarzian is 2 - n/2:')
for n in range(0, 8):
    out('       n = %d:  weight %s   order of the operator %d %s'
        % (n, sp.nsimplify(sp.Rational(4 - n, 2)), 6 - n,
           '' if n <= 4 else '  <-- no invariant operator'))

# ===================================================================== F
out()
out('=' * 78)
out('F.  The numbers.  sdim of the versal base and of the space of')
out('    LF-structures (= H^0 of S), for a general rigging.')
out('=' * 78)

out('    genus g >= 2, trivial rigging:')
for n in range(0, 5):
    K = schw.contact_layers(n)
    S = schw.schwarzian_layers(n)
    out('      n=%d   H^1(K) = %-28s H^0(S) = %s'
        % (n, srs.sdim(schw.sdim_h1(K, g)), srs.sdim(schw.sdim_h0(S, g))))

out('    genus g >= 2, independent rigging (all nu_i independent):')
for n in range(1, 5):
    rig = [srs.eta('n%d' % i) for i in range(n)]
    K = schw.contact_layers(n, rig)
    S = schw.schwarzian_layers(n, rig)
    out('      n=%d   H^1(K) = %-28s H^0(S) = %s'
        % (n, srs.sdim(schw.sdim_h1(K, g)), srs.sdim(schw.sdim_h0(S, g))))

out('    genus 0 (the projective line, one theta characteristic O(-1)):')
for n in range(0, 7):
    K = schw.contact_layers(n)
    S = schw.schwarzian_layers(n)
    out('      n=%d   H^1(K) = %-20s H^0(S) = %-20s  h^0 of the base layer '
        'O(%d) = %d'
        % (n, srs.sdim(schw.sdim_h1(K, 0)), srs.sdim(schw.sdim_h0(S, 0)),
           n - 4, max(0, n - 4 + 1)))

out('    genus 1:')
for sector in ('NS', 'R'):
    srs.set_g1_sector(sector)
    for n in range(1, 5):
        K = schw.contact_layers(n)
        S = schw.schwarzian_layers(n)
        out('      %-2s n=%d   H^1(K) = %-16s H^0(S) = %s'
            % (sector, n, srs.sdim(schw.sdim_h1(K, 1)),
               srs.sdim(schw.sdim_h0(S, 1))))
srs.set_g1_sector('NS')

out()
out('=' * 78)
out('%d checks, %d failed' % (CHECKS[0], FAILS[0]))
out('=' * 78)
LOG.close()
sys.exit(1 if FAILS[0] else 0)
