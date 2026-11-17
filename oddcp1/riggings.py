"""
riggings.py -- Section 23: riggings that do not split into theta
characteristics (Theorem 23.6, its Corollaries, Propositions 23.11-23.21,
Statement 23.17).

Part A  self-tests: Riemann-Roch, Serre duality, and the exterior algebra
Part B  calibration: the diagonal riggings reproduce srs.py exactly
Part C  the closed formula for a semistable rigging, on every split one
Part D  the two tails, and where they can fail to vanish
Part E  odd dimension 2: the complete answer, diagonal and not
Part F  odd dimensions 3 and 4: what a non-diagonal rigging changes
Part G  the generic rigging, and the diagonal one as the maximal jump
Part H  genus 0: over the projective line every non-trivial rigging is
        unstable, and the deformations it gives are ODD
Part I  genus 1

[Va1] below is Vaintrob's 1988 survey, which the text cites as [Va2].
"""
import sys

sys.path.insert(0, '.')

import sympy as sp

import srs
from srs import G, eta, TRIV
from orth import (Curve, Line, ONE, L, K, THETA, torsion, diagonal,
                  hyperbolic, lam, tw, h0V, h1V, det, layers, sdim_versal,
                  sdim_closed, tails, semistable_split, sdim)

LOG = open('riggings.log', 'w')


def say(*a):
    t = ' '.join(str(x) for x in a)
    print(t)
    LOG.write(t + '\n')


FAIL = [0]


def check(name, got, want):
    ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0 \
        if not isinstance(got, (tuple, list)) else \
        all(sp.simplify(sp.sympify(x) - sp.sympify(y)) == 0
            for x, y in zip(got, want))
    say('  [%s] %-54s %s' % ('ok ' if ok else 'FAIL', name,
                             '' if ok else '%s vs %s' % (got, want)))
    if not ok:
        FAIL[0] += 1


def head(t):
    say('')
    say('=' * 74)
    say(t)
    say('=' * 74)


GENERA = [2, 3, 4, 5, 6, 7, 8]

# ============================================================ Part A
head('Part A.  Self-tests')

say("""
A1.  Riemann-Roch h^0 - h^1 = deg + 1 - g on every line bundle that occurs
     below: L^k (x) nu for -6 <= k <= 6, and M^j L^k with deg M = d.
""")
bad = 0
for g in GENERA:
    C = Curve(g, deg={'M': 0})
    for k in range(-6, 7):
        for t in [(), ('a',), ('a', 'b')]:
            b = L.pow(k) * torsion(*t)
            if sp.simplify(C.h0(b) - C.h1(b) - C.chi(b)) != 0:
                bad += 1
    for d in range(-4, 9):
        C = Curve(g, deg={'M': d})
        for j in (-1, 1):
            for k in range(-4, 5):
                b = Line({'M': j, 'L': k})
                if sp.simplify(C.h0(b) - C.h1(b) - C.chi(b)) != 0:
                    bad += 1
check('Riemann-Roch on every line bundle used', bad, 0)

say("""
A2.  Serre duality is an involution: h^1(F) = h^0(K F^{-1}) and
     h^0(F) = h^1(K F^{-1}).
""")
bad = 0
for g in GENERA:
    C = Curve(g, deg={'M': 2})
    for k in range(-4, 5):
        for j in (-1, 0, 1):
            b = Line({'M': j, 'L': k})
            if sp.simplify(C.h1(b) - C.h0(K * b.inv())) != 0:
                bad += 1
            if sp.simplify(C.h0(b) - C.h1(K * b.inv())) != 0:
                bad += 1
check('Serre duality, both ways', bad, 0)

say("""
A3.  The exterior algebra of a split orthogonal bundle: rank Lambda^q V
     = C(n,q); Lambda^n V = det V is 2-torsion; deg Lambda^q V = 0;
     and Lambda^q V is SELF-DUAL, which is what Serre duality is applied to.
""")
bad_rank = bad_deg = bad_dual = bad_det = 0
C = Curve(4, deg={'M': 3, 'N': 0})
TESTV = [
    diagonal((), (), ()),
    diagonal(('a',), ('a',), ('b',)),
    hyperbolic('M'),
    hyperbolic('M') + diagonal(()),
    hyperbolic('M', 'N'),
    hyperbolic('M') + diagonal(('a',), ('b',)),
    hyperbolic('M', 'N') + diagonal(()),
]
for V in TESTV:
    n = len(V)
    for q in range(n + 1):
        Lq = lam(V, q)
        if len(Lq) != sp.binomial(n, q):
            bad_rank += 1
        if sum(C.degree(b) for b in Lq) != 0:
            bad_deg += 1
        if sorted(b.key() for b in Lq) != sorted(b.inv().key() for b in Lq):
            bad_dual += 1
    d = det(V)
    if (d * d).key() != ONE.key():
        bad_det += 1
check('rank Lambda^q V = C(n,q)', bad_rank, 0)
check('deg Lambda^q V = 0', bad_deg, 0)
check('Lambda^q V is self-dual', bad_dual, 0)
check('det V is 2-torsion', bad_det, 0)

say("""
A4.  ad V = Lambda^2 V: rank C(n,2) = dim so(n), degree 0, self-dual -- and
     h^1(ad V) - h^0(ad V) = C(n,2)(g-1), which is the dimension of the
     moduli of O(n)-bundles when h^0(ad V) = 0.
""")
bad = 0
for g in GENERA:
    C = Curve(g, deg={'M': 0, 'N': 0})
    for V in TESTV:
        n = len(V)
        ad = lam(V, 2)
        if sp.simplify(h1V(C, ad) - h0V(C, ad)
                       - sp.binomial(n, 2) * (g - 1)) != 0:
            bad += 1
check('h^1(ad V) - h^0(ad V) = C(n,2)(g-1)', bad, 0)

# ============================================================ Part B
head('Part B.  Calibration: the diagonal riggings reproduce srs.py')

say("""
B1.  For V = (+) nu_i the layers Lambda^q V (x) L^{q-2} of this module are
     the layers L^{q-2} (x) nu_S of srs.py, and the two versal
     superdimensions agree, for every genus 2 <= g <= 8 and every rigging
     of 1 <= n <= 5 odd coordinates drawn from the list below.
""")
RIGS = {
    1: [[TRIV], [eta('a')]],
    2: [[TRIV, TRIV], [TRIV, eta('a')], [eta('a'), eta('a')],
        [eta('a'), eta('b')]],
    3: [[TRIV] * 3, [TRIV, TRIV, eta('a')], [eta('a'), eta('b'), eta('c')],
        [eta('a'), eta('a'), eta('b')]],
    4: [[TRIV] * 4, [TRIV, TRIV, eta('a'), eta('a')],
        [eta('a'), eta('b'), eta('c'), eta('d')]],
    5: [[TRIV] * 5, [TRIV, eta('a'), eta('a'), eta('b'), eta('b')]],
}
bad_layers = bad_sdim = 0
for n, rl in RIGS.items():
    for rig in rl:
        Vo = [torsion(*r) for r in rig]
        srs_sheaf = srs.contact(n, rig)
        for g in GENERA:
            C = Curve(g)
            # the layers, as multisets of (line bundle, parity)
            mine = sorted((b.free.get('L', 0), tuple(sorted(b.tors)), p)
                          for pieces, p in layers(Vo) for b in pieces)
            theirs = sorted((b.k, tuple(sorted(b.e)), p)
                            for b, p in srs_sheaf.pieces)
            if mine != theirs:
                bad_layers += 1
            a1 = sdim_versal(C, Vo)
            a2 = tuple(sp.expand(x.subs(G, g))
                       for x in srs_sheaf.sdim_versal())
            if any(sp.simplify(x - y) != 0 for x, y in zip(a1, a2)):
                bad_sdim += 1
check('the layers are srs.py layers', bad_layers, 0)
check('the versal superdimensions agree', bad_sdim, 0)

say("""
B2.  ...and for the trivial rigging both are the printed closed formula
     ( 3g-3 + C(n,2)g + C(n,4) | n(2g-2) + C(n,3)s ).
""")
bad = 0
sO = srs.s(TRIV)
for n in range(1, 7):
    Vo = [torsion() for _ in range(n)]
    for g in GENERA:
        C = Curve(g)
        ev, od = sdim_versal(C, Vo)
        want_ev = 3 * g - 3 + sp.binomial(n, 2) * g + sp.binomial(n, 4)
        want_od = n * (2 * g - 2) + sp.binomial(n, 3) * sO
        if sp.simplify(ev - want_ev) != 0 or sp.simplify(od - want_od) != 0:
            bad += 1
check('the trivial rigging gives the printed formula', bad, 0)

# ============================================================ Part C
head('Part C.  The closed formula for a rigging that need not be diagonal')

say("""
C1.  For every SPLIT orthogonal bundle V -- a sum of anisotropic lines nu_i
     and of hyperbolic planes M (+) M^{-1} -- the direct layer-by-layer
     computation agrees with

       ( 3g-3 + h^1(ad V) + h^0(Lambda^4 V) + (even tail)
              | h^0(V (x) L^3) + h^0(Lambda^3 V (x) L) + (odd tail) ),

     the tails being the terms with r >= 3, resp. r >= 2.  Tested on 1 <= n
     <= 6, on hyperbolic planes of degree -3 <= d <= 6, and on 2 <= g <= 8.
""")
FAMILIES = []
for t in [(), ('a',)]:
    FAMILIES.append(('diag nu=%s' % (t or 'O',), lambda t=t: diagonal(t)))
FAMILIES += [
    ('diag (O,O)', lambda: diagonal((), ())),
    ('diag (a,b)', lambda: diagonal(('a',), ('b',))),
    ('hyp M', lambda: hyperbolic('M')),
    ('hyp M + O', lambda: hyperbolic('M') + diagonal(())),
    ('hyp M + a', lambda: hyperbolic('M') + diagonal(('a',))),
    ('hyp M + (O,O)', lambda: hyperbolic('M') + diagonal((), ())),
    ('hyp M,N', lambda: hyperbolic('M', 'N')),
    ('hyp M,N + O', lambda: hyperbolic('M', 'N') + diagonal(())),
    ('hyp M,N + (a,b)', lambda: hyperbolic('M', 'N') + diagonal(('a',), ('b',))),
    ('hyp M,N,P', lambda: hyperbolic('M', 'N', 'P')),
]
bad = 0
cases = 0
for name, mk in FAMILIES:
    V = mk()
    for d in range(-3, 7):
        for g in GENERA:
            C = Curve(g, deg={'M': d, 'N': 0, 'P': 1})
            direct = sdim_versal(C, V)
            closed = sdim_closed(C, V)
            tl = tails(C, V)
            got = (sp.expand(closed[0] + tl[0]), sp.expand(closed[1] + tl[1]))
            cases += 1
            if any(sp.simplify(x - y) != 0 for x, y in zip(direct, got)):
                bad += 1
                say('      MISMATCH %s d=%d g=%d: %s vs %s'
                    % (name, d, g, sdim(direct), sdim(got)))
say('     %d cases' % cases)
check('closed formula + tails = the layer sum', bad, 0)

say("""
C2.  For a SEMISTABLE V the tails vanish and h^0(V (x) L^3) = n(2g-2), so
     the formula collapses to

       ( 3g-3 + h^1(ad V) + h^0(Lambda^4 V) | n(2g-2) + h^0(Lambda^3 V (x) L) ).
""")
bad_t = bad_c = 0
cases = 0
for name, mk in FAMILIES:
    V = mk()
    for g in GENERA:
        C = Curve(g, deg={'M': 0, 'N': 0, 'P': 0})
        if not semistable_split(C, V):
            continue
        cases += 1
        if tails(C, V) != (0, 0):
            bad_t += 1
        n = len(V)
        if sp.simplify(h0V(C, tw(V, L.pow(3))) - n * (2 * g - 2)) != 0:
            bad_c += 1
say('     %d semistable cases' % cases)
check('the tails vanish on a semistable rigging', bad_t, 0)
check('h^0(V (x) L^3) = n(2g-2) there', bad_c, 0)

say("""
C3.  The tails vanish for n <= 5 WHATEVER the rigging: Lambda^q V = 0 for
     q > n, and for n = 4, 5 the only surviving term of a tail carries
     Lambda^5 V = det V (x) (a factor of degree 0) against L^{-1}.
""")
bad = 0
for name, mk in FAMILIES:
    V = mk()
    if len(V) > 5:
        continue
    for d in range(-4, 8):
        for g in GENERA:
            C = Curve(g, deg={'M': d, 'N': d, 'P': d})
            if tails(C, V) != (0, 0):
                bad += 1
check('the tails vanish identically for n <= 5', bad, 0)

# ============================================================ Part D
head('Part D.  Where a tail CAN fail to vanish')

say("""
D1.  The ODD tail is the first to wake up, and it needs n >= 6: for n <= 5
     its only term carries Lambda^5 V = det V, of degree 0, against L^{-1}.
     At n = 6, with three hyperbolic planes of degree d, Lambda^5 V has
     summands of degree +-d, and h^0(Lambda^5 V (x) L^{-1}) is non-zero as
     soon as d > 2g-2; it then equals 3(d-2g+2).
""")
for g in (2, 3):
    for d in (0, 1, 2, 3, 9):
        C = Curve(g, deg={'M': d, 'N': d, 'P': d})
        V = hyperbolic('M', 'N', 'P')
        tl = tails(C, V)
        say('     g=%d d=%-2d n=6  tails = (%s | %s)   sdim = %s'
            % (g, d, tl[0], tl[1], sdim(sdim_versal(C, V))))

say("""
D2.  The EVEN tail needs n >= 7: at n = 6 its only term carries
     Lambda^6 V = det V, again of degree 0, against K^{-1}.  With n = 7 --
     three hyperbolic planes and one anisotropic line -- Lambda^6 V acquires
     summands of degree +-d and the tail is non-zero once d > 3g-3.
""")
for g in (2, 3):
    for d in (0, 2, 3, 4, 6, 9):
        C = Curve(g, deg={'M': d, 'N': d, 'P': d, 'Q': d})
        V7 = hyperbolic('M', 'N', 'P') + diagonal(())
        V8 = hyperbolic('M', 'N', 'P', 'Q')
        t7, t8 = tails(C, V7), tails(C, V8)
        say('     g=%d d=%-2d n=7 tails = (%s | %s)    n=8 tails = (%s | %s)'
            % (g, d, t7[0], t7[1], t8[0], t8[1]))

say("""
     In every one of these the rigging is unstable: for a semistable V the
     tails vanish for all n, which is C2.  So the collapsed formula fails
     only off the semistable locus, and only from n = 6 on.
""")

# ============================================================ Part E
head('Part E.  Odd dimension 2: the complete answer')

say("""
E1.  For n = 2 and ANY rigging,

        sdim B = ( 3g-3 + h^1(det V) | 4g-4 + h^0(V (x) L^{-1}) ),

     since Lambda^2 V = det V is 2-torsion and the only odd layer is
     Theta (x) E = V (x) L^{-1}, of Euler characteristic 4-4g.  Verified
     against the layer sum for every split V of rank 2 and 2 <= g <= 8.
""")
bad = 0
cases = 0
RANK2 = [('diag (O,O)', lambda: diagonal((), ())),
         ('diag (a,a)', lambda: diagonal(('a',), ('a',))),
         ('diag (a,b)', lambda: diagonal(('a',), ('b',))),
         ('hyp M', lambda: hyperbolic('M'))]
for name, mk in RANK2:
    V = mk()
    for d in range(-6, 10):
        for g in GENERA:
            C = Curve(g, deg={'M': d})
            ev, od = sdim_versal(C, V)
            want_ev = 3 * g - 3 + C.h1(det(V))
            want_od = 4 * g - 4 + h0V(C, tw(V, L.pow(-1)))
            cases += 1
            if sp.simplify(ev - want_ev) != 0 or sp.simplify(od - want_od) != 0:
                bad += 1
say('     %d cases' % cases)
check('n = 2: ( 3g-3+h^1(det V) | 4g-4+h^0(V L^{-1}) )', bad, 0)

say("""
E2.  Hence the census.  The even entry sees nothing but det V; the odd one
     is 4g-4 exactly when V has no line subbundle of degree >= g-1, which is
     implied by semistability.  Both values of the even entry are already
     attained by DIAGONAL riggings, so at n = 2 a non-diagonal rigging
     produces no superdimension a diagonal one does not -- as long as V is
     semistable.  Outside that range the odd entry is unbounded.
""")
for g in (2, 3, 5):
    say('     g = %d' % g)
    C = Curve(g)
    for name, mk in RANK2[:3]:
        say('       %-14s det V = %-4s  sdim = %s'
            % (name, det(mk()), sdim(sdim_versal(C, mk()))))
    for d in range(0, 3 * 5 + 1):
        C = Curve(g, deg={'M': d})
        say('       hyp M, deg M = %-2d det V = O     sdim = %s'
            % (d, sdim(sdim_versal(C, hyperbolic('M')))))

say("""
E3.  The special riggings inside the hyperbolic family: M = L^j is not
     general, and there the odd entry jumps.  M = L gives E = K (+) O.
""")
for g in (2, 3, 5):
    for j in (1, 2, 3):
        C = Curve(g)
        V = [L.pow(j), L.pow(-j)]
        say('     g = %d  M = L^%d (deg %d)  sdim = %s'
            % (g, j, j * (g - 1), sdim(sdim_versal(C, V))))

say("""
E4.  The component det V = delta =/= O.  A split representative is the
     diagonal rigging (nu_1, nu_2) with nu_1 nu_2 = delta; the general
     member is V = pi_* N for the etale double cover pi : X_delta --> X
     defined by delta and N of norm O, which is semistable and does not
     split.  For both, Lambda^2 V = delta and h^0(V (x) L^{-1}) = 0, so the
     answer is (4g-4 | 4g-4) throughout the component.
""")
for g in GENERA[:4]:
    C = Curve(g)
    V = diagonal(('a',), ('b',))
    say('     g = %d  split representative: sdim = %s   h^1(delta) = %s'
        % (g, sdim(sdim_versal(C, V)), C.h1(det(V))))

# ============================================================ Part F
head('Part F.  Odd dimensions 3 and 4')

say("""
F1.  n = 3.  Lambda^3 V = det V and Lambda^4 V = 0, so

       sdim B = ( 3g-3 + h^1(ad V) | 6g-6 + h^0(det V (x) L) ),
       h^1(ad V) = 3(g-1) + h^0(ad V).

     The odd entry does not see the rigging beyond det V.  The even one does,
     through h^0(ad V), which is 3 for the trivial diagonal rigging and 0 for
     a V that is stable as a vector bundle -- the two extremes being 6g-3 and
     6g-6.  A split SO(3)-bundle O (+) M (+) M^{-1} = sl(W) for W = A (+) B,
     M = A B^{-1}, sits in between at 6g-5.
""")
for g in (2, 3, 4):
    C = Curve(g, deg={'M': 0})
    for name, V in [('diagonal (O,O,O)', diagonal((), (), ())),
                    ('diagonal (a,a,b)', diagonal(('a',), ('a',), ('b',))),
                    ('diagonal (a,b,c)', diagonal(('a',), ('b',), ('c',))),
                    ('sl(A (+) B)  = O + M + M^{-1}',
                     diagonal(()) + hyperbolic('M'))]:
        ad = lam(V, 2)
        say('     g=%d  %-28s h^0(ad V)=%-2s sdim = %s'
            % (g, name, h0V(C, ad), sdim(sdim_versal(C, V))))
say("""
     and, for a V stable as a vector bundle, h^0(ad V) = 0 because
     h^0(End V) = 1 and the identity is symmetric, whence
""")
for g in (2, 3, 4):
    say('     g=%d  stable V, det V = delta       sdim = (%s | %s)'
        % (g, 6 * g - 6, sp.sympify(6 * g - 6) + srs.s(eta('delta'))))

say("""
F2.  n = 4.  Now Lambda^4 V = det V enters the EVEN entry: h^0(det V) is 1
     if V is special orthogonal and 0 otherwise.  So at n = 4 the parity of
     the rigging -- its first Stiefel-Whitney class -- is felt twice, in
     h^1(ad V) and in h^0(Lambda^4 V).
""")
for g in (2, 3):
    C = Curve(g, deg={'M': 0, 'N': 0})
    for name, V in [('diagonal (O,O,O,O)', diagonal((), (), (), ())),
                    ('diagonal (a,a,b,b)',
                     diagonal(('a',), ('a',), ('b',), ('b',))),
                    ('diagonal (a,b,c,d)',
                     diagonal(('a',), ('b',), ('c',), ('d',))),
                    ('two hyperbolic planes', hyperbolic('M', 'N')),
                    ('hyperbolic + (a,b)',
                     hyperbolic('M') + diagonal(('a',), ('b',)))]:
        say('     g=%d  %-24s det V = %-6s h^0(L^4V)=%-2s sdim = %s'
            % (g, name, det(V), h0V(C, lam(V, 4)),
               sdim(sdim_versal(C, V))))

# ============================================================ Part G
head('Part G.  The generic rigging, and the diagonal one as the maximal jump')

say("""
G1.  The three numbers h^0(ad V), h^0(Lambda^4 V), h^0(Lambda^3 V (x) L) are
     the whole dependence of the answer on the rigging (for V semistable),
     and all three are upper semicontinuous.  Inside the DIAGONAL family
     they vanish simultaneously as soon as the nu_i are linearly independent
     in H^1(X; Z/2) = (Z/2)^{2g}, which asks for n <= 2g and makes every
     nu_S with |S| = 2, 4 non-trivial and every s(nu_S) with |S| = 3 that of
     a theta characteristic in general position.  So the general answer is

             ( 3g-3 + C(n,2)(g-1) | n(2g-2) ),

     and the trivial diagonal rigging exceeds it by exactly
     C(n,2) + C(n,4) = h^0(ad V) + h^0(Lambda^4 V) in the even entry.
""")
bad = 0
for n in range(1, 7):
    for g in GENERA:
        if n > 2 * g:
            continue
        C = Curve(g)
        V = [torsion('e%d' % i) for i in range(n)]   # independent classes
        ev, od = sdim_versal(C, V)
        # a theta characteristic in general position has s = 0
        od = od.subs({srs.s(b.tors): 0 for b in lam(V, 3)})
        want = (3 * g - 3 + sp.binomial(n, 2) * (g - 1), n * (2 * g - 2))
        if sp.simplify(ev - want[0]) != 0 or sp.simplify(od - want[1]) != 0:
            bad += 1
check('a general diagonal rigging gives (3g-3+C(n,2)(g-1) | n(2g-2))', bad, 0)
bad = 0
for n in range(1, 7):
    for g in GENERA:
        C = Curve(g)
        triv = [torsion() for _ in range(n)]
        ev, _ = sdim_versal(C, triv)
        if sp.simplify(ev - (3 * g - 3 + sp.binomial(n, 2) * (g - 1)
                             + sp.binomial(n, 2) + sp.binomial(n, 4))) != 0:
            bad += 1
check('the trivial rigging exceeds it by C(n,2)+C(n,4)', bad, 0)

say("""
G2.  And the same general value is what a STABLE V gives, by h^0(ad V) = 0
     and h^0(Lambda^4 V) = h^0(Lambda^3 V (x) L) = 0 in general position.
     So a non-diagonal rigging produces no superdimension that a diagonal
     one does not, as long as n <= 2g; what it produces is MODULI -- the
     riggings form a family of dimension C(n,2)(g-1), inside which the
     diagonal ones are the 2^{2gn} points of an O(1)^n-reduction.
""")
for n in (2, 3, 4):
    for g in (2, 3, 5):
        say('     n=%d g=%d  dim Bun_{O(n)} = %s   general sdim = (%s | %s)'
            % (n, g, sp.binomial(n, 2) * (g - 1),
               3 * g - 3 + sp.binomial(n, 2) * (g - 1), n * (2 * g - 2)))

say("""
G3.  H^0(X; Theta) = 0 for g >= 2, so the Atiyah sequence
     0 --> ad V --> At(V) --> Theta --> 0 gives
     h^1(At V) = h^1(Theta) + h^1(ad V): the EVEN entry of the versal base
     is, for n <= 3, exactly the dimension of the deformations of the PAIR
     (X, V), curve and orthogonal bundle deforming together.
""")
bad = 0
for g in GENERA:
    C = Curve(g)
    if C.h0(THETA) != 0:
        bad += 1
check('h^0(Theta) = 0 for 2 <= g <= 8', bad, 0)


# ============================================================ Part H
head('Part H.  Genus 0: every non-trivial rigging over CP^1 is unstable')

say("""
H1.  On CP^1 every bundle splits and there is no 2-torsion, so an
     orthogonal bundle is (+)_j (O(d_j) (+) O(-d_j)) (+) O^{n-2k} and the
     only theta characteristic is L = O(-1).  Hence the diagonal rigging is
     unique -- E = O(-1)^n, i.e. CP^{1|n} -- and every other rigging is
     unstable.  For n = 2, V = O(d) (+) O(-d):
""")
for d in range(0, 9):
    C = Curve(0, deg={'M': d})
    V = hyperbolic('M')
    E = tw(V, L)
    say('     d = %-2d E = %-24s sdim = %s'
        % (d, ' (+) '.join('O(%d)' % C.degree(b) for b in E),
           sdim(sdim_versal(C, V))))

say("""
     So the deformations given by an unstable rigging over CP^1 are purely
     ODD, of dimension d-2, and they appear exactly at d = 3, i.e. exactly
     when a summand of E reaches O(-4).  That is the same threshold as the
     one at which the odd obstructions of Subsec. 19.2
     appear over CP^1: there h^1(Theta (x) F) = h^1(O(2) (x) F) is non-zero
     first for F = O(-4).
""")
Cm = Curve(0, deg={'F': -4})
check('h^1(Theta (x) O(-4)) = 1 over CP^1', Cm.h1(THETA * Line({'F': 1})), 1)
Cm3 = Curve(0, deg={'F': -3})
check('h^1(Theta (x) O(-3)) = 0 over CP^1', Cm3.h1(THETA * Line({'F': 1})), 0)
bad = 0
for d in range(0, 12):
    C = Curve(0, deg={'M': d})
    ev, od = sdim_versal(C, hyperbolic('M'))
    if ev != 0 or od != max(0, d - 2):
        bad += 1
check('n=2 over CP^1: sdim = (0 | max(0,d-2))', bad, 0)

say("""
H2.  n = 3 and n = 4 over CP^1, one hyperbolic plane of degree d and the
     rest trivial.  The rigid cases n <= 3 of Proposition 22.6
     ("Contact superstrings over CP^1") are the ones with d = 0.
""")
for n, mk in [(3, lambda: hyperbolic('M') + diagonal(())),
              (4, lambda: hyperbolic('M') + diagonal((), ())),
              (4, lambda: hyperbolic('M', 'N'))]:
    for d in range(0, 5):
        C = Curve(0, deg={'M': d, 'N': d})
        say('     n = %d  d = %-2d sdim = %s' % (n, d, sdim(sdim_versal(C, mk()))))

# ============================================================ Part I
head('Part I.  Genus 1')

say("""
I1.  At g = 1 one has K = O and L is one of the four points of order 2:
     L = O is the odd theta characteristic (Ramond), the three others are
     even (Neveu-Schwarz).  For n = 2 and V = M (+) M^{-1} with deg M = 0:
     the even entry is 2 for every M, and the odd one is 2 if M = L and 0
     otherwise.  So the Ramond locus of the rank-2 rigging is a single point
     of a g-dimensional family.
""")
for sector, trivial in (('Ramond   (L = O) ', True),
                        ('Neveu-Schwarz    ', False)):
    C = Curve(1, deg={'M': 0}, L_at_g1_trivial=trivial)
    for name, V in [('M general      ', hyperbolic('M')),
                    ('M = L          ', [L, L.inv()]),
                    ('diagonal (O,O) ', diagonal((), ())),
                    ('diagonal (a,a) ', diagonal(('a',), ('a',))),
                    ('diagonal (a,b) ', diagonal(('a',), ('b',)))]:
        say('     %s %s sdim = %s' % (sector, name, sdim(sdim_versal(C, V))))

say("""
I2.  ...and the case M = L reproduces, in the Ramond sector, the (2|2) of
     item 2) of [Va1] Subsec. 3.3.2, while a general M gives its (2|0).
""")
C = Curve(1, deg={'M': 0}, L_at_g1_trivial=True)
check('g=1, M = L, Ramond: (2|2)', sdim_versal(C, [L, L.inv()]), (2, 2))
check('g=1, M general, Ramond: (2|0)', sdim_versal(C, hyperbolic('M')), (2, 0))

# ============================================================
head('%d checks failed' % FAIL[0])
LOG.close()
