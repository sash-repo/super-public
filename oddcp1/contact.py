"""
contact.py -- the versal bases of the structured superstrings, Sections 18-26
(Theorem 21.1, Propositions 22.4, 22.6, 24.1, 24.2, 25.3, Statement 22.2).

Part A  self-tests: Riemann-Roch and Serre duality on every bundle used
Part B  calibration against Vaintrob's own answers (3.3.2 (2), 3.4.5) and
        against the numbers over CP^1 (Proposition 7.1)
Part C  the contact superstrings k(1|n), n = 1..4, trivial rigging, g >= 2
Part D  the rigging: where NS and Ramond part company
Part E  the pericontact m(1), and vect(1|1), vect(1|2)
Part F  genus 0 and genus 1 for every type

[Va1] below is Vaintrob's 1988 survey (Itogi Nauki 32), which the text cites
as [Va2].
"""
import sys
sys.path.insert(0, '.')

import sympy as sp
from srs import (G, LB, TRIV, eta, emul, s, h0, h1, Sheaf, sdeg,
                 contact, pericontact, vect, sdim, set_g1_sector)

LOG = open('contact.log', 'w')


def say(*a):
    t = ' '.join(str(x) for x in a)
    print(t)
    LOG.write(t + '\n')


FAIL = [0]


def check(name, got, want):
    ok = sp.simplify(sp.sympify(got) - sp.sympify(want)) == 0
    if not ok:
        FAIL[0] += 1
    say('  [%s] %-52s %s' % ('ok ' if ok else 'FAIL', name,
                             '' if ok else 'got %s, want %s' % (got, want)))


def checkpair(name, got, want):
    ok = all(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0
             for a, b in zip(got, want))
    if not ok:
        FAIL[0] += 1
    say('  [%s] %-52s %s' % ('ok ' if ok else 'FAIL', name,
                             sdim(got) if ok else 'got %s, want %s'
                             % (sdim(got), sdim(want))))


# =========================================================== A. self-tests
say('=' * 74)
say('PART A.  Riemann-Roch and Serre duality, on every bundle that occurs')
say('=' * 74)

say('\nA1.  h^0 - h^1 = deg + 1 - g  for L^k (x) eta, -6 <= k <= 6, g >= 2')
for k in range(-6, 7):
    for e in (TRIV, eta('a')):
        b = LB(k, e)
        lhs = sp.expand(h0(b) - h1(b))
        rhs = sp.expand(sdeg(b) + 1 - G)
        check('RR  %-8s eta=%s' % (b, 'O' if e == TRIV else 'a'), lhs, rhs)

say('\nA2.  the same at g = 0 and g = 1 (both riggings at g = 1)')
for gg in (0, 1):
    for k in range(-6, 7):
        classes = [TRIV] if gg == 0 else [TRIV, eta('a')]
        for e in classes:
            b = LB(k, e)
            check('RR  g=%d  %-8s' % (gg, b),
                  h0(b, gg) - h1(b, gg), sdeg(b, gg) + 1 - gg)

say('\nA3.  Serre duality is symmetric: h^1(F) = h^0(K F^{-1}) and back')
for k in range(-6, 7):
    for e in (TRIV, eta('a')):
        b, d = LB(k, e), LB(2 - k, e)
        check('Serre  %-8s' % b, h1(b), h0(d))
        check('Serre  %-8s (back)' % b, h0(b), h1(d))

say('\nA4.  chi computed two ways agrees, for every structure below')
for sh in [contact(1), contact(2), contact(3), contact(4), pericontact(),
           vect([LB(0)]), vect([LB(1), LB(-1)])]:
    checkpair('chi  %-12s' % sh.name, sh.chi(), sh.chi_from_cohomology())

say('\nA5.  the sanity of h^0(L) : it is s_o, and h^1(L) = s_o too (self-dual)')
check('h^0(L)', h0(LB(1)), s(TRIV))
check('h^1(L)', h1(LB(1)), s(TRIV))
check('h^0(K)', h0(LB(2)), G)
check('h^1(O)', h1(LB(0)), G)
check('h^1(Theta)', h1(LB(-2)), 3 * G - 3)
check('h^1(L^{-1})', h1(LB(-1)), 2 * G - 2)
check('h^1(K)', h1(LB(2)), 1)
check('h^1(L^3)', h1(LB(3)), 0)


# ===================================================== B. calibration on Vaintrob
say('')
say('=' * 74)
say('PART B.  Calibration: Vaintrob\'s own answers, and the answers over CP^1')
say('=' * 74)

say('\nB1.  [Va1] Thm. 3.4.5: a SUSY curve of genus g >= 2 has a universal')
say('     deformation with smooth base of dimension 3g-3 | 2g-2.')
say('     Here: the contact structure k(1|1); E = L, L^2 = K.')
checkpair('k(1|1), g >= 2', contact(1).sdim_versal(), (3 * G - 3, 2 * G - 2))

say('\nB2.  [Va1] 3.3.2 (2): X of genus g, F invertible of degree 0,')
say('     the supermanifold (X, Lambda F) deformed AS A SUPERMANIFOLD.')
say('     A non-trivial 2-torsion F models a generic degree-0 F exactly:')
say('     both have h^0(F) = h^0(F^*) = 0, and h^1 is then forced by RR.')
Fo = vect([LB(0)], split=True)                 # F = O      : T_0 = Theta (+) O splits
Fn = vect([LB(0, eta('a'))], split=True)       # F non-trivial, degree 0: flat, splits
checkpair('g >= 2, F = O          [Va1: 4g-3 | 4g-3]',
          Fo.sdim_versal(), (4 * G - 3, 4 * G - 3))
checkpair('g >= 2, F /= O, deg 0  [Va1: 4g-3 | 4g-4]',
          Fn.sdim_versal(), (4 * G - 3, 4 * G - 4))
checkpair('g = 1,  F = O          [Va1: 2 | 2]',
          Fo.sdim_versal(1), (2, 2))
checkpair('g = 1,  F /= O         [Va1: 2 | 0]',
          Fn.sdim_versal(1), (2, 0))
checkpair('g = 0,  F = O          [Va1: rigid]',
          Fo.sdim_versal(0), (0, 0))

say('\n     NB.  T_0 is the Atiyah algebroid of F, split iff F carries')
say('     a holomorphic connection -- true for deg F = 0 flat, which is')
say('     exactly Vaintrob\'s hypothesis.  Outside it only chi is safe.')

say('\nB3.  the text, Proposition 7.1: E = O(-k) over CP^1 (g = 0, L = O(-1)),')
say('     the odd obstructions H^1(T_1) = H^1(Theta (x) E) = max(0, k-3).')
for k in range(0, 12):
    got = h1(LB(k - 2), 0)
    check('H^1(Theta (x) O(-%d))' % k, got, max(0, k - 3))


# ============================================= C. the contact table, trivial rigging
say('')
say('=' * 74)
say('PART C.  n-extended contact superstrings, trivial rigging, g >= 2')
say('=' * 74)
say('''
   The sheaf is K = Theta (x) Lambda^. E, the piece of Lambda-degree q having
   parity q mod 2; for E = L (x) C^n it is (L^{q-2})^{binom(n,q)}.  So

       even = (3g-3) + binom(n,2).g + binom(n,4)
       odd  = n(2g-2) + binom(n,3).s,      s = h^0(L).
''')
for n in range(0, 7):
    sh = contact(n)
    ev, od = sh.sdim_versal()
    pred_ev = (3 * G - 3) + sp.binomial(n, 2) * G + sp.binomial(n, 4)
    pred_od = n * (2 * G - 2) + sp.binomial(n, 3) * s(TRIV)
    checkpair('k(1|%d)' % n, (ev, od), (pred_ev, pred_od))

say('\nC1.  the layer-by-layer table, k(1|n), trivial rigging, g >= 2')
say('     q   bundle      parity   h^1')
for q in range(0, 7):
    b = LB(q - 2)
    say('     %d   %-10s  %-6s   %s' % (q, b, 'ev' if q % 2 == 0 else 'od',
                                        sp.simplify(h1(b))))

say('\nC2.  where the rigging can possibly be felt: the layers whose h^1')
say('     depends on the class, i.e. q = 2, 3, 4.  n < 2 has none.')


# ================================================ D. the rigging: NS vs Ramond
say('')
say('=' * 74)
say('PART D.  The rigging.  NS and Ramond: when is the answer the same?')
say('=' * 74)
say('''
   A rigging of an n-extended contact superstring is eta_1,...,eta_n in
   H^1(X; Z/2): E = (+)_i L (x) eta_i, and L (x) eta_i is again a theta
   characteristic, so each odd coordinate is separately NS or Ramond around
   each of the 2g cycles.  All eta_i = O is one extreme; the answer below is
   the general one.
''')
for n in (1, 2, 3, 4):
    say('\n   n = %d' % n)
    triv = contact(n, [TRIV] * n)
    say('     all eta_i = O      : %s' % sdim(triv.sdim_versal()))
    distinct = contact(n, [eta('e%d' % i) for i in range(n)])
    say('     eta_i independent  : %s' % sdim(distinct.sdim_versal()))
    same = sp.simplify(triv.sdim_versal()[0] - distinct.sdim_versal()[0]) == 0 \
        and sp.simplify(triv.sdim_versal()[1] - distinct.sdim_versal()[1]) == 0
    say('     rigging-independent: %s' % ('YES' if same else 'NO'))

say('''
   Reading of the above.  For n = 1 the two agree identically: this is the
   proof that Theorem 3.4.5 does not depend on the rigging, and the reason is
   that the rigging can only be felt in the layers q = 2, 3, 4, of which
   n = 1 has none.  For n = 2 the EVEN part already parts company (the layer
   q = 2 contributes g when eta_1 = eta_2 and g-1 otherwise); for n >= 3 the
   ODD part does too, through the theta characteristics L (x) eta_S, |S| = 3.
''')

say('   D1.  n = 2, the even part, the two cases side by side:')
for tag, rig in (('eta_1 = eta_2', [eta('a'), eta('a')]),
                 ('eta_1 /= eta_2', [eta('a'), eta('b')])):
    say('        %-16s %s' % (tag, sdim(contact(2, rig).sdim_versal())))

say('\n   D2.  n = 3, the odd part carries one theta characteristic:')
for tag, rig in (('all trivial', [TRIV] * 3),
                 ('eta_1 = eta_2 = eta_3 = a', [eta('a')] * 3),
                 ('independent', [eta('a'), eta('b'), eta('c')])):
    say('        %-26s %s' % (tag, sdim(contact(3, rig).sdim_versal())))

say('\n   D3.  n = 4:')
for tag, rig in (('all trivial', [TRIV] * 4),
                 ('independent', [eta('a'), eta('b'), eta('c'), eta('d')])):
    say('        %-26s %s' % (tag, sdim(contact(4, rig).sdim_versal())))


# ============================================ E. pericontact, vect, and the rest
say('')
say('=' * 74)
say('PART E.  The pericontact m(1), and the vectorial types')
say('=' * 74)

say('\nE1.  m(1): pieces and answer')
m = pericontact()
for b, p in m.pieces:
    say('     %-10s %s' % (b, 'ev' if p == 0 else 'od'))
say('     sdim of the versal base, g >= 2 : %s' % sdim(m.sdim_versal()))
checkpair('m(1), g >= 2', m.sdim_versal(), (4 * G - 3, 4 * G - 4))
say('     with a rigging eta            : %s'
    % sdim(pericontact(eta('a')).sdim_versal()))

say('\nE2.  the three (1|2)-type answers side by side, g >= 2:')
say('     k(1|2), trivial rigging : %s' % sdim(contact(2).sdim_versal()))
say('     m(1)                    : %s' % sdim(m.sdim_versal()))
say('     vect(1|1), F /= O deg 0 : %s' % sdim(Fn.sdim_versal()))
say('     ranks   k(1|2) %s   m(1) %s   vect(1|1) %s'
    % (contact(2).rank(), m.rank(), Fn.rank()))
say('     chi     k(1|2) %s   m(1) %s   vect(1|1) %s'
    % (sdim(contact(2).chi()), sdim(m.chi()), sdim(Fn.chi())))

say('\nE3.  vect(1|2): E = L^a (+) L^b.  Only chi is claimed (T_q is an')
say('     extension, and T_0 is the Atiyah algebroid of E).')
for a, b in ((0, 0), (1, -1), (2, -2), (1, 1)):
    v = vect([LB(a), LB(b)])
    say('     E = L^%d (+) L^%d : rank %s  chi %s' % (a, b, v.rank(), sdim(v.chi())))

say("\nE4.  svect(1|2): a nowhere-zero section of Ber = K (x) (det E)^{-1} (x) Lambda E")
say('     needs det E = K, i.e. E = L^a (+) L^{2-a}.  Then')
say('     0 -> S -> T -> O_X -> 0 is exact, chi is additive, so')
say('     chi(S) = chi(T) - chi(O_X), and h^1(S) = h^0(S) + (8g-8).')
for a in (1, 0, 2, 3):
    b = 2 - a
    v = vect([LB(a), LB(b)])
    OX = Sheaf([(LB(0), 0), (LB(a + b), 0), (LB(a), 1), (LB(b), 1)], 'O_X')
    cv, co = v.chi(), OX.chi()
    ch = (sp.expand(cv[0] - co[0]), sp.expand(cv[1] - co[1]))
    say('     E = L^%d (+) L^%d : chi(T) %s  chi(O_X) %s  chi(S) %s'
        % (a, b, sdim(cv), sdim(co), sdim(ch)))
    checkpair('        chi(svect(1|2)), det E = K', ch, (8 - 8 * G, 8 - 8 * G))

# ================================================== F. the small genera
say('')
say('=' * 74)
say('PART F.  Genus 0 and genus 1')
say('=' * 74)

say('\nF1.  genus 0.  There is one theta characteristic, L = O(-1), and no')
say('     2-torsion: no rigging at all, so no NS/Ramond question.')
for n in range(1, 6):
    say('     k(1|%d) : %s' % (n, sdim(contact(n).sdim_versal(0))))
say('     m(1)   : %s' % sdim(pericontact().sdim_versal(0)))

say('\nF2.  genus 1.  L is itself 2-torsion -- one of the four spin')
say('     structures.  L = O is the ODD theta characteristic (h^0 = 1),')
say('     periodic around both cycles, i.e. Ramond; the three others are')
say('     even (h^0 = 0), i.e. Neveu-Schwarz.')
for sector in ('R', 'NS'):
    set_g1_sector(sector)
    say('     --- sector %s ---' % sector)
    for n in range(1, 6):
        got = contact(n).sdim_versal(1)
        want = (2 ** (n - 1), 2 ** (n - 1) if sector == 'R' else 0)
        checkpair('     k(1|%d), g=1' % n, got, want)
    say('     m(1)   : %s' % sdim(pericontact().sdim_versal(1)))
set_g1_sector('NS')

say('\nF3.  the SUSY curve (n = 1) in every genus, both sectors.  This is')
say('     the answer to the question which rigging Thm. 3.4.5 is about:')
set_g1_sector('R')
r1 = contact(1).sdim_versal(1)
set_g1_sector('NS')
n1 = contact(1).sdim_versal(1)
say('     g = 0,  one rigging only : %s' % sdim(contact(1).sdim_versal(0)))
say('     g = 1,  Ramond           : %s' % sdim(r1))
say('     g = 1,  Neveu-Schwarz    : %s' % sdim(n1))
say('     g >= 2, EVERY rigging    : %s' % sdim(contact(1).sdim_versal()))
say('')
say('     So: at g >= 2 the answer is the same in both sectors, and the')
say('     proof is that the rigging is felt only in the layers q = 2,3,4,')
say('     which k(1|1) does not have.  At g = 1 the answer is NOT the same.')

say('\nF4.  genus 0 again: CP^{1|n} IS the split n-extended contact')
say('     superstring of genus 0, since E = O(-1)^n = L^n there.  Compare')
say('     [Va1] 3.3.2 (1): P^{1|n} is rigid AS A SUPERMANIFOLD for n <= 3.')
for n in range(1, 6):
    say('     n = %d : as a contact superstring %s' % (n, sdim(contact(n).sdim_versal(0))))
say('     The single even class at n = 4 lives in H^1(CP^1; Theta (x) Lambda^4 E)')
say('     = H^1(CP^1; O(-2)) = C, spanned by the Cech class 1/t, i.e. by the')
say('     generating function t^{-1}theta_1theta_2theta_3theta_4 -- which is')
say('     exactly the element spanning k(1|4)/k\'(1|4).')
check('H^1(Theta (x) Lambda^4 E) at g=0, n=4', h1(LB(2), 0), 1)
check('...and it vanishes for n=3', h1(LB(1), 0), 0)

say('')
say('=' * 74)
say('%d checks failed' % FAIL[0])
say('=' * 74)
LOG.close()
sys.exit(1 if FAIL[0] else 0)
