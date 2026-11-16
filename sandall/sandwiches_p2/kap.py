#!/usr/bin/env python3
r"""
kap.py --- the Kaplansky algebras, their 2-closures, and their linear and
NONLINEAR superizations, built from scratch over F_2 and machine-checked.

Source of the definitions: Bouarroudj, Lebedev, Leites, Shchepochkina,
"Lie algebra deformations in characteristic 2", Math. Res. Lett. 22 (2015)
353--402 = arXiv:1301.2781; the equation numbers below are its own.

J-systems (eq. (64)).  Over F_2, a J-system Gamma in a space V with a symmetric
inner product B is a set of non-zero vectors such that u + v in Gamma whenever
u, v in Gamma are distinct with B(u, v) = 1.  It defines the Lie algebra

    g_Gamma = Span(e_u | u in Gamma),   [e_u, e_v] = B(u, v) e_{u+v}
                                        (0 if u = v or u + v not in Gamma).

The Kaplansky algebras (eqs. (65), (66)) are, with dim V = 2m and B alternate
non-degenerate, resp. dim V = n and B = I non-alternate:

    Kap_1(n)     : Gamma = V \ {0, e},  e the unique vector with B(e,y)=B(y,y);
    Kap_2(2m)    : Gamma = V \ {0};
    Kap_3(n)     : = o'_I(n) (Kaplansky's own observation);
    Kap_{4,a}(2m): Gamma = {u | Q(u) = 1}, Arf(Q) = a, B(u,v)=Q(u+v)+Q(u)+Q(v);
    Kap_{4,B}(2m): Gamma = V (so e_0 is central).

What is checked here (all of it exactly, over F_2, no floating point and no
sampling unless said so):

  1. the Jacobi identity and [x,x] = 0 for every algebra built;
  2. the dimension formula (73) and the inclusions
     Kap_{4,a}(2m) < Kap_2(2m) < Kap_{4,B}(2m);
  3. Prop. 5.1.1: Kap_{4,B}(2m) is isomorphic to O(2m; N_s) with the bracket
     (74) --- checked by the explicit map e_u |--> f_u, not by a search;
  4. eq. (69): Kap_{4,B}(2m) = Kap_2(2m) + c, c the center;
  5. the small isomorphisms (76) and the exceptions of Subsec. 5.2, by an
     isomorphism search over F_2 (dims 3, 6, 10) and by invariants (dim 15);
  6. the 2-closure (eqs. (80)--(83)) is a restricted Lie algebra;
  7. Subsec. 5.1.2: the operator D: e_u |--> (Q(u) + 1)e_u IS a derivation with
     D^2 = D, i.e., a Z/2-grading, and it is NOT linear in the weights: no
     phi in V^* has phi(u) = Q(u) + 1 on Gamma;
  8. the linear superization KapLS_2(2m) and the nonlinear ones
     KapS_{2,a}(2m) (eq. (88)) and KapS_{4,a,eps}(2m) (eqs. (89), (90)):
     that each is a Lie superalgebra (grading + squaring + [x^2,y]=[x,[x,y]]),
     that it is simple, and its superdimension;
  9. the sandwich data --- dim s(L), codim n_L(s), maximality --- of every
     algebra of the family within reach of an exhaustive scan.

Output: runs/kap.log (never /tmp).

Usage:  python3 kap.py [m_max]      (default 2, i.e. dim V <= 4; m=3 does the
                                     structure checks only, no 2^n scans)
"""

import os
import sys
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from css import LieF2, rref, in_span, all_elements   # noqa: E402

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', 'kap.log')
_log = None


def say(msg=''):
    print(msg)
    if _log:
        _log.write(msg + '\n')


# --------------------------------------------------------------- V, B, Q

def bitcount(x):
    return bin(x).count('1')


def B_alt(m):
    """the alternate non-degenerate form on V = F_2^{2m}: bits 0..m-1 vs m..2m-1"""
    def B(u, v):
        return (bitcount((u & (v >> m)) & ((1 << m) - 1)) +
                bitcount(((u >> m) & v) & ((1 << m) - 1))) & 1
    return B


def B_id(n):
    """the non-alternate form B = I on V = F_2^n"""
    return lambda u, v: bitcount(u & v) & 1


def Q_arf(m, a):
    """Q_0(u) = sum u_i u_{i+m};  Q_1(u) = u_0 + u_m + Q_0(u)   (eq. (71))"""
    def Q(u):
        q = bitcount((u & (u >> m)) & ((1 << m) - 1)) & 1
        if a:
            q ^= (u & 1) ^ ((u >> m) & 1)
        return q
    return Q


# ------------------------------------------------------- J-system algebras

class Jalg(LieF2):
    """the Lie algebra g_Gamma of a J-system, with the vectors kept"""

    def __init__(self, gamma, B, name):
        gamma = list(gamma)
        idx = {u: i for i, u in enumerate(gamma)}
        n = len(gamma)
        tab = [[0] * n for _ in range(n)]
        for i, u in enumerate(gamma):
            for j, v in enumerate(gamma):
                if u == v or not B(u, v):
                    continue
                k = idx.get(u ^ v)
                if k is not None:
                    tab[i][j] = 1 << k
        LieF2.__init__(self, tab, name)
        self.gamma, self.idx, self.B = gamma, idx, B


def kap1(n):
    B = B_id(n)
    e = (1 << n) - 1
    return Jalg([u for u in range(1, 1 << n) if u != e], B, 'Kap_1(%d)' % n)


def kap2(m):
    return Jalg(range(1, 1 << (2 * m)), B_alt(m), 'Kap_2(%d)' % (2 * m))


def kapB(m):
    return Jalg(range(1 << (2 * m)), B_alt(m), 'Kap_{4,B}(%d)' % (2 * m))


def kap4(m, a):
    Q = Q_arf(m, a)
    return Jalg([u for u in range(1, 1 << (2 * m)) if Q(u)], B_alt(m),
                'Kap_{4,%d}(%d)' % (a, 2 * m))


def oI(n):
    """o'_I(n): the symmetric n x n matrices with zero diagonal, over F_2"""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = {p: k for k, p in enumerate(pairs)}
    N = len(pairs)

    def mat(k):
        i, j = pairs[k]
        return {(i, j), (j, i)}

    tab = [[0] * N for _ in range(N)]
    for k in range(N):
        A = mat(k)
        for l in range(N):
            C = mat(l)
            prod = {}
            for (i, j) in A:
                for (r, s) in C:
                    if j == r:
                        prod[(i, s)] = prod.get((i, s), 0) ^ 1
            for (i, j) in C:
                for (r, s) in A:
                    if j == r:
                        prod[(i, s)] = prod.get((i, s), 0) ^ 1
            # the result is symmetric with zero diagonal, so the entries with
            # i < j already name the basis elements: reading both (i,j) and
            # (j,i) would cancel them
            v = 0
            for (i, j), c in prod.items():
                if c and i < j:
                    v ^= 1 << idx[(i, j)]
            tab[k][l] = v
    return LieF2(tab, "o'_I(%d)" % n)


def centre_dim(L):
    """the centre, by linear algebra: the kernel of x |--> (ad_x e_j)_j"""
    rows = []
    for i in range(L.n):
        r = 0
        for j in range(L.n):
            r |= L.tab[i][j] << (j * L.n)
        rows.append(r)
    return L.n - len(rref([r for r in rows if r]))


def sumalg(L1, L2):
    n1, n2 = L1.n, L2.n
    tab = [[0] * (n1 + n2) for _ in range(n1 + n2)]
    for i in range(n1):
        for j in range(n1):
            tab[i][j] = L1.tab[i][j]
    for i in range(n2):
        for j in range(n2):
            tab[n1 + i][n1 + j] = L2.tab[i][j] << n1
    return LieF2(tab, '%s + %s' % (L1.name, L2.name))


# ------------------------------------------------- the divided-power model

def kapB_model(m):
    r"""O(2m; N_s) with the bracket (74):  [f,g] = sum x_i y_i (d_{x_i}f d_{y_i}g
    + d_{y_i}f d_{x_i}g), in the coordinates x_i = 1 + p_i, y_i = 1 + q_i.
    Since p_i^2 = 0, we have x_i^2 = 1, so monomials multiply by symmetric
    difference of their supports; a monomial is a subset of {x_1..x_m,y_1..y_m}
    encoded by a 2m-bit integer, bit i = x_{i+1}, bit m+i = y_{i+1}."""
    N = 1 << (2 * m)
    tab = [[0] * N for _ in range(N)]
    for f in range(N):
        for g in range(N):
            # the exclusive-or of the exponent vectors is the product of the
            # monomials (x_i^2 = 1), and each summand is one basis element
            w = 0
            for i in range(m):
                bx, by = 1 << i, 1 << (m + i)
                if (f & bx) and (g & by):
                    w ^= 1 << (((f ^ bx) ^ (g ^ by)) ^ bx ^ by)
                if (f & by) and (g & bx):
                    w ^= 1 << (((f ^ by) ^ (g ^ bx)) ^ bx ^ by)
            tab[f][g] = w
    return LieF2(tab, 'O(%d; N_s) with (74)' % (2 * m))


# ------------------------------------------------------------- 2-closure

class Closure(LieF2):
    r"""the 2-closure g + V^* of a J-system algebra, eqs. (80)--(83):
    [alpha, beta] = 0, [alpha, e_u] = alpha(u) e_u, e_u^{[2]} = B_u in V^*."""

    def __init__(self, J, dimV, name):
        n, k = J.n, dimV
        tab = [[0] * (n + k) for _ in range(n + k)]
        for i in range(n):
            for j in range(n):
                tab[i][j] = J.tab[i][j]
        for t in range(k):                       # alpha_t = the t-th coordinate
            for i, u in enumerate(J.gamma):
                if (u >> t) & 1:
                    tab[n + t][i] = tab[i][n + t] = 1 << i
        LieF2.__init__(self, tab, name)
        self.J, self.dimV, self.nJ = J, k, n

    def sq(self, x):
        """the 2-structure: (sum a_u e_u + alpha)^[2] over F_2"""
        out = 0
        for i in range(self.nJ):
            if (x >> i) & 1:
                u = self.J.gamma[i]
                for t in range(self.dimV):       # B_u = sum_t B(u, eps_t) alpha_t
                    if self.J.B(u, 1 << t):
                        out ^= 1 << (self.nJ + t)
        out ^= (x >> self.nJ) << self.nJ         # alpha^[2] = alpha
        # the cross terms: (x+y)^[2] = x^[2] + y^[2] + [x,y]
        bits = [1 << i for i in range(self.n) if (x >> i) & 1]
        for a in range(len(bits)):
            for b in range(a + 1, len(bits)):
                out ^= self.mul(bits[a], bits[b])
        return out


# --------------------------------------------------------- superalgebras

class SuperF2:
    """a Lie superalgebra over F_2: an ambient LieF2 with a Z/2-grading given
    by the bit mask `odd`, and a squaring on the odd part."""

    def __init__(self, L, odd, sq, name):
        self.L, self.odd, self.sqfun, self.name = L, odd, sq, name
        self.n = L.n
        self.ev = ((1 << L.n) - 1) ^ odd

    def sdim(self):
        return (bitcount(self.ev), bitcount(self.odd))

    def check(self):
        r"""the grading, and the squaring axiom [x^2, y] = [x, [x, y]].

        The axiom needs to be verified on the basis only: with
        x = sum_{i in S} e_i and x^2 := sum_{i in S} e_i^{[2]} + sum_{i<k in S}
        [e_i, e_k], which is how the squaring is defined, the Jacobi identity
        gives [x^2, y] = [x, [x, y]] for every x as soon as it holds for the
        basis vectors, because the cross terms of both sides are
        [[e_i, e_k], y] = [e_i, [e_k, y]] + [e_k, [e_i, y]]."""
        L = self.L
        for i in range(self.n):
            for j in range(self.n):
                pi, pj = (self.odd >> i) & 1, (self.odd >> j) & 1
                v = L.tab[i][j]
                for k in range(self.n):
                    if (v >> k) & 1 and ((self.odd >> k) & 1) != (pi ^ pj):
                        return False, 'grading fails at (%d,%d,%d)' % (i, j, k)
        for i in range(self.n):
            if not (self.odd >> i) & 1:
                continue
            s = self.sqfun(1 << i)
            if s & self.odd:
                return False, 'x^2 not even'
            for j in range(self.n):
                if L.mul(s, 1 << j) != L.mul(1 << i, L.mul(1 << i, 1 << j)):
                    return False, '[x^2,y] != [x,[x,y]]'
        return True, 'ok'

    def super_ideal(self, gens):
        """the ideal generated by `gens`: closed under ad and under the squaring
        of its odd elements.  Squaring is quadratic with polarization the
        bracket, so closing on a basis of the odd part suffices."""
        L = self.L
        rows = rref([g for g in gens if g])
        changed = True
        while changed:
            changed = False
            rows = L.ideal(rows)
            for y in self._odd_part(rows):        # a basis of I n g_odd
                s = self.sqfun(y)
                if s and not in_span(s, rows):
                    rows = rref(rows + [s])
                    changed = True
        return rows

    def _odd_part(self, rows):
        """a basis of the intersection of Span(rows) with the odd subspace"""
        piv, out = {}, []
        for r in rows:
            v = r
            while True:
                e = v & self.ev
                if not e:
                    break
                top = e.bit_length() - 1
                if top in piv:
                    v ^= piv[top]
                else:
                    piv[top] = v
                    v = 0
                    break
            if v:
                out.append(v)
        return out

    def is_simple(self, bound=16, sample=3000):
        """no non-zero proper ideal.  An ideal need not be graded a priori, but
        it is a subspace, so running over all the non-zero elements settles it;
        beyond dimension `bound` we only sample, and say so."""
        if self.n <= bound:
            for u in range(1, 1 << self.n):
                if len(self.super_ideal([u])) != self.n:
                    return 'NO'
            return 'yes'
        import random
        rnd = random.Random(20260816)
        for _ in range(sample):
            u = rnd.randrange(1, 1 << self.n)
            if len(self.super_ideal([u])) != self.n:
                return 'NO'
        return 'on %d random elements' % sample


def superization(cl, parity, name):
    """the superization of a 2-closure: parity(u) for e_u, 0 for V^*"""
    odd = 0
    for i, u in enumerate(cl.J.gamma):
        if parity(u):
            odd |= 1 << i
    return SuperF2(cl, odd, cl.sq, name)


# ------------------------------------------------------------ sandwiches

def fast_sandwiches(cl):
    r"""the exhaustive sandwich scan of a 2-closure, done in O(n) per element.

    The 2-closure is restricted and (checked here) centreless, so
    (ad_x)^2 = ad_{x^{[2]}} vanishes iff x^{[2]} = 0; and x |--> x^{[2]} is
    quadratic with polarization the bracket, so a Gray-code walk over the 2^n
    subsets of the basis updates it with one exclusive-or per basis element.
    The survivors are then tested for (san2)."""
    n = cl.n
    centre = [u for u in range(1, 1 << n)
              if all(cl.mul(u, 1 << j) == 0 for j in range(n))]
    if centre:
        return None                                   # the shortcut is invalid
    sqb = [cl.sq(1 << i) for i in range(n)]
    cols = [[cl.tab[i][b] for b in range(n)] for i in range(n)]
    out, x, v = [], 0, 0
    for i in range(1, 1 << n):
        b = (i & -i).bit_length() - 1
        w = 0
        y = x
        while y:
            j = (y & -y).bit_length() - 1
            w ^= cols[j][b]
            y &= y - 1
        v ^= sqb[b] ^ w
        x ^= 1 << b
        if v == 0:
            img = rref(cl.ad(x))
            els = all_elements(img)
            if all(cl.mul(a, c) == 0 for a in els for c in els):
                out.append(x)
    return out


def sandwich_data(L, do_max=True):
    """dim s(L), codim n_L(s), and whether n_L(s) is a maximal subalgebra"""
    S = L.sandwiches()
    if not S:
        return 0, None, None, 0
    s = L.subalgebra_generated(S)
    nor = L.normaliser(s)
    mx = L.is_maximal_subalgebra(nor) if (do_max and len(nor) < L.n) else None
    return len(s), L.n - len(nor), mx, len(S)


# ------------------------------------------------------ isomorphism search

def is_monomial(L):
    """every [e_i, e_j] is 0 or one basis element"""
    return all(L.tab[i][j] == 0 or (L.tab[i][j] & (L.tab[i][j] - 1)) == 0
               for i in range(L.n) for j in range(L.n))


def monomial_iso(A, C):
    r"""an isomorphism A --> C carrying basis elements to basis elements, found
    by backtracking with propagation.  Both algebras must be monomial (all the
    structure constants 0 or 1 in the given bases), which every algebra here is
    --- a J-system algebra, o'_I(n), or a direct sum of these.  Existence of
    such a map is a proof of the isomorphism; its non-existence only says that
    no isomorphism of this restricted shape exists.  Returns the map (a list of
    target indices) or None."""
    if A.n != C.n or not (is_monomial(A) and is_monomial(C)):
        return None
    n = A.n
    bit = {1 << i: i for i in range(n)}

    def rec(phi, used):
        if all(x is not None for x in phi):
            # a full check of the multiplication table
            for i in range(n):
                for j in range(n):
                    v = A.tab[i][j]
                    w = 0 if v == 0 else 1 << phi[bit[v]]
                    if C.tab[phi[i]][phi[j]] != w:
                        return None
            return list(phi)
        i = min(k for k in range(n) if phi[k] is None)
        for y in range(n):
            if y in used:
                continue
            ph, us = list(phi), set(used)
            ph[i], ok = y, True
            us.add(y)
            queue = [i]
            while queue and ok:                   # propagate
                a = queue.pop()
                for b in range(n):
                    if ph[b] is None:
                        continue
                    v = A.tab[a][b]
                    w = C.tab[ph[a]][ph[b]]
                    if v == 0:
                        if w:
                            ok = False
                            break
                        continue
                    if w == 0 or (w & (w - 1)):
                        ok = False
                        break
                    k, l = bit[v], bit[w]
                    if ph[k] is None:
                        if l in us:
                            ok = False
                            break
                        ph[k] = l
                        us.add(l)
                        queue.append(k)
                    elif ph[k] != l:
                        ok = False
                        break
            if ok:
                r = rec(ph, us)
                if r:
                    return r
        return None

    return rec([None] * n, set())


def invariants(L):
    return (L.n, L.num_2nilps(), L.dim_der(), len(L.derived()))


# ------------------------------------------------------------------ main

def main(mmax=2):
    global _log
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    _log = open(LOG, 'w', buffering=1)

    say('=' * 78)
    say('The Kaplansky algebras of arXiv:1301.2781 = Math. Res. Lett. 22 (2015)')
    say('=' * 78)

    # ---------------------------------------------------------- 1. structure
    say('\n--- 1. the algebras, the Jacobi identity, the dimensions (73)')
    for m in range(1, mmax + 1):
        d = 2 * m
        A = {'2': kap2(m), 'B': kapB(m), '4,0': kap4(m, 0), '4,1': kap4(m, 1)}
        for key in ['2', 'B', '4,0', '4,1']:
            L = A[key]
            msg = L.check_jacobi()[1] if L.n <= 40 else 'not run (dim > 40)'
            say('%-18s dim %3d   Jacobi: %s' % (L.name, L.n, msg))
        # (73)
        for a in (0, 1):
            pred = 2 ** (m - 1) * (2 ** m - (-1) ** a)
            got = A['4,%d' % a].n
            say('   (73) dim Kap_{4,%d}(%d) = %d, predicted %d : %s'
                % (a, d, got, pred, 'OK' if got == pred else 'WRONG'))
        # inclusions
        g2 = set(A['2'].gamma)
        gB = set(A['B'].gamma)
        for a in (0, 1):
            ga = set(A['4,%d' % a].gamma)
            say('   Kap_{4,%d}(%d) < Kap_2(%d) < Kap_{4,B}(%d) : %s'
                % (a, d, d, d, 'OK' if ga <= g2 <= gB else 'WRONG'))
        # (69): Kap_{4,B} = Kap_2 + centre
        zdim = centre_dim(A['B'])
        say('   (69) centre of Kap_{4,B}(%d): dim %d, and Kap_{4,B} = Kap_2 + c : %s'
            % (d, zdim, 'OK' if zdim == 1 and A['B'].n == A['2'].n + 1
               else 'CHECK'))
        # (74): the divided-power model
        M = kapB_model(m)
        okj = M.check_jacobi()[0] if M.n <= 40 else 'not run'
        same = all(M.tab[u][v] == A['B'].tab[A['B'].idx[u]][A['B'].idx[v]]
                   for u in range(1 << d) for v in range(1 << d))
        say('   Prop. 5.1.1: O(%d;N_s) with (74) is Kap_{4,B}(%d) under e_u <-> f_u'
            ' : %s  (Jacobi: %s)' % (d, d, 'OK' if same else 'WRONG', okj))

    # Kap_1 and Kap_3
    say('')
    for n in (4, 5, 6):
        K1 = kap1(n)
        ok, msg = K1.check_jacobi()
        say('%-18s dim %3d   Jacobi: %s' % (K1.name, K1.n, msg))
    for n in (3, 5, 6):
        O = oI(n)
        ok, msg = O.check_jacobi()
        say("%-18s dim %3d   Jacobi: %s   simple: %s"
            % (O.name, O.n, msg, O.is_simple() if O.n <= 16 else '?'))

    # ------------------------------------------------- 2. the small isomorphisms
    say('\n--- 2. the isomorphisms (76) and those of Subsec. 5.2')
    say("(an isomorphism is looked for among the maps carrying basis to basis;")
    say(" finding one is a proof, and every algebra here is monomial)")
    K41_2 = kap4(1, 1)
    say("Kap_{4,1}(2) (dim %d) = o'_I(3) (dim %d): %s"
        % (K41_2.n, oI(3).n, 'YES' if monomial_iso(K41_2, oI(3)) else 'no'))
    say("dim Kap_{4,0}(2) = %d (the degenerate case, Span(x_1y_1))" % kap4(1, 0).n)
    if mmax >= 2:
        K40_4, K41_4 = kap4(2, 0), kap4(2, 1)
        say("Kap_{4,0}(4) (dim %d) = o'_I(3) + o'_I(3) (dim %d): %s"
            % (K40_4.n, 2 * oI(3).n,
               'YES' if monomial_iso(K40_4, sumalg(oI(3), oI(3))) else 'no'))
        say("Kap_{4,1}(4) (dim %d) = o'_I(5) (dim %d): %s"
            % (K41_4.n, oI(5).n, 'YES' if monomial_iso(K41_4, oI(5)) else 'no'))
        K2_4, O6 = kap2(2), oI(6)
        say("Kap_2(4) (dim %d) = Kap_3(6) = o'_I(6) (dim %d): %s"
            % (K2_4.n, O6.n, 'YES' if monomial_iso(K2_4, O6) else 'no'))
        say('invariants (dim, non-zero 2-nilpotents, dim Der, dim derived):')
        say('   Kap_2(4)  %s' % (invariants(K2_4),))
        say("   o'_I(6)   %s" % (invariants(O6),))
        say('   the CSS basis called Kap_2(4) in css-bases/dim15: '
            '(15, 151, 20, 15) --- our independent check')

    # --------------------------------------------- 3. the 2-closures, gradings
    say('\n--- 3. the 2-closures (80)-(83) and the NONLINEAR Z/2-gradings')
    for m in range(1, mmax + 1):
        d = 2 * m
        for key, J in (('2', kap2(m)), ('4,0', kap4(m, 0)), ('4,1', kap4(m, 1))):
            if J.n == 1:
                continue
            cl = Closure(J, d, 'Kap_{%s}(%d) + V^*' % (key, d))
            ok, msg = cl.check_jacobi()
            # restrictedness: [x^[2], y] = [x,[x,y]] for all x
            rest = all(cl.mul(cl.sq(1 << i), 1 << j) ==
                       cl.mul(1 << i, cl.mul(1 << i, 1 << j))
                       for i in range(cl.n) for j in range(cl.n))
            say('%-24s dim %3d  Jacobi: %-4s  restricted on the basis: %s'
                % (cl.name, cl.n, msg, rest))
        # the nonlinear grading of Kap_2(2m)
        J = kap2(m)
        for a in (0, 1):
            Q = Q_arf(m, a)
            D = lambda u, Q=Q: (Q(u) + 1) & 1
            der = all((D(u ^ v) == (D(u) ^ D(v))) for u in J.gamma for v in J.gamma
                      if u != v and J.B(u, v))
            # is D linear in the weights, i.e. is there phi in V^* with phi=D?
            rows, rhs = [], []
            for u in J.gamma:
                rows.append(u)
                rhs.append(D(u))
            lin = solvable_linear(rows, rhs, d)
            say('   Kap_2(%d): D = Q_%d + 1 is a Z/2-grading: %s;'
                ' linear in the weights: %s'
                % (d, a, der, lin))

    # ------------------------------------------------------ 4. superizations
    say('\n--- 4. the superizations, their sdim and simplicity')
    rows = []
    for m in range(1, mmax + 1):
        d = 2 * m
        J2 = kap2(m)
        cl2 = Closure(J2, d, 'Kap_2(%d)+V^*' % d)
        # the linear one, KapLS_2(2m): parity phi_v(u) = B(v,u), v = eps_1
        v = 1
        S = superization(cl2, lambda u, v=v: J2.B(v, u), 'KapLS_2(%d)' % d)
        rows.append((S.name, S, cl2))
        # the two nonlinear ones, KapS_{2,a}(2m): parity Q_a(u) + 1
        for a in (0, 1):
            Q = Q_arf(m, a)
            S = superization(cl2, lambda u, Q=Q: (Q(u) + 1) & 1,
                             'KapS_{2,%d}(%d)' % (a, d))
            rows.append((S.name, S, cl2))
        # KapS_{4,a,eps}(2m), eqs. (89), (90)
        for a in (0, 1):
            J4 = kap4(m, a)
            if J4.n <= 1:
                continue
            cl4 = Closure(J4, d, 'Kap_{4,%d}(%d)+V^*' % (a, d))
            Q = Q_arf(m, a)
            for eps in (0, 1):
                vv = va_eps(m, a, eps)
                if vv is None:
                    continue
                S = superization(cl4, lambda u, vv=vv: J4.B(vv, u),
                                 'KapS_{4,%d,%d}(%d)' % (a, eps, d))
                rows.append((S.name, S, cl4))
    for name, S, cl in rows:
        ok, msg = S.check()
        sd = S.sdim()
        simple = S.is_simple(bound=14)
        say('%-22s sdim (%d|%d)  superalgebra: %-6s simple: %s'
            % (name, sd[0], sd[1], msg, simple))

    # -------------------------------------------------------- 5. sandwiches
    say('\n--- 5. the sandwich data (exhaustive over F_2)')
    say('%-24s %5s %8s %8s %8s %s' % ('L', 'dim', '#sand', 'dim s', 'codim n', 'n max?'))
    todo = []
    for m in range(1, mmax + 1):
        d = 2 * m
        todo.append(kap2(m))
        todo.append(kapB(m))
        for a in (0, 1):
            J = kap4(m, a)
            if J.n > 1:
                todo.append(J)
        todo.append(Closure(kap2(m), d, 'Kap_2(%d)+V^*' % d))
        for a in (0, 1):
            J = kap4(m, a)
            if J.n > 1:
                todo.append(Closure(J, d, 'Kap_{4,%d}(%d)+V^*' % (a, d)))
    for L in todo:
        if L.n > 22:
            say('%-24s %5d   (scan of 2^%d out of reach)' % (L.name, L.n, L.n))
            continue
        if L.n > 17:
            if not isinstance(L, Closure):
                say('%-24s %5d   (scan of 2^%d skipped)' % (L.name, L.n, L.n))
                continue
            S = fast_sandwiches(L)
            if S is None:
                say('%-24s %5d   (has a centre: the fast scan does not apply)'
                    % (L.name, L.n))
                continue
            if not S:
                say('%-24s %5d %8d %8d %8s' % (L.name, L.n, 0, 0, '-'))
                continue
            s = L.subalgebra_generated(S)
            nor = L.normaliser(s)
            mx = L.is_maximal_subalgebra(nor) if len(nor) < L.n else None
            say('%-24s %5d %8d %8d %8s %s'
                % (L.name, L.n, len(S), len(s), L.n - len(nor),
                   '' if mx is None else ('yes' if mx else 'no')))
            continue
        ds, cod, mx, ns = sandwich_data(L)
        say('%-24s %5d %8d %8d %8s %s'
            % (L.name, L.n, ns, ds, '-' if cod is None else cod,
               '' if mx is None else ('yes' if mx else 'no')))

    say('\ndone.')
    _log.close()


def va_eps(m, a, eps):
    """the vectors v_{a,eps} of eq. (89)"""
    if a == 0:
        return 1 if eps == 0 else (1 | (1 << m))
    if eps == 1:
        return 1
    return 2 if m > 1 else None


def solvable_linear(rows, rhs, d):
    """is there phi in V^* (a d-bit vector) with <phi, rows[i]> = rhs[i]?"""
    aug = [(rows[i] << 1) | rhs[i] for i in range(len(rows))]
    piv = {}
    for v in aug:
        w = v
        for p in sorted(piv, reverse=True):
            if (w >> p) & 1:
                w ^= piv[p]
        if w == 1:                      # 0 = 1
            return False
        if w:
            piv[w.bit_length() - 1] = w
    return True


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
