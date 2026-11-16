"""periprim.py -- primitive forms over a periplectic supermanifold, in parallel
with the symplectic case.

Coordinates: x_1..x_n even, xi_1..xi_n odd; the odd symplectic (periplectic) form is

    om = sum_i dx_i d xi_i,      p(dy) = p(y)+1,

so that  Xh_i := dx_i  is ODD and  Zh_i := d xi_i  is EVEN, and the forms with
constant coefficients are

    Omega = Lambda(Xh_1..Xh_n) (x) C[Zh_1..Zh_n].

Xp := om ^ .            (odd, degree (+1,+1) in the bidegree (deg_Xh, deg_Zh))
Xm := sum_i d/dXh_i d/dZh_i    (odd, degree (-1,-1)) -- convolution with the
                               bivector dual to om.

What is checked here, exactly, over Q:

 1) Xp^2 = 0 and Xm^2 = 0  (there is no sl(2) over an odd form);
 2) z := [Xp,Xm] is CENTRAL and scalar in each bidegree:  z = n + q - p;
    hence z = 0 exactly on the line Xh_1...Xh_n  (bidegree (n,0));
 3) Xp, Xm, z and the degree operator span gl(1|1);
 4) where z != 0 the module is free: P := z^{-1} Xm Xp and Q := z^{-1} Xp Xm are
    complementary idempotents, Omega = Ker Xm (+) Im Xp = Ker Xm (+) Ker Xp;
 5) hence the PERIPLECTIC HODGE-LEPAGE decomposition
        Omega^{(p,q)} = Prim^{(p,q)} (+) om ^ Prim^{(p-1,q-1)},
    Prim := Ker Xm, and dim Prim^{(p,q)} = dim Om^{(p,q)} - dim Om^{(p-1,q-1)};
 6) the filtration I^i = om Omega^{i-2} <= K^i = Ker(om.) <= Omega^i of [L3]
    SPLITS: Omega^i/K^i is isomorphic to Ker Xm in Omega^i, canonically, and
    K^i/I^i = 0 unless i = n, where it is the line Xh_1...Xh_n;
 7) invariance: [L_D, Xp] = 0 for every D in pe(n) -- but [L_D, Xm] = 0 FAILS on
    the odd part of pe(n), so the splitting of 4)-6) is NOT invariant.  That it
    fails for every choice of the lowering operator, and not just for this one,
    is settled in invhom.py;
 8) the same computation for an EVEN form on a (2m|0)- and on a (2m|2k)-manifold
    reproduces sl(2) and the classical Hodge-Lepage -- the control experiment.

Run:  python3 periprim.py
"""
from fractions import Fraction
from itertools import combinations
import sys

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
try:
    from exactlin import rank as _rank
except Exception:                                        # pragma: no cover
    _rank = None

LOG = open(os.devnull, 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    LOG.write(s + '\n')
    LOG.flush()


CHECKS = [0, 0]


def check(name, cond):
    CHECKS[0] += 1
    if cond:
        CHECKS[1] += 1
        say('  ok   %s' % name)
    else:
        say('  FAIL %s' % name)


# ----------------------------------------------------------------- monomials
# a monomial is (I, a):  I a sorted tuple of Xh-indices, a a tuple of Zh-degrees
# a form is a dict monomial -> Fraction

def bideg(m):
    I, a = m
    return len(I), sum(a)


def add(d, m, c):
    if c:
        d[m] = d.get(m, 0) + c
        if d[m] == 0:
            del d[m]


def basis(n, p, q):
    """the monomials of bidegree (p,q)"""
    out = []
    for I in combinations(range(n), p):
        for a in compositions(n, q):
            out.append((I, a))
    return out


def compositions(n, q):
    if n == 1:
        yield (q,)
        return
    for k in range(q + 1):
        for rest in compositions(n - 1, q - k):
            yield (k,) + rest


def sign_insert(I, i):
    """sign of Xh_i * Xh_I,  0 if i in I"""
    if i in I:
        return 0
    return (-1) ** sum(1 for j in I if j < i)


# ------------------------------------------------------------------ operators

def Xp(n, f):
    """multiplication by om = sum_i Xh_i Zh_i"""
    out = {}
    for (I, a), c in f.items():
        for i in range(n):
            s = sign_insert(I, i)
            if s:
                J = tuple(sorted(I + (i,)))
                b = list(a)
                b[i] += 1
                add(out, (J, tuple(b)), s * c)
    return out


def Xm(n, f):
    """sum_i d/dXh_i d/dZh_i, the convolution with the bivector dual to om"""
    out = {}
    for (I, a), c in f.items():
        for i in range(n):
            if i not in I or a[i] == 0:
                continue
            s = (-1) ** sum(1 for j in I if j < i)
            J = tuple(j for j in I if j != i)
            b = list(a)
            b[i] -= 1
            add(out, (J, tuple(b)), s * a[i] * c)
    return out


def comb(f, g, cf=1, cg=1):
    out = dict((m, cf * c) for m, c in f.items())
    for m, c in g.items():
        add(out, m, cg * c)
    return out


def anti(n, A, B, f):
    return comb(A(n, B(n, f)), B(n, A(n, f)))


def mono(m):
    return {m: Fraction(1)}


# ------------------------------------------------------ 1)-3) the relations

def relations(nmax=4, qmax=4):
    say('\n1)--3)  the relations of gl(1|1), and z = n + q - p')
    for n in range(1, nmax + 1):
        ok2p = ok2m = okz = True
        for p in range(n + 1):
            for q in range(qmax + 1):
                for m in basis(n, p, q):
                    f = mono(m)
                    if Xp(n, Xp(n, f)):
                        ok2p = False
                    if Xm(n, Xm(n, f)):
                        ok2m = False
                    z = anti(n, Xp, Xm, f)
                    want = {m: Fraction(n + q - p)} if n + q - p else {}
                    if z != want:
                        okz = False
        check('n=%d: Xp^2 = 0' % n, ok2p)
        check('n=%d: Xm^2 = 0' % n, ok2m)
        check('n=%d: [Xp,Xm] = n + q - p' % n, okz)
    say('    so z = 0 exactly in the bidegree (n,0): the line Xh_1...Xh_n.')


# ------------------------------------------------- 4)-5) the decomposition

def matrix(n, op, src, dst):
    """the matrix of op: span(src) -> span(dst), columns indexed by src"""
    idx = dict((m, k) for k, m in enumerate(dst))
    M = [[Fraction(0)] * len(src) for _ in dst]
    for j, m in enumerate(src):
        for mm, c in op(n, mono(m)).items():
            M[idx[mm]][j] = c
    return M


def rank(M):
    if not M or not M[0]:
        return 0
    if _rank is not None:
        return _rank(M)
    M = [row[:] for row in M]
    r = 0
    rows, cols = len(M), len(M[0])
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i][c]), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [v / pv for v in M[r]]
        for i in range(rows):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        r += 1
    return r


def hodge(nmax=4, qmax=5):
    say('\n4)--5)  the periplectic Hodge--Lepage decomposition')
    for n in range(1, nmax + 1):
        okdim = okfree = True
        rows = []
        for p in range(n + 1):
            for q in range(qmax + 1):
                B = basis(n, p, q)
                Bd = basis(n, p - 1, q - 1) if p >= 1 and q >= 1 else []
                Bu = basis(n, p + 1, q + 1) if p < n else []
                rp = rank(matrix(n, Xp, B, Bu)) if Bu else 0
                rm = rank(matrix(n, Xm, B, Bd)) if Bd else 0
                kerm = len(B) - rm
                # 5) dim Om^(p,q) = pi^(p,q) + pi^(p-1,q-1): the alternating sum
                alt, j = 0, 0
                while p - j >= 0 and q - j >= 0:
                    alt += (-1) ** j * len(basis(n, p - j, q - j))
                    j += 1
                if kerm != alt:
                    okdim = False
                # 4) rank Xp + rank Xm = dim  <=>  the module is free
                if n + q - p != 0:
                    if rp + rm != len(B):
                        okfree = False
                else:
                    if rp or rm:
                        okfree = False
                rows.append((p, q, len(B), kerm))
        check('n=%d: dim Om^(p,q) = dim Prim^(p,q) + dim Prim^(p-1,q-1)' % n, okdim)
        check('n=%d: rank Xp + rank Xm = dim Om^(p,q) off (n,0), and 0 on it' % n, okfree)
        if n == 3:
            say('    n=3, (p,q; dim Om^(p,q), dim Prim^(p,q)):')
            say('    ' + '  '.join('(%d,%d;%d,%d)' % r for r in rows if r[1] <= 3))


# ------------------------------------------- 6) the filtration of [L3] splits

def filtration(nmax=4, imax=6):
    say('\n6)  the filtration I^i <= K^i <= Omega^i of [L3] splits by Xm')
    for n in range(1, nmax + 1):
        oksplit = okline = True
        data = []
        for i in range(imax + 1):
            B = [m for p in range(min(i, n) + 1) if i - p >= 0
                 for m in basis(n, p, i - p)]
            Bu = [m for p in range(min(i + 2, n) + 1) if i + 2 - p >= 0
                  for m in basis(n, p, i + 2 - p)]
            Bd = [m for p in range(min(i - 2, n) + 1) if i - 2 - p >= 0
                  for m in basis(n, p, i - 2 - p)] if i >= 2 else []
            dimOm = len(B)
            rp = rank(matrix(n, Xp, B, Bu)) if B and Bu else 0
            dimI = rank(matrix(n, Xp, Bd, B)) if Bd else 0
            dimK = dimOm - rp
            dimP = dimOm - dimK                      # = rank Xp = dim Om/K
            dimPrim = dimOm - (rank(matrix(n, Xm, B, [m for p in range(min(i - 2, n) + 1)
                                                      if i - 2 - p >= 0
                                                      for m in basis(n, p, i - 2 - p)]))
                               if i >= 2 else 0)
            # Om^i/K^i and Ker Xm in Om^i have the same dimension: the splitting
            if dimP != dimPrim - (1 if i == n else 0):
                oksplit = False
            if (dimK - dimI) != (1 if i == n else 0):
                okline = False
            data.append((i, dimOm, dimI, dimK, dimP, dimPrim))
        check('n=%d: dim Om^i/K^i = dim (Ker Xm)^i, but for the line at i=n' % n, oksplit)
        check('n=%d: dim K^i/I^i = 1 for i=n, 0 otherwise' % n, okline)
        if n == 3:
            say('    n=3  i: dim Om^i, I^i, K^i, P^i=Om/K, Prim^i=Ker Xm')
            for r in data:
                say('      %d: %4d %4d %4d %4d %4d' % r)


# --------------------------------------------------- 7) pe(n)-invariance

def pe_basis(n):
    """pe(n) inside gl(n|n) preserving om: a list of (parity, action on the
    generators), the action given as a dict  y -> list of (coeff, z)."""
    out = []
    # even part gl(n):  x_i -> x_j (i.e. E_ij), xi -> -xi^t
    for i in range(n):
        for j in range(n):
            act = {('x', i): [(Fraction(1), ('x', j))],
                   ('xi', j): [(Fraction(-1), ('xi', i))]}
            out.append((0, act))
    # odd, symmetric B: xi_i -> x_j  and xi_j -> x_i
    for i in range(n):
        for j in range(i, n):
            act = {('xi', i): [(Fraction(1), ('x', j))]}
            if j != i:
                act.setdefault(('xi', j), []).append((Fraction(1), ('x', i)))
            out.append((1, act))
    # odd, antisymmetric C: x_i -> xi_j, x_j -> -xi_i
    for i in range(n):
        for j in range(i + 1, n):
            act = {('x', i): [(Fraction(1), ('xi', j))],
                   ('x', j): [(Fraction(-1), ('xi', i))]}
            out.append((1, act))
    return out


def gen_index(n, y):
    """the position of a generator in the ordered list Xh_0..Xh_{n-1},Zh_0.."""
    kind, i = y
    return ('X', i) if kind == 'x' else ('Z', i)


def act_gen(n, act, m, par=1):
    """the derivation of Omega induced by a linear vector field, applied to one
    monomial; the derivation sends  hat y -> hat (D y)  and is a superderivation
    of parity p(D)."""
    I, a = m
    out = {}
    # the Xh-part, left to right, then the Zh-part
    def word(m):
        I, a = m
        w = [('X', i) for i in I]
        for i in range(len(a)):
            w += [('Z', i)] * a[i]
        return w

    w = word(m)
    for pos, g in enumerate(w):
        y = ('x', g[1]) if g[0] == 'X' else ('xi', g[1])
        img = act.get(y)
        if not img:
            continue
        # parity of the letters to the left of pos: Xh odd, Zh even
        left = sum(1 for gg in w[:pos] if gg[0] == 'X') % 2
        for c, z in img:
            gz = ('X', z[1]) if z[0] == 'x' else ('Z', z[1])
            nw = w[:pos] + [gz] + w[pos + 1:]
            s, mm = normalize(n, nw)
            if s:
                add(out, mm, c * s * ((-1) ** (par * left)) * Fraction(1))
    return out


def normalize(n, w):
    """sort a word of letters ('X',i) odd and ('Z',i) even into a monomial"""
    xs = [g[1] for g in w if g[0] == 'X']
    zs = [g[1] for g in w if g[0] == 'Z']
    # moving every Z past the X's on its left costs nothing (Z is even)
    # sort the X's, counting transpositions
    s = 1
    xs = xs[:]
    for i in range(len(xs)):
        for j in range(len(xs) - 1):
            if xs[j] > xs[j + 1]:
                xs[j], xs[j + 1] = xs[j + 1], xs[j]
                s = -s
    if len(set(xs)) != len(xs):
        return 0, None
    a = [0] * n
    for z in zs:
        a[z] += 1
    return s, (tuple(xs), tuple(a))


def invariance(nmax=3, pqmax=3):
    say('\n7)  pe(n)-invariance: Xp yes, Xm no')
    for n in range(1, nmax + 1):
        okom = okp = okm = True
        for par, act in pe_basis(n):
            # L_D om = 0
            om = Xp(n, mono((tuple(), tuple([0] * n))))
            r = {}
            for m, c in om.items():
                for mm, cc in act_gen(n, act, m, par).items():
                    add(r, mm, c * cc)
            if r:
                okom = False
            for p in range(n + 1):
                for q in range(pqmax + 1):
                    for m in basis(n, p, q):
                        f = mono(m)
                        # [L_D, Xp] with the sign (-1)^{p(D) p(Xp)}
                        a1 = {}
                        for mm, c in Xp(n, f).items():
                            for m3, c3 in act_gen(n, act, mm, par).items():
                                add(a1, m3, c * c3)
                        a2 = {}
                        for mm, c in act_gen(n, act, m, par).items():
                            for m3, c3 in Xp(n, mono(mm)).items():
                                add(a2, m3, c * c3)
                        sgn = (-1) ** (par * 1)
                        if comb(a1, a2, 1, -sgn):
                            okp = False
                        b1 = {}
                        for mm, c in Xm(n, f).items():
                            for m3, c3 in act_gen(n, act, mm, par).items():
                                add(b1, m3, c * c3)
                        b2 = {}
                        for mm, c in act_gen(n, act, m, par).items():
                            for m3, c3 in Xm(n, mono(mm)).items():
                                add(b2, m3, c * c3)
                        if comb(b1, b2, 1, -sgn):
                            okm = False
        check('n=%d: L_D om = 0 for every D in pe(n)' % n, okom)
        check('n=%d: [L_D, Xp] = 0, the multiplication by om IS invariant' % n, okp)
        check('n=%d: [L_D, Xm] != 0: the lowering operator is NOT invariant' % n,
              not okm)


# ------------------------------------------ 8) the control: an even 2-form

def even_control(mmax=3):
    """Control experiment.  On a $2m$-dimensional symplectic manifold the very
    same recipe -- multiply by om, convolve with the dual bivector -- gives
    sl(2): there both hat p_i and hat q_i are ODD, om is EVEN, and
    H = [Xp,Xm] = deg - m is NOT central.  Over an odd om, z is."""
    say('\n8)  control: over an EVEN form the same recipe gives sl(2)')

    def gmul(I, J):
        """the product of two ordered Grassmann monomials, with its sign"""
        if set(I) & set(J):
            return 0, None
        s, K = 1, list(I) + list(J)
        for a in range(len(K)):
            for b in range(len(K) - 1):
                if K[b] > K[b + 1]:
                    K[b], K[b + 1] = K[b + 1], K[b]
                    s = -s
        return s, tuple(K)

    def dl(a, I):
        """the left derivative d/d(letter a)"""
        if a not in I:
            return 0, None
        return (-1) ** sum(1 for j in I if j < a), tuple(j for j in I if j != a)

    for m in range(1, mmax + 1):
        N = 2 * m

        def XP(I):
            out = {}
            for i in range(m):
                s, K = gmul((2 * i, 2 * i + 1), I)
                if s:
                    add(out, K, s)
            return out

        def XM(I):
            out = {}
            for i in range(m):
                s1, J = dl(2 * i + 1, I)
                if not s1:
                    continue
                s2, K = dl(2 * i, J)
                if s2:
                    add(out, K, s1 * s2)
            return out

        okH = okad = True
        for d in range(N + 1):
            for I in combinations(range(N), d):
                h = {}
                for J, c in XP(I).items():
                    for K, cc in XM(J).items():
                        add(h, K, c * cc)
                for J, c in XM(I).items():
                    for K, cc in XP(J).items():
                        add(h, K, -c * cc)
                want = {I: d - m} if d != m else {}
                if h != want:
                    okH = False
                # [H,Xp] = 2Xp : H acts on Om^{d+2} by d+2-m and on Om^d by d-m
                for K, c in XP(I).items():
                    if len(K) - m - (d - m) != 2:
                        okad = False
        check('m=%d: [Xp,Xm] = deg - m on the 2m-dimensional symplectic manifold' % m, okH)
        check('m=%d: [H,Xp] = 2Xp, so the three span sl(2)' % m, okad)
    say('    over an even form H = [Xp,Xm] is NOT central: [H,Xp] = 2Xp,')
    say('    over an odd form z = [Xp,Xm] IS central -- that is the whole difference,')
    say('    and it turns the infinite sl(2)-string into a two-term one.')


if __name__ == '__main__':
    say('periprim.py -- primitive forms over a periplectic supermanifold')
    say('backend: %s' % ('exactlin' if _rank else 'pure python'))
    relations()
    hodge()
    filtration()
    invariance()
    even_control()
    say('\n%d/%d checks passed' % (CHECKS[1], CHECKS[0]))
    say('the splitting of 4)--6) is real as linear algebra and is not invariant;')
    say('see invhom.py for Hom_{pe(n)}(Om^i, Om^{i-2}) = 0.')
