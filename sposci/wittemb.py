#!/usr/bin/env python3
r"""
wittemb.py -- the graded embeddings witt --> k^L(1|n) and witt --> k^M(1|n),
i.e. the text's note on Conjecture (233):

  "for n = 2, perhaps, not any b in C (too many cases), but just one case b = 1?
   why b = 0 is excluded?  why should all c_* vanish?  why e f''?  to be
   completed also for k^M(1|n) with n <= 4"

Method.  Elements of k(1|n) are the K_f, and [K_f, K_g] = K_{{f,g}}, where the
contact bracket {.,.} is eq. (48) of the text, valid verbatim for R^L(n) and
for R^M(n) (in the latter case theta_n = sqrt t . zeta, so a monomial containing
that generator carries a half-integer power of t).  We look for GRADED
embeddings, i.e. ones sending the standard basis L_m = t^{m+1} d_t of witt to
elements of k-degree m; the most general such image is

  G_m = (1/2) t^{m+1} + sum_M gamma_M(m) t^{m+1-deg(M)/2} M,

M running over the even monomials in the odd coordinates, gamma_M an unknown
polynomial in m (we allow degrees up to deg(M)/2 + 2 and let the equations kill
the surplus).  The 1/2 is forced: the t-part of {a t^{m+1}, a t^{n+1}} is
2a^2 (n-m) t^{m+n+1}, which must be a(n-m) t^{m+n+1}.

The conditions {G_m, G_n} = (n-m) G_{m+n} are then collected, for all m, n in a
window, as polynomial equations in the unknowns and printed.
"""

from fractions import Fraction as F
from itertools import combinations


# ------------------------------------------------------------------ polynomials

class Poly(dict):
    """polynomial in the unknowns VARS: {exponent tuple: Fraction}."""
    NV = 0

    @staticmethod
    def zero():
        return Poly()

    @staticmethod
    def const(c):
        c = F(c)
        return Poly() if c == 0 else Poly({(0,) * Poly.NV: c})

    @staticmethod
    def var(i):
        e = [0] * Poly.NV
        e[i] = 1
        return Poly({tuple(e): F(1)})

    def __add__(a, b):
        r = Poly(a)
        for e, c in b.items():
            r[e] = r.get(e, F(0)) + c
            if r[e] == 0:
                del r[e]
        return r

    def __neg__(a):
        return Poly({e: -c for e, c in a.items()})

    def __sub__(a, b):
        return a + (-b)

    def __mul__(a, b):
        if isinstance(b, (int, F)):
            b = Poly.const(b)
        r = Poly()
        for e1, c1 in a.items():
            for e2, c2 in b.items():
                e = tuple(x + y for x, y in zip(e1, e2))
                r[e] = r.get(e, F(0)) + c1 * c2
                if r[e] == 0:
                    del r[e]
        return r
    __rmul__ = __mul__

    def subs(self, vals):
        """vals: dict index -> Fraction; returns Poly in the remaining vars."""
        r = Poly()
        for e, c in self.items():
            cc, ee = c, list(e)
            for i, v in vals.items():
                cc *= F(v) ** e[i]
                ee[i] = 0
            if cc:
                ee = tuple(ee)
                r[ee] = r.get(ee, F(0)) + cc
                if r[ee] == 0:
                    del r[ee]
        return r

    def show(self, names):
        if not self:
            return '0'
        out = []
        for e, c in sorted(self.items()):
            m = '*'.join(('%s^%d' % (names[i], k) if k > 1 else names[i])
                         for i, k in enumerate(e) if k)
            out.append(('%s' % c) if not m else
                       ('%s*%s' % (c, m) if c != 1 else m))
        return ' + '.join(out)


# ------------------------------------------------- superfunctions on the string
#
# A function is {(a, S): Poly}, a in (1/2)Z the power of t, S a sorted tuple of
# odd-generator labels.

def fadd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, Poly.zero()) + v
    return {k: v for k, v in out.items() if v}


def fscal(c, f):
    c = c if isinstance(c, Poly) else Poly.const(c)
    return {k: c * v for k, v in f.items() if c * v}


def sortS(S):
    S, sgn = list(S), 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] > S[j + 1]:
                S[j], S[j + 1] = S[j + 1], S[j]
                sgn = -sgn
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), sgn


def fmul(f, g):
    out = {}
    for (a1, S1), c1 in f.items():
        for (a2, S2), c2 in g.items():
            S, sg = sortS(S1 + S2)
            if not sg:
                continue
            k = (a1 + a2, S)
            out[k] = out.get(k, Poly.zero()) + Poly.const(sg) * c1 * c2
    return {k: v for k, v in out.items() if v}


def d_t(f):
    out = {}
    for (a, S), c in f.items():
        if a != 0:
            out[(a - 1, S)] = out.get((a - 1, S), Poly.zero()) + Poly.const(a) * c
    return {k: v for k, v in out.items() if v}


def d_odd(j, f):
    out = {}
    for (a, S), c in f.items():
        if j in S:
            p = S.index(j)
            T = S[:p] + S[p + 1:]
            out[(a, T)] = out.get((a, T), Poly.zero()) + Poly.const((-1) ** p) * c
    return {k: v for k, v in out.items() if v}


def theta_deg(f):
    """the operator sum theta_i d_theta_i."""
    out = {}
    for (a, S), c in f.items():
        if S:
            out[(a, S)] = Poly.const(len(S)) * c
    return {k: v for k, v in out.items() if v}


def parity(f):
    ps = {len(S) % 2 for (a, S) in f}
    assert len(ps) <= 1, 'inhomogeneous parity'
    return ps.pop() if ps else 0


def kbracket(f, g, odd, pairs):
    """eq. (48): the contact bracket.

    odd  = list of odd generator labels
    pairs= list of ('orth', a) or ('hyp', a, b) describing the quadratic form:
           orthonormal generators are self-paired, hyperbolic ones in pairs."""
    two_f = fadd(fscal(2, f), fscal(-1, theta_deg(f)))
    two_g = fadd(fscal(2, g), fscal(-1, theta_deg(g)))
    out = fadd(fmul(two_f, d_t(g)), fscal(-1, fmul(d_t(f), two_g)))
    last = {}
    for p in pairs:
        if p[0] == 'orth':
            a = p[1]
            last = fadd(last, fmul(d_odd(a, f), d_odd(a, g)))
        else:
            a, b = p[1], p[2]
            last = fadd(last, fmul(d_odd(a, f), d_odd(b, g)),
                        fmul(d_odd(b, f), d_odd(a, g)))
    return fadd(out, fscal(-(-1) ** parity(f), last))


# ------------------------------------------------------------------- the ansatz

class Setting:
    def __init__(self, name, odd, pairs, halfint=(), maxdeg_extra=2):
        self.name, self.odd, self.pairs = name, odd, pairs
        self.halfint = set(halfint)      # generators forcing t^(1/2 + Z)
        self.extra = maxdeg_extra

    def monomials(self):
        """even monomials M in the odd generators, deg >= 2."""
        out = []
        for d in range(2, len(self.odd) + 1, 2):
            for S in combinations(sorted(self.odd), d):
                out.append(S)
        return out

    def tshift(self, S):
        """the power of t accompanying M in degree m: m + 1 - deg/2 (+1/2 if
        the monomial contains an odd number of half-integer generators)."""
        h = len([s for s in S if s in self.halfint])
        return F(-len(S), 2) + (F(1, 2) if h % 2 else 0)

    def setup(self):
        """returns (names, unknown index table)."""
        names, idx = [], {}
        for S in self.monomials():
            dmax = len(S) // 2 + self.extra
            for s in range(dmax + 1):
                idx[(S, s)] = len(names)
                names.append('g[%s]_%d' % (''.join(map(str, S)), s))
        Poly.NV = len(names)
        return names, idx

    def G(self, m, idx):
        """the candidate image of L_m = t^{m+1} d_t."""
        out = {(F(m + 1), ()): Poly.const(F(1, 2))}
        for S in self.monomials():
            dmax = len(S) // 2 + self.extra
            coef = Poly.zero()
            for s in range(dmax + 1):
                coef = coef + Poly.var(idx[(S, s)]) * Poly.const(F(m) ** s)
            if coef:
                a = F(m + 1) + self.tshift(S)
                out[(a, S)] = out.get((a, S), Poly.zero()) + coef
        return {k: v for k, v in out.items() if v}

    def equations(self, idx, window=4):
        eqs = []
        for m in range(-window, window + 1):
            for n in range(-window, window + 1):
                if m >= n:
                    continue
                lhs = kbracket(self.G(m, idx), self.G(n, idx),
                               self.odd, self.pairs)
                res = fadd(lhs, fscal(-(n - m), self.G(m + n, idx)))
                for k, v in res.items():
                    if v:
                        eqs.append((m, n, k, v))
        return eqs


SETTINGS = [
    Setting('k^L(1|1)', ['th'], [('orth', 'th')]),
    Setting('k^L(1|2)', ['x', 'y'], [('hyp', 'x', 'y')]),
    Setting('k^L(1|3)', ['x', 'y', 'th'], [('hyp', 'x', 'y'), ('orth', 'th')]),
    Setting('k^L(1|4)', ['x1', 'y1', 'x2', 'y2'],
            [('hyp', 'x1', 'y1'), ('hyp', 'x2', 'y2')]),
    Setting('k^M(1|1)', ['z'], [('orth', 'z')], halfint=['z']),
    Setting('k^M(1|2)', ['t1', 'z'], [('orth', 't1'), ('orth', 'z')],
            halfint=['z']),
    Setting('k^M(1|3)', ['t1', 't2', 'z'],
            [('orth', 't1'), ('orth', 't2'), ('orth', 'z')], halfint=['z']),
    Setting('k^M(1|4)', ['t1', 't2', 't3', 'z'],
            [('orth', 't1'), ('orth', 't2'), ('orth', 't3'), ('orth', 'z')],
            halfint=['z']),
]


def solve_report(st):
    names, idx = st.setup()
    eqs = st.equations(idx)
    # split into linear and higher-degree parts
    lin, quad = [], []
    for (m, n, k, v) in eqs:
        deg = max(sum(e) for e in v) if v else 0
        (lin if deg <= 1 else quad).append((m, n, k, v))
    # unknowns forced to vanish by the linear equations (each linear equation
    # here turns out to be a multiple of a single unknown or of a difference)
    forced = set()
    for (m, n, k, v) in lin:
        terms = [e for e in v if sum(e) == 1]
        if len(v) == len(terms) == 1:
            forced.add(terms[0].index(1))
    free = [i for i in range(len(names)) if i not in forced]
    print('  %s:  %d unknowns, %d equations (%d linear, %d quadratic)'
          % (st.name, len(names), len(eqs), len(lin), len(quad)))
    print('     forced to 0 by the linear part: %d of %d'
          % (len(forced), len(names)))
    # substitute 0 for the forced ones and see what is left
    vals = {i: 0 for i in forced}
    left = []
    for (m, n, k, v) in eqs:
        w = v.subs(vals)
        if w:
            left.append((m, n, k, w))
    seen = set()
    uniq = []
    for (m, n, k, w) in left:
        s = w.show(names)
        if s not in seen:
            seen.add(s)
            uniq.append((m, n, k, s))
    print('     surviving unknowns: %s'
          % ', '.join(names[i] for i in free) or '(none)')
    print('     remaining conditions on them: %d distinct' % len(uniq))
    for (m, n, k, s) in uniq[:12]:
        print('        (m,n)=(%d,%d) coeff of t^%s %s :  %s = 0'
              % (m, n, k[0], ''.join(k[1]) or '1', s))
    return names, idx, free, uniq


def check_family(st, idx, assign, window=6, label=''):
    """verify that a concrete assignment of the unknowns gives a homomorphism."""
    Poly.NV = Poly.NV
    vals = {i: F(0) for i in range(Poly.NV)}
    vals.update(assign)
    bad = 0
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            if m >= n:
                continue
            lhs = kbracket(st.G(m, idx), st.G(n, idx), st.odd, st.pairs)
            res = fadd(lhs, fscal(-(n - m), st.G(m + n, idx)))
            for k, v in res.items():
                if v.subs(vals):
                    bad += 1
    print('     %s: %d failures on |m|,|n| <= %d' % (label, bad, window))
    return bad


if __name__ == '__main__':
    print('=' * 78)
    print('graded embeddings of witt into the distinguished stringy superalgebras')
    print('=' * 78)
    for st in SETTINGS:
        print()
        names, idx, free, uniq = solve_report(st)
        # the conjecture's own shape, for the k^L cases
        if st.name == 'k^L(1|2)':
            # b f' xi eta = b (m+1) t^m xi eta  ->  g[xy]_0 = b, g[xy]_1 = b
            for b in (F(0), F(1), F(-1), F(3, 7)):
                a = {idx[(('x', 'y'), 0)]: b / 2, idx[(('x', 'y'), 1)]: b / 2}
                check_family(st, idx, a, label='b = %s' % b)
        if st.name == 'k^L(1|4)':
            S = ('x1', 'x2', 'y1', 'y2')
            for e in (F(0), F(1), F(5, 3)):
                a = {idx[(S, 0)]: F(0), idx[(S, 1)]: e / 2, idx[(S, 2)]: e / 2}
                check_family(st, idx, a, window=4, label='e = %s (b=c=0)' % e)


# ------------------------------------------------- stage 2: exact solutions
#
# For the settings whose conditions are all linear the classification of graded
# embeddings is complete (Gaussian elimination); for the others we plug in the
# shape of Conjecture (233) itself and see which coefficients survive.

def linear_kernel(st, idx, names):
    """solve the subsystem of equations that are linear in the unknowns."""
    eqs = st.equations(idx)
    rows = []
    for (m, n, k, v) in eqs:
        if v and max(sum(e) for e in v) <= 1:
            row = [F(0)] * len(names)
            ok = True
            for e, c in v.items():
                if sum(e) == 0:
                    ok = False
                else:
                    row[e.index(1)] = c
            if ok:
                rows.append(row)
    N = len(names)
    piv, r = [], 0
    M = [row[:] for row in rows]
    for c in range(N):
        p = next((rr for rr in range(r, len(M)) if M[rr][c]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        f0 = M[r][c]
        M[r] = [x / f0 for x in M[r]]
        for rr in range(len(M)):
            if rr != r and M[rr][c]:
                g = M[rr][c]
                M[rr] = [a - g * b for a, b in zip(M[rr], M[r])]
        piv.append(c)
        r += 1
    free = [c for c in range(N) if c not in piv]
    basis = []
    for fc in free:
        vec = {fc: F(1)}
        for rr, c in enumerate(piv):
            vec[c] = -M[rr][fc]
        basis.append({k: v for k, v in vec.items() if v})
    return free, basis, len(rows)


def stage2():
    print()
    print('=' * 78)
    print('STAGE 2 -- exact solutions')
    print('=' * 78)
    for st in SETTINGS:
        names, idx = st.setup()
        if not names:
            print()
            print('  %s: no even monomials in the odd coordinates at all, so the'
                  % st.name)
            print('     only graded embedding is f d_t |--> K_{f/2}: the list is'
                  ' complete here.')
            continue
        free, basis, nlin = linear_kernel(st, idx, names)
        eqs = st.equations(idx)
        nquad = sum(1 for (m, n, k, v) in eqs
                    if v and max(sum(e) for e in v) > 1)
        print()
        print('  %s: %d unknowns; linear part (%d equations) has kernel of'
              ' dimension %d' % (st.name, len(names), nlin, len(free)))
        print('     free parameters: %s' % ', '.join(names[i] for i in free))
        if nquad == 0:
            print('     no quadratic conditions ==> the family above is the'
                  ' complete list of graded embeddings')
            for b in basis:
                a = {i: v for i, v in b.items()}
                check_family(st, idx, a, window=6,
                             label='basis vector %s' %
                             ','.join('%s=%s' % (names[i], v)
                                      for i, v in sorted(b.items())))
        else:
            print('     plus %d quadratic conditions (not solved in full here)'
                  % nquad)
    # --- the shape of the Conjecture itself, for n = 3 and n = 4
    print()
    print('  the shape of Conjecture (233) itself:')
    for name, mons in (('k^L(1|3)', {('th', 'x'): 'c1', ('th', 'y'): 'c2',
                                     ('x', 'y'): 'b'}),
                       ('k^L(1|4)', {('x1', 'y1'): 'b1', ('x2', 'y2'): 'b2',
                                     ('x1', 'y2'): 'c12', ('x2', 'y1'): 'c21',
                                     ('x1', 'x2', 'y1', 'y2'): 'e'})):
        st = next(s for s in SETTINGS if s.name == name)
        names, idx = st.setup()
        # gamma_M(m) = coef/2 * (m+1) for the degree-2 monomials,
        # gamma_M(m) = coef/2 * (m+1)m  for the degree-4 one
        for trial in ({'b': F(1)}, {'b': F(3)}, {'c1': F(1)},
                      {'b': F(1), 'c1': F(1)}, {'c1': F(1), 'c2': F(1)},
                      {'b1': F(1), 'b2': F(1)}, {'b1': F(1), 'b2': F(-1)},
                      {'c12': F(1)}, {'c12': F(1), 'c21': F(1)}, {'e': F(1)},
                      {'b1': F(1), 'e': F(2)}):
            if not set(trial) <= set(mons.values()):
                continue
            a = {}
            for S, nm in mons.items():
                if nm not in trial:
                    continue
                v = trial[nm] / 2
                if len(S) == 2:                      # v (m+1)
                    a[idx[(S, 0)]] = a.get(idx[(S, 0)], F(0)) + v
                    a[idx[(S, 1)]] = a.get(idx[(S, 1)], F(0)) + v
                else:                                # v (m+1) m = v(m^2+m)
                    a[idx[(S, 1)]] = a.get(idx[(S, 1)], F(0)) + v
                    a[idx[(S, 2)]] = a.get(idx[(S, 2)], F(0)) + v
            check_family(st, idx, a, window=4,
                         label='%s with %s' %
                         (name, ', '.join('%s=%s' % kv for kv in sorted(trial.items()))))


if __name__ == '__main__':
    stage2()
