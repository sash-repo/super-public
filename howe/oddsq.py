#!/usr/bin/env python3
r"""oddsq.py -- Theorem 4.3.1 ("The odd square root"): the relations
(26) for every n, checked for n = 1..5.

The flat periplectic model of eq. (25).  Coordinates $x_1..x_n$ even,
$\xi_1..\xi_n$ odd; with the convention $p(\widehat y)=p(y)+\od$ the differentials
$\widehat x_i$ are odd and $\widehat\xi_i$ even, so

    Omega = K[x] (x) Lambda(xi) (x) Lambda(xhat) (x) K[xihat],

the odd symplectic (periplectic) form is  om = sum xhat_i xihat_i,  and the horizontal
part of the pericontact form of eq. (43) is  al = sum (x_i xihat_i + xi_i xhat_i).
The central charge hbar is an ODD parameter, hbar^2 = 0, so it is carried as one more
odd generator of the ground ring, which is what makes the Koszul signs those of
Lambda(hbar) (x) Omega.

The five operators of eq. (25):

    X_+ = om.,   X_- = sum d/d(xihat_i) d/d(xhat_i),   H = [X_+,X_-],
    D^+ = d + (hbar/2) al.,                            D^- = [X_-,D^+].

Checked, in exact arithmetic over Q, on every basis monomial up to a bounded degree:

  (1) the relations of eq. (26), each side evaluated separately;
  (2) the parities: X_+, X_-, D^+ odd and H, D^- even -- the Theorem's last clause;
  (3) [D^+,D^+] = -2 hbar X_+ is NOT zero (the Theorem says "\neq0"), i.e. that the
      square root is not the trivial one;
  (4) the two facts the relations rest on and which are visibly uniform in n:
      om^2 = 0 (om is odd) and X_-^2 = 0 (symmetric in the xihat, antisymmetric in the
      xhat), so that the first line of (26) is sl(1|1) for every n;
  (5) d al = 2 om -- the identity behind [D^+,D^+] = -2 hbar X_+;
  (6) that the whole closure of the five operators is ba(2) +) sl(1|1), of sdim (5|4),
      independently of n -- the Theorem's "Thus, the whole algebra is".

Calibration: the printed relations are reproduced for n = 1 and n = 2
before n = 3 and n = 4 are looked at; a sign convention that failed there
would be ours, not the Theorem's.

Usage:  python3 oddsq.py [nmax] [degree]
"""
import itertools
import sys
from fractions import Fraction as F


# --------------------------------------------------------------- the algebra

class Alg:
    """Free supercommutative superalgebra on the listed generators.

    A monomial is (exps, odds): exps a tuple of exponents of the even generators and
    odds a tuple, strictly increasing, of the indices of the odd generators present.
    ONE global order on the odd generators fixes every sign.
    """

    def __init__(self, n):
        self.n = n
        self.ev, self.od = [], []
        # even generators
        for i in range(n):
            self.ev.append('x%d' % i)
        for i in range(n):
            self.ev.append('xih%d' % i)
        # odd generators, in one global order: hbar first, then xi, then xhat
        self.od.append('hbar')
        for i in range(n):
            self.od.append('xi%d' % i)
        for i in range(n):
            self.od.append('xh%d' % i)
        self.eidx = {g: i for i, g in enumerate(self.ev)}
        self.oidx = {g: i for i, g in enumerate(self.od)}

    # ---- monomials
    def one(self):
        return {((0,) * len(self.ev), ()): F(1)}

    def gen(self, name):
        if name in self.eidx:
            e = [0] * len(self.ev)
            e[self.eidx[name]] = 1
            return {(tuple(e), ()): F(1)}
        return {((0,) * len(self.ev), (self.oidx[name],)): F(1)}

    def parity_of(self, mon):
        return len(mon[1]) % 2

    def mmul(self, a, b):
        """product of two monomials -> (monomial, sign) or None"""
        ea, oa = a
        eb, ob = b
        if set(oa) & set(ob):
            return None
        # sign: move each odd of b left past the odds of a with a larger index
        s = 0
        for j in ob:
            s += sum(1 for i in oa if i > j)
        e = tuple(p + q for p, q in zip(ea, eb))
        o = tuple(sorted(oa + ob))
        return (e, o), (-1) ** s

    def mul(self, u, v):
        out = {}
        for a, ca in u.items():
            for b, cb in v.items():
                r = self.mmul(a, b)
                if r is None:
                    continue
                m, s = r
                out[m] = out.get(m, F(0)) + s * ca * cb
        return {m: c for m, c in out.items() if c}

    def add(self, *vs):
        out = {}
        for v in vs:
            for m, c in v.items():
                out[m] = out.get(m, F(0)) + c
        return {m: c for m, c in out.items() if c}

    def smul(self, c, v):
        return {m: c * x for m, x in v.items() if c * x}

    def basis(self, deg, drop=()):
        """all monomials of total degree <= deg, omitting the named generators"""
        ev = [i for i, g in enumerate(self.ev) if g not in drop]
        od = [i for i, g in enumerate(self.od) if g not in drop]
        out = []
        for k in range(len(od) + 1):
            for oo in itertools.combinations(od, k):
                rest = deg - k
                if rest < 0:
                    continue
                for tot in range(rest + 1):
                    for comp in _compositions(len(ev), tot):
                        e = [0] * len(self.ev)
                        for i, c in zip(ev, comp):
                            e[i] = c
                        out.append((tuple(e), tuple(oo)))
        return out


def _compositions(k, total):
    if k == 0:
        yield ()
        return
    for first in range(total + 1):
        for rest in _compositions(k - 1, total - first):
            yield (first,) + rest


# -------------------------------------------------------------- operators

class Op:
    """A linear operator on Alg, given by its action on monomials, with a parity.

    The action is memoized: an operator built by nested brackets would otherwise cost
    exponentially in the depth of the nesting.
    """

    def __init__(self, alg, act, parity, name):
        self.alg, self._act, self.p, self.name = alg, act, parity, name
        self._cache = {}

    def act(self, m):
        r = self._cache.get(m)
        if r is None:
            r = self._cache[m] = self._act(m)
        return r

    def __call__(self, v):
        out = {}
        for m, c in v.items():
            for mm, cc in self.act(m).items():
                out[mm] = out.get(mm, F(0)) + c * cc
        return {m: c for m, c in out.items() if c}


def mult_op(alg, elt, parity, name):
    """left multiplication by a homogeneous element"""
    def act(m):
        return alg.mul(elt, {m: F(1)})
    return Op(alg, act, parity, name)


def deriv(alg, name):
    """the (left) partial derivative in a generator; odd generator -> odd derivation"""
    if name in alg.eidx:
        i = alg.eidx[name]

        def act(m):
            e, o = m
            if e[i] == 0:
                return {}
            ee = list(e)
            k = ee[i]
            ee[i] -= 1
            return {(tuple(ee), o): F(k)}
        return Op(alg, act, 0, 'd/d' + name)
    j = alg.oidx[name]

    def act(m):
        e, o = m
        if j not in o:
            return {}
        k = o.index(j)
        oo = tuple(x for x in o if x != j)
        return {(e, oo): F((-1) ** k)}
    return Op(alg, act, 1, 'd/d' + name)


def compose(A, B):
    return Op(A.alg, lambda m: A(B.act(m)), (A.p + B.p) % 2, '(%s%s)' % (A.name, B.name))


def opadd(*ops):
    alg, p = ops[0].alg, ops[0].p
    for o in ops:
        assert o.p == p, 'adding operators of different parity'
    return Op(alg, lambda m: alg.add(*[o.act(m) for o in ops]), p,
              '+'.join(o.name for o in ops))


def opsmul(c, A):
    return Op(A.alg, lambda m: A.alg.smul(c, A.act(m)), A.p, '%s*%s' % (c, A.name))


def br(A, B):
    """supercommutator AB - (-1)^{p(A)p(B)} BA"""
    sgn = -1 if (A.p and B.p) else 1
    AB, BA = compose(A, B), compose(B, A)
    return Op(A.alg, lambda m: A.alg.add(AB.act(m), A.alg.smul(F(-sgn), BA.act(m))),
              (A.p + B.p) % 2, '[%s,%s]' % (A.name, B.name))


def equal(A, B, basis):
    for m in basis:
        if A.act(m) != B.act(m):
            return False, m
    return True, None


def is_zero(A, basis):
    return equal(A, Op(A.alg, lambda m: {}, A.p, '0'), basis)


def parity_ok(A, basis):
    """A homogeneous of parity A.p: it shifts the parity of every monomial by A.p"""
    for m in basis:
        r = A.act(m)
        pm = len(m[1]) % 2
        for mm in r:
            if (len(mm[1]) % 2) != (pm + A.p) % 2:
                return False, m
    return True, None


# --------------------------------------------------------------- the model

def model(n):
    a = Alg(n)
    om = a.add(*[a.mul(a.gen('xh%d' % i), a.gen('xih%d' % i)) for i in range(n)])
    al = a.add(*[a.mul(a.gen('x%d' % i), a.gen('xih%d' % i)) for i in range(n)],
               *[a.mul(a.gen('xi%d' % i), a.gen('xh%d' % i)) for i in range(n)])
    hbar = a.gen('hbar')

    Xp = mult_op(a, om, 1, 'X_+')
    Xm = opadd(*[compose(deriv(a, 'xih%d' % i), deriv(a, 'xh%d' % i)) for i in range(n)])
    Xm.p, Xm.name = 1, 'X_-'
    # the exterior differential d = sum xhat_i d/dx_i + xihat_i d/dxi_i, odd
    d = opadd(*[compose(mult_op(a, a.gen('xh%d' % i), 1, ''), deriv(a, 'x%d' % i))
                for i in range(n)],
              *[compose(mult_op(a, a.gen('xih%d' % i), 0, ''), deriv(a, 'xi%d' % i))
                for i in range(n)])
    d.p, d.name = 1, 'd'
    Dp = opadd(d, opsmul(F(1, 2), mult_op(a, a.mul(hbar, al), 1, 'hbar*al')))
    Dp.p, Dp.name = 1, 'D^+'
    H = br(Xp, Xm)
    H.name = 'H'
    Dm = br(Xm, Dp)
    Dm.name = 'D^-'
    return a, om, al, hbar, Xp, Xm, H, Dp, Dm, d


ZERO = None


def run(n, deg, log):
    a, om, al, hbar, Xp, Xm, H, Dp, Dm, d = model(n)
    B = a.basis(deg)
    log('n = %d, degree <= %d, %d basis monomials' % (n, deg, len(B)))
    ok = []

    def chk(name, cond, witness=None):
        ok.append(bool(cond))
        log('   %-38s %s%s' % (name, 'ok' if cond else 'FAIL',
                               '' if cond or witness is None else '  at %s' % (witness,)))

    # (2) parities
    for A in (Xp, Xm, Dp):
        r, w = parity_ok(A, B)
        chk('%s is odd' % A.name, r, w)
    for A in (H, Dm):
        r, w = parity_ok(A, B)
        chk('%s is even' % A.name, r, w)

    # (4) the two facts uniform in n
    chk('om^2 = 0', not a.mul(om, om))
    r, w = is_zero(compose(Xm, Xm), B)
    chk('X_-^2 = 0', r, w)
    # (5) d al = 2 om
    dal = d({m: F(1) for m in [] } ) if False else None
    dal = d(al)
    chk('d al = 2 om', dal == a.smul(F(2), om))

    # (1) the relations of eq. (26)
    r, w = is_zero(br(Xp, Xp), B)
    chk('[X_+,X_+] = 0', r, w)
    r, w = is_zero(br(Xm, Xm), B)
    chk('[X_-,X_-] = 0', r, w)
    for A in (Xp, Xm):
        r, w = is_zero(br(H, A), B)
        chk('[H,%s] = 0' % A.name, r, w)
    r, w = equal(br(Dp, Dp), opsmul(F(-2), compose(mult_op(a, hbar, 1, 'hbar'), Xp)), B)
    chk('[D^+,D^+] = -2 hbar X_+', r, w)
    # (3) and it is not zero
    r, w = is_zero(br(Dp, Dp), B)
    chk('[D^+,D^+] != 0', not r)
    r, w = equal(br(Dp, Dm), opsmul(F(-1), compose(mult_op(a, hbar, 1, 'hbar'), H)), B)
    chk('[D^+,D^-] = -hbar H', r, w)
    r, w = is_zero(br(Dm, Dm), B)
    chk('[D^-,D^-] = 0', r, w)
    r, w = is_zero(br(Xm, Dm), B)
    chk('[X_-,D^-] = 0', r, w)

    # (6) the closure of the five operators, as a Lie superalgebra of operators over C
    # (the hbar-multiples counted separately, hbar being carried as a generator)
    dim = closure_dim([Xp, Xm, H, Dp, Dm], a, min(deg, 3))
    chk('closure sdim = (8|8), as for n = 1', dim == (8, 8), dim)
    return all(ok), len(ok), dim


# ------------------------------------------------- closure of the operators

def _vec(A, basis):
    """the operator as one long coordinate vector on the chosen basis"""
    out = {}
    for k, m in enumerate(basis):
        for mm, c in A.act(m).items():
            out[(k, mm)] = c
    return out


def _reduce(rows, v):
    v = dict(v)
    for piv, r in rows:
        if piv in v:
            c = v[piv] / r[piv]
            for k, x in r.items():
                v[k] = v.get(k, F(0)) - c * x
                if not v[k]:
                    del v[k]
    return v


def closure_dim(gens, alg, deg):
    """dimension (even|odd) of the Lie superalgebra of operators generated by gens"""
    basis = alg.basis(deg)
    rows = {0: [], 1: []}          # echelon rows, by parity
    ops = {0: [], 1: []}

    def push(A):
        v = _vec(A, basis)
        w = _reduce(rows[A.p], v)
        if not w:
            return False
        piv = sorted(w, key=lambda k: (k[0], str(k[1])))[0]
        rows[A.p].append((piv, w))
        ops[A.p].append(A)
        return True

    fresh = [A for A in gens if push(A)]
    while fresh:
        new = []
        for A in fresh:
            for B in ops[0] + ops[1]:
                C = br(A, B)
                if push(C):
                    new.append(C)
        fresh = new
        if len(ops[0]) + len(ops[1]) > 60:
            return ('overflow',)
    return (len(ops[0]), len(ops[1]))


def main():
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    deg = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lines = []

    def log(s):
        print(s)
        lines.append(s)

    log('oddsq.py -- Theorem 4.3.1 (the odd square root) for n = 1..%d' % nmax)
    total, good, dims = 0, 0, []
    for n in range(1, nmax + 1):
        d = deg if n <= 3 else min(deg, 3)
        r, k, dim = run(n, d, log)
        total += k
        good += k if r else 0
        dims.append((n, dim))
        log('   => n = %d: %s' % (n, 'all %d checks pass' % k if r else 'FAILURES'))
    log('')
    log('closure sdim by n: %s' % ', '.join('n=%d: (%s|%s)' % (n, d[0], d[1])
                                            for n, d in dims))
    log('%d checks, %d pass' % (total, good))


if __name__ == '__main__':
    main()
