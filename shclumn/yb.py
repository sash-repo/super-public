"""
Exact engine: solutions of the classical Yang-Baxter
equation with values in the distinguished simple stringy Lie superalgebras.

Everything is exact rational arithmetic; the only import is `stringy.py`
(the superstring function calculus).

Contents
--------
* the twelve distinguished stringy Lie superalgebras as bracket oracles on
  monomial keys:
      KL(n)     k^L(1|n),  n = 1..4   (Neveu-Schwarz), theta-coordinates
      KM(n)     k^M(1|n),  n = 1..4   (Ramond),        theta-zeta-coordinates
      Vect(m)   vect^L(1|m), m = 1,2
      SVect(lam) svect^L_lambda(1|2)
      MPer()    m^L(1)      (pericontact, alpha_0 = d tau + q d xi + xi d q)
  plus Witt for control.  Every bracket is checked against the super Jacobi
  identity before use.

* the Schouten bracket [[r,r]] = [r12,r13] + [r12,r23] + [r13,r23] of an even
  element r of g (x) g, in the super convention

      [r12,r13] = sum (-1)^{p_a p_b} [u_a,u_b] (x) v_a (x) v_b
      [r12,r23] = sum              u_a (x) [v_a,u_b] (x) v_b
      [r13,r23] = sum (-1)^{p_a p_b} u_a (x) u_b (x) [v_a,v_b]

  for r = sum_a u_a (x) v_a with p(u_a) = p(v_a) (evenness of r).

* skew-supersymmetry: tau(u (x) v) = (-1)^{p(u)p(v)} v (x) u, r is skew iff
  tau(r) = -r.  For u,v even this is the usual u^v; for u,v odd it is the
  SYMMETRIC tensor u (x) v + v (x) u -- in particular x (x) x for x odd.
"""

from fractions import Fraction as Q
from itertools import combinations, product
import sys

from stringy import mono, add, mul, smul, d_t, d_odd, parity


# ===================================================================== algebras
#
# An algebra object provides
#     keys(...)     an iterable of monomial keys
#     par(key)      the parity of the basis vector
#     brk(f, g)     the bracket of two elements (dicts key -> coeff)
# Elements are dicts {key: Fraction}; a basis vector is {key: 1}.


class Contact:
    """k^L(1|n) in the theta-coordinates: alpha_1 = dt + sum_i theta_i d theta_i.

    Generating functions f = t^a * theta^S, Laurent in t; the bracket is

        {f,g} = (2-E)(f) d_t g - d_t f (2-E)(g)
                - (-1)^{p(f)} sum_i d_{theta_i} f d_{theta_i} g,

    E = sum_i theta_i d_{theta_i}.  K_f <-> f.
    """

    name = "k^L(1|n)"

    def __init__(self, n):
        self.n = n
        self.odd = tuple(range(1, n + 1))
        self.name = "k^L(1|%d)" % n

    def nabla(self, h):
        out = {}
        for (a, S), c in h.items():
            v = (2 - len(S)) * c
            if v:
                out[(a, S)] = v
        return out

    def brk(self, f, g):
        res = add(mul(self.nabla(f), d_t(g)), smul(-1, mul(d_t(f), self.nabla(g))))
        s = {}
        for i in self.odd:
            s = add(s, mul(d_odd(i, f), d_odd(i, g)))
        return add(res, smul(-((-1) ** parity(f)), s))

    def par(self, key):
        return len(key[1]) % 2

    def expo(self, key):
        """the Laurent exponent of the key (used to cut windows)."""
        return key[0]

    def deg(self, key):
        """contact degree: deg t = 2, deg theta = 1, deg K_f = deg f - 2."""
        a, S = key
        return 2 * a + len(S) - 2

    def keys(self, arange, parities=(0, 1)):
        out = []
        for a in arange:
            for r in range(self.n + 1):
                for S in combinations(self.odd, r):
                    if r % 2 in parities:
                        out.append((a, S))
        return out

    def homog_keys(self, d, parities=(0, 1)):
        """basis of the homogeneous component of contact degree d."""
        out = []
        for r in range(self.n + 1):
            if (d + 2 - r) % 2:
                continue
            a = (d + 2 - r) // 2
            for S in combinations(self.odd, r):
                if r % 2 in parities:
                    out.append((a, S))
        return out


class Ramond:
    """k^M(1|n): the Moebius (Ramond) twist, theta_n replaced by sqrt(t) zeta.

    Odd labels: 0 = zeta, 1..n-1 = theta_i.  With
        D = d_t - (zeta / 2t) d_zeta,
        {f,g} = (2-E)(f) Dg - Df (2-E)(g)
                + sigma (-1)^{p(f)} [ (1/t) d_zeta f d_zeta g
                                      + sum_{i>=1} d_{theta_i} f d_{theta_i} g ],
    the constants being fixed by the Jacobi identity (scanned in check_jacobi).
    """

    def __init__(self, n, sigma=1, kz=Q(1), k=Q(1)):
        self.n = n
        self.odd = tuple(range(0, n))          # 0 = zeta, 1..n-1 = theta
        self.thetas = tuple(range(1, n))
        self.sigma, self.kz, self.k = sigma, kz, k
        self.name = "k^M(1|%d)" % n

    def nabla(self, h):
        out = {}
        for (a, S), c in h.items():
            v = (2 - len(S)) * c
            if v:
                out[(a, S)] = v
        return out

    def D(self, f):
        return add(d_t(f), smul(Q(-1, 2), mul(mono(-1, (0,)), d_odd(0, f))))

    def brk(self, f, g):
        res = add(mul(self.nabla(f), self.D(g)), smul(-1, mul(self.D(f), self.nabla(g))))
        s = smul(self.kz, mul(mono(-1), mul(d_odd(0, f), d_odd(0, g))))
        for i in self.thetas:
            s = add(s, smul(self.k, mul(d_odd(i, f), d_odd(i, g))))
        return add(res, smul(self.sigma * ((-1) ** parity(f)), s))

    def par(self, key):
        return len(key[1]) % 2

    def expo(self, key):
        return key[0]

    def deg(self, key):
        """zeta carries weight 0, theta weight 1, t weight 2."""
        a, S = key
        return 2 * a + len(S) - (1 if 0 in S else 0) - 2

    def keys(self, arange, parities=(0, 1)):
        out = []
        for a in arange:
            for r in range(self.n + 1):
                for S in combinations(self.odd, r):
                    if r % 2 in parities:
                        out.append((a, S))
        return out

    def homog_keys(self, d, parities=(0, 1)):
        out = []
        for r in range(self.n + 1):
            for S in combinations(self.odd, r):
                w = len(S) - (1 if 0 in S else 0)
                if (d + 2 - w) % 2:
                    continue
                a = (d + 2 - w) // 2
                if r % 2 in parities:
                    out.append((a, S))
        return out


class Vect:
    """vect^L(1|m): vector fields f d_mu, f Laurent in t, mu in {'t', 1..m}.

    Keys are (mu, a, S) meaning t^a theta^S d_mu.
    """

    def __init__(self, m):
        self.m = m
        self.odd = tuple(range(1, m + 1))
        self.coords = ('t',) + self.odd
        self.name = "vect^L(1|%d)" % m

    def _apply(self, mu, f):
        return d_t(f) if mu == 't' else d_odd(mu, f)

    def par_coord(self, mu):
        return 0 if mu == 't' else 1

    def par(self, key):
        mu, a, S = key
        return (len(S) + self.par_coord(mu)) % 2

    def expo(self, key):
        return key[1]

    def deg(self, key):
        mu, a, S = key
        return 2 * a + len(S) - (2 if mu == 't' else 1)

    def brk(self, X, Y):
        """[f d_mu, g d_nu] = f d_mu(g) d_nu - (-1)^{p(X)p(Y)} g d_nu(f) d_mu."""
        out = {}
        for (mu, a, S), c in X.items():
            f = {(a, S): c}                  # keep the coefficient type (may be symbolic)
            pX = (len(S) + self.par_coord(mu)) % 2
            for (nu, b, T), e in Y.items():
                g = {(b, T): e}
                pY = (len(T) + self.par_coord(nu)) % 2
                h1 = mul(f, self._apply(mu, g))
                for k, v in h1.items():
                    key = (nu,) + k
                    out[key] = out.get(key, 0) + v
                h2 = mul(g, self._apply(nu, f))
                sgn = -((-1) ** (pX * pY))
                for k, v in h2.items():
                    key = (mu,) + k
                    out[key] = out.get(key, 0) + sgn * v
        return {k: v for k, v in out.items() if v}

    def keys(self, arange, parities=(0, 1)):
        out = []
        for mu in self.coords:
            for a in arange:
                for r in range(self.m + 1):
                    for S in combinations(self.odd, r):
                        key = (mu, a, S)
                        if self.par(key) in parities:
                            out.append(key)
        return out

    def homog_keys(self, d, parities=(0, 1)):
        out = []
        for mu in self.coords:
            sh = 2 if mu == 't' else 1
            for r in range(self.m + 1):
                if (d + sh - r) % 2:
                    continue
                a = (d + sh - r) // 2
                for S in combinations(self.odd, r):
                    key = (mu, a, S)
                    if self.par(key) in parities:
                        out.append(key)
        return out


class SVect(Vect):
    """svect^L_lambda(1|2): divergence-free w.r.t. the volume element t^lambda vol.

    div_lambda(f d_t + sum g_i d_{theta_i})
        = d_t f + lambda f / t - sum_i (-1)^{p(g_i)} d_{theta_i} g_i.
    We work with a rational lambda; the elements are the kernel of div_lambda.
    """

    def __init__(self, lam=Q(1, 3)):
        Vect.__init__(self, 2)
        self.lam = lam
        self.name = "svect^L_lambda(1|2)"

    def div(self, X):
        """div(f d_t + sum g_i d_{theta_i}) = d_t f + lambda f / t
                                              + sum_i (-1)^{p(g_i)} d_{theta_i} g_i.

        The sign of the odd summand is (-1)^{p(g_i)}, p(g_i) = len(S) mod 2 --
        it is what makes Ker div_lambda a SUBALGEBRA (checked in `check_div`
        below: with the opposite sign, brackets of divergence-free fields are
        not divergence-free).
        """
        out = {}
        for (mu, a, S), c in X.items():
            f = mono(a, S, c)
            if mu == 't':
                out = add(out, d_t(f), smul(self.lam, mul(mono(-1), f)))
            else:
                out = add(out, smul((-1) ** (len(S) % 2), d_odd(mu, f)))
        return out

    def check_div(self, A=2):
        """0 = Ker div_lambda is closed under the bracket (a window test)."""
        keys = [k for k in self.keys(range(-A, A + 1))]
        ker = _kernel_basis(keys, self.div)
        bad = sum(1 for X in ker for Y in ker if self.div(self.brk(X, Y)))
        return bad, len(ker)

    def homog_basis(self, d, parities=(0, 1)):
        """A basis of the degree-d part of the kernel of div_lambda."""
        keys = Vect.homog_keys(self, d, parities)
        return _kernel_basis(keys, lambda X: self.div(X))


class MPer:
    """m^L(1): the pericontact superstring, alpha_0 = d tau + q d xi + xi d q.

    Coordinates: q even (Laurent), tau and xi odd (labels 1 = tau, 2 = xi).
    Generating functions f(q, tau, xi); the bracket is the pericontact
    (Buttin) bracket

      {f,g} = (2-E)(f) d_tau g + (-1)^{p(f)} d_tau f (2-E)(g)
              - sigma [ d_q f d_xi g + (-1)^{p(f)} d_xi f d_q g ],

    E = q d_q + xi d_xi (the Euler operator of the non-tau coordinates).
    sigma is fixed by the Jacobi identity.  Note M_f has parity p(f) + 1.
    """

    TAU, XI = 1, 2

    def __init__(self, sigma=1):
        self.sigma = sigma
        self.name = "m^L(1)"
        self.odd = (1, 2)

    def nabla(self, h):
        """(2 - E)(h) with E = q d_q + xi d_xi; q is the 't' slot of stringy.py."""
        out = {}
        for (a, S), c in h.items():
            e = a + (1 if self.XI in S else 0)
            v = (2 - e) * c
            if v:
                out[(a, S)] = v
        return out

    def brk(self, f, g):
        pf = parity(f)
        res = add(mul(self.nabla(f), d_odd(self.TAU, g)),
                  smul((-1) ** pf, mul(d_odd(self.TAU, f), self.nabla(g))))
        s = add(mul(d_t(f), d_odd(self.XI, g)),
                smul((-1) ** pf, mul(d_odd(self.XI, f), d_t(g))))
        return add(res, smul(-self.sigma, s))

    def par(self, key):
        """M_f is odd when f is even: the parity of M_f is p(f) + 1."""
        return (len(key[1]) + 1) % 2

    def expo(self, key):
        return key[0]

    def deg(self, key):
        """The Weisfeiler grading of m(1): deg q = deg xi = 1, deg tau = 2, and
        deg M_f = deg f - 2.  (Checked: deg[x,y] = deg x + deg y, no violation;
        the earlier convention 2a + xi - tau - 1 is not a grading at all.)"""
        a, S = key
        return a + 2 * (1 if self.TAU in S else 0) + (1 if self.XI in S else 0) - 2

    def keys(self, arange, parities=(0, 1)):
        out = []
        for a in arange:
            for r in range(3):
                for S in combinations((self.TAU, self.XI), r):
                    if self.par((a, S)) in parities:
                        out.append((a, S))
        return out


class Witt:
    """witt = vect^L(1|0), the control case (a Lie algebra)."""

    name = "witt"

    def brk(self, f, g):
        # f, g are Laurent polynomials standing for f d_t
        return add(mul(f, d_t(g)), smul(-1, mul(d_t(f), g)))

    def par(self, key):
        return 0

    def expo(self, key):
        return key[0]

    def deg(self, key):
        return 2 * key[0] - 2

    def keys(self, arange, parities=(0, 1)):
        return [(a, ()) for a in arange] if 0 in parities else []


# ------------------------------------------------------------------- utilities

def _kernel_basis(keys, op):
    """Basis (as coefficient dicts over `keys`) of the kernel of a linear op."""
    rows = {}
    for j, k in enumerate(keys):
        for t, c in op({k: Q(1)}).items():
            rows.setdefault(t, {})[j] = Q(c)
    mat = [[r.get(j, Q(0)) for j in range(len(keys))] for r in rows.values()]
    return [dict(zip(keys, v)) for v in _nullspace(mat, len(keys))]


def _nullspace(mat, ncols):
    m = [row[:] for row in mat]
    piv, row = {}, 0
    for col in range(ncols):
        p = None
        for i in range(row, len(m)):
            if m[i][col]:
                p = i
                break
        if p is None:
            continue
        m[row], m[p] = m[p], m[row]
        inv = Q(1) / m[row][col]
        m[row] = [x * inv for x in m[row]]
        for i in range(len(m)):
            if i != row and m[i][col]:
                f = m[i][col]
                m[i] = [a - f * b for a, b in zip(m[i], m[row])]
        piv[col] = row
        row += 1
    out = []
    for free in range(ncols):
        if free in piv:
            continue
        v = [Q(0)] * ncols
        v[free] = Q(1)
        for col, r in piv.items():
            v[col] = -m[r][free]
        out.append(v)
    return out


def basis(alg, key):
    return {key: Q(1)}


def check_jacobi(alg, keys, sample=None, seed=0):
    """super Jacobi [f,[g,h]] = [[f,g],h] + (-1)^{p(f)p(g)} [g,[f,h]]; 0 = ok.

    With `sample` set, a random sample of that many triples is tested instead
    of all |keys|^3 of them (the full test is cubic and only feasible for the
    small algebras).
    """
    bad = 0
    if sample is not None:
        import random
        rnd = random.Random(seed)
        triples = [(rnd.choice(keys), rnd.choice(keys), rnd.choice(keys))
                   for _ in range(sample)]
    else:
        triples = product(keys, repeat=3)
    for kf, kg, kh in triples:
        f, g, h = {kf: Q(1)}, {kg: Q(1)}, {kh: Q(1)}
        lhs = alg.brk(f, alg.brk(g, h))
        rhs = add(alg.brk(alg.brk(f, g), h),
                  smul((-1) ** (alg.par(kf) * alg.par(kg)), alg.brk(g, alg.brk(f, h))))
        if add(lhs, smul(-1, rhs)):
            bad += 1
    return bad


def check_skew(alg, keys):
    """[f,g] = -(-1)^{p(f)p(g)} [g,f]."""
    bad = 0
    for kf, kg in product(keys, repeat=2):
        f, g = {kf: Q(1)}, {kg: Q(1)}
        if add(alg.brk(f, g), smul((-1) ** (alg.par(kf) * alg.par(kg)), alg.brk(g, f))):
            bad += 1
    return bad


# ======================================================== the Yang-Baxter side

def tensor_add(out, key, c):
    if c:
        out[key] = out.get(key, 0) + c
        if not out[key]:
            del out[key]


def schouten(r, alg):
    """[[r,r]] = [r12,r13] + [r12,r23] + [r13,r23] in g (x) g (x) g.

    r is a dict {(ku, kv): coeff} with p(ku) = p(kv) (r even).
    """
    out = {}
    for (ka, kb), ca in r.items():
        pa = alg.par(ka)
        for (kc, kd), cb in r.items():
            pb = alg.par(kc)
            sgn = (-1) ** (pa * pb)
            for k1, c1 in alg.brk({ka: Q(1)}, {kc: Q(1)}).items():
                tensor_add(out, (k1, kb, kd), sgn * ca * cb * c1)
            for k2, c2 in alg.brk({kb: Q(1)}, {kc: Q(1)}).items():
                tensor_add(out, (ka, k2, kd), ca * cb * c2)
            for k3, c3 in alg.brk({kb: Q(1)}, {kd: Q(1)}).items():
                tensor_add(out, (ka, kc, k3), sgn * ca * cb * c3)
    return out


def is_skew(r, alg):
    """tau(r) = -r, where tau(u (x) v) = (-1)^{p(u)p(v)} v (x) u."""
    out = dict(r)
    for (ku, kv), c in r.items():
        tensor_add(out, (kv, ku), ((-1) ** (alg.par(ku) * alg.par(kv))) * c)
    return not out


def is_even(r, alg):
    return all(alg.par(ku) == alg.par(kv) for (ku, kv) in r)


def wedge(x, y, alg):
    """the skew-supersymmetric x ^ y = x (x) y - (-1)^{p(x)p(y)} y (x) x."""
    out = {}
    for ku, cu in x.items():
        for kv, cv in y.items():
            tensor_add(out, (ku, kv), cu * cv)
            tensor_add(out, (kv, ku), -((-1) ** (alg.par(ku) * alg.par(kv))) * cu * cv)
    return out


def square(x, alg):
    """x (x) x for x odd -- skew-supersymmetric, of 'rank 1'."""
    out = {}
    for ku, cu in x.items():
        for kv, cv in x.items():
            tensor_add(out, (ku, kv), cu * cv)
    return out


def cybe_ok(r, alg):
    return not schouten(r, alg)
