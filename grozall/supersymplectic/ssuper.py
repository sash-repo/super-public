"""Exact engine for the superization of Grozman's problem: bilinear invariant
differential operators on a SYMPLECTIC SUPERMANIFOLD.

Setting.  Coordinates z_0..z_{2m-1} even (x_1..x_m, y_1..y_m) and z_{2m}..z_{2m+n-1}
odd (xi_1..xi_n).  The even symplectic form

        omega = sum_i dx_i ^ dy_i + (1/2) sum_a dxi_a dxi_a

is even and super-antisymmetric: omega_{ij} = -(-1)^{p(i)p(j)} omega_{ji}, so it is
antisymmetric on the even block and SYMMETRIC on the odd one.  Its automorphism
algebra is osp(n|2m); the Poisson superalgebra is po(2m|n) and the Hamiltonian one
h(2m|n).

Fibers.  A covariant super-symmetric 2-tensor field ("metric") A_{ij},
A_{ij} = (-1)^{p(i)p(j)}A_{ji}, is encoded as the quadratic polynomial

        A~(z,u) = sum_{ij} A_{ij}(z) u^i u^j ,      p(u^i) = p(i),

in auxiliary variables u; supercommutativity of the u's IS the super-symmetry, so
no index bookkeeping and no symmetrisation is ever needed.  The fiber is S^2(W^*),
which for the even form is again the adjoint module of osp(n|2m):
        sdim S^2(W^*) = ( m(2m+1) + n(n-1)/2 | 2mn ) = sdim osp(n|2m).
Setting p(u^i) = p(i)+1 instead ("Lambda mode") encodes super-ANTIsymmetric
2-tensors; omega itself lives there, and that is what calibrates the signs.

Everything is exact rational arithmetic.  Nothing here is assumed: the Poisson
bracket, the fact that a Hamiltonian field is a derivation, the Lie derivative on
tensor fields and the generation of h(2m|n) by its degree <= 3 part are all
verified by the self-tests in test_ssuper.py.
"""

from fractions import Fraction


def popcount(x):
    return bin(x).count("1")


# --------------------------------------------------------------------- algebra


class Super:
    """Supercommutative polynomial algebra over Q: nev even, nod odd generators.

    A monomial is a pair (exponent tuple for the even generators, bitmask of the
    odd ones).  The bitmask is read in ONE global order -- bit a is the a-th odd
    generator -- and every sign in the module comes from that single convention.
    """

    def __init__(self, nev, nod):
        self.nev = nev
        self.nod = nod
        self.ez = (0,) * nev

    # --- constructors
    def zero(self):
        return {}

    def const(self, c):
        c = Fraction(c)
        return {} if c == 0 else {(self.ez, 0): c}

    def gen_ev(self, i):
        e = [0] * self.nev
        e[i] = 1
        return {(tuple(e), 0): Fraction(1)}

    def gen_od(self, a):
        return {(self.ez, 1 << a): Fraction(1)}

    # --- arithmetic
    @staticmethod
    def add(p, q):
        r = dict(p)
        for k, c in q.items():
            v = r.get(k, 0) + c
            if v:
                r[k] = v
            elif k in r:
                del r[k]
        return r

    @staticmethod
    def summ(ps):
        r = {}
        for p in ps:
            for k, c in p.items():
                v = r.get(k, 0) + c
                if v:
                    r[k] = v
                elif k in r:
                    del r[k]
        return r

    @staticmethod
    def scale(p, c):
        c = Fraction(c)
        if c == 0:
            return {}
        return {k: v * c for k, v in p.items()}

    @staticmethod
    def _msign(m1, m2):
        """Sign of moving the odd part m2 past m1 (both sorted ascending)."""
        inv = 0
        b = 0
        mm = m2
        while mm:
            if mm & 1:
                inv += popcount(m1 >> (b + 1))
            mm >>= 1
            b += 1
        return -1 if inv & 1 else 1

    def mul(self, p, q):
        r = {}
        for (e1, m1), c1 in p.items():
            for (e2, m2), c2 in q.items():
                if m1 & m2:
                    continue
                s = self._msign(m1, m2)
                key = (tuple(a + b for a, b in zip(e1, e2)), m1 | m2)
                v = r.get(key, 0) + (c1 * c2 if s > 0 else -c1 * c2)
                if v:
                    r[key] = v
                elif key in r:
                    del r[key]
        return r

    def d_ev(self, p, i):
        r = {}
        for (e, m), c in p.items():
            if e[i] == 0:
                continue
            f = list(e)
            f[i] -= 1
            key = (tuple(f), m)
            v = r.get(key, 0) + c * e[i]
            if v:
                r[key] = v
            elif key in r:
                del r[key]
        return r

    def d_od(self, p, a):
        """LEFT derivative with respect to the a-th odd generator."""
        bit = 1 << a
        r = {}
        for (e, m), c in p.items():
            if not (m & bit):
                continue
            s = -1 if popcount(m & (bit - 1)) & 1 else 1
            key = (e, m ^ bit)
            v = r.get(key, 0) + (c if s > 0 else -c)
            if v:
                r[key] = v
            elif key in r:
                del r[key]
        return r

    # --- parity
    @staticmethod
    def parity_parts(p):
        """Split into (even part, odd part)."""
        ev, od = {}, {}
        for k, c in p.items():
            if popcount(k[1]) & 1:
                od[k] = c
            else:
                ev[k] = c
        return ev, od

    @staticmethod
    def is_homog(p):
        pars = {popcount(k[1]) & 1 for k in p}
        return len(pars) <= 1

    @staticmethod
    def parity(p):
        pars = {popcount(k[1]) & 1 for k in p}
        if not pars:
            return 0
        assert len(pars) == 1, "polynomial is not parity-homogeneous"
        return pars.pop()

    @staticmethod
    def iszero(p):
        return not p


# ------------------------------------------------------------------- the setup


class SSetup:
    """Coordinates z (2m even + n odd) and fiber variables u, in a chosen mode.

    mode = "S": p(u^i) = p(i)      -- super-symmetric 2-tensors, fiber S^2(W^*)
    mode = "L": p(u^i) = p(i)+1    -- super-antisymmetric ones, fiber Lambda^2(W^*)
    """

    def __init__(self, m, n, mode="S", odd_coef=1, ham_sign=1, form="even"):
        self.m, self.n, self.mode, self.form = m, n, mode, form
        if form == "odd":
            # the PERIPLECTIC case: sdim (k|k), k = m = n, odd symplectic form
            # omega = sum_i dq_i dxi_i, aut(omega) = pe(k), the bracket is
            # Buttin's antibracket and the Hamiltonian algebra is le(k).
            assert m == n, "the odd symplectic form needs sdim (k|k)"
            self.k = m
            self.nc = 2 * m
            self.par = [0] * m + [1] * m
        else:
            self.nc = 2 * m + n                  # number of coordinates
            self.par = [0] * (2 * m) + [1] * n   # parity of each coordinate
        sh = 0 if mode == "S" else 1
        self.upar = [(p + sh) % 2 for p in self.par]
        self.odd_coef = odd_coef                 # the coefficient in {f,g}'s odd half
        self.ham_sign = ham_sign

        # variable layout: even = [z even] + [u even];  odd = [z odd] + [u odd]
        self.zev = {}
        self.zod = {}
        nev = 0
        nod = 0
        for i in range(self.nc):
            if self.par[i] == 0:
                self.zev[i] = nev
                nev += 1
            else:
                self.zod[i] = nod
                nod += 1
        self.uev = {}
        self.uod = {}
        for i in range(self.nc):
            if self.upar[i] == 0:
                self.uev[i] = nev
                nev += 1
            else:
                self.uod[i] = nod
                nod += 1
        self.A = Super(nev, nod)

        # omega^{ij}: on the even block the standard symplectic inverse, on the
        # odd one the identity (times odd_coef).
        Om = [[Fraction(0)] * self.nc for _ in range(self.nc)]
        if form == "odd":
            for i in range(self.k):
                Om[i][self.k + i] = Fraction(1)
                Om[self.k + i][i] = Fraction(odd_coef)
        else:
            for i in range(m):
                Om[i][m + i] = Fraction(-1)
                Om[m + i][i] = Fraction(1)
            for a in range(n):
                Om[2 * m + a][2 * m + a] = Fraction(odd_coef)
        self.Om = Om
        self.Omnz = [(i, j, Om[i][j]) for i in range(self.nc)
                     for j in range(self.nc) if Om[i][j]]

    # --- coordinates as polynomials
    def z(self, i):
        return (self.A.gen_ev(self.zev[i]) if self.par[i] == 0
                else self.A.gen_od(self.zod[i]))

    def u(self, i):
        return (self.A.gen_ev(self.uev[i]) if self.upar[i] == 0
                else self.A.gen_od(self.uod[i]))

    def dz(self, p, i):
        return (self.A.d_ev(p, self.zev[i]) if self.par[i] == 0
                else self.A.d_od(p, self.zod[i]))

    def du(self, p, i):
        return (self.A.d_ev(p, self.uev[i]) if self.upar[i] == 0
                else self.A.d_od(p, self.uod[i]))

    def mulu(self, i, p):
        """u^i * p (multiplication on the LEFT)."""
        return self.A.mul(self.u(i), p)

    # --- the Poisson superalgebra po(2m|n)
    def _bracket_homog(self, f, g, pf):
        """{f,g} for f of parity pf."""
        A = self.A
        m, n = self.m, self.n
        if self.form == "odd":
            # Buttin's antibracket; it is ODD, so p({f,g}) = p(f)+p(g)+1.
            k = self.k
            s = Fraction(self.ham_sign) * (-1 if pf else 1)
            terms = []
            for i in range(k):
                terms.append(A.mul(self.dz(f, i), self.dz(g, k + i)))
                terms.append(A.scale(A.mul(self.dz(f, k + i), self.dz(g, i)), s))
            return A.summ(terms)
        terms = []
        for i in range(m):
            terms.append(A.mul(self.dz(f, i), self.dz(g, m + i)))
            terms.append(A.scale(A.mul(self.dz(f, m + i), self.dz(g, i)), -1))
        s = Fraction(self.ham_sign) * (-1 if pf else 1)
        for a in range(n):
            terms.append(A.scale(A.mul(self.dz(f, 2 * m + a),
                                       self.dz(g, 2 * m + a)), s))
        return A.summ(terms)

    def bracket(self, f, g):
        ev, od = self.A.parity_parts(f)
        out = []
        if ev:
            out.append(self._bracket_homog(ev, g, 0))
        if od:
            out.append(self._bracket_homog(od, g, 1))
        return self.A.summ(out)

    def ham(self, f):
        """The components X_f^k = {f, z_k} of the Hamiltonian vector field."""
        return [self.bracket(f, self.z(k)) for k in range(self.nc)]

    def act(self, f, g):
        """X_f applied to g, from the components (equals {f,g} on functions)."""
        X = self.ham(f)
        return self.A.summ([self.A.mul(X[k], self.dz(g, k)) for k in range(self.nc)])

    # --- the Lie derivative on tensor fields, in the u-encoding
    #
    # (L_X A~) = sum_k X^k d_{z^k} A~  +  sum_{i,k} eps(i,k,X) u^i (d_i X^k)(d_{u^k}A~)
    #
    # The sign eps is NOT guessed.  Searching all 256 sign patterns that are
    # Z/2-polynomial in (p(i),p(k),p(X)), test_ssuper.py finds exactly one that
    # both is a representation of h(2m|n) and satisfies L_X(df) = d(X f) on
    # 1-forms -- namely eps = +1.  So the classical formula is the right one
    # verbatim, every sign being carried by the supercommutative arithmetic.
    LIE_SIGN = (0,) * 8

    def set_lie_sign(self, bits):
        self.LIE_SIGN = bits

    def _eps(self, pi, pk, pX):
        c, ei, ek, eX, eik, eiX, ekX, eikX = self.LIE_SIGN
        s = (c + ei * pi + ek * pk + eX * pX + eik * pi * pk + eiX * pi * pX
             + ekX * pk * pX + eikX * pi * pk * pX) % 2
        return -1 if s else 1

    def lie(self, f, T):
        """L_{X_f} T for a tensor field T given in the u-encoding."""
        A = self.A
        X = self.ham(f)
        pX = A.parity(f)
        terms = [A.mul(X[k], self.dz(T, k)) for k in range(self.nc)]
        for i in range(self.nc):
            for k in range(self.nc):
                dXk = self.dz(X[k], i)
                if not dXk:
                    continue
                dT = self.du(T, k)
                if not dT:
                    continue
                t = self.mulu(i, A.mul(dXk, dT))
                e = self._eps(self.par[i], self.par[k], pX)
                terms.append(t if e > 0 else A.scale(t, -1))
        return A.summ(terms)

    # --- 1-forms: the calibration object
    def dform(self, f):
        """df = sum_i u^i (d_i f), a 1-form in the same u-encoding."""
        A = self.A
        return A.summ([A.mul(self.u(i), self.dz(f, i)) for i in range(self.nc)])

    def free_deriv(self, T):
        """U(T) = sum_i u^i (d_{z^i} T): one derivative index left free."""
        A = self.A
        return A.summ([A.mul(self.u(i), self.dz(T, i)) for i in range(self.nc)])

    # --- monomials and random fields
    def monomials(self, deg):
        """All coordinate monomials of exact degree deg."""
        out = []
        self._mon(deg, 0, self.A.const(1), out)
        return out

    def _mon(self, deg, start, acc, out):
        if deg == 0:
            out.append(acc)
            return
        for i in range(start, self.nc):
            p = self.A.mul(acc, self.z(i))
            if not p:
                continue
            self._mon(deg - 1, i if self.par[i] == 0 else i + 1, p, out)

    def fiber_monomials(self):
        """A basis of the fiber: the quadratic monomials in u."""
        out = []
        for i in range(self.nc):
            for j in range(i, self.nc):
                p = self.A.mul(self.u(i), self.u(j))
                if p:
                    out.append(((i, j), p))
        return out

    def sdim_fiber(self):
        ev = od = 0
        for (i, j), p in self.fiber_monomials():
            if (self.upar[i] + self.upar[j]) % 2:
                od += 1
            else:
                ev += 1
        return ev, od

    def random_field(self, deg, rng, spread=4, nterms=None):
        """A random EVEN tensor field: a quadratic-in-u, parity-even polynomial."""
        A = self.A
        mons = [p for d in range(deg + 1) for p in self.monomials(d)]
        out = []
        for (i, j), fib in self.fiber_monomials():
            pf = A.parity(fib)
            pool = [p for p in mons if A.parity(p) == pf]   # keep the field even
            if not pool:
                continue
            if nterms is not None and len(pool) > nterms:
                pool = rng.sample(pool, nterms)
            for p in pool:
                c = rng.randint(-spread, spread)
                if c:
                    out.append(A.scale(A.mul(p, fib), c))
        return A.summ(out)
