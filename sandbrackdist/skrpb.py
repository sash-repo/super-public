#!/usr/bin/env python3
"""
skrpb.py -- exact engine for the Poisson brackets attached to Skryabin's normal
shapes of symplectic forms, and for the p-structures of the corresponding
Hamiltonian Lie algebras.

Everything is built from scratch over F_p in the divided power algebra
O(2k; N) = K[u_1,...,u_{2k}; N], u^{(a)} with 0 <= a_i < p^{N_i}.

Conventions (fixed once and used everywhere below).
  * tilde: ~i = i+k for 1<=i<=k, ~(i+k) = i.  (0-based in the code: i <-> i+k)
  * omega_0 = sum_{i<=k} du_i ^ du_{~i}; its matrix is J, J_{i,~i}=1, J_{~i,i}=-1.
  * A symplectic form of Skryabin's 1st type is
        omega = omega_0 + eps * sum_{i,j<=2k} A_{ij} d(ubar_i) ^ d(ubar_j),
        ubar_i := u_i^{(p^{N_i})},   so d(ubar_i) = xbar_i du_i,
        xbar_i := u_i^{(p^{N_i}-1)},
    with A = antidiag(B, -B^T) and B one of J_k(0), J_{k,r}(lambda), C_k.
    Its matrix is Omega = J + eps*Atilde, Atilde_{ij} = A_{ij} xbar_i xbar_j,
    i.e. in block form Omega = [[0, I+eps S],[-(I+eps S)^T, 0]],
        S_{ij} = B_{ij} xbar_i xbar_{~j}.
  * H_f is defined by i_{H_f} omega = df, and {f,g} := H_f(g).  Then
        {f,g} = sum_{i,j<=k} G_{ij} ( d_{~i} f  d_j g  -  d_j f  d_{~i} g ),
        G := (I + eps S)^{-1} = sum_{m>=0} (-eps)^m S^m   (finite: S nilpotent).
    At eps=0 this is the bracket of Lemma "ssSpanH" of the text.
  * p-structure: H_f^p = H_{f^[p]}; f^[p] is recovered from the components
    c_i = (ad_f)^p(u_i) by d_j(f^[p]) = sum_i c_i Omega_{ij}, then integrated.

No claim below is asserted without being checked here.
"""

import sys, itertools, random
from functools import lru_cache

# ---------------------------------------------------------------- arithmetic

def binom_mod(n, k, p):
    """C(n,k) mod p by Lucas."""
    if k < 0 or k > n:
        return 0
    r = 1
    while n or k:
        ni, ki = n % p, k % p
        if ki > ni:
            return 0
        # small binomial
        num = den = 1
        for t in range(ki):
            num = num * (ni - t) % p
            den = den * (t + 1) % p
        r = r * num % p * pow(den, p - 2, p) % p
        n //= p
        k //= p
    return r


class DP:
    """Divided power algebra O(n; N) over F_p.  Polynomials are dicts
    {exponent tuple: coefficient mod p}."""

    def __init__(self, p, N):
        self.p = p
        self.N = tuple(N)
        self.n = len(N)
        self.top = tuple(p ** Ni for Ni in N)          # exponents are < top[i]

    # --- constructors
    def zero(self):
        return {}

    def mono(self, exps, c=1):
        e = tuple(exps)
        assert len(e) == self.n
        if any(x < 0 or x >= self.top[i] for i, x in enumerate(e)):
            return {}
        c %= self.p
        return {e: c} if c else {}

    def var(self, i, a=1):
        e = [0] * self.n
        e[i] = a
        return self.mono(e)

    def one(self):
        return self.mono([0] * self.n)

    # --- linear algebra
    def add(self, f, g):
        r = dict(f)
        for e, c in g.items():
            v = (r.get(e, 0) + c) % self.p
            if v:
                r[e] = v
            else:
                r.pop(e, None)
        return r

    def scal(self, c, f):
        c %= self.p
        if c == 0:
            return {}
        return {e: (v * c) % self.p for e, v in f.items()}

    def sub(self, f, g):
        return self.add(f, self.scal(-1, g))

    # --- product
    def mul(self, f, g):
        p, top, n = self.p, self.top, self.n
        r = {}
        for e1, c1 in f.items():
            for e2, c2 in g.items():
                e = tuple(a + b for a, b in zip(e1, e2))
                if any(e[i] >= top[i] for i in range(n)):
                    continue
                c = c1 * c2 % p
                for i in range(n):
                    if c == 0:
                        break
                    c = c * binom_mod(e[i], e1[i], p) % p
                if c:
                    v = (r.get(e, 0) + c) % p
                    if v:
                        r[e] = v
                    else:
                        r.pop(e, None)
        return r

    def prod(self, *fs):
        r = self.one()
        for f in fs:
            r = self.mul(r, f)
            if not r:
                return {}
        return r

    # --- calculus
    def diff(self, i, f):
        r = {}
        for e, c in f.items():
            if e[i] == 0:
                continue
            e2 = list(e)
            e2[i] -= 1
            r[tuple(e2)] = c
        return r

    def integrate(self, i, f):
        """antiderivative in u_i; returns (poly, obstruction) where the
        obstruction collects the monomials of top degree in u_i (which would
        need u_i^{(p^{N_i})}, outside O)."""
        r, obs = {}, {}
        for e, c in f.items():
            if e[i] + 1 >= self.top[i]:
                obs[e] = c
                continue
            e2 = list(e)
            e2[i] += 1
            r[tuple(e2)] = c
        return r, obs

    def const_term(self, f):
        return f.get(tuple([0] * self.n), 0)

    # --- printing
    def tex(self, f, names=None):
        if not f:
            return "0"
        names = names or ["u_{%d}" % (i + 1) for i in range(self.n)]
        parts = []
        for e in sorted(f, reverse=True):
            c = f[e]
            s = ""
            for i, a in enumerate(e):
                if a == 0:
                    continue
                s += names[i] if a == 1 else "%s^{(%d)}" % (names[i], a)
            cc = c if c <= self.p // 2 else c - self.p
            if s == "":
                parts.append("%d" % cc)
            elif cc == 1:
                parts.append(s)
            elif cc == -1:
                parts.append("-" + s)
            else:
                parts.append("%d%s" % (cc, s))
        out = parts[0]
        for t in parts[1:]:
            out += (" " + t) if t.startswith("-") else (" + " + t)
        return out


# ---------------------------------------------------------------- the bracket

class Ham:
    """Hamiltonian Lie algebra h_omega(2k; N) for omega = omega_0 + eps*(B-part).

    B is a k x k integer matrix (entries reduced mod p); eps in F_p^*.
    Setting B = None (or eps = 0) gives the undeformed h(2k; N)."""

    def __init__(self, p, N, B=None, eps=1, off=1):
        """off=1: xbar_i = u_i^{(p^{N_i}-1)}, i.e. omega = omega_0 + eps sum A_ij
                  d(u_i^{(p^{N_i})}) ^ d(u_j^{(p^{N_j})})  -- the convention of
                  eqs. (0.3) and (1st_type) of the text;
           off=2: xbar_i = u_i^{(p^{N_i}-2)}, i.e. the same with u^{(p^{N}-1)}
                  in place of u^{(p^{N})} -- the convention behind the displayed
                  formulas (J1,0n=4), (J2p), (O4p=3), (J4p=*), (J421p=*)."""
        assert len(N) % 2 == 0
        self.p = p
        self.N = tuple(N)
        self.k = len(N) // 2
        self.A = DP(p, N)
        self.eps = eps % p
        self.off = off
        k, A = self.k, self.A
        self.B = [[0] * k for _ in range(k)] if B is None else [[x % p for x in row] for row in B]
        # xbar_i = u_i^{(p^{N_i}-off)}
        self.xbar = [A.var(i, A.top[i] - off) for i in range(2 * k)]
        # S_{ij} = B_{ij} xbar_i xbar_{~j}
        self.S = [[A.scal(self.B[i][j], A.mul(self.xbar[i], self.xbar[j + k]))
                   for j in range(k)] for i in range(k)]
        self.G = self._inverse_I_plus_epsS()
        self.Omega = self._omega_matrix()

    # G = (I + eps S)^{-1} = sum_m (-eps)^m S^m
    def _inverse_I_plus_epsS(self):
        A, k, p = self.A, self.k, self.p
        G = [[A.one() if i == j else A.zero() for j in range(k)] for i in range(k)]
        term = [[A.one() if i == j else A.zero() for j in range(k)] for i in range(k)]  # S^m
        sign = 1
        cap = sum(t - 1 for t in A.top) + 3     # S has entries in the maximal ideal
        for m in range(1, cap):
            new = [[A.zero() for _ in range(k)] for _ in range(k)]
            nonzero = False
            for i in range(k):
                for j in range(k):
                    acc = A.zero()
                    for t in range(k):
                        if term[i][t] and self.S[t][j]:
                            acc = A.add(acc, A.mul(term[i][t], self.S[t][j]))
                    new[i][j] = acc
                    if acc:
                        nonzero = True
            if not nonzero:
                break
            term = new
            sign = (-sign * self.eps) % p
            for i in range(k):
                for j in range(k):
                    if term[i][j]:
                        G[i][j] = A.add(G[i][j], A.scal(sign, term[i][j]))
        return G

    def _omega_matrix(self):
        """Omega_{ij}, a 2k x 2k matrix over O."""
        A, k, p = self.A, self.k, self.p
        n = 2 * k
        Om = [[A.zero() for _ in range(n)] for _ in range(n)]
        for i in range(k):
            Om[i][i + k] = A.one()
            Om[i + k][i] = A.scal(-1, A.one())
        for i in range(k):
            for j in range(k):
                if self.B[i][j]:
                    s = A.scal(self.eps * self.B[i][j], A.mul(self.xbar[i], self.xbar[j + k]))
                    Om[i][j + k] = A.add(Om[i][j + k], s)
                    Om[j + k][i] = A.sub(Om[j + k][i], s)
        return Om

    # {f,g} = sum_{i,j<k} G_ij ( d_{~i} f d_j g - d_j f d_{~i} g )
    def br(self, f, g):
        A, k = self.A, self.k
        r = A.zero()
        for i in range(k):
            dif = A.diff(i + k, f)
            for j in range(k):
                Gij = self.G[i][j]
                if not Gij:
                    continue
                t = A.sub(A.mul(dif, A.diff(j, g)),
                          A.mul(A.diff(j, f), A.diff(i + k, g)))
                if t:
                    r = A.add(r, A.mul(Gij, t))
        return r

    # ------------------------------------------------------------ checks
    def check_jacobi(self, trials=40, seed=0):
        """Jacobi identity on random elements (equivalent to d omega = 0)."""
        rnd = random.Random(seed)
        A = self.A
        allm = list(itertools.product(*[range(t) for t in A.top]))
        bad = 0
        for _ in range(trials):
            fs = []
            for _ in range(3):
                f = A.zero()
                for _ in range(3):
                    f = A.add(f, A.mono(rnd.choice(allm), rnd.randrange(1, self.p)))
                fs.append(f)
            a, b, c = fs
            s = A.add(A.add(self.br(a, self.br(b, c)),
                            self.br(b, self.br(c, a))),
                      self.br(c, self.br(a, b)))
            if s:
                bad += 1
        return bad

    def check_antisym(self, trials=20, seed=1):
        rnd = random.Random(seed)
        A = self.A
        allm = list(itertools.product(*[range(t) for t in A.top]))
        bad = 0
        for _ in range(trials):
            a = A.mono(rnd.choice(allm), rnd.randrange(1, self.p))
            b = A.mono(rnd.choice(allm), rnd.randrange(1, self.p))
            if A.add(self.br(a, b), self.br(b, a)):
                bad += 1
        return bad

    # ------------------------------------------------------------ p-structure
    def ad_pow(self, f, g, m):
        for _ in range(m):
            g = self.br(f, g)
        return g

    def p_power(self, f):
        """Return (f^[p], obstruction, ok) with H_f^p = H_{f^[p]}.
        obstruction is non-empty when the antiderivative leaves O(2k;N)."""
        A, k, p = self.A, self.k, self.p
        n = 2 * k
        c = [self.ad_pow(f, A.var(i), p) for i in range(n)]     # components
        # d_j h = sum_i c_i Omega_{ij}
        dh = []
        for j in range(n):
            acc = A.zero()
            for i in range(n):
                if c[i] and self.Omega[i][j]:
                    acc = A.add(acc, A.mul(c[i], self.Omega[i][j]))
            dh.append(acc)
        h, obs = self._integrate_gradient(dh)
        # verify
        ok = all(not A.sub(A.diff(j, h), dh[j]) for j in range(n)) and not obs
        return h, obs, ok

    def _integrate_gradient(self, dh):
        """Reconstruct h from its partial derivatives."""
        A, n = self.A, 2 * self.k
        h = A.zero()
        obs = A.zero()
        rem = list(dh)
        for j in range(n):
            part, o = A.integrate(j, rem[j])
            obs = A.add(obs, o)
            if not part:
                continue
            h = A.add(h, part)
            for s in range(n):
                rem[s] = A.sub(rem[s], A.diff(s, part))
        return h, obs

    def is_p_closed(self, verbose=False):
        """Check on all monomials that f^[p] exists inside O(2k;N)."""
        A = self.A
        allm = list(itertools.product(*[range(t) for t in A.top]))
        bad = []
        for e in allm:
            f = A.mono(e)
            h, obs, ok = self.p_power(f)
            if not ok:
                bad.append((e, obs))
        return bad


# ---------------------------------------------------------------- shapes

def J_k_0(k):
    """J_k(0): 1's on the superdiagonal."""
    B = [[0] * k for _ in range(k)]
    for i in range(k - 1):
        B[i][i + 1] = 1
    return B


def C_k(k):
    """C_k: 1's on the superdiagonal and 1 in the lower-left corner."""
    B = J_k_0(k)
    B[k - 1][0] = 1
    return B


def J_kr(k, r, lam):
    """J_{k,r}(lambda), k = r n: blocks 1_r on the block superdiagonal and
    J_r(lambda) in the lower-left block."""
    assert k % r == 0
    n = k // r
    B = [[0] * k for _ in range(k)]
    for c in range(n - 1):
        for a in range(r):
            B[c * r + a][(c + 1) * r + a] = 1
    for a in range(r):
        B[(n - 1) * r + a][a] = lam
        if a + 1 < r:
            B[(n - 1) * r + a][a + 1] = 1
    return B


# ------------------------------------------------ Skryabin's forms of 2nd type

class HamExp:
    """h_{omega_{2,j}}(2k; N), the Lie algebra preserving

        omega_{2,j} = d( exp(eps u_j^{(p^{N_j})}) sum_{i<=k} u_i du_{~i} )
                    = exp(eps v) ( omega_0 + eps xbar_j du_j ^ alpha_0 ),
        v := u_j^{(p^{N_j})},  xbar_j := u_j^{(p^{N_j}-1)},  alpha_0 := sum u_i du_{~i}.

    Generating functions are a = atilde exp(eps v) with atilde in O(2k;N); the
    induced bracket on the atilde's is

        {a,b} = sum_{i,l} (Omega^{-1})_{li} (d_l a + eps a xbar_l delta_{lj})
                                            (d_i b + eps b xbar_i delta_{ij}),

    Omega being the matrix of omega_{2,j} with the factor exp(eps v) stripped.
    (jj below is the 0-based index j.)"""

    def __init__(self, p, N, jj, eps=1):
        assert len(N) % 2 == 0
        self.p, self.N, self.k = p, tuple(N), len(N) // 2
        self.A = DP(p, N)
        self.eps = eps % p
        self.jj = jj
        A, k = self.A, self.k
        self.xbarj = A.var(jj, A.top[jj] - 1)
        n = 2 * k
        Om = [[A.zero() for _ in range(n)] for _ in range(n)]
        for i in range(k):
            Om[i][i + k] = A.one()
            Om[i + k][i] = A.scal(-1, A.one())
        # eps * xbar_j du_j ^ alpha_0 = eps xbar_j sum_i u_i du_j ^ du_{~i}
        for i in range(k):
            t = A.scal(self.eps, A.mul(self.xbarj, A.var(i)))
            if jj == i + k:
                continue                      # du_j ^ du_j = 0
            Om[jj][i + k] = A.add(Om[jj][i + k], t)
            Om[i + k][jj] = A.sub(Om[i + k][jj], t)
        self.Omega = Om
        self.Oinv = self._invert(Om)

    def _invert(self, Om):
        """Om = J + C with C nilpotent (entries in the maximal ideal):
        Om^{-1} = (I + J^{-1}C)^{-1} J^{-1}."""
        A, k, p = self.A, self.k, self.p
        n = 2 * k
        # J^{-1}: J_{i,~i}=1, J_{~i,i}=-1  =>  (J^{-1})_{i,~i}=-1, (J^{-1})_{~i,i}=1
        Ji = [[A.zero() for _ in range(n)] for _ in range(n)]
        for i in range(k):
            Ji[i][i + k] = A.scal(-1, A.one())
            Ji[i + k][i] = A.one()
        C = [[A.sub(Om[i][j], A.one() if (j == i + k and i < k) else
                    (A.scal(-1, A.one()) if (i >= k and j == i - k) else A.zero()))
              for j in range(n)] for i in range(n)]
        X = self._matmul(Ji, C)                       # J^{-1} C, nilpotent
        # (I+X)^{-1} = sum (-X)^m
        R = [[A.one() if i == j else A.zero() for j in range(n)] for i in range(n)]
        T = [[A.one() if i == j else A.zero() for j in range(n)] for i in range(n)]
        cap = sum(t - 1 for t in A.top) + 3
        sign = 1
        for _ in range(cap):
            T = self._matmul(T, X)
            if all(not T[i][j] for i in range(n) for j in range(n)):
                break
            sign = -sign
            for i in range(n):
                for j in range(n):
                    if T[i][j]:
                        R[i][j] = A.add(R[i][j], A.scal(sign, T[i][j]))
        return self._matmul(R, Ji)

    def _matmul(self, X, Y):
        A = self.A
        n = len(X)
        Z = [[A.zero() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for t in range(n):
                if not X[i][t]:
                    continue
                for j in range(n):
                    if Y[t][j]:
                        Z[i][j] = A.add(Z[i][j], A.mul(X[i][t], Y[t][j]))
        return Z

    def _D(self, l, a):
        """d_l a + eps a xbar_l delta_{l,j}"""
        A = self.A
        r = A.diff(l, a)
        if l == self.jj:
            r = A.add(r, A.scal(self.eps, A.mul(a, self.xbarj)))
        return r

    def br(self, a, b):
        A, n = self.A, 2 * self.k
        r = A.zero()
        Da = [self._D(l, a) for l in range(n)]
        Db = [self._D(i, b) for i in range(n)]
        for l in range(n):
            if not Da[l]:
                continue
            for i in range(n):
                if self.Oinv[l][i] and Db[i]:
                    r = A.add(r, A.mul(self.Oinv[l][i], A.mul(Da[l], Db[i])))
        return r

    def check_jacobi(self, trials=30, seed=0):
        rnd = random.Random(seed)
        A = self.A
        allm = list(itertools.product(*[range(t) for t in A.top]))
        bad = 0
        for _ in range(trials):
            fs = []
            for _ in range(3):
                f = A.zero()
                for _ in range(2):
                    f = A.add(f, A.mono(rnd.choice(allm), rnd.randrange(1, self.p)))
                fs.append(f)
            a, b, c = fs
            s = A.add(A.add(self.br(a, self.br(b, c)), self.br(b, self.br(c, a))),
                      self.br(c, self.br(a, b)))
            if s:
                bad += 1
        return bad

    def ad_pow(self, f, g, m):
        for _ in range(m):
            g = self.br(f, g)
        return g
