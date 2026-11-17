#!/usr/bin/env python3
"""The worked example for the superized Bott theorem: G = U(2|1), M = C^{2|1}.

Theorem 2.2 1) of the text says  chi(D) = i_*(s(D)).  This engine computes the
right-hand side for the smallest unitary supergroup with a != b, taking for D
the Dolbeault-type operator on the flag supermanifold G/T twisted by a
character of the maximal torus T, and for M the tautological module C^{2|1},
which is ATYPICAL.

WHAT IS COMPUTED, all in exact arithmetic:

  1. gl(2|1), its root system, and the atypicality of the tautological weight.
  2. The Kac modules K(lambda) = Lambda(g_{-1}) (x) L^0_lambda, built from
     scratch; the representation property is verified on all 81 brackets.
  3. The irreducibles L_lambda = K(lambda)/rad and their characters, with the
     parity carried in Z[eps]/(eps^2 - 1) as the R(G) of the text requires.
  4. The symbol of D_mu in R(T):

        s(D_mu) = e^mu * prod_{even alpha>0} (1 - e^alpha)
                        * prod_{odd gamma>0} (1 + eps e^gamma)^{-1},

     the last factor being an infinite series -- Lambda^k of an ODD line is its
     k-th symmetric power, so the alternating sum is 1/(1 + eps e^gamma).  This
     is why the target of i_* is the completion R^(G) and not R(G).
  5. i_*(s(D_mu)) = sum over the dominant nu of <ch L_nu, s(D_mu)> [L_nu],
     the pairing being the delta pairing on weights over Z[eps]/(eps^2-1).
     The sum is over an infinite set of nu; the engine exhibits the finitely
     many survivors.

THE POINT.  For TYPICAL nu the odd factor of the symbol cancels the odd factor
of the Kac character exactly, and the pairing reduces to the classical Weyl
computation, giving delta_{nu, mu}.  For ATYPICAL nu it does not cancel:
ch L_nu is a proper quotient of ch K(nu), so ch L_nu * prod (1 + eps e^-gamma)^{-1}
is an infinite series, and what makes the answer finite again is the resolution
of L_nu by Kac modules -- the Bernstein-Leites formula.

Subsecs. 2.6-2.7 of the text (Theorem 2.6.4, Statement 2.7, formula (11)).

Run:  python3 bott21.py
"""

import sys
from fractions import Fraction
from itertools import product

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, ok, detail))
    print(("  ok   " if ok else "  FAIL") + f"  {name}" + (f"   [{detail}]" if detail else ""))
    return ok


# ------------------------------------------------------ Z[eps]/(eps^2-1) ---

class Z2:
    """a + b*eps with eps^2 = 1 -- the ring the superdimension takes values in."""

    __slots__ = ("a", "b")

    def __init__(self, a=0, b=0):
        self.a, self.b = a, b

    def __add__(self, o):
        o = o if isinstance(o, Z2) else Z2(o)
        return Z2(self.a + o.a, self.b + o.b)

    __radd__ = __add__

    def __neg__(self):
        return Z2(-self.a, -self.b)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, Z2) else Z2(o)))

    def __mul__(self, o):
        o = o if isinstance(o, Z2) else Z2(o)
        return Z2(self.a * o.a + self.b * o.b, self.a * o.b + self.b * o.a)

    __rmul__ = __mul__

    def __eq__(self, o):
        o = o if isinstance(o, Z2) else Z2(o)
        return (self.a, self.b) == (o.a, o.b)

    def __bool__(self):
        return bool(self.a or self.b)

    def __hash__(self):
        return hash((self.a, self.b))

    def at(self, e):
        """evaluation at eps = e (e = 1 or -1): the two homomorphisms onto Z"""
        return self.a + e * self.b

    def __repr__(self):
        if not self.b:
            return str(self.a)
        if not self.a:
            return ("" if self.b == 1 else ("-" if self.b == -1 else str(self.b))) + "eps"
        s = "+" if self.b > 0 else "-"
        m = abs(self.b)
        return f"{self.a}{s}{'' if m == 1 else m}eps"


ONE, EPS = Z2(1, 0), Z2(0, 1)


# --------------------------------------------------------------- gl(2|1) ---
# indices 0, 1 even; index 2 odd.  E_ij is the matrix unit.

PAR = [0, 0, 1]
UNITS = [(i, j) for i in range(3) for j in range(3)]


def upar(u):
    return (PAR[u[0]] + PAR[u[1]]) % 2


def ubracket(u, v):
    """[E_ij, E_kl] as {unit: int}."""
    (i, j), (k, l) = u, v
    out = {}
    if j == k:
        out[(i, l)] = out.get((i, l), 0) + 1
    if l == i:
        s = -((-1) ** (upar(u) * upar(v)))
        out[(k, j)] = out.get((k, j), 0) + s
    return {a: b for a, b in out.items() if b}


# weights are triples (w1, w2, w3) meaning w1 eps1 + w2 eps2 + w3 delta
def wt_of_unit(u):
    i, j = u
    w = [0, 0, 0]
    w[i] += 1
    w[j] -= 1
    return tuple(w)


ALPHA = (1, -1, 0)                       # the even positive root eps1 - eps2
GAMMA1 = (1, 0, -1)                      # the odd positive roots eps_i - delta
GAMMA2 = (0, 1, -1)
POS_EV = [ALPHA]
POS_OD = [GAMMA1, GAMMA2]
RHO_EV = (Fraction(1, 2), Fraction(-1, 2), 0)
RHO_OD = (Fraction(1, 2), Fraction(1, 2), -1)
RHO = tuple(a - b for a, b in zip(RHO_EV, RHO_OD))     # = (0, -1, 1) = -eps2 + delta


def form(x, y):
    """the invariant form: (eps_i, eps_j) = delta_ij, (delta, delta) = -1."""
    return x[0] * y[0] + x[1] * y[1] - x[2] * y[2]


def wadd(x, y, c=1):
    return tuple(a + c * b for a, b in zip(x, y))


# ------------------------------------------------------- the g_0-module ---

class L0:
    """The irreducible g_0 = gl(2) (+) gl(1) module of highest weight
    (a1, a2 | b), realized on the binary forms of degree d = a1 - a2."""

    def __init__(self, a1, a2, b):
        assert a1 >= a2
        self.a1, self.a2, self.b, self.d = a1, a2, b, a1 - a2
        self.dim = self.d + 1
        self.wts = [(a1 - k, a2 + k, b) for k in range(self.dim)]

    def act(self, u, k):
        """E_u . u_k  as {index: coefficient}; u_k = x1^{d-k} x2^k."""
        i, j = u
        d = self.d
        if i == 2 or j == 2:                       # E_i3, E_3i, E_33
            if (i, j) == (2, 2):
                return {k: Fraction(self.b)}
            return {}
        # x_i d/dx_j  on x1^{d-k} x2^k, plus the det^{a2} twist
        exps = [d - k, k]
        out = {}
        e = exps[j]
        if e:
            new = exps[:]
            new[j] -= 1
            new[i] += 1
            out[new[1]] = out.get(new[1], Fraction(0)) + e
        if i == j:
            out[k] = out.get(k, Fraction(0)) + self.a2
        return {a: b for a, b in out.items() if b}


# ------------------------------------------------------- the Kac module ---

class Kac:
    """K(lambda) = Lambda(g_{-1}) (x) L^0_lambda, g_{-1} = <f_1, f_2>,
    f_i = E_{3i} (odd, of weight -gamma_i)."""

    def __init__(self, a1, a2, b):
        self.L0 = L0(a1, a2, b)
        self.hw = (a1, a2, b)
        self.basis = [(S, k) for S in [(), (0,), (1,), (0, 1)]
                      for k in range(self.L0.dim)]
        self.index = {v: t for t, v in enumerate(self.basis)}
        self.dim = len(self.basis)
        self.par = [len(S) % 2 for (S, k) in self.basis]
        self.wt = [self._wt(S, k) for (S, k) in self.basis]
        self.rho = {u: self._matrix(u) for u in UNITS}

    def _wt(self, S, k):
        w = self.L0.wts[k]
        for i in S:
            w = wadd(w, (GAMMA1 if i == 0 else GAMMA2), -1)
        return w

    # -- the action, built by the recursion described in the docstring -----
    def _act_f(self, i, vec):
        """f_i . v for v a dict {(S, k): coef}"""
        out = {}
        for (S, k), c in vec.items():
            if i in S:
                continue
            pos = sum(1 for t in S if t < i)
            sgn = (-1) ** pos
            T = tuple(sorted(S + (i,)))
            out[(T, k)] = out.get((T, k), Fraction(0)) + sgn * c
        return {a: b for a, b in out.items() if b}

    def _act_g0(self, u, vec):
        """E_u . v for u in g_0 = {E_11, E_12, E_21, E_22, E_33}"""
        out = {}
        for (S, k), c in vec.items():
            # on L^0
            for k2, c2 in self.L0.act(u, k).items():
                out[(S, k2)] = out.get((S, k2), Fraction(0)) + c * c2
            # on the f's:  [E_u, f_i] = sum_j c_ji f_j
            for pos, i in enumerate(S):
                fi = (2, i)                                  # f_i = E_{3i}
                for tgt, coef in ubracket(u, fi).items():
                    if tgt[0] != 2 or tgt[1] == 2:
                        continue
                    j = tgt[1]
                    if j in S and j != i:
                        continue
                    T = list(S)
                    T[pos] = j
                    sgn = 1
                    Ts = sorted(T)
                    # sign of sorting a list of distinct odd generators
                    perm = list(T)
                    for x in range(1, len(perm)):
                        y = x
                        while y > 0 and perm[y - 1] > perm[y]:
                            perm[y - 1], perm[y] = perm[y], perm[y - 1]
                            sgn = -sgn
                            y -= 1
                    key = (tuple(Ts), k)
                    out[key] = out.get(key, Fraction(0)) + c * coef * sgn
        return {a: b for a, b in out.items() if b}

    def _act_e(self, j, vec):
        """E_{j3} . v, by  e f_i = [e, f_i] - f_i e  and  e . (1 (x) u) = 0."""
        out = {}
        for (S, k), c in vec.items():
            out = _addv(out, self._e_on(j, S, k), c)
        return {a: b for a, b in out.items() if b}

    def _e_on(self, j, S, k):
        if not S:
            return {}
        i, rest = S[0], S[1:]
        e = (j, 2)
        fi = (2, i)
        res = {}
        # [e, f_i] . (f_rest (x) u_k)
        for tgt, coef in ubracket(e, fi).items():
            res = _addv(res, self._act_g0(tgt, {(rest, k): Fraction(1)}), coef)
        # - f_i . (e . (f_rest (x) u_k))
        inner = self._e_on(j, rest, k)
        res = _addv(res, self._act_f(i, inner), -1)
        return res

    def _matrix(self, u):
        i, j = u
        cols = []
        for v in self.basis:
            vec = {v: Fraction(1)}
            if i == 2 and j != 2:
                img = self._act_f(j, vec)
            elif j == 2 and i != 2:
                img = self._act_e(i, vec)
            else:
                img = self._act_g0(u, vec)
            cols.append(img)
        M = [[Fraction(0)] * self.dim for _ in range(self.dim)]
        for c, img in enumerate(cols):
            for key, val in img.items():
                M[self.index[key]][c] = val
        return M

    # -- structure ---------------------------------------------------------
    def check_rep(self):
        ok = True
        for u in UNITS:
            for v in UNITS:
                A, B = self.rho[u], self.rho[v]
                s = (-1) ** (upar(u) * upar(v))
                lhs = [[sum(A[r][t] * B[t][c] for t in range(self.dim))
                        - s * sum(B[r][t] * A[t][c] for t in range(self.dim))
                        for c in range(self.dim)] for r in range(self.dim)]
                rhs = [[Fraction(0)] * self.dim for _ in range(self.dim)]
                for tgt, coef in ubracket(u, v).items():
                    T = self.rho[tgt]
                    for r in range(self.dim):
                        for c in range(self.dim):
                            rhs[r][c] += coef * T[r][c]
                if lhs != rhs:
                    ok = False
        return ok

    def character(self):
        ch = {}
        for t, w in enumerate(self.wt):
            ch[w] = ch.get(w, Z2()) + (EPS if self.par[t] else ONE)
        return {w: c for w, c in ch.items() if c}

    def singular_vectors(self):
        """vectors killed by E_12, E_13, E_23, computed weight space by weight
        space; the highest-weight line is excluded."""
        raise_ops = [(0, 1), (0, 2), (1, 2)]
        out = []
        spaces = {}
        for t, w in enumerate(self.wt):
            spaces.setdefault(w, []).append(t)
        for w, idxs in spaces.items():
            rows = []
            for u in raise_ops:
                M = self.rho[u]
                for r in range(self.dim):
                    rows.append([M[r][c] for c in idxs])
            ns = nullspace(rows, len(idxs))
            for v in ns:
                if w == self.hw:
                    continue
                out.append((w, {idxs[t]: v[t] for t in range(len(idxs)) if v[t]}))
        return out

    def submodule(self, gens):
        """the g-submodule generated by the given vectors (as dicts)"""
        vecs = []
        for g in gens:
            vecs.append([g.get(t, Fraction(0)) for t in range(self.dim)])
        changed = True
        basis = reduce_rows(vecs, self.dim)
        while changed:
            changed = False
            new = []
            for b in basis:
                for u in UNITS:
                    M = self.rho[u]
                    img = [sum(M[r][c] * b[c] for c in range(self.dim))
                           for r in range(self.dim)]
                    if any(img):
                        new.append(img)
            cand = reduce_rows(basis + new, self.dim)
            if len(cand) > len(basis):
                basis, changed = cand, True
        return basis

    def irreducible_character(self):
        sv = self.singular_vectors()
        if not sv:
            return self.character(), 0
        sub = self.submodule([g for _, g in sv])
        # character of the submodule: it is weight-graded, so count by weights
        chsub = {}
        for row in echelon_by_weight(sub, self.wt, self.par):
            w, p = row
            chsub[w] = chsub.get(w, Z2()) + (EPS if p else ONE)
        ch = dict(self.character())
        for w, c in chsub.items():
            ch[w] = ch.get(w, Z2()) - c
        return {w: c for w, c in ch.items() if c}, len(sub)


def _addv(a, b, c=1):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Fraction(0)) + c * v
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------- linear algebra ---

def reduce_rows(rows, n):
    """row echelon basis of the span"""
    basis = []
    piv = []
    for row in rows:
        r = row[:]
        for b, p in zip(basis, piv):
            if r[p]:
                f = r[p] / b[p]
                r = [x - f * y for x, y in zip(r, b)]
        p = next((i for i in range(n) if r[i]), None)
        if p is not None:
            basis.append(r)
            piv.append(p)
    return basis


def nullspace(rows, n):
    if not rows:
        return [[Fraction(1) if i == j else Fraction(0) for i in range(n)]
                for j in range(n)]
    M = [r[:] for r in rows]
    piv, r = [], 0
    for c in range(n):
        p = next((i for i in range(r, len(M)) if M[i][c]), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        inv = Fraction(1) / M[r][c]
        M[r] = [x * inv for x in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c]:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == len(M):
            break
    free = [c for c in range(n) if c not in piv]
    out = []
    for fc in free:
        v = [Fraction(0)] * n
        v[fc] = Fraction(1)
        for row, pc in zip(M[:r], piv):
            v[pc] = -row[fc]
        out.append(v)
    return out


def echelon_by_weight(basis, wt, par):
    """the multiset of (weight, parity) of a weight-graded subspace"""
    out = []
    groups = {}
    for t, w in enumerate(wt):
        groups.setdefault((w, par[t]), []).append(t)
    for key, idxs in groups.items():
        rows = [[b[t] for t in idxs] for b in basis]
        rk = len(reduce_rows(rows, len(idxs)))
        out.extend([key] * rk)
    return out


# ------------------------------------------------------ the Bott side ----

def symbol_coeff(mu, w):
    """the coefficient of e^w in
       s(D_mu) = e^mu (1 - e^alpha) prod_{gamma>0 odd} (1 + eps e^gamma)^{-1}
                = e^mu (1 - e^alpha) sum_{j,k>=0} (-eps)^{j+k} e^{j gamma1 + k gamma2}
    """
    tot = Z2()
    for theta in (0, 1):
        base = wadd(mu, ALPHA, theta)
        j = w[0] - base[0]
        k = w[1] - base[1]
        m = base[2] - w[2]
        if j < 0 or k < 0 or j + k != m:
            continue
        c = (-1) ** theta * (-1) ** (j + k)
        tot = tot + (Z2(0, c) if (j + k) % 2 else Z2(c, 0))
    return tot


def pairing(ch, mu):
    """<ch, s(D_mu)> = sum_w ch_w * s_w  in Z[eps]/(eps^2 - 1)"""
    tot = Z2()
    for w, c in ch.items():
        s = symbol_coeff(mu, w)
        if s:
            tot = tot + c * s
    return tot


# ------------------------------------------------- the classical control --

def classical_character(a1, a2):
    """gl(2): the irreducible of highest weight (a1, a2)"""
    return {(a1 - k, a2 + k): 1 for k in range(a1 - a2 + 1)}


def classical_pairing(ch, mu):
    """<ch, e^mu (1 - e^alpha)> for gl(2)"""
    tot = 0
    for w, c in ch.items():
        if w == mu:
            tot += c
        if w == (mu[0] + 1, mu[1] - 1):
            tot -= c
    return tot


# ---------------------------------------------------------------- main ----

def main():
    print("The superized Bott theorem for G = U(2|1), M = C^{2|1}\n")

    # ---------------------------------------------------------------- 1 ---
    print("  1. The root system, and the atypicality of the tautological weight\n")
    lam = (1, 0, 0)                          # eps_1: the tautological module
    lr = wadd(lam, RHO)
    print(f"    rho = {RHO},  lambda + rho = {lr}")
    at = [g for g in POS_OD if form(lr, g) == 0]
    check("the tautological weight eps_1 is atypical",
          len(at) == 1, f"atypical root {at[0] if at else None}")
    check("its atypicality root is gamma_2 = eps_2 - delta", at == [GAMMA2])
    for mu in [(2, 0, 0), (1, 1, 0), (0, 0, 1), (2, 1, 5)]:
        mr = wadd(mu, RHO)
        n = sum(1 for g in POS_OD if form(mr, g) == 0)
        print(f"    {mu}: atypicality degree {n}")

    # ---------------------------------------------------------------- 2 ---
    print("\n  2. The Kac module K(eps_1), built from scratch\n")
    K = Kac(1, 0, 0)
    check("K(eps_1): the representation property on all 81 brackets",
          K.check_rep())
    ch = K.character()
    de = sum(1 for p in K.par if p == 0)
    do = sum(1 for p in K.par if p == 1)
    print(f"    dim K = {K.dim}, sdim = ({de}|{do})")
    check("sdim K(eps_1) = (4|4)", (de, do) == (4, 4))

    chL, dsub = K.irreducible_character()
    print(f"    the maximal submodule has dimension {dsub}")
    print(f"    ch L(eps_1) = " + " + ".join(
        f"{c}e^{w}" for w, c in sorted(chL.items())))
    want = {(1, 0, 0): ONE, (0, 1, 0): ONE, (0, 0, 1): EPS}
    check("ch L(eps_1) = e^{eps_1} + e^{eps_2} + eps e^{delta}: "
          "the tautological module C^{2|1}", chL == want)
    sd = sum((c for c in chL.values()), Z2())
    check("sdim C^{2|1} = 2 + eps", sd == Z2(2, 1), str(sd))
    check("its two evaluations are dim = 3 and dim_ev - dim_od = 1",
          (sd.at(1), sd.at(-1)) == (3, 1))

    # ---------------------------------------------------------------- 3 ---
    print("\n  3. A typical weight, for contrast\n")
    Kt = Kac(2, 0, 5)
    check("K(2,0|5): the representation property", Kt.check_rep())
    chLt, dsubt = Kt.irreducible_character()
    mr = wadd((2, 0, 5), RHO)
    typ = all(form(mr, g) != 0 for g in POS_OD)
    check("(2, 0 | 5) is typical", typ)
    check("K(2,0|5) is irreducible: no proper maximal submodule", dsubt == 0)
    check("dim K = 4 dim L^0 = 12", Kt.dim == 12)

    # ---------------------------------------------------------------- 4 ---
    print("\n  4. The classical control: gl(2), where i_* is Borel-Weil-Bott\n")
    ok = True
    for mu in [(0, 0), (1, 0), (2, 0), (3, 1)]:
        hits = []
        for a1 in range(-2, 7):
            for a2 in range(-2, a1 + 1):
                v = classical_pairing(classical_character(a1, a2), mu)
                if v:
                    hits.append(((a1, a2), v))
        print(f"    mu = {mu}: i_*(e^mu (1 - e^alpha)) = " +
              " + ".join(f"{v}[M{n}]" for n, v in hits))
        if hits != [(mu, 1)]:
            ok = False
    check("gl(2): i_*(e^mu (1 - e^alpha)) = [M_mu] for every dominant mu", ok)

    # ---------------------------------------------------------------- 5 ---
    print("\n  5. The symbol of D_mu, and the factor 1/(1 + eps e^gamma)\n")
    mu = (1, 0, 0)
    print("    the first weights of s(D_mu) for mu = eps_1:")
    shown = 0
    for m in range(4):
        for j in range(m + 1):
            k = m - j
            for theta in (0, 1):
                w = wadd(wadd(wadd(mu, ALPHA, theta), GAMMA1, j), GAMMA2, k)
                c = symbol_coeff(mu, w)
                if c and shown < 12:
                    print(f"      e^{w}: {c}")
                    shown += 1
    check("the symbol is an infinite series: e^{mu + j gamma_2} != 0 for all j",
          all(symbol_coeff(mu, wadd(mu, GAMMA2, j)) for j in range(40)))
    check("individual coefficients do cancel: e^{mu + gamma_1} = 0",
          not symbol_coeff(mu, wadd(mu, GAMMA1)))

    # ---------------------------------------------------------------- 6 ---
    print("\n  6. For a TYPICAL nu the odd factor cancels: <ch L_nu, s> = delta\n")
    ok = True
    for nu in [(2, 0, 5), (3, 1, 4), (0, 0, 7), (2, 0, 5)]:
        Kn = Kac(*nu)
        chn, _ = Kn.irreducible_character()
        for mu2 in [(2, 0, 5), (1, 0, 0), (3, 1, 4)]:
            v = pairing(chn, mu2)
            want = ONE if nu == mu2 else Z2()
            if v != want:
                ok = False
                print(f"      nu = {nu}, mu = {mu2}: {v}")
    check("typical nu: <ch L_nu, s(D_mu)> = delta_{nu, mu}", ok)

    # ---------------------------------------------------------------- 7 ---
    print("\n  7. i_*(s(D_mu)): the survivors, over a range of atypical nu\n")
    print("     (only atypical nu can survive, by 6; nu is atypical iff")
    print("      a_1 + b + 1 = 0 (gamma_1) or a_2 + b = 0 (gamma_2))\n")
    counts = {}
    for mu2, kind in [((1, 0, 0), "atypical"), ((0, 0, 0), "atypical"),
                      ((2, 0, 5), "typical"), ((1, 1, 0), "typical")]:
        mr = wadd(mu2, RHO)
        got = "typical" if all(form(mr, g) != 0 for g in POS_OD) else "atypical"
        check(f"mu = {mu2} is {kind}", got == kind)
        for N in (5, 8):
            hits = []
            for a1 in range(-N, N + 1):
                for a2 in range(max(-N, a1 - 6), a1 + 1):
                    for b in sorted({-a1 - 1, -a2, mu2[2]}):
                        Km = Kac(a1, a2, b)
                        chn, _ = Km.irreducible_character()
                        v = pairing(chn, mu2)
                        if v:
                            hits.append(((a1, a2, b), v))
            counts[(mu2, N)] = hits
        h5, h8 = counts[(mu2, 5)], counts[(mu2, 8)]
        print(f"    mu = {mu2} ({kind}), range {8}: " +
              " + ".join(f"{v}[L{n}]" for n, v in sorted(h8)))
        if kind == "typical":
            check(f"mu = {mu2} typical: i_*(s(D_mu)) = [L_mu], one term",
                  h8 == [(mu2, ONE)], f"{len(h8)} survivor(s)")
        else:
            check(f"mu = {mu2} atypical: the survivors grow with the range "
                  f"({len(h5)} at 5, {len(h8)} at 8), so the sum is infinite",
                  len(h8) > len(h5))

    # ---------------------------------------------------------------- 8 ---
    print("\n  8. The mechanism: the chain of atypical weights\n")
    print("     nu |--> the weight of the singular vector of K(nu)\n")
    for nu in [(1, 0, 0), (1, 1, -1), (2, 2, -3), (3, 2, -4), (4, 2, -5)]:
        Kn = Kac(*nu)
        sv = Kn.singular_vectors()
        wts = sorted({w for w, _ in sv})
        print(f"    K{nu}: singular weights besides the top: {wts}")
    check("K(eps_1) has exactly one singular weight besides the top",
          len({w for w, _ in Kac(1, 0, 0).singular_vectors()}) == 1)

    bad = [c for c in CHECKS if not c[1]]
    print(f"\n  {len(CHECKS) - len(bad)}/{len(CHECKS)} checks pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
