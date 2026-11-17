#!/usr/bin/env python3
"""
The Z-graded basis of po(2|m) as a direct sum of g_0-modules, for m = 0, 1, 2.

  m=0:  g_0 = sl(2)      -- reproduces eq. (12)  of the text
  m=1:  g_0 = osp(1|2)   -- reproduces eq. (21)  of the text
  m=2:  g_0 = osp(2|2)   -- Statement 2.3.5 of the text

Everything is exact arithmetic on the monomial basis of C[p,q;odd vars].

Conventions are those of the text, i.e., Leites's "symmetric" contact form
  alpha = dt + (p dq + q dp) + theta d theta + (xi d eta + eta d xi),
so that, by eq. (33) of the text,
  {f,g} = d_q f d_p g + d_p f d_q g
          - (-1)^{p(f)} ( d_theta f d_theta g + d_xi f d_eta g + d_eta f d_xi g ).

A monomial is stored as (i, j, e) = p^i q^j * (odd part), where e is a tuple of
0/1 exponents of the odd indeterminates in their fixed order.

Irreducibility test.  Any nonzero submodule of a finite-dimensional module over
g_0 contains a nonzero vector killed by n^+ (n^+ acts nilpotently).  So if
dim ker n^+ = 1 and the generator of ker n^+ generates the whole component,
the component is irreducible.  Both are computed exactly below.
"""
import sys
from fractions import Fraction
from itertools import product

from exactlin import rank, nullspace, rref          # noqa: E402

# odd indeterminates, in order, and the pairing used by the contact form
ODD = {0: [],
       1: ["theta"],
       2: ["xi", "eta"]}


class Po:
    def __init__(self, m):
        assert m in ODD
        self.m = m
        self.odd = ODD[m]
        self.n = len(self.odd)

    # ------------------------------------------------------------- monomials
    def parity(self, mon):
        return sum(mon[2]) % 2

    def mmul(self, m1, m2):
        i, j, e1 = m1
        k, l, e2 = m2
        e = []
        sign = 1
        # move e2 letters left past the tail of e1
        for t in range(self.n):
            if e1[t] and e2[t]:
                return None
            e.append(e1[t] + e2[t])
        for t in range(self.n):
            if e2[t]:
                sign *= (-1) ** sum(e1[t + 1:])
        return (sign, (i + k, j + l, tuple(e)))

    def d_p(self, mon):
        i, j, e = mon
        return None if i == 0 else (i, (i - 1, j, e))

    def d_q(self, mon):
        i, j, e = mon
        return None if j == 0 else (j, (i, j - 1, e))

    def d_odd(self, mon, t):
        """Left derivative in the t-th odd indeterminate."""
        i, j, e = mon
        if not e[t]:
            return None
        sign = (-1) ** sum(e[:t])
        ne = list(e)
        ne[t] = 0
        return (sign, (i, j, tuple(ne)))

    def _tmul(self, t1, t2):
        if t1 is None or t2 is None:
            return None
        s1, a = t1
        s2, b = t2
        r = self.mmul(a, b)
        return None if r is None else (s1 * s2 * r[0], r[1])

    def bracket_mon(self, m1, m2):
        out = {}

        def add(t, c=1):
            if t is None:
                return
            s, mm = t
            out[mm] = out.get(mm, 0) + c * s

        add(self._tmul(self.d_q(m1), self.d_p(m2)))
        add(self._tmul(self.d_p(m1), self.d_q(m2)))
        sgn = -((-1) ** self.parity(m1))
        if self.m == 1:                       # theta d theta
            add(self._tmul(self.d_odd(m1, 0), self.d_odd(m2, 0)), sgn)
        elif self.m == 2:                     # xi d eta + eta d xi
            add(self._tmul(self.d_odd(m1, 0), self.d_odd(m2, 1)), sgn)
            add(self._tmul(self.d_odd(m1, 1), self.d_odd(m2, 0)), sgn)
        return {k: v for k, v in out.items() if v != 0}

    def bracket(self, f, g):
        out = {}
        for a, ca in f.items():
            for b, cb in g.items():
                for mm, c in self.bracket_mon(a, b).items():
                    out[mm] = out.get(mm, 0) + ca * cb * c
        return {k: v for k, v in out.items() if v != 0}

    # --------------------------------------------------------------- grading
    def basis_of_degree(self, k):
        ev, od = [], []
        for e in product((0, 1), repeat=self.n):
            d = k - sum(e)
            if d < 0:
                continue
            for i in range(d + 1):
                mon = (i, d - i, tuple(e))
                (ev if self.parity(mon) == 0 else od).append(mon)
        return ev + od

    def sdim_of_degree(self, k):
        b = self.basis_of_degree(k)
        return (sum(1 for m in b if self.parity(m) == 0),
                sum(1 for m in b if self.parity(m) == 1))

    # ------------------------------------------------------------ the Borel
    def height(self, mon):
        """A functional positive on exactly one half of the roots of g_0."""
        i, j, e = mon
        h = 3 * (i - j)
        if self.m == 2:
            h += (e[0] - e[1])
        elif self.m == 1:
            h += 0
        return h

    def n_plus(self):
        return [{m: 1} for m in self.basis_of_degree(2) if self.height(m) > 0]

    def g0(self):
        return [{m: 1} for m in self.basis_of_degree(2)]


# ----------------------------------------------------------- linear algebra
def as_vector(f, basis, index):
    v = [Fraction(0)] * len(basis)
    for m, c in f.items():
        v[index[m]] = Fraction(c)
    return v


def act_matrix(P, x, basis, index):
    cols = [as_vector(P.bracket(x, {m: 1}), basis, index) for m in basis]
    n = len(basis)
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def echelon(vectors):
    if not vectors:
        return []
    R, _ = rref([[Fraction(x) for x in v] for v in vectors])
    return [row for row in R if any(c != 0 for c in row)]


def common_kernel(mats, n):
    stacked = [row for M in mats for row in M]
    if not stacked:
        return [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    return nullspace(stacked)


def generated_submodule(P, v, gens, basis, index):
    cur = echelon([v])
    while True:
        new = list(cur)
        for x in gens:
            for w in cur:
                f = {basis[i]: w[i] for i in range(len(basis)) if w[i] != 0}
                new.append(as_vector(P.bracket(x, f), basis, index))
        new = echelon(new)
        if len(new) == len(cur):
            return len(cur)
        cur = new


def fmt(P, terms):
    names = ["p", "q"] + P.odd
    out = []
    for mon, c in sorted(terms.items()):
        exps = [mon[0], mon[1]] + list(mon[2])
        s = "" if c == 1 else ("-" if c == -1 else str(c))
        for nm, e in zip(names, exps):
            if e == 1:
                s += nm
            elif e > 1:
                s += "%s^%d" % (nm, e)
        out.append(s if s not in ("", "-") else s + "1")
    return " + ".join(out)


G0NAME = {0: "sl(2)", 1: "osp(1|2)", 2: "osp(2|2) = sl(1|2)"}


def run(m, kmax=8):
    P = Po(m)
    print("=" * 72)
    print("po(2|%d):  g_0 = <degree-2 polynomials> = %s" % (m, G0NAME[m]))
    sd0 = P.sdim_of_degree(2)
    print("  sdim g_0 = (%d|%d);  n^+ = %s"
          % (sd0[0], sd0[1], [fmt(P, x) for x in P.n_plus()]))
    gens = P.g0()
    npos = P.n_plus()
    print("\n   k   sdim of po(2|%d)_{k-2}   dim ker n^+   generated   irreducible?" % m)
    print("   " + "-" * 68)
    rows = []
    for k in range(kmax + 1):
        basis = P.basis_of_degree(k)
        index = {mm: i for i, mm in enumerate(basis)}
        n = len(basis)
        mats = [act_matrix(P, x, basis, index) for x in npos]
        ker = common_kernel(mats, n)
        dk = len(ker)
        gen = generated_submodule(P, ker[0], gens, basis, index) if dk == 1 else None
        irr = (dk == 1 and gen == n)
        sd = P.sdim_of_degree(k)
        hw = None
        if dk == 1:
            terms = {basis[i]: ker[0][i] for i in range(n) if ker[0][i] != 0}
            mon = next(iter(terms))
            hw = (fmt(P, terms), P.parity(mon))
        rows.append((k, sd, dk, gen, irr, hw))
        print("   %-3d (%d|%d)%s %-13d %-11s %s"
              % (k, sd[0], sd[1], " " * max(1, 21 - len("(%d|%d)" % sd)),
                 dk, gen if gen is not None else "-", "YES" if irr else "NO"))
    print("\n   highest weight vector of each component (parity in brackets):")
    for k, sd, dk, gen, irr, hw in rows:
        if hw:
            print("     k=%-2d  %-12s [%s]   sdim (%d|%d)"
                  % (k, hw[0], "even" if hw[1] == 0 else "ODD", sd[0], sd[1]))
    return rows


if __name__ == '__main__':
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    for m in (0, 1, 2):
        run(m, kmax)
        print()
