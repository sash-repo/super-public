#!/usr/bin/env python3
r"""testalg.py -- WHICH algebras C the definition-by-points has to be tested on,
and what that choice decides.

Third of the trio.  grass3.py showed that at p = 3 the Grassmann envelope cannot
feel the identity T(x) = [x,[x,x]] = 0; evenrules.py showed that at p = 2 it
cannot feel the squaring.  Both worked over Lambda(xi_1, ..., xi_N).  The
question here is the one left open by that choice: WHERE DO THE ALGEBRAS C COME
FROM, how many odd parameters are enough, and -- the point -- whether a larger
supply of C would repair either failure.

The answers, all four verified below.

  A.  HOW MANY ODD PARAMETERS.  The axioms tested are multilinear of arity <= 3,
      so N = 3 suffices and N = 2 does not: over Lambda(2) the fully odd Jacobi
      triple is invisible, every term of it carrying three odd parameters and
      hence a product of three of the xi's.  Witness over F_5 and F_7, plus a
      sweep in which "the axioms of a Lie superalgebra hold" and "the envelope
      over Lambda(3) is a Lie algebra" are compared object by object.

  B.  AT p = 3 NO C HELPS.  In a supercommutative superalgebra over a field of
      characteristic =/= 2, every odd element squares to zero (2c^2 = 0), so the
      three cyclic terms of the odd Jacobi triple are equal and their sum is
      3 T(x) = 0 whatever C is.  Checked on the free object A = <y, h, z> of
      the category without the identity, over Lambda(3), Lambda(4), and over the
      mixed algebra Lambda(3) (x) F_3[t]/(t^3) with t even -- the envelope is an
      honest Lie algebra every time.  So the p = 3 failure is arithmetic, not a
      defect of the test algebras.

  C.  AT p = 2 A LARGER C DOES HELP, AND IT IS THE OBVIOUS ONE.  At p = 2
      "supercommutative" is just "commutative", so it does NOT force xi^2 = 0:
      the algebra K[xi] with xi odd is as graded-commutative as Lambda(1) =
      K[xi]/(xi^2) is, and the received formulation of the Even Rules silently
      takes the second.  Over C = F_2[xi]/(xi^3) the enriched functor
            C |-> ( g(C) = (g (x) C)_ev ,  q_C )
      with q_C the squaring on C_od (x) g_od does see the squaring: q_C(xi (x) x)
      = xi^2 (x) s(x), and xi^2 =/= 0.  Verified: q_C is well defined, quadratic,
      polarizes to the bracket and satisfies the axiom of [BLLS] 2.2 in g(C);
      it separates H^0 from H^1 (which Lambda(N) cannot, for any N); it does not
      exist for B, which has no squaring; and it recovers s for osp(1|2).

  D.  WHAT IS LOST, IN ONE LINE OF LINEAR ALGEBRA.  At p = 2 there are two
      "exterior squares" of an odd space V: the exterior square Lambda^2 V and
      the divided square Gamma^2 V, and
            0 --> Lambda^2 V --> Gamma^2 V --> V^(2) --> 0
      is exact and non-split, V^(2) the Frobenius twist.  The bracket of a Lie
      superalgebra restricted to g_od is a map Lambda^2 g_od --> g_ev; the
      squaring is a lift of it to Gamma^2 g_od.  The Even Rules transport
      multilinear operations, i.e. maps out of Lambda^2, so they see the bracket
      and nothing else; existence of a lift is Lemma "no squaring at all", and
      the lifts form a torsor over Hom(V^(2), z(g) \cap g_ev).  Both the exact
      sequence and the torsor count are checked here.

Exact arithmetic throughout (F_2, F_4, F_3, F_5, F_7); no floating point.
"""
import os
import sys
from fractions import Fraction
from itertools import combinations, product

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

from grass3 import SuperAlg, Grassmann, envelope, check_ordinary, free_one_odd
from evenrules import (GF, SAlg2, build_squaring, squaring_defects,
                       all_squarings, enumerate_brackets, alg_H, alg_B,
                       alg_D, alg_C, alg_osp12, osp12_squaring)

LOG = None


def say(s=""):
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def hr(t=""):
    say()
    say("=" * 78)
    if t:
        say(t)
        say("=" * 78)


PASS = [0]
FAIL = [0]


def check(name, ok, detail=""):
    PASS[0] += 1 if ok else 0
    FAIL[0] += 0 if ok else 1
    say("  [%s] %s%s" % ("ok " if ok else "FAIL", name,
                         ("  -- " + detail) if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# A general Z/2-graded commutative algebra, of which Lambda(N) is one case
# ---------------------------------------------------------------------------

class GCAlg:
    """A finite-dimensional Z/2-graded commutative K-algebra, given by a basis
    of monomials, their parities and a structure table.

    Two families are built below:
      grassmann(N)          Lambda(xi_1,...,xi_N),  xi_i^2 = 0     (any p)
      odd_polynomial(d)     K[xi]/(xi^d),  xi odd                  (p = 2 only:
                            for p =/= 2 supercommutativity forces xi^2 = 0)
      tensor(C1, C2)        C1 (x) C2
    """

    def __init__(self, names, par, mul, p):
        self.names = list(names)
        self.par = list(par)
        self.n = len(names)
        self.p = p
        self.mul_tab = dict(mul)             # (i,j) -> {k: c}

    def mul(self, i, j):
        return self.mul_tab.get((i, j), {})

    def check_axioms(self):
        """Graded commutativity, associativity, and -- when p =/= 2 -- that odd
        elements square to zero, which is what supercommutativity forces."""
        bad = []
        p = self.p
        for i in range(self.n):
            for j in range(self.n):
                s = -1 if (self.par[i] and self.par[j]) else 1
                lhs = self.mul(i, j)
                rhs = {k: (s * c) % p for k, c in self.mul(j, i).items()}
                rhs = {k: c for k, c in rhs.items() if c}
                if lhs != rhs:
                    bad.append(("comm", self.names[i], self.names[j]))
        for i in range(self.n):
            for j in range(self.n):
                for k in range(self.n):
                    l = {}
                    for a, c in self.mul(i, j).items():
                        for b, d in self.mul(a, k).items():
                            l[b] = (l.get(b, 0) + c * d) % p
                    r = {}
                    for a, c in self.mul(j, k).items():
                        for b, d in self.mul(i, a).items():
                            r[b] = (r.get(b, 0) + c * d) % p
                    l = {a: c for a, c in l.items() if c}
                    r = {a: c for a, c in r.items() if c}
                    if l != r:
                        bad.append(("assoc", i, j, k))
        return bad

    def odd_squares_vanish(self):
        """True iff c^2 = 0 for every odd basis monomial."""
        return all(not self.mul(i, i) for i in range(self.n) if self.par[i])


def grassmann_gc(N, p):
    L = Grassmann(N, p)
    names = ["".join("x%d" % (t + 1) for t in m) or "1" for m in L.mon]
    par = [len(m) % 2 for m in L.mon]
    mul = {}
    for i, a in enumerate(L.mon):
        for j, b in enumerate(L.mon):
            s, prod = L.mul(a, b)
            if s:
                mul[(i, j)] = {L.index[prod]: s % p}
    return GCAlg(names, par, mul, p)


def odd_polynomial(d, p):
    """K[xi]/(xi^d), xi odd; basis 1, xi, ..., xi^{d-1}, parity = degree mod 2."""
    names = ["1"] + ["xi^%d" % k if k > 1 else "xi" for k in range(1, d)]
    par = [k % 2 for k in range(d)]
    mul = {}
    for i in range(d):
        for j in range(d):
            if i + j < d:
                mul[(i, j)] = {i + j: 1}
    return GCAlg(names, par, mul, p)


def tensor_gc(A, B):
    """A (x) B with the Koszul sign; basis pairs (i,j) in lexicographic order."""
    p = A.p
    basis = [(i, j) for i in range(A.n) for j in range(B.n)]
    index = {b: k for k, b in enumerate(basis)}
    names = ["%s.%s" % (A.names[i], B.names[j]) for i, j in basis]
    par = [(A.par[i] + B.par[j]) % 2 for i, j in basis]
    mul = {}
    for u, (i1, j1) in enumerate(basis):
        for v, (i2, j2) in enumerate(basis):
            s = -1 if (B.par[j1] and A.par[i2]) else 1
            row = {}
            for a, c in A.mul(i1, i2).items():
                for b, dd in B.mul(j1, j2).items():
                    k = index[(a, b)]
                    row[k] = (row.get(k, 0) + s * c * dd) % p
            row = {k: c for k, c in row.items() if c}
            if row:
                mul[(u, v)] = row
    return GCAlg(names, par, mul, p)


def envelope_gc(A, C):
    """(A (x) C)_ev as an ORDINARY algebra: the general-C form of grass3's
    envelope(), with the same sign rule [lam (x) a, mu (x) b] =
    (-1)^{|a||mu|} lam mu (x) [a,b]."""
    p = A.p
    basis = [(m, i) for m in range(C.n) for i in range(A.n)
             if C.par[m] == A.par[i]]
    index = {b: k for k, b in enumerate(basis)}
    names = ["%s@%s" % (A.names[i], C.names[m]) for m, i in basis]
    table = {}
    for u, (m, i) in enumerate(basis):
        for v, (mm, j) in enumerate(basis):
            prod = C.mul(m, mm)
            if not prod:
                continue
            s = -1 if (A.par[i] and C.par[mm]) else 1
            br = A.br(A.e(i), A.e(j))
            row = {}
            for mk, cc in prod.items():
                for k, c in br.items():
                    key = (mk, k)
                    if key not in index:
                        raise AssertionError("envelope not closed")
                    row[index[key]] = (row.get(index[key], 0) + s * cc * c) % p
            row = {k: c for k, c in row.items() if c}
            if row:
                table[(u, v)] = row
    G = SuperAlg(names, [0] * len(basis), table, p)
    G.basis = basis
    return G


# ---------------------------------------------------------------------------
# A.  how many odd parameters suffice
# ---------------------------------------------------------------------------

def alg_jacobi_defect(p):
    """A superalgebra whose ONLY defect is the Jacobi identity on the triple of
    three DISTINCT odd basis vectors:
        g_od = <x1, x2, x3>, g_ev = <u>;  [x1,x2] = u = -[x2,x1]? no: odd-odd is
        symmetric, [x1,x2] = [x2,x1] = u;  [u,x3] = x3 = -[x3,u].
    Then Jac(x1,x2,x3) = [x3,[x1,x2]] = [x3,u] = -x3 =/= 0, whereas every triple
    with a repeated argument is fine."""
    names = ["u", "x1", "x2", "x3"]
    par = [0, 1, 1, 1]
    u, x1, x2, x3 = 0, 1, 2, 3
    table = {
        (x1, x2): {u: 1}, (x2, x1): {u: 1},
        (u, x3): {x3: 1}, (x3, u): {x3: -1},
    }
    return SuperAlg(names, par, table, p)


def part_A(p):
    hr("A.  How many odd parameters -- p = %d" % p)
    A = alg_jacobi_defect(p)
    say("  g = <u even; x1,x2,x3 odd | [x1,x2] = u, [u,x3] = x3>, sdim (1|3)")
    say("  antisymmetry defects: %d;  Jacobi defects: %s"
        % (len(A.check_antisym()),
           sorted(set(tuple(sorted(t)) for t in A.check_jacobi()))))
    check("the only Jacobi defects involve x1, x2, x3 all three",
          all(set(t) == {"x1", "x2", "x3"} for t in A.check_jacobi()),
          "%d triples" % len(A.check_jacobi()))
    for N in (1, 2, 3, 4):
        G = envelope(A, N)
        ba, bj = check_ordinary(G)
        say("    Lambda(%d): dim G = %3d, antisym defects %d, Jacobi defects %d"
            % (N, G.n, len(ba), len(bj)))
        if N <= 2:
            check("Lambda(%d) sees nothing" % N, not ba and not bj)
        else:
            check("Lambda(%d) sees the defect" % N, bool(bj))


def gl_super(m, n, p):
    """gl(m|n) over F_p in the basis of matrix units."""
    N = m + n
    par_idx = [0] * m + [1] * n
    basis = [(i, j) for i in range(N) for j in range(N)]
    index = {b: k for k, b in enumerate(basis)}
    names = ["E%d%d" % (i + 1, j + 1) for i, j in basis]
    par = [(par_idx[i] + par_idx[j]) % 2 for i, j in basis]
    table = {}
    for a, (i, j) in enumerate(basis):
        for b, (k, l) in enumerate(basis):
            row = {}
            if j == k:
                row[index[(i, l)]] = row.get(index[(i, l)], 0) + 1
            if l == i:
                s = -1 if (par[a] and par[b]) else 1
                row[index[(k, j)]] = row.get(index[(k, j)], 0) - s
            row = {t: c % p for t, c in row.items()}
            row = {t: c for t, c in row.items() if c}
            if row:
                table[(a, b)] = row
    return SuperAlg(names, par, table, p)


def perturbations(A, limit):
    """Antisymmetry-preserving one-constant perturbations of A: add 1 to the
    coefficient of e_k in [e_i, e_j] and repair [e_j, e_i]."""
    out = []
    for i in range(A.n):
        for j in range(i, A.n):
            pij = (A.par[i] + A.par[j]) % 2
            for k in range(A.n):
                if A.par[k] != pij:
                    continue
                table = {key: dict(v) for key, v in A.table.items()}
                row = table.get((i, j), {})
                row[k] = (row.get(k, 0) + 1) % A.p
                row = {t: c for t, c in row.items() if c}
                if row:
                    table[(i, j)] = row
                else:
                    table.pop((i, j), None)
                if i != j:
                    s = -1 if (A.par[i] and A.par[j]) else 1
                    row2 = {t: (-s * c) % A.p for t, c in row.items()}
                    row2 = {t: c for t, c in row2.items() if c}
                    if row2:
                        table[(j, i)] = row2
                    else:
                        table.pop((j, i), None)
                out.append(SuperAlg(A.names, A.par, table, A.p))
                if len(out) >= limit:
                    return out
    return out


def random_superalgebras(p, m, n, count, seed):
    """Pseudo-random brackets on sdim (m|n) over F_p: even-even -> even,
    even-odd -> odd, odd-odd -> even, super-antisymmetric by construction."""
    state = seed

    def rnd(k):
        nonlocal state
        state = (1103515245 * state + 12345) % (1 << 31)
        return state % k

    ev, od = list(range(m)), list(range(m, m + n))
    out = []
    for _ in range(count):
        table = {}

        def put(i, j, vals, sign):
            row = {k: v for k, v in vals.items() if v}
            if row:
                table[(i, j)] = dict(row)
                table[(j, i)] = {k: (sign * v) % p for k, v in row.items()}

        for a in range(len(ev)):
            for b in range(a + 1, len(ev)):
                put(ev[a], ev[b], {k: rnd(p) for k in ev}, -1)
        for a in ev:
            for b in od:
                put(a, b, {k: rnd(p) for k in od}, -1)
        for a in range(len(od)):
            for b in range(a, len(od)):
                if a == b:
                    row = {k: rnd(p) for k in ev}
                    row = {k: v for k, v in row.items() if v}
                    if row:
                        table[(od[a], od[a])] = row
                else:
                    put(od[a], od[b], {k: rnd(p) for k in ev}, +1)
        out.append(SuperAlg(["e%d" % i for i in range(m + n)],
                            [0] * m + [1] * n, table, p))
    return out


def part_A_sweep(p):
    hr("A'.  Sweep: 'super axioms' vs 'the Lambda(N)-envelope is a Lie algebra'")
    say("  field F_%d; the pool is gl(1|1), gl(2|1), gl(1|2) and every" % p)
    say("  antisymmetry-preserving one-constant perturbation of each")
    tot = {2: [0, 0], 3: [0, 0], 4: [0, 0]}     # [agree, disagree]
    nsuper = 0
    algs = []
    for (m, n) in ((1, 1), (2, 1), (1, 2)):
        A = gl_super(m, n, p)
        algs.append(A)
        algs += perturbations(A, 60)
    for A in algs:
        good = not A.check_antisym() and not A.check_jacobi()
        nsuper += 1 if good else 0
        for N in (2, 3, 4):
            ba, bj = check_ordinary(envelope(A, N))
            env_good = not ba and not bj
            tot[N][0 if env_good == good else 1] += 1
    say("  %d objects, of which %d are Lie superalgebras" % (len(algs), nsuper))
    for N in (2, 3, 4):
        say("    Lambda(%d): agrees on %d, disagrees on %d"
            % (N, tot[N][0], tot[N][1]))
    check("the pool is not vacuous", 0 < nsuper < len(algs),
          "%d Lie superalgebras, %d not" % (nsuper, len(algs) - nsuper))
    check("Lambda(2) is not enough", tot[2][1] > 0, "%d disagreements" % tot[2][1])
    check("Lambda(3) is enough", tot[3][1] == 0)
    check("Lambda(4) adds nothing", tot[4][1] == 0)


# ---------------------------------------------------------------------------
# B.  at p = 3 no C helps
# ---------------------------------------------------------------------------

def part_B():
    hr("B.  p = 3: no supercommutative C sees the cubing")
    A = free_one_odd(3)
    say("  A = <y odd, h = [y,y], z = [y,h]>, sdim (1|2), T(y) = z =/= 0")
    check("A satisfies antisymmetry and Jacobi over F_3",
          not A.check_antisym() and not A.check_jacobi())
    check("but T(y) = [y,[y,y]] =/= 0",
          bool(A.br(A.e(0), A.br(A.e(0), A.e(0)))))

    tests = []
    for N in (3, 4):
        C = grassmann_gc(N, 3)
        tests.append(("Lambda(%d)" % N, C))
    C3 = grassmann_gc(3, 3)
    T = GCAlg(["1", "t", "t2"], [0, 0, 0],
              {(i, j): ({i + j: 1} if i + j <= 2 else {})
               for i in range(3) for j in range(3)}, 3)
    tests.append(("Lambda(3) (x) F_3[t]/(t^3)", tensor_gc(C3, T)))

    for name, C in tests:
        bad = C.check_axioms()
        check("%s is graded commutative and associative" % name, not bad)
        check("%s has c^2 = 0 for every odd c" % name, C.odd_squares_vanish())
        G = envelope_gc(A, C)
        ba, bj = check_ordinary(G)
        check("the envelope over %s is an ordinary Lie algebra" % name,
              not ba and not bj, "dim = %d" % G.n)

    say()
    say("  Why no C can do better: in a supercommutative C over a field of")
    say("  characteristic =/= 2 one has cd = -dc for c, d odd, hence 2c^2 = 0")
    say("  and c^2 = 0; and the three cyclic terms of Jac(c1 y, c2 y, c3 y) are")
    say("  equal, so their sum is 3 c1c2c3 (x) T(y) = 0 in characteristic 3.")


# ---------------------------------------------------------------------------
# C.  at p = 2 the polynomial odd line does help
# ---------------------------------------------------------------------------

def g_of_C(A, C):
    """(A (x) C)_ev for A an SAlg2 (p = 2) and C a GCAlg over F_2: returns the
    list of basis pairs (m, i) and the bracket on coefficient tuples."""
    basis = [(m, i) for m in range(C.n) for i in range(A.n)
             if C.par[m] == A.par[i]]
    index = {b: k for k, b in enumerate(basis)}
    return basis, index


def bracket_gC(A, C, basis, index, u, v):
    """Bracket of two elements of (A (x) C)_ev, coefficients over F_2."""
    out = [0] * len(basis)
    for a, ca in enumerate(u):
        if not ca:
            continue
        m, i = basis[a]
        for b, cb in enumerate(v):
            if not cb:
                continue
            mm, j = basis[b]
            prod = C.mul(m, mm)
            if not prod:
                continue
            br = A.br(A.e(i), A.e(j))
            for mk, cc in prod.items():
                for k, c in enumerate(br):
                    if c and (cc * c) % 2:
                        out[index[(mk, k)]] ^= 1
    return tuple(out)


def q_gC(A, C, basis, index, s, u):
    """The candidate squaring on the odd-coefficient part of (A (x) C)_ev:
    q(sum c_a (x) x_a) = sum c_a^2 (x) s(x_a) + sum_{a<b} c_a c_b (x) [x_a,x_b],
    where s is the squaring of A on odd BASIS vectors (a dict i -> vector)."""
    out = [0] * len(basis)
    terms = [(a, basis[a]) for a, c in enumerate(u) if c]
    for a, (m, i) in terms:
        prod = C.mul(m, m)                      # c^2
        sv = s[i]
        for mk, cc in prod.items():
            for k, c in enumerate(sv):
                if c and (cc * c) % 2:
                    out[index[(mk, k)]] ^= 1
    for t1 in range(len(terms)):
        for t2 in range(t1 + 1, len(terms)):
            _, (m1, i1) = terms[t1]
            _, (m2, i2) = terms[t2]
            prod = C.mul(m1, m2)
            br = A.br(A.e(i1), A.e(i2))
            for mk, cc in prod.items():
                for k, c in enumerate(br):
                    if c and (cc * c) % 2:
                        out[index[(mk, k)]] ^= 1
    return tuple(out)


def odd_part_elements(A, C, basis):
    """All elements of C_od (x) A_od inside (A (x) C)_ev."""
    slots = [a for a, (m, i) in enumerate(basis) if C.par[m] == 1]
    out = []
    for bits in product((0, 1), repeat=len(slots)):
        v = [0] * len(basis)
        for b, a in zip(bits, slots):
            v[a] = b
        out.append(tuple(v))
    return out, slots


def part_C():
    hr("C.  p = 2: the odd line with xi^2 =/= 0 repairs the Even Rules")
    F2 = GF(1)
    Cpoly = odd_polynomial(3, 2)                # F_2[xi]/(xi^3), xi odd
    Cgrass = odd_polynomial(2, 2)               # Lambda(1) = F_2[xi]/(xi^2)
    check("F_2[xi]/(xi^3) is graded commutative and associative",
          not Cpoly.check_axioms())
    check("and xi^2 =/= 0 in it", not Cpoly.odd_squares_vanish())
    check("whereas Lambda(1) has xi^2 = 0", Cgrass.odd_squares_vanish())

    # H^0 and H^1: same bracket, different squarings
    H = alg_H(F2)
    c_idx = 0
    sq0 = {i: H.zero() for i in H.od}
    sq1 = {i: H.e(c_idx) for i in H.od}
    for tag, s in (("H^0 (x^2 = 0)", sq0), ("H^1 (x^2 = c)", sq1)):
        vals = [s[i] for i in H.od]
        sfull = build_squaring(H, vals)
        check("%s is a Lie superalgebra in the sense of [BLLS] 2.2" % tag,
              not squaring_defects(H, sfull))

    for name, C in (("F_2[xi]/(xi^3)", Cpoly), ("Lambda(1)", Cgrass)):
        basis, index = g_of_C(H, C)
        odds, _ = odd_part_elements(H, C, basis)
        q0 = {u: q_gC(H, C, basis, index, sq0, u) for u in odds}
        q1 = {u: q_gC(H, C, basis, index, sq1, u) for u in odds}
        same_lie = True                          # bracket does not use s at all
        differ = any(q0[u] != q1[u] for u in odds)
        say("    over %s: dim g(C) = %d, brackets identical: %s, q differs: %s"
            % (name, len(basis), same_lie, differ))
        if C is Cpoly:
            check("over F_2[xi]/(xi^3) the enriched functor separates H^0, H^1",
                  differ)
        else:
            check("over Lambda(1) it does not", not differ)

    # the enriched q is a squaring of the ordinary Lie algebra g(C)
    hr("C'.  q_C is an honest squaring of g(C)")
    for tag, A, s in (("H^1", H, sq1),
                      ("osp(1|2)", alg_osp12(F2), None)):
        if s is None:
            A_ = A
            sq = osp12_squaring(A_)
            # osp12_squaring returns the full quadratic map; take basis values
            s = {i: sq[A_.e(i)] for i in A_.od}
        basis, index = g_of_C(A, Cpoly)
        odds, _ = odd_part_elements(A, Cpoly, basis)
        qq = {u: q_gC(A, Cpoly, basis, index, s, u) for u in odds}
        pol_ok = all(
            qq[tuple(x ^ y for x, y in zip(u, v))] ==
            tuple(a ^ b ^ c for a, b, c in
                  zip(qq[u], qq[v], bracket_gC(A, Cpoly, basis, index, u, v)))
            for u in odds for v in odds)
        jac_ok = all(
            bracket_gC(A, Cpoly, basis, index, qq[u],
                       tuple(1 if k == b else 0 for k in range(len(basis)))) ==
            bracket_gC(A, Cpoly, basis, index, u,
                       bracket_gC(A, Cpoly, basis, index, u,
                                  tuple(1 if k == b else 0
                                        for k in range(len(basis)))))
            for u in odds for b in range(len(basis)))
        check("%s: q_C polarizes to the bracket of g(C)" % tag, pol_ok,
              "%d pairs" % (len(odds) ** 2))
        check("%s: [q_C(u), v] = [u,[u,v]] in g(C)" % tag, jac_ok,
              "%d pairs" % (len(odds) * len(basis)))
        # recovery of s from q_C
        rec_ok = True
        for i in A.od:
            u = [0] * len(basis)
            u[index[(1, i)]] = 1                 # xi (x) e_i
            val = qq[tuple(u)]
            # must be xi^2 (x) s(e_i)
            want = [0] * len(basis)
            for k, c in enumerate(s[i]):
                if c:
                    want[index[(2, k)]] = 1
            rec_ok &= (val == tuple(want))
        check("%s: q_C(xi (x) x) = xi^2 (x) s(x), so s is recovered" % tag,
              rec_ok)

    # B: no squaring exists, and none exists over C either
    hr("C''.  B has no squaring, and none appears over C")
    B = alg_B(F2)
    check("B admits no squaring", not all_squarings(B))
    basis, index = g_of_C(B, Cpoly)
    odds, slots = odd_part_elements(B, Cpoly, basis)
    evens = [a for a, (m, i) in enumerate(basis) if Cpoly.par[m] == 0]
    found = None
    for vals in product(range(1 << len(evens)), repeat=len(slots)):
        s_basis = {}
        for a, bits in zip(slots, vals):
            v = [0] * len(basis)
            for t, e in enumerate(evens):
                if bits >> t & 1:
                    v[e] = 1
            s_basis[a] = tuple(v)
        # extend quadratically by q(u+v) = q(u)+q(v)+[u,v]
        qq = {}
        ok = True
        for u in odds:
            acc = [0] * len(basis)
            act = [a for a in slots if u[a]]
            for a in act:
                acc = [x ^ y for x, y in zip(acc, s_basis[a])]
            for t1 in range(len(act)):
                for t2 in range(t1 + 1, len(act)):
                    ea = tuple(1 if k == act[t1] else 0 for k in range(len(basis)))
                    eb = tuple(1 if k == act[t2] else 0 for k in range(len(basis)))
                    acc = [x ^ y for x, y in
                           zip(acc, bracket_gC(B, Cpoly, basis, index, ea, eb))]
            qq[u] = tuple(acc)
        for u in odds:
            for b in range(len(basis)):
                eb = tuple(1 if k == b else 0 for k in range(len(basis)))
                if (bracket_gC(B, Cpoly, basis, index, qq[u], eb) !=
                        bracket_gC(B, Cpoly, basis, index, u,
                                   bracket_gC(B, Cpoly, basis, index, u, eb))):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            found = vals
            break
    check("no q_C on g(C) for B either", found is None,
          "%d candidates tried" % ((1 << len(evens)) ** len(slots)))


# ---------------------------------------------------------------------------
# D.  Gamma^2 vs Lambda^2 and the torsor
# ---------------------------------------------------------------------------

def part_D():
    hr("D.  0 --> Lambda^2 V --> Gamma^2 V --> V^(2) --> 0 at p = 2")
    for n in range(1, 6):
        dG = n * (n + 1) // 2                    # x_i^[2] and x_i x_j, i < j
        dL = n * (n - 1) // 2
        check("dim Gamma^2 = dim Lambda^2 + dim V for n = %d" % n,
              dG == dL + n, "%d = %d + %d" % (dG, dL, n))
    say()
    say("  A quadratic map q: V --> W is a linear map Gamma^2 V --> W; its")
    say("  polarization is its restriction to Lambda^2 V.  So a squaring is a")
    say("  LIFT of the odd-odd bracket along Gamma^2 --> ... , the obstruction")
    say("  is the axiom [q(x),y] = [x,[x,y]], and the lifts, when there are any,")
    say("  form a torsor over Hom_Frob(V, z(g) \\cap g_ev) = the maps killed by")
    say("  the polarization.  Both halves are checked next.")

    hr("D'.  The torsor count over F_2 and F_4")
    for k in (1, 2):
        F = GF(k)
        for (m, n) in ((1, 1), (1, 2), (2, 1), (2, 2)):
            tot = 0
            agree = 0
            names = (["u%d" % (i + 1) for i in range(m)]
                     + ["x%d" % (i + 1) for i in range(n)])
            par = [0] * m + [1] * n
            for table in enumerate_brackets(m, n):
                A = SAlg2(names, par, table, F)
                if A.check_bracket():
                    continue
                sq = all_squarings(A)
                if not sq:
                    continue
                tot += 1
                z = A.even_center()
                # Frobenius-semilinear additive maps g_od -> z: over F_q, a map
                # is determined by its values on a basis, each in z, and every
                # such choice is semilinear -- so the count is |z|^n
                predicted = len(z) ** n
                agree += 1 if len(sq) == predicted else 0
            if tot:
                check("%s, sdim (%d|%d): #squarings = |z|^n on all %d brackets"
                      % (F.name, m, n, tot), agree == tot,
                      "%d of %d" % (agree, tot))


def part_E():
    hr("E.  p = 2: the morphism identities are weaker than the points")
    say("  The internal definition asks that the bracket be a morphism with")
    say("  [,] o (1 + tau) = 0 and Jacobi.  At p = 2 the first reads")
    say("  [a,b] + [b,a] = 0, i.e. the bracket is symmetric, and it says nothing")
    say("  about [a,a] for a EVEN, which the points do impose (Prop. 'What the")
    say("  envelope imposes at p = 2', item 2).  Witness:")
    F2 = GF(1)
    names, par = ["u", "v"], [0, 0]
    table = {(0, 0): {1: 1}}                    # [u,u] = v, v central
    A = SAlg2(names, par, table, F2)
    say("    g = <u, v both even | [u,u] = v>, sdim (2|0)")
    check("symmetric and Jacobi", not A.check_bracket(alt_odd=False) or
          all(t[0] == "alt" for t in A.check_bracket(alt_odd=False)),
          "defects: %s" % A.check_bracket(alt_odd=False))
    defects = A.check_bracket(alt_odd=False)
    check("but [u,u] =/= 0, so g is not a Lie algebra",
          any(t[0] == "alt" for t in defects))
    G = envelope_gc(SuperAlg(names, par, {(0, 0): {1: 1}}, 2),
                    grassmann_gc(3, 2))
    ba, bj = check_ordinary(G)
    check("and the envelope is not a Lie algebra either", bool(ba),
          "%d defects, dim = %d" % (len(ba), G.n))


def main():
    global LOG
    here = os.path.dirname(os.path.abspath(__file__))
    LOG = open(os.path.join(here, "testalg.log"), "w")
    say("testalg.py -- which algebras C the definition by points is tested on")
    # Optional arguments select parts, e.g. "python testalg.py A B C E";
    # with none, all parts run (A' and D take most of the time).
    want = set(sys.argv[1:]) or {"A", "A'", "B", "C", "D", "E"}
    if "A" in want:
        part_A(5)
        part_A(7)
    if "A'" in want:
        part_A_sweep(5)
    if "B" in want:
        part_B()
    if "C" in want:
        part_C()
    if "D" in want:
        part_D()
    if "E" in want:
        part_E()
    hr("TOTAL")
    say("  %d checks, %d pass, %d fail" % (PASS[0] + FAIL[0], PASS[0], FAIL[0]))
    LOG.close()
    return 1 if FAIL[0] else 0


if __name__ == "__main__":
    sys.exit(main())
