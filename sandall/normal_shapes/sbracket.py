#!/usr/bin/env python3
"""The bracket of each standard shape: the dual bivector, read off the machine.

A classification of normal shapes is worth little without the bracket the
shape induces, so for each of the three cases we compute

    iota_{H_f} omega = df,      {f, g} := H_f(g),

from the definition, and match it against a closed formula.  The formula is
not guessed: every candidate is tested on ALL pairs of basis monomials.
"""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S

LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "sbracket.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def partial(A, a, f):
    """d/dg_a applied to f in O(m|n;u), a = 0..m+n-1 (x's then xi's)."""
    imgs = [{}] * A.N
    idx = a if a < A.m else A.ixi + (a - A.m)
    imgs = list(imgs)
    imgs[idx] = A.one()
    par = 0 if a < A.m else 1
    return A.derivation(imgs, par)(f)


class Ham:
    """iota_{H_f} omega = df, with the linear system built once."""

    def __init__(self, A, om):
        self.A, self.om = A, om
        B1 = A.basis(1)
        self.i1 = {mm: i for i, mm in enumerate(B1)}
        self.fields = S.vfield_basis(A)
        rows = [S.vec(A, A.iota(D)(om), self.i1) for D in self.fields]
        self.T = [[rows[k][i] for k in range(len(rows))]
                  for i in range(len(B1))]
        self.mons = A.fun_basis()
        self.idx = {mm: i for i, mm in enumerate(self.mons)}
        self.cache = {}

    def field(self, f):
        from exactlin import solve as _solve
        A = self.A
        v = _solve(self.T, S.vec(A, A.d()(f), self.i1), A.p)
        assert v is not None, "omega degenerate: no Hamiltonian field"
        acc = [0] * ((A.m + A.n) * len(self.mons))
        pars = set()
        for k, c in enumerate(v):
            c = int(c) % A.p
            if not c:
                continue
            pars.add(self.fields[k][2])
            w = S.vf_vec(A, self.fields[k], self.idx)
            for t in range(len(acc)):
                acc[t] = (acc[t] + c * w[t]) % A.p
        par = pars.pop() if len(pars) == 1 else 0
        return S.vec_vf(A, acc, self.mons, par)

    def of(self, mon):
        if mon not in self.cache:
            self.cache[mon] = self.field({mon: 1})
        return self.cache[mon]


def apply_field(A, D, g):
    fs, gs, par = D
    imgs = [{}] * A.N
    imgs = list(imgs)
    for i in range(A.m):
        imgs[A.ix + i] = fs[i]
    for j in range(A.n):
        imgs[A.ixi + j] = gs[j]
    return A.derivation(imgs, par)(g)


def test(A, om, formula, label):
    """(test.ham must be set to a Ham(A, om) before calling)"""
    """compare {f,g} against formula(f,g) on all pairs of basis monomials."""
    B = A.fun_basis()
    bad = 0
    H = formula.__self__ if hasattr(formula, "__self__") else None
    ham = test.ham
    for a in B:
        f = {a: 1}
        D = ham.of(a)
        for b in B:
            g = {b: 1}
            lhs = apply_field(A, D, g)
            rhs = formula(f, g)
            if lhs != rhs:
                bad += 1
    say(f"      {label}: {bad} mismatches on {len(B)**2} pairs "
        f"{'-- ok' if bad == 0 else ''}")
    return bad


def par_of(A, f):
    pars = set(A.parity(mm) for mm in f)
    return pars.pop() if len(pars) == 1 else None


# ------------------------------------------------------------------ the cases
say("The brackets of the standard shapes")
bad = 0

say("")
say("(1) p > 2, omega_0 = sum_{i<=k} dx_i dx_{k+i} + sum_j (dxi_j)^{(2)}  (even)")
say("    the candidates: {f,g} = eps_x * [sign]^{p(f)} sum_i (d_{k+i} f d_i g")
say("                            - d_i f d_{k+i} g) + eps_e * [sign]^{p(f)}")
say("                            sum_j d_{xi_j} f d_{xi_j} g,  all 16 tried")
for (m, n, p, u) in [(2, 1, 3, (1, 1)), (2, 2, 3, (1, 1)), (2, 1, 5, (1, 1)),
                     (4, 1, 3, (1, 1, 1, 1)), (2, 3, 3, (1, 1))]:
    A = S.Om(m, n, p, u, dh=2, dxh=2)
    k = m // 2
    om = {}
    for i in range(k):
        om = A.add(om, A.mul(A.dx(i), A.dx(k + i)))
    for j in range(n):
        om = A.add(om, A.dxi(j, 2))
    say(f"    O({m}|{n};{u}) p={p}:")
    test.ham = Ham(A, om)
    found = []
    for ex in (1, -1):
        for tx in (0, 1):
            for ee in (1, -1):
                for te in (0, 1):
                    def formula(f, g, ex=ex, tx=tx, ee=ee, te=te):
                        pf = par_of(A, f) or 0
                        sx = ex * ((-1) ** (pf * tx))
                        se = ee * ((-1) ** (pf * te))
                        out = {}
                        for i in range(k):
                            out = A.add(out,
                                        A.smul(sx, A.mul(partial(A, k + i, f),
                                                         partial(A, i, g))),
                                        A.smul(-sx, A.mul(partial(A, i, f),
                                                          partial(A, k + i, g))))
                        for j in range(n):
                            out = A.add(out, A.smul(se,
                                                    A.mul(partial(A, m + j, f),
                                                          partial(A, m + j, g))))
                        return out
                    B = A.fun_basis()
                    nb = 0
                    for a in B:
                        D = test.ham.of(a)
                        for b in B:
                            if apply_field(A, D, {b: 1}) != formula({a: 1}, {b: 1}):
                                nb += 1
                                break
                        if nb:
                            break
                    if nb == 0:
                        b = test(A, om, formula,
                                 f"eps_x={ex:+d} (-1)^{{p(f)}}^{tx}, "
                                 f"eps_xi={ee:+d} (-1)^{{p(f)}}^{te}")
                        if b == 0:
                            found.append((ex, tx, ee, te))
    if not found:
        bad += 1
        say("      NO candidate matched")
    else:
        say(f"      matched: {found}")

say("")
say("(2) p = 2, omega_I = sum_a (dg_a)^{(2)}  (even; no analog for p > 2)")
say("    candidate: {f,g} = sum_a d_a f d_a g,  a running over ALL m+n")
for (m, n, u) in [(1, 1, (1,)), (2, 1, (1, 1)), (0, 2, ()), (2, 2, (1, 1)),
                  (1, 2, (1,)), (1, 1, (2,)), (3, 0, (1, 1, 1))]:
    A = S.Om(m, n, 2, u, dh=2, dxh=2)
    om = {}
    for i in range(m):
        om = A.add(om, A.dx(i, 2))
    for j in range(n):
        om = A.add(om, A.dxi(j, 2))

    def formula(f, g):
        out = {}
        for a in range(m + n):
            out = A.add(out, A.mul(partial(A, a, f), partial(A, a, g)))
        return out
    say(f"    O({m}|{n};{u}) p=2:")
    test.ham = Ham(A, om)
    bad += test(A, om, formula, "sum_a d_a f d_a g")

say("")
say("(3) p = 2, omega_Pi (alternating, m and n even) and the odd shape")
for (m, n, u) in [(2, 0, (1, 1)), (0, 2, ()), (2, 2, (1, 1))]:
    A = S.Om(m, n, 2, u, dh=2, dxh=2)
    om = {}
    for i in range(0, m - 1, 2):
        om = A.add(om, A.mul(A.dx(i), A.dx(i + 1)))
    for j in range(0, n - 1, 2):
        om = A.add(om, A.mul(A.dxi(j), A.dxi(j + 1)))

    def formula(f, g):
        out = {}
        for i in range(0, m - 1, 2):
            out = A.add(out, A.mul(partial(A, i, f), partial(A, i + 1, g)),
                        A.mul(partial(A, i + 1, f), partial(A, i, g)))
        for j in range(0, n - 1, 2):
            out = A.add(out, A.mul(partial(A, m + j, f), partial(A, m + j + 1, g)),
                        A.mul(partial(A, m + j + 1, f), partial(A, m + j, g)))
        return out
    say(f"    O({m}|{n};{u}) p=2, omega_Pi:")
    test.ham = Ham(A, om)
    bad += test(A, om, formula, "the pairwise (Hamiltonian) bracket")

for (n, u) in [(1, (1,)), (2, (1, 1))]:
    A = S.Om(n, n, 2, u, dh=2, dxh=2)
    om = {}
    for i in range(n):
        om = A.add(om, A.mul(A.dx(i), A.dxi(i)))

    def formula(f, g):
        out = {}
        for i in range(n):
            out = A.add(out, A.mul(partial(A, i, f), partial(A, n + i, g)),
                        A.mul(partial(A, n + i, f), partial(A, i, g)))
        return out
    say(f"    O({n}|{n};{u}) p=2, omega odd (the antibracket):")
    test.ham = Ham(A, om)
    bad += test(A, om, formula, "the Buttin antibracket, signs killed by p=2")

say("")
say("TOTAL failures:", bad, " (must be 0)")
LOG.close()
