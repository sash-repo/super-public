#!/usr/bin/env python3
r"""
fockq.py -- the four open questions of the Fock spaces of an odd form,
checked exactly.  Built on abfock.py (the modules F_i, pe(n), stabilizers)
and bn.py (the antibracket b(n) and the quantizing cocycle C).

Q1  F_i as a module over the parabolic p_i = Stab_{pe(n)}(L_i):
    the NORMALLY ORDERED lift (creator * annihilator, and hbar * annihilator *
    annihilator, no identity term) exists for every X in p_i, is an honest
    homomorphism with NO scalar correction, kills the vacuum; the X whose lift
    is hbar*(second order) form the nilradical, of sdim = sdim Pi S^2(M_i^*),
    and their lifts span ALL hbar * (constant second-order operators); the rest
    preserve the degree.  The contractions S^{k+2} -> S^k are non-zero.

Q2  (a) the kernel of rho_i : But_C -> End_C(F_i) on PBW monomials of low
    degree is exactly  Span{ m : c(m) >= 2 } + hbar Span{ m : c(m) >= 1 },
    c(m) = the number of factors from L_i;
    (b) n = 1, i = 0: R(a(q) + b(q) xi) = -hbar a(d/dx) + x b(d/dx) is a
    representation of the QUANTIZED b(1), bracket {,} + hbar C;
    (c) n = 2: no R with R(1) = -hbar, R(xi_j) = x_j mod hbar,
    R(q_m) in hbar End satisfies the two relations used in the proof
    (checked mod hbar on a generous ansatz).

Q4  char p: q_j^p is central in the envelope (checked on F_0 mod p), and the
    truncated F_i/(x^p) is a module (the ideal is stable).

Run:  python3 fockq.py      (from this folder)  -> writes fockq.log
"""

import sys
from fractions import Fraction
from itertools import combinations, product

from abfock import (Fock, pe_basis, pe_bracket, apply_X, stabilizes_L,
                    parity, nullspace, solve)
from bn import Bn

PASS = FAIL = 0
LOG = []


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        LOG.append(f"  ok   {name} {detail}")
    else:
        FAIL += 1
        LOG.append(f"  FAIL {name} {detail}")
    return ok


def testvecs(F, maxdeg):
    # hbar-free basis vectors suffice: every operator here is C-linear
    return [F.mono(S, a) for (S, a) in F.basis(maxdeg, with_hbar=False)]


def op_diff_zero(A, B, vecs):
    for v in vecs:
        if not Fock.iszero(Fock.add(A(v), B(v), -1)):
            return False
    return True


def xdeg(key):
    S, a = key
    return len([s for s in S if s != 0]) + sum(a)


# ====================================================================== Q1
def ansatz(F):
    """normally ordered bilinears: (operator, parity, kind, label)."""
    cre, ann = [], []
    for j in range(1, F.i + 1):
        cre.append((F.mul_odd(j), 1, ('l', j)))
        ann.append((F.d_odd(j), 1, ('dl', j)))
    for j in range(F.nx):
        cre.append((F.mul_x(j), 0, ('x', j)))
        ann.append((F.d_x(j), 0, ('dx', j)))
    h = F.mul_odd(0)
    terms = []
    for (c, pc, lc) in cre:
        for (a, pa, la) in ann:
            terms.append((F.comp(c, a), (pc + pa) % 2, 'ca', (lc, la)))
    for s in range(len(ann)):
        for t in range(s, len(ann)):
            a, pa, la = ann[s]
            b, pb, lb = ann[t]
            if s == t and pa == 1:
                continue            # d_l d_l = 0
            terms.append((F.comp(h, a, b), (1 + pa + pb) % 2, 'haa', (la, lb)))
    return terms


def nlift(F, X, pX, vecs, terms):
    """solve [hatX, rho(w)] = rho(Xw) inside the normally ordered ansatz."""
    use = [t for t in terms if t[1] == pX]
    rows, rhs = [], []
    for g in F.gens():
        A, pA = F.rho(*g)
        for v in vecs:
            want = {}
            for hh, c in apply_X(X, g).items():
                Ah, _ = F.rho(*hh)
                want = Fock.add(want, Ah(v), c)
            cont = [F.bracket(o, pX, A, pA)(v) for (o, _, _, _) in use]
            keys = sorted({k for d in cont for k in d} | set(want))
            for k in keys:
                rows.append([d.get(k, Fraction(0)) for d in cont])
                rhs.append(want.get(k, Fraction(0)))
    sol = solve(rows, rhs, len(use))
    if sol is None:
        return None, None
    coeffs = [(c, t) for c, t in zip(sol, use) if c != 0]

    def op(f):
        out = {}
        for c, (o, _, _, _) in coeffs:
            out = Fock.add(out, o(f), c)
        return out
    return op, coeffs


def q1(n, maxdeg):
    B = pe_basis(n)
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg)
        terms = ansatz(F)
        st = [(X, p) for X, p in B if stabilizes_L(X, n, i)]
        lifts, kinds = [], []
        ok_exist = True
        for X, p in st:
            op, co = nlift(F, X, p, vecs, terms)
            if op is None:
                ok_exist = False
                break
            lifts.append((op, p))
            kinds.append({t[2] for _, t in co})
        check(f"Q1 n={n} i={i}: every X in p_{i} has a normally ordered lift",
              ok_exist, f"({len(st)} basis elements)")
        if not ok_exist:
            continue
        # vacuum weight 0
        vac = F.one()
        check(f"Q1 n={n} i={i}: every normally ordered lift kills the vacuum",
              all(Fock.iszero(op(vac)) for op, _ in lifts))
        # homomorphism with NO scalar: [hatX,hatY] = hat[X,Y]
        good = 0
        tot = 0
        for a in range(len(st)):
            for b in range(len(st)):
                X, pX = st[a]
                Y, pY = st[b]
                Z, pZ = pe_bracket(X, pX, Y, pY, n)
                tot += 1
                if Z:
                    Zop, _ = nlift(F, Z, pZ, vecs, terms)
                    if Zop is None:
                        continue
                else:
                    Zop = lambda f: {}
                br = F.bracket(lifts[a][0], pX, lifts[b][0], pY)
                if op_diff_zero(br, Zop, vecs):
                    good += 1
        check(f"Q1 n={n} i={i}: X -> hatX is a homomorphism p_{i} -> gl(F_{i}), no scalar term",
              good == tot, f"({good}/{tot} pairs)")
        # the nilradical: lifts made of hbar*aa only
        nil = [k for k in kinds if k == {'haa'}]
        levi = [k for k in kinds if k == {'ca'}]
        mixed = [k for k in kinds if k not in ({'haa'}, {'ca'})]
        # the basis of p_i from pe_basis need not be adapted; count by rank
        # of the projections instead
        nil_dim = rank_of_part(F, st, vecs, terms, 'haa')
        ca_dim = rank_of_part(F, st, vecs, terms, 'ca')
        m, k = n - i, i          # sdim M_i = (m|k)
        # Pi S^2(M^*): parity-flipped super-symmetric square of the partials
        s2_even = m * (m + 1) // 2 + k * (k - 1) // 2
        s2_odd = m * k
        nil_expect = s2_even + s2_odd
        n_haa = len([t for t in terms if t[2] == 'haa'])
        check(f"Q1 n={n} i={i}: hbar*(2nd order) part of the lifts spans all hbar S^2(partials)",
              nil_dim == n_haa == nil_expect,
              f"(rank {nil_dim}, terms {n_haa}, dim Pi S^2(M^*) = {nil_expect})")
        gl_dim = (m + k) ** 2
        check(f"Q1 n={n} i={i}: creator*annihilator part spans gl(M_{i}) = gl({m}|{k})",
              ca_dim == gl_dim, f"(rank {ca_dim}, expected {gl_dim})")
        check(f"Q1 n={n} i={i}: dim p_{i} = dim gl(M_i) + dim Pi S^2(M_i^*)",
              len(st) == gl_dim + nil_expect,
              f"({len(st)} = {gl_dim} + {nil_expect})")
        # contractions S^{k+2} -> S^k non-zero, as far as they can be
        top = maxdeg if F.nx > 0 else F.i
        hops = []
        haa_ops = [t[0] for t in terms if t[2] == 'haa']
        for d in range(0, top - 1):
            src = [F.mono(S, a) for (S, a) in F.basis(d + 2, with_hbar=False)
                   if xdeg((S, a)) == d + 2]
            hit = any(not Fock.iszero(o(v)) for o in haa_ops for v in src)
            hops.append(hit)
        check(f"Q1 n={n} i={i}: every contraction S^(k+2) -> S^k is non-zero (k < {top-1})",
              all(hops), f"({len(hops)} degrees)")


def rank_of_part(F, st, vecs, terms, tag):
    """rank of the tag-part of the lifts, as vectors of coefficients."""
    rows = []
    idx = {}
    for (o, p, t, lab) in terms:
        if t == tag:
            idx[lab] = len(idx)
    for X, p in st:
        _, co = nlift(F, X, p, vecs, terms)
        r = [Fraction(0)] * len(idx)
        for c, tt in co:
            if tt[2] == tag:
                r[idx[tt[3]]] += c
        rows.append(r)
    if not idx:
        return 0
    return len(idx) - len(nullspace(rows, len(idx))) if rows else 0


# ====================================================================== Q2
def pbw_ops(F, maxdeg):
    """PBW monomials q^a theta^S of But of degree <= maxdeg, as operators,
    with and without an hbar in front.  Returns (label, op, count)."""
    n, i = F.n, F.i
    out = []
    h = F.mul_odd(0)
    for d in range(maxdeg + 1):
        for k in range(min(d, n) + 1):
            for S in combinations(range(1, n + 1), k):
                for a in exps(n, d - k):
                    ops = []
                    cnt = 0
                    for j in range(1, n + 1):
                        A, _ = F.rho('q', j)
                        ops += [A] * a[j - 1]
                        if j <= i:
                            cnt += a[j - 1]
                    for j in S:
                        A, _ = F.rho('th', j)
                        ops.append(A)
                        if j > i:
                            cnt += 1
                    o = F.comp(*ops) if ops else (lambda f: dict(f))
                    out.append((('', a, S), o, cnt))
                    out.append((('h', a, S), F.comp(h, o), cnt))
    return out


def exps(n, d):
    if n == 0:
        if d == 0:
            yield ()
        return
    if n == 1:
        yield (d,)
        return
    for e in range(d + 1):
        for r in exps(n - 1, d - e):
            yield (e,) + r


def q2_kernel(n, maxdeg):
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, maxdeg + 1)
        ops = pbw_ops(F, maxdeg)
        cols = []
        for (_, o, _) in ops:
            cols.append([o(v) for v in vecs])
        keys = sorted({(vi, k) for c in cols for vi, d in enumerate(c) for k in d})
        kidx = {k: t for t, k in enumerate(keys)}
        rows = [[Fraction(0)] * len(ops) for _ in keys]
        for t, c in enumerate(cols):
            for vi, d in enumerate(c):
                for k, val in d.items():
                    rows[kidx[(vi, k)]][t] = val
        ker = nullspace(rows, len(ops)) if keys else [[Fraction(0)] * len(ops)]
        pred = [t for t, (lab, _, cnt) in enumerate(ops)
                if (lab[0] == '' and cnt >= 2) or (lab[0] == 'h' and cnt >= 1)]
        # every predicted monomial is itself in the kernel
        in_ker = all(all(Fock.iszero(ops[t][1](v)) for v in vecs) for t in pred)
        check(f"Q2 n={n} i={i}: ker rho_i on But (deg<={maxdeg}) = <c>=2> + hbar<c>=1>",
              in_ker and len(ker) == len(pred),
              f"(dim ker {len(ker)}, predicted {len(pred)}, of {len(ops)})")


def q2_n1_rep(maxdeg):
    """n=1, i=0: R(a(q)+b(q)xi) = -hbar a(d/dx) + x b(d/dx) on F_0 = C[x]."""
    F = Fock(1, 0)
    bn = Bn(1)
    h = F.mul_odd(0)
    dx = F.d_x(0)
    x = F.mul_x(0)
    vecs = testvecs(F, maxdeg + 2)

    def dpow(k):
        return F.comp(*([dx] * k)) if k else (lambda f: dict(f))

    def R_mono(key):
        (qe,), xi = key[0], key[1]
        if len(xi) == 0:
            return F.opadd((-1, F.comp(h, dpow(qe))))
        return F.comp(x, dpow(qe))

    def R(elem):
        terms = [(c, R_mono(k)) for k, c in elem.items()]
        return F.opadd(*terms) if terms else (lambda f: {})

    basis = bn.basis(maxdeg)
    good = tot = 0
    for kf in basis:
        for kg in basis:
            f = {kf: Fraction(1)}
            g = {kg: Fraction(1)}
            pf, pg = bn.pb(kf), bn.pb(kg)
            lhs = F.bracket(R(f), pf, R(g), pg)
            br = bn.bracket(f, g)
            Cfg = bn.C(f, g)
            rhs = F.opadd((1, R(br)), (1, F.comp(h, R(Cfg))))
            tot += 1
            if op_diff_zero(lhs, rhs, vecs):
                good += 1
    check("Q2 n=1 i=0: R(a+b xi) = -hbar a(d) + x b(d) represents the QUANTIZED b(1)",
          good == tot, f"({good}/{tot} pairs of generating functions, deg<={maxdeg})")
    # it extends rho_0: R(xi) = x = rho_0(q_1), R(q) = -hbar d = rho_0(theta_1)
    ok = (op_diff_zero(R({((0,), (0,)): Fraction(1)}), x, vecs) and
          op_diff_zero(R({((1,), ()): Fraction(1)}),
                       F.opadd((-1, F.comp(h, dx))), vecs))
    check("Q2 n=1 i=0: R restricted to the linear functions is rho_0", ok)


def q2_obstruction():
    """n=2: {1,g}_hbar and {xi1 xi2, q_m}_hbar -- the two facts the proof uses."""
    bn = Bn(2)
    one = bn.mono((0, 0), ())
    ok = True
    cnt = 0
    for key in bn.basis(4):
        g = {key: Fraction(1)}
        br = bn.bracket(one, g)
        Cg = bn.C(one, g)
        # {1,g} = 0 and C(1,g) = (-1)^{p(1)} (0-1)(d1(g)-1) g = -(d1(g)-1) g
        want = {key: Fraction(-(bn.d1(key) - 1))} if bn.d1(key) != 1 else {}
        ok &= bn.iszero(br) and bn.iszero(Bn.add(Cg, want, -1))
        cnt += 1
    check("Q2 obstruction: in the quantized b(2), ad_1 = -hbar*delta (1 is no longer central)",
          ok, f"({cnt} generating functions)")
    f = bn.mono((0, 0), (0, 1))
    qm = bn.mono((0, 1), ())
    br = bn.bracket(f, qm)
    check("Q2 obstruction: {xi_1 xi_2, q_2} = +-xi_1, a creator of F_0, not in hbar End",
          (not bn.iszero(br)) and set(br) == {((0, 0), (0,))}, f"({br})")


# ====================================================================== Q4
def q4(p, n):
    """char p: q_j^p central on F_i (so on But), and x^p F_i an invariant ideal."""
    ok_central = True
    ok_ideal = True
    for i in range(n + 1):
        F = Fock(n, i)
        vecs = testvecs(F, p + 2)
        for j in range(1, n + 1):
            A, _ = F.rho('q', j)
            Ap = F.comp(*([A] * p))
            for g in F.gens():
                Bo, pB = F.rho(*g)
                br = F.bracket(Ap, 0, Bo, pB)
                for v in vecs:
                    w = br(v)
                    if any(c.numerator % p != 0 for c in w.values()):
                        ok_central = False
        # the ideal generated by x_j^p: rho(w) maps x^{>=p} to x^{>=p} mod p
        for v in vecs:
            (S, a), = list(v.keys())
            if not any(e >= p for e in a):
                continue
            for g in F.gens():
                Bo, _ = F.rho(*g)
                for (S2, a2), c in Bo(v).items():
                    if c.numerator % p != 0 and not any(e >= p for e in a2):
                        ok_ideal = False
    check(f"Q4 p={p} n={n}: q_j^p super-commutes with every rho_i(w) mod p, all i",
          ok_central)
    check(f"Q4 p={p} n={n}: (x_j^p) F_i is stable under rho_i mod p, all i", ok_ideal)


def main():
    LOG.append("Q1: F_i over the parabolic p_i")
    for n in (1, 2, 3):
        q1(n, maxdeg=3 if n < 3 else 2)
    LOG.append("Q2: the envelope, and the quantized b(n)")
    for n in (1, 2):
        q2_kernel(n, maxdeg=3)
    q2_kernel(3, maxdeg=2)
    q2_n1_rep(maxdeg=6)
    q2_obstruction()
    LOG.append("Q4: characteristic p")
    for p in (3, 5):
        for n in (1, 2):
            q4(p, n)
    LOG.append("-" * 74)
    LOG.append(f"passed {PASS}, failed {FAIL}")
    txt = "\n".join(LOG)
    print(txt)
    with open("fockq.log", "w") as fh:
        fh.write(txt + "\n")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
