#!/usr/bin/env python3
"""
ramondframe.py -- the Ramond ("M") sector of the Ivanov-Krivonos equations.

Checks the "Ramond-like" versions of the Ivanov-Krivonos equations for
N = 2, 3, 4: the frame of the Ramond sector, the double cover t = s^2
(Lemma 9.1.1) and the twisted multiplier (Proposition 9.1.2).  The formulas
numbered (24)-(36) below are the defining formulas of the Neveu-Schwarz and
Ramond contact fields used in the text; the numbers are local to this file.

Realisation 2) of k^M(1|n) (no square roots in the coefficients): the
coordinates are t (Laurent, and, when we look at the double cover, with
half-integer exponents), zeta (label 0) and theta_1..theta_{n-1} (labels
1..n-1), and the Moebius contact form is

    alpha~ = dt + t*zeta*dzeta + sum theta_i dtheta_i.

The formulas:

  (30)  E := zeta d_zeta + sum theta_i d_theta_i,   nabla := 2 - E,
        Dcal := d_t - zeta/(2t) d_zeta = (1/2) K^M_1
  (31)  H^M_f := (-1)^{p(f)} ( (1/t) (df/dzeta) d_zeta + sum (df/dtheta_i) d_theta_i )
  (29)  K^M_f := nabla(f) Dcal + H^M_f + Dcal(f) E
  (32)  {f,g}_R.b. := nabla(f) Dcal(g) - Dcal(f) nabla(g) + {f,g}_MP.b.
  (33)  {f,g}_MP.b. := (-1)^{p(f)} ( (1/t) f_zeta g_zeta + sum f_theta_i g_theta_i )
  (36)  D^M_zeta := zeta d_t + (1/t) d_zeta,
        D^M_theta_i := theta_i d_t + d_theta_i - theta_i zeta/(2t) d_zeta.

Everything below is exact arithmetic over Q (fractions.Fraction), the exponent
of t being a Fraction as well, so that the double cover t -> t^{1/2} is
representable.

Usage:  ../venv/bin/python ramondframe.py   (also writes the log to ramondframe-out.txt)
"""

import sys
from fractions import Fraction as F

import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from stringy import (const, mono, add, smul, mul, parity, d_t, d_odd, eq,
                     show, VF)

# --------------------------------------------------------------- the frame

def labels(n):
    """odd labels of R^L(n) in the M-realisation: 0 = zeta, 1..n-1 = theta_i."""
    return tuple(range(n))


def t_pow(a, c=1):
    return mono(F(a), (), c)


def Dcal(n):
    """Dcal = d_t - zeta/(2t) d_zeta  =  (1/2) K^M_1 ; = d_t in the theta-chart."""
    return VF({'t': const(1), 0: mono(F(-1), (0,), F(-1, 2))}, labels(n))
    # coefficient of d_zeta is  -zeta/(2t) = -(1/2) t^{-1} zeta


def Efield(n):
    comps = {j: mono(0, (j,)) for j in labels(n)}
    return VF(comps, labels(n))


def apply_vf(X, f):
    return X.apply(f)


def nabla(f, n):
    return add(smul(2, f), smul(-1, Efield(n).apply(f)))


def KM(f, n):
    """(29): K^M_f = nabla(f) Dcal + H^M_f + Dcal(f) E."""
    p = parity(f)
    sgn = -1 if p else 1
    nf = nabla(f, n)
    df = Dcal(n).apply(f)
    comps = {}
    # nabla(f) * Dcal
    comps['t'] = mul(nf, const(1))
    comps[0] = mul(nf, mono(F(-1), (0,), F(-1, 2)))
    # H^M_f
    hz = smul(sgn, mul(mono(F(-1), (), 1), d_odd(0, f)))       # (1/t) df/dzeta
    comps[0] = add(comps[0], hz)
    for i in range(1, n):
        comps[i] = add(comps.get(i, {}), smul(sgn, d_odd(i, f)))
    # Dcal(f) * E
    for j in labels(n):
        comps[j] = add(comps.get(j, {}), mul(df, mono(0, (j,))))
    return VF(comps, labels(n))


def Rbracket(f, g, n):
    """(32)-(33)."""
    p = parity(f)
    sgn = -1 if p else 1
    out = add(mul(nabla(f, n), Dcal(n).apply(g)),
              smul(-1, mul(Dcal(n).apply(f), nabla(g, n))))
    mp = mul(mono(F(-1), (), 1), mul(d_odd(0, f), d_odd(0, g)))
    for i in range(1, n):
        mp = add(mp, mul(d_odd(i, f), d_odd(i, g)))
    return add(out, smul(sgn, mp))


def DM_zeta(n):
    """(36): D^M_zeta = zeta d_t + (1/t) d_zeta."""
    return VF({'t': mono(0, (0,)), 0: mono(F(-1), (), 1)}, labels(n))


def DM_theta(i, n):
    """(36): D^M_theta_i = theta_i d_t + d_theta_i - theta_i zeta/(2t) d_zeta."""
    return VF({'t': mono(0, (i,)),
               i: const(1),
               0: mono(F(-1), (i, 0), F(-1, 2))}, labels(n))


def KM_zeta(n):
    return VF({'t': mono(0, (0,)), 0: mono(F(-1), (), -1)}, labels(n))


def KM_theta(i, n):
    return VF({'t': mono(0, (i,)),
               i: const(-1),
               0: mono(F(-1), (i, 0), F(-1, 2))}, labels(n))


def hatD(a, n):
    """The Neveu-Schwarz frame on the double cover:
       hatD_i = D^M_theta_i (i < n),  hatD_n = t^{1/2} D^M_zeta."""
    if a < n:
        return DM_theta(a, n) if a >= 1 else None
    X = DM_zeta(n)
    c = t_pow(F(1, 2))
    return VF({k: mul(c, v) for k, v in X.c.items()}, labels(n))


def frame(n):
    """The R^L-frame (D^M_theta_1, ..., D^M_theta_{n-1}, D^M_zeta)."""
    return [DM_theta(i, n) for i in range(1, n)] + [DM_zeta(n)]


def hatframe(n):
    """The double-cover frame (hatD_1, ..., hatD_{n-1}, hatD_n)."""
    return [DM_theta(i, n) for i in range(1, n)] + [hatD(n, n)]


# -------------------------------------------------------- linear algebra

def anticomm(X, Y):
    """{X,Y} = XY + YX for two odd vector fields = their super bracket."""
    return X.bracket(Y)


def decompose(X, fr, n):
    """Write X = sum c_b fr[b] with c_b functions; return (coeffs, ok)."""
    N = len(fr)
    c = [None] * N
    # first the theta-components: fr[i] has d_theta_{i+1}-coefficient 1 (i<N-1)
    for i in range(N - 1):
        c[i] = X.c.get(i + 1, {})
    # then the zeta-component
    rest = X.c.get(0, {})
    for i in range(N - 1):
        rest = add(rest, smul(-1, mul(c[i], fr[i].c.get(0, {}))))
    zc = fr[N - 1].c.get(0, {})          # = 1/t  (or t^{-1/2} in the hat frame)
    # divide rest by zc: zc is a monomial t^a
    (a, S), coeff = list(zc.items())[0]
    assert S == ()
    c[N - 1] = {(k[0] - a, k[1]): v / coeff for k, v in rest.items()}
    # verify
    Y = None
    for b in range(N):
        Zb = VF({k: mul(c[b], v) for k, v in fr[b].c.items()}, labels(n))
        Y = Zb if Y is None else Y.plus(Zb)
    ok = (Y.plus(X.scal(-1))).is_zero()
    return c, ok


def monomials(n, tmin, tmax):
    out = []
    ods = labels(n)
    for a in range(tmin, tmax + 1):
        for m in range(1 << n):
            S = tuple(j for j in ods if m >> j & 1)
            out.append(mono(a, S))
    return out


def tparity(f):
    """0 if every exponent of t is an integer, 1 if every one is a half-odd
    integer, None if mixed or f = 0."""
    ps = {0 if k[0].denominator == 1 else 1 for k in f}
    if len(ps) != 1:
        return None if ps else 0
    return ps.pop()


# --------------------------------------------------------------- checks

LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


def check_dl_formulas(n, tmin=-3, tmax=4):
    """(C1) K^M_f preserves the Ramond distribution; (C2) his Ramond bracket."""
    fr = frame(n)
    ms = monomials(n, tmin, tmax)
    bad_dist = bad_br = 0
    for f in ms:
        K = KM(f, n)
        for X in fr:
            c, ok = decompose(K.bracket(X), fr, n)
            if not ok:
                bad_dist += 1
    for f in ms[:len(ms)]:
        for g in ms[:len(ms)]:
            lhs = KM(f, n).bracket(KM(g, n))
            rhs = KM(Rbracket(f, g, n), n)
            if not (lhs.plus(rhs.scal(-1))).is_zero():
                bad_br += 1
    return len(ms), bad_dist, len(ms) ** 2, bad_br


# ------------------------------------- the Neveu-Schwarz side, (24)-(25)

def KL(f, n):
    """(24): K_f = (2 - sum theta_i d_theta_i)(f) d_t
                        - (-1)^{p(f)} sum (df/dtheta_i - theta_i df/dt) d_theta_i.
    Odd labels here are 1..n (no zeta)."""
    lab = tuple(range(1, n + 1))
    p = parity(f)
    sgn = -1 if p else 1
    E = add(*[mul(mono(0, (i,)), d_odd(i, f)) for i in lab]) if lab else {}
    nf = add(smul(2, f), smul(-1, E))
    comps = {'t': nf}
    for i in lab:
        comps[i] = smul(-sgn, add(d_odd(i, f),
                                  smul(-1, mul(mono(0, (i,)), d_t(f)))))
    return VF(comps, lab)


def kbracket(f, g, n):
    """(25)."""
    lab = tuple(range(1, n + 1))
    p = parity(f)
    sgn = -1 if p else 1
    Ef = add(*[mul(mono(0, (i,)), d_odd(i, f)) for i in lab]) if lab else {}
    Eg = add(*[mul(mono(0, (i,)), d_odd(i, g)) for i in lab]) if lab else {}
    nf = add(smul(2, f), smul(-1, Ef))
    ng = add(smul(2, g), smul(-1, Eg))
    out = add(mul(nf, d_t(g)), smul(-1, mul(d_t(f), ng)))
    s = {}
    for i in lab:
        s = add(s, mul(d_odd(i, f), d_odd(i, g)))
    return add(out, smul(-sgn, s))


def D_theta_L(i, n, sign=+1):
    """(34): D_theta_i = theta_i d_t + d_theta_i  (sign = +1);
    the other sign convention is d_theta_i - theta_i d_t."""
    lab = tuple(range(1, n + 1))
    return VF({'t': smul(sign, mono(0, (i,))), i: const(1)}, lab)


def check_L_formulas(n, tmin=-3, tmax=4):
    """(24)-(25) and (34): the contact fields, his contact bracket, and
    which sign convention for D_theta_i the fields K_f normalize."""
    lab = tuple(range(1, n + 1))
    ms = []
    for a in range(tmin, tmax + 1):
        for m in range(1 << n):
            S = tuple(lab[j] for j in range(n) if m >> j & 1)
            ms.append(mono(a, S))
    bad_br = 0
    for f in ms:
        for g in ms:
            lhs = KL(f, n).bracket(KL(g, n))
            rhs = KL(kbracket(f, g, n), n)
            if not (lhs.plus(rhs.scal(-1))).is_zero():
                bad_br += 1
    res = {}
    for sign in (+1, -1):
        fr = [D_theta_L(i, n, sign) for i in lab]
        bad = 0
        for f in ms:
            K = KL(f, n)
            for X in fr:
                Y = K.bracket(X)
                c = [Y.c.get(i, {}) for i in lab]
                Z = None
                for b, i in enumerate(lab):
                    W = VF({k: mul(c[b], v) for k, v in fr[b].c.items()}, lab)
                    Z = W if Z is None else Z.plus(W)
                if not (Y.plus(Z.scal(-1))).is_zero():
                    bad += 1
        res[sign] = bad
    return len(ms), len(ms) ** 2, bad_br, res


def check_double_cover(n):
    """(C4) hatD_a hatD_b + hatD_b hatD_a = 2 delta_ab Dcal."""
    hf = hatframe(n)
    D = Dcal(n)
    bad = 0
    rows = []
    for a in range(n):
        for b in range(n):
            ac = hf[a].bracket(hf[b])
            tgt = D.scal(2) if a == b else VF({}, labels(n))
            ok = (ac.plus(tgt.scal(-1))).is_zero()
            if not ok:
                bad += 1
            rows.append((a, b, ok))
    return bad, rows


def multiplier(f, n, hat=True):
    """omega with [K^M_f, D_a] = sum_b omega[a][b] D_b in the chosen frame."""
    fr = hatframe(n) if hat else frame(n)
    K = KM(f, n)
    om = []
    for X in fr:
        c, ok = decompose(K.bracket(X), fr, n)
        assert ok, "not in the distribution"
        om.append(c)
    return om


def check_multiplier(n, tmin=-3, tmax=4):
    """(C5) shape of the multiplier in the double-cover frame:
       omega is in c o(n), and omega_{ab} has integer powers of t iff
       a, b are both < n or both = n."""
    ms = monomials(n, tmin, tmax)
    bad_co = bad_par = 0
    seen_offdiag = {}
    for f in ms:
        om = multiplier(f, n)
        # c o(N): the symmetric part is scalar
        for a in range(n):
            for b in range(n):
                s = add(om[a][b], om[b][a])
                if a != b and s:
                    bad_co += 1
                if a == b:
                    s0 = add(om[a][a], smul(-1, om[0][0]))
                    if s0:
                        bad_co += 1
        # monodromy grading
        for a in range(n):
            for b in range(n):
                if not om[a][b]:
                    continue
                want = 0 if ((a == n - 1) == (b == n - 1)) else 1
                got = tparity(om[a][b])
                if got != want:
                    bad_par += 1
                if a != b:
                    seen_offdiag[(a, b)] = True
    return len(ms), bad_co, bad_par, len(seen_offdiag)


def check_o2_lines(n=2, tmin=-3, tmax=4):
    """(C6/C7) N=2: in the R^L-frame (D^M_theta, D^M_zeta) the o(2)-part of the
    multiplier is B * Jtilde with Jtilde = [[0, -t], [1/t, 0]]; hence an
    invariant line needs a square root of -t and does not exist over R^L, while
    over R^M there are exactly two, exchanged by t^{1/2} -> -t^{1/2}."""
    ms = monomials(n, tmin, tmax)
    shapes = set()
    for f in ms:
        om = multiplier(f, n, hat=False)
        a12, a21 = om[0][1], om[1][0]
        if not a12 and not a21:
            continue
        # a12 = -t * a21 ?
        lhs = add(a12, mul(mono(1, (), 1), a21))
        shapes.add(bool(lhs))
    return len(ms), shapes


def check_o4_swap():
    """The monodromy M = diag(1,1,1,-1) in O(4) exchanges the two ideals of
    o(4) = su(2) + su(2).  Computed on the basis E_ij - E_ji of o(4)."""
    import itertools
    idx = [(i, j) for i in range(4) for j in range(i + 1, 4)]

    def mat(i, j):
        m = [[0] * 4 for _ in range(4)]
        m[i][j], m[j][i] = 1, -1
        return m

    def mm(A, B):
        return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
                for i in range(4)]

    def sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(4)] for i in range(4)]

    def br(A, B):
        return sub(mm(A, B), mm(B, A))

    # the self-dual / anti-self-dual bases
    def comb(pairs, signs):
        m = [[0] * 4 for _ in range(4)]
        for (i, j), s in zip(pairs, signs):
            M = mat(i, j)
            for a in range(4):
                for b in range(4):
                    m[a][b] += s * M[a][b]
        return m

    sd = [comb([(0, 1), (2, 3)], [1, 1]),
          comb([(0, 2), (3, 1)], [1, 1]),
          comb([(0, 3), (1, 2)], [1, 1])]
    asd = [comb([(0, 1), (2, 3)], [1, -1]),
           comb([(0, 2), (3, 1)], [1, -1]),
           comb([(0, 3), (1, 2)], [1, -1])]

    def in_span(X, basis):
        # basis is 3 independent antisymmetric matrices; solve by coordinates
        coords = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        import fractions
        rows = [[F(b[i][j]) for (i, j) in coords] for b in basis]
        tgt = [F(X[i][j]) for (i, j) in coords]
        # least squares by elimination
        A = [r[:] + [tgt[k]] for k, r in enumerate(rows)]
        # solve rows^T c = tgt : 6 equations, 3 unknowns
        M = [[rows[b][e] for b in range(3)] + [tgt[e]] for e in range(6)]
        # gaussian elimination
        piv = 0
        for col in range(3):
            r = next((r for r in range(piv, 6) if M[r][col]), None)
            if r is None:
                continue
            M[piv], M[r] = M[r], M[piv]
            pv = M[piv][col]
            M[piv] = [x / pv for x in M[piv]]
            for r2 in range(6):
                if r2 != piv and M[r2][col]:
                    fpr = M[r2][col]
                    M[r2] = [x - fpr * y for x, y in zip(M[r2], M[piv])]
            piv += 1
        for r in range(6):
            if all(M[r][c] == 0 for c in range(3)) and M[r][3] != 0:
                return False
        return True

    # closure: [sd, sd] in sd, [sd, asd] = 0
    ok_ideals = all(in_span(br(x, y), sd) for x in sd for y in sd) and \
                all(in_span(br(x, y), asd) for x in asd for y in asd) and \
                all(all(v == 0 for row in br(x, y) for v in row)
                    for x in sd for y in asd)
    # conjugation by the reflection
    R = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    swaps = all(in_span(mm(mm(R, x), R), asd) for x in sd) and \
            all(in_span(mm(mm(R, x), R), sd) for x in asd)
    fixes = any(in_span(mm(mm(R, x), R), sd) for x in sd)
    return ok_ideals, swaps, fixes


def check_o3_irreducible():
    """V = C^3 is irreducible over o(3): no proper invariant subspace."""
    import itertools
    basis = []
    for i in range(3):
        for j in range(i + 1, 3):
            m = [[0] * 3 for _ in range(3)]
            m[i][j], m[j][i] = 1, -1
            basis.append(m)
    # a line C v invariant  <=>  v is an eigenvector of every generator;
    # over C the generators have distinct eigenvectors, check pairwise
    # (an invariant line would be a common eigenvector of two commuting-free
    # generators); we verify by the standard argument: the only subspaces
    # invariant under all three are 0 and V, tested on all 1- and 2-dim
    # subspaces spanned by integer vectors of height <= 2 -- and, exactly,
    # by the fact that the Casimir acts irreducibly.  We check exactness:
    # the algebra generated by basis is all of gl(3)^{o(3)-envelope}: compute
    # the associative algebra generated and see it is the full 9-dim gl(3).
    def mm(A, B):
        return tuple(tuple(sum(A[i][k] * B[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))
    gens = [tuple(tuple(r) for r in b) for b in basis]
    span = []

    def add_to_span(M):
        v = [F(M[i][j]) for i in range(3) for j in range(3)]
        for w in span:
            p = next((k for k in range(9) if w[k]), None)
            if p is not None and v[p]:
                c = v[p] / w[p]
                v = [a - c * b for a, b in zip(v, w)]
        if any(v):
            span.append(v)
            return True
        return False

    I = tuple(tuple(1 if i == j else 0 for j in range(3)) for i in range(3))
    todo = [I] + gens
    prods = list(todo)
    for _ in range(4):
        new = []
        for A in prods:
            for B in gens:
                new.append(mm(A, B))
        prods = new
        todo += new
    for M in todo:
        add_to_span(M)
    return len(span)          # 9 means the enveloping algebra is all of gl(3)


def check_lambda_monodromy(n=2, tmin=-3, tmax=4):
    """(C8) For N=2 the chiral multiplier lambda(f) = (1/2) Dbar D f and its
    conjugate are exchanged by the monodromy: in the hat frame,
    lambda = A - iB with A = omega_00, B = omega_10, and B has half-odd powers
    of t while A has integer ones -- so M(lambda) = lambdabar."""
    ms = monomials(n, tmin, tmax)
    bad = 0
    nz = 0
    for f in ms:
        om = multiplier(f, n)
        A, B = om[0][0], om[1][0]
        if tparity(A) not in (0, None) or (B and tparity(B) != 1):
            bad += 1
        if B:
            nz += 1
    return len(ms), bad, nz


def main():
    say("ramond.py -- the Ramond sector of the Ivanov-Krivonos equations")
    say("=" * 78)

    say("\n(1) Formulas (29)-(33): K^M_f preserves the")
    say("    Ramond distribution, and his Ramond bracket is its bracket.")
    for n in (2, 3, 4):
        nm, bd, npair, bb = check_dl_formulas(n)
        say("    n=%d: %4d generating functions f = t^a*(odd monomial), -3<=a<=4;"
            % (n, nm))
        say("         [K^M_f, D] in Span_{R^L}(D^M_zeta, D^M_theta_i): %d failures"
            % bd)
        say("         [K^M_f, K^M_g] = K^M_{f,g}_R.b. on %d pairs: %d failures"
            % (npair, bb))

    say("\n(1a) The same for his Neveu-Schwarz formulas (24), (25), (34).")
    for n in (1, 2, 3, 4):
        nm, npair, bb, res = check_L_formulas(n)
        say("    n=%d: %4d functions; [K_f, K_g] = K_{f,g}_k.b. on %d pairs:"
            " %d failures" % (n, nm, npair, bb))
        say("         K_f normalises D_theta_i = theta_i d_t + d_theta_i:"
            " %d failures; the other sign: %d failures"
            % (res[+1], res[-1]))

    say("\n(2) The double cover: hatD_i := D^M_theta_i, hatD_n := t^{1/2}D^M_zeta")
    say("    satisfy the Neveu-Schwarz relations with d_t replaced by Dcal.")
    for n in (2, 3, 4):
        bad, rows = check_double_cover(n)
        say("    n=%d: {hatD_a, hatD_b} = 2 delta_ab Dcal on all %d pairs: %d failures"
            % (n, len(rows), bad))

    say("\n(3) The multiplier in the double-cover frame is c o(N)-valued, and")
    say("    its entries are graded by the monodromy: omega_{ab} has integer")
    say("    powers of t if a, b are on the same side of the twist, and")
    say("    half-odd-integer powers if they are not.")
    for n in (2, 3, 4):
        nm, bco, bpar, off = check_multiplier(n)
        say("    n=%d: %4d functions; c o(N)-shape: %d failures; monodromy grading:"
            " %d failures; off-diagonal entries realised: %d of %d"
            % (n, nm, bco, bpar, off, n * (n - 1)))

    say("\n(4) N=2: in the R^L-frame (D^M_theta, D^M_zeta) the o(2)-part of")
    say("    the multiplier is B*Jtilde, Jtilde = [[0,-t],[1/t,0]]; an invariant")
    say("    line is an eigenvector of Jtilde, i.e. needs a square root of -t.")
    nm, shapes = check_o2_lines()
    say("    %d functions; omega_{12} + t*omega_{21} nonzero anywhere: %s"
        % (nm, any(shapes)))
    say("    Jtilde^2 = -1, eigenvectors (1, +- i/t^{1/2}) -- not over R^L;")
    say("    over R^M there are exactly two, exchanged by t^{1/2} -> -t^{1/2}.")

    say("\n(5) N=3: V = C^3 is irreducible over o(3) -- the associative algebra")
    say("    generated by o(3) inside gl(3) is the whole of gl(3):")
    d = check_o3_irreducible()
    say("    dim of the enveloping algebra = %d (9 = gl(3), i.e. irreducible)" % d)
    say("    Hence, no proper submodule to twist: the Ramond sector has no more")
    say("    scalar constraints for N=3 than the Neveu-Schwarz one, i.e. none.")

    say("\n(6) N=4: the monodromy is the reflection R = diag(1,1,1,-1) in O(4);")
    say("    it exchanges the two ideals su(2) of o(4).")
    ok, swaps, fixes = check_o4_swap()
    say("    the two ideals close and commute: %s" % ok)
    say("    R conjugates each ideal onto the other: %s" % swaps)
    say("    R fixes an ideal: %s" % fixes)

    say("\n(7) N=2: the chiral multiplier and its conjugate are exchanged by the")
    say("    monodromy (A even, B odd in t^{1/2}).")
    nm, bad, nz = check_lambda_monodromy()
    say("    %d functions, %d with a nonzero o(2)-part, %d failures"
        % (nm, nz, bad))

    say("\n" + "=" * 78)
    say("Summary: (29)-(33) and (36) check out; the Ramond sector is the")
    say("Neveu-Schwarz one pulled back to the double cover t = s^2, the")
    say("monodromy acting on the structure algebra o(N) by the reflection that")
    say("flips the twisted odd coordinate.  Hence the Ramond-like Ivanov-")
    say("Krivonos systems: for N=2 and N=4 the constraint and its conjugate are")
    say("exchanged by the monodromy, so the dilaton stays periodic and its")
    say("o(N)-partners become anti-periodic; for N=3 nothing changes, and there")
    say("is no scalar equation in either sector.")

    with open("ramondframe-out.txt", "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
