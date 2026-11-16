"""
Explicit bases of the annihilators of the Schwarzians of Theorem 4.5.1 of the
text (the five cases other than s_4).

The "annihilator" of a Schwarzian s is the radical of the cocycle
c(f,g) = Res(f s(g)) in its second argument:
        Ann := {g in g | c(f,g) = 0 for all f in g}.
When Res(f h) = 0 for all admissible f forces h = 0, this is just Ker s;
we compute BOTH (as the radical of an explicit matrix over a symmetric
window of t-degrees) so that the difference, if any, is visible.

  (1) k^L'(1|4), cocycle (2), s_{4^0} = (2-E)/t
  (2) k^L'(1|4), cocycle (3), s_{4'}  = d_t
  (3) vect^L(1|1) inside k^L(1|2), s = D_xi D_eta d_t
  (4) vect^L(1|2) and svect^L_lambda(1|2) inside k^L'(1|4), cocycle (3)
  (5) svect^L'(1|2), same cocycle
Plus, as a control, (0) k^L'(1|4), cocycle (1), s_4 = Delta_IS (the case
already printed in the text, eq. (93)).
"""

from fractions import Fraction as F
from itertools import combinations
from stringy import mono, add, mul, smul, d_t, d_odd, parity, eq, show
from check_cocycles import L_bracket, L_res, L_ops, monomials

# ===================================================================== tools

def rank_and_radical(rows, cols, cmat):
    """cmat[(i,j)] = c(rows[i], cols[j]); return the indices j spanning the
    radical {y : c(x,y)=0 for all x} -- here: the null space of the matrix."""
    m, n = len(rows), len(cols)
    A = [[F(cmat.get((i, j), 0)) for j in range(n)] for i in range(m)]
    # Gaussian elimination, tracking pivot columns
    piv, r = [], 0
    for j in range(n):
        p = next((i for i in range(r, m) if A[i][j] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        inv = F(1) / A[r][j]
        A[r] = [x * inv for x in A[r]]
        for i in range(m):
            if i != r and A[i][j] != 0:
                f = A[i][j]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv.append(j)
        r += 1
        if r == m:
            break
    free = [j for j in range(n) if j not in piv]
    # null space basis: one vector per free column
    ns = []
    for j0 in free:
        v = [F(0)] * n
        v[j0] = F(1)
        for k, jp in enumerate(piv):
            v[jp] = -A[k][j0]
        ns.append(v)
    return r, ns

def fmt_L(key):
    a, S = key
    s = "".join("th%d" % j for j in S)
    base = "" if a == 0 else ("t" if a == 1 else "t^%d" % a)
    return (base + s) if (base or s) else "1"

# ============================================ part 1: the three s on k^L'(1|4)

n4 = 4
D1, Dth, Kth = L_ops(4)

def d_t_inv(f):
    out = {}
    for (a, S), c in f.items():
        if a == -1:
            raise ValueError("D_1^{-1} undefined on t^{-1}: %s" % ((a, S),))
        out[(a + 1, S)] = out.get((a + 1, S), 0) + c / (a + 1)
    return {k: v for k, v in out.items() if v}

def P_IS(f):
    """the differential part D_th1 D_th2 D_th3 D_th4."""
    g = f
    for i in (4, 3, 2, 1):
        g = Dth(i)(g)
    return g

def _kappa(S):
    """P(t^a th_S) = kappa_S * a(a-1)...(a-3+s) * t^{a-4+s} * th_{S^c},
    s = |S|:  only the subset T = S^c of the expansion of P survives, since
    th_T th_{S cap T} = 0 unless S cap T is empty.  Read kappa off at a = 10."""
    a0, s = 10, len(S)
    Sc = tuple(i for i in (1, 2, 3, 4) if i not in S)
    out = P_IS(mono(a0, S))
    key = (a0 - 4 + s, Sc)
    ff = F(1)
    for j in range(4 - s):
        ff *= (a0 - j)
    if not out:
        return F(0), Sc
    assert set(out) == {key}, (S, out)
    return F(out[key]) / ff, Sc

_KAPPA = {S: _kappa(S) for r in range(5) for S in combinations((1, 2, 3, 4), r)}

def Delta_IS(f):
    """eq. (58) as a SINGLE operator: in
          P(t^a th_S) = kappa * a(a-1)...(a-3+s) * t^{a-4+s} th_{S^c}
    the last factor of the falling factorial is exactly (a-4+s)+1, i.e. the
    denominator produced by D_1^{-1}; the two cancel, and

          Delta_IS(t^a th_S) = kappa * a(a-1)...(a-(2-s)) * t^{a-3+s} th_{S^c}
                                                       [3-s factors]  (s < 4),
          Delta_IS(t^a th_1th_2th_3th_4) = kappa * t^{a+1}/(a+1)      (s = 4),

    the only pole being at the element t^{-1}th_1th_2th_3th_4 that is thrown
    away when passing to k^L'(1|4).  Composing the two factors literally
    (D_1^{-1} applied last) loses the cancellation at the zeros of the falling
    factorial; see Delta_IS_literal."""
    out = {}
    for (a, S), c in f.items():
        s = len(S)
        kappa, Sc = _KAPPA[S]
        if not kappa:
            continue
        if s == 4:
            if a == -1:
                raise ValueError("Delta_IS has a pole at t^{-1}th_1..th_4")
            coef = c * kappa / (a + 1)
            key = (a + 1, Sc)
        else:
            ff = F(1)
            for j in range(3 - s):
                ff *= (a - j)
            coef = c * kappa * ff
            key = (a - 3 + s, Sc)
        if coef:
            out[key] = out.get(key, 0) + coef
    return {k: v for k, v in out.items() if v}

def Delta_IS_literal(f):
    g = f
    for i in (4, 3, 2, 1):
        g = Dth(i)(g)
    return d_t_inv(g)

def s40(f):
    """(2-E)/t  with E = sum theta_i d_theta_i."""
    out = {}
    for (a, S), c in f.items():
        k = (2 - len(S)) * c
        if k:
            out[(a - 1, S)] = out.get((a - 1, S), 0) + k
    return {k: v for k, v in out.items() if v}

def s4p(f):
    return d_t(f)

def kprime_keys(A):
    """monomials of k^L'(1|4): all t^a phi(theta) except t^{-1}th1th2th3th4."""
    ks = monomials(4, range(-A, A + 1), (1, 2, 3, 4))
    return [k for k in ks if k != (-1, (1, 2, 3, 4))]

def report_kL4(name, s, A=5, guard=2):
    """rows: a wider window (the test functions f); cols: the window we
    classify (the arguments g)."""
    rows = kprime_keys(A + guard)
    cols = kprime_keys(A)
    cmat = {}
    for j, kg in enumerate(cols):
        sg = s(mono(*kg))
        for i, kf in enumerate(rows):
            v = L_res(4, mul(mono(*kf), sg))
            if v:
                cmat[(i, j)] = v
    r, ns = rank_and_radical(rows, cols, cmat)
    ker = [kg for kg in cols if not s(mono(*kg))]
    print("-" * 72)
    print("k^L'(1|4), %s :  window |deg t| <= %d" % (name, A))
    print("   dim Ker s (in the window)      = %d" % len(ker))
    print("   dim radical of c (in the window) = %d" % len(ns))
    # is the radical spanned by monomials?
    monos = []
    for v in ns:
        nz = [j for j, x in enumerate(v) if x]
        if len(nz) == 1:
            monos.append(cols[nz[0]])
        else:
            monos.append(("MIX", [(cols[j], v[j]) for j in nz]))
    print("   radical basis: %s" % ", ".join(
        fmt_L(m) if not isinstance(m, tuple) or m[0] != "MIX" else str(m) for m in monos))
    ev = sum(1 for m in monos if not isinstance(m[0], str) and len(m[1]) % 2 == 0)
    od = len(monos) - ev
    print("   sdim of the radical in the window = (%d|%d)" % (ev, od))
    ke = sum(1 for k in ker if len(k[1]) % 2 == 0)
    print("   Ker s basis  : %s" % ", ".join(fmt_L(k) for k in ker))
    print("   sdim of Ker s = (%d|%d)" % (ke, len(ker) - ke))
    return ker

def check_closed(keys, tag):
    """is Span(K_f, f in keys) closed under the contact bracket?"""
    S = set(keys)
    bad = []
    for kf in keys:
        for kg in keys:
            b = L_bracket(4, mono(*kf), mono(*kg))
            for k in b:
                if k not in S:
                    bad.append((kf, kg, k))
    print("   closed under the contact bracket: %s" % ("YES" if not bad else "NO %s" % bad[:3]))

def check_loop_o4(keys):
    """{f,g} for f,g of theta-degree 2 equals -sum d_i f d_i g: the bracket
    does not involve d_t, so Span(t^m th_i th_j) = o(4) (x) C[t,t^{-1}]."""
    bad = 0
    for kf in keys:
        for kg in keys:
            lhs = L_bracket(4, mono(*kf), mono(*kg))
            rhs = {}
            for i in range(1, 5):
                rhs = add(rhs, mul(d_odd(i, mono(*kf)), d_odd(i, mono(*kg))))
            rhs = smul(-1, rhs)
            if not eq(lhs, rhs):
                bad += 1
    print("   {f,g} = -sum_i d_i f d_i g on theta-degree 2 (loop bracket): %s"
          % ("OK" if bad == 0 else "%d failures" % bad))
    # the m-independence: the structure constants of t^m th_i th_j only
    # depend on i,j,k,l -- exactly the loop algebra o(4) (x) C[t,t^{-1}]
    ok = True
    for (i, j) in combinations(range(1, 5), 2):
        for (k, l) in combinations(range(1, 5), 2):
            base = L_bracket(4, mono(0, (i, j)), mono(0, (k, l)))
            for m in (-2, -1, 1, 3):
                for nn in (-3, 0, 2):
                    b = L_bracket(4, mono(m, (i, j)), mono(nn, (k, l)))
                    shifted = {(a + m + nn, S): c for (a, S), c in base.items()}
                    if not eq(b, shifted):
                        ok = False
    print("   structure constants independent of the t-degrees: %s" % ("OK" if ok else "NO"))

def check_po04(keys):
    """{f,g} = -(-1)^{p(f)} sum d_i f d_i g on Lambda(4) = po(0|4)."""
    bad = 0
    for kf in keys:
        for kg in keys:
            f, g = mono(*kf), mono(*kg)
            lhs = L_bracket(4, f, g)
            rhs = {}
            for i in range(1, 5):
                rhs = add(rhs, mul(d_odd(i, f), d_odd(i, g)))
            rhs = smul(-(-1) ** parity(f), rhs)
            if not eq(lhs, rhs):
                bad += 1
    print("   {f,g} = -(-1)^p(f) sum_i d_i f d_i g on Lambda(4) = po(0|4): %s"
          % ("OK" if bad == 0 else "%d failures" % bad))

# ======= the hyperbolic realisation of k^L(1|2n): alpha_1 = dt + sum(xi dEta + eta dXi)
# (this is the realisation in which the embedding (28) is written: labels
#  xi_i = 2i-1, eta_i = 2i)

def L_bracket_h(n, f, g):
    """contact bracket for alpha_1 = dt + sum_i (xi_i d eta_i + eta_i d xi_i)."""
    def nabla(h):
        return {k: (2 - len(k[1])) * c for k, c in h.items() if (2 - len(k[1]))}
    res = add(mul(nabla(f), d_t(g)), smul(-1, mul(d_t(f), nabla(g))))
    s = {}
    for i in range(1, n + 1):
        xi, eta = 2 * i - 1, 2 * i
        s = add(s, mul(d_odd(xi, f), d_odd(eta, g)), mul(d_odd(eta, f), d_odd(xi, g)))
    return add(res, smul(-(-1) ** parity(f), s))

def D_xi(i):
    return lambda f: add(d_odd(2 * i - 1, f), mul(mono(0, (2 * i,)), d_t(f)))

def D_eta(i):
    return lambda f: add(d_odd(2 * i, f), mul(mono(0, (2 * i - 1,)), d_t(f)))

# =========================================== part 2: the embedding i, vect^L

def make_i(n):
    """i: vect^L(1|n) -> k^L(1|2n), eq. (28).  xi_i = 2i-1, eta_i = 2i."""
    xi = [2 * i - 1 for i in range(1, n + 1)]
    eta = [2 * i for i in range(1, n + 1)]
    Phi = {}
    for i in range(n):
        Phi = add(Phi, mono(0, (xi[i], eta[i])))

    def tPhi(m):
        if m >= 0:
            out, base = mono(0), add(mono(1), smul(-1, Phi))
            for _ in range(m):
                out = mul(out, base)
            return out
        out = {}
        for k in range(0, n + 1):
            cb = F(1)
            for j in range(k):
                cb *= F(m - j, j + 1)
            P = mono(0)
            for _ in range(k):
                P = mul(P, Phi)
            out = add(out, mul(mono(m - k, (), cb * (-1) ** k), P))
        return out

    def i_dx(m, T=()):
        """i( theta_T x^m d_x ) ; T a tuple of theta-indices (1-based)"""
        f = mono(0, tuple(xi[j - 1] for j in T))
        return smul(F(1, 2 ** m) if m >= 0 else F(2 ** (-m)), mul(f, tPhi(m)))

    def i_dth(m, j, T=()):
        """i( theta_T x^m d_{theta_j} )"""
        f = mono(0, tuple(xi[k - 1] for k in T))
        return smul(F(1, 2 ** m) if m >= 0 else F(2 ** (-m)),
                    mul(mul(f, mono(0, (eta[j - 1],))), tPhi(m)))
    return i_dx, i_dth

def vect_basis(n, M):
    """basis of vect^L(1|n) with |deg x| <= M, as (kind, m, j, T) tokens."""
    out = []
    Ts = [T for r in range(n + 1) for T in combinations(range(1, n + 1), r)]
    for m in range(-M, M + 1):
        for T in Ts:
            out.append(('dx', m, None, T))
        for j in range(1, n + 1):
            for T in Ts:
                out.append(('dth', m, j, T))
    return out

def name_vect(tok):
    kind, m, j, T = tok
    th = "".join("th%d" % k for k in T)
    xm = "" if m == 0 else ("x" if m == 1 else "x^%d" % m)
    d = "d_x" if kind == 'dx' else "d_th%d" % j
    return (th + xm + " " + d).strip()

def par_vect(tok):
    kind, m, j, T = tok
    return (len(T) + (0 if kind == 'dx' else 1)) % 2

def tok_VF(n, tok):
    """the token as an honest vector field on (x | theta_1..theta_n)."""
    from stringy import VF
    kind, m, j, T = tok
    f = mono(m, tuple(T))
    return VF({'t': f} if kind == 'dx' else {j: f}, tuple(range(1, n + 1)))

def img_tok(n, tok):
    i_dx, i_dth = make_i(n)
    kind, m, j, T = tok
    return i_dx(m, T) if kind == 'dx' else i_dth(m, j, T)

def VF_to_gen(n, X):
    """i(X) for an arbitrary X in vect^L(1|n), by linearity over the tokens."""
    out = {}
    Ts = [T for r in range(n + 1) for T in combinations(range(1, n + 1), r)]
    for coord, f in X.c.items():
        for (a, S), c in f.items():
            # write t^a th_S as th_S x^a; the token carries th_S on the LEFT
            tok = ('dx', a, None, S) if coord == 't' else ('dth', a, coord, S)
            out = add(out, smul(c, img_tok(n, tok)))
    return out

def check_i_hom(n, M=3):
    """i is a homomorphism vect^L(1|n) -> k^L(1|2n) on the whole window."""
    B = vect_basis(n, M)
    ims = [img_tok(n, t) for t in B]
    keys = sorted({k for f in ims for k in f})
    idx = {k: i for i, k in enumerate(keys)}
    r, _ = rank_and_radical(list(range(len(keys))), B,
                            {(idx[k], j): ims[j][k] for j in range(len(B)) for k in ims[j]})
    print("   i injective on the window (%d basis elements, rank %d): %s"
          % (len(B), r, "OK" if r == len(B) else "NO"))
    bad = 0
    for t1 in B:
        for t2 in B:
            X, Y = tok_VF(n, t1), tok_VF(n, t2)
            lhs = L_bracket_h(n, img_tok(n, t1), img_tok(n, t2))
            rhs = VF_to_gen(n, X.bracket(Y))
            if not eq(lhs, rhs):
                bad += 1
                if bad <= 3:
                    print("      [i(%s),i(%s)] != i([.,.]) : %s  vs  %s"
                          % (name_vect(t1), name_vect(t2), show(lhs), show(rhs)))
    print("   i([X,Y]) = [i(X),i(Y)] on the window: %s"
          % ("OK" if bad == 0 else "%d failures out of %d" % (bad, len(B) ** 2)))

def radical_vect(n, M, sfun, resfun, label, guard=2):
    """radical of c(X,Y) = Res(i(X) s(i(Y))) on vect^L(1|n)."""
    i_dx, i_dth = make_i(n)

    def img(tok):
        kind, m, j, T = tok
        return i_dx(m, T) if kind == 'dx' else i_dth(m, j, T)

    rows = vect_basis(n, M + guard)
    cols = vect_basis(n, M)
    imr = [img(t) for t in rows]
    cmat = {}
    for j, tg in enumerate(cols):
        sg = sfun(img(tg))
        for i, f in enumerate(imr):
            v = resfun(mul(f, sg))
            if v:
                cmat[(i, j)] = v
    r, ns = rank_and_radical(rows, cols, cmat)
    print("-" * 72)
    print("%s :  window |deg x| <= %d,  dim radical = %d" % (label, M, len(ns)))
    ev = od = 0
    for v in ns:
        nz = [j for j, x in enumerate(v) if x]
        ps = {par_vect(cols[j]) for j in nz}
        if len(ps) == 1:
            (ev, od) = (ev + 1, od) if ps == {0} else (ev, od + 1)
        terms = " + ".join(("%s*%s" % (v[j], name_vect(cols[j])) if v[j] != 1
                            else name_vect(cols[j])) for j in nz)
        print("     %s" % terms)
    print("   sdim = (%d|%d)" % (ev, od))
    return ns, cols

# ============================================================= svect^L_lambda

def svect_basis(lam, M, derived=False):
    """basis (52) of svect^L_lambda(1|2), each element written as a
    list of (coeff, token) in the vect^L(1|2) basis."""
    out = []
    for m in range(-M, M + 1):
        # LL_m = x^{m+1} d_x + (lam+m+1)/2 (th1 x^m d_1 + th2 x^m d_2)
        out.append(("LL_%d" % m, 0,
                    [(F(1), ('dx', m + 1, None, ())),
                     (F(lam + m + 1, 1) / 2, ('dth', m, 1, (1,))),
                     (F(lam + m + 1, 1) / 2, ('dth', m, 2, (2,)))]))
        for j in (1, 2):
            # S_m^j = th_j x^{m+1} d_x + (lam+m+1)(th_j th_1 x^m d_1 + th_j th_2 x^m d_2)
            terms = [(F(1), ('dx', m + 1, None, (j,)))]
            for k in (1, 2):
                if k != j:
                    # th_j th_k  ->  sorted with a sign
                    sgn = 1 if j < k else -1
                    terms.append((F(lam + m + 1, 1) * sgn, ('dth', m, k, tuple(sorted((j, k))))))
            out.append(("S_%d^%d" % (m, j), 1, terms))
        out.append(("x^%d th1 d_2" % m, 0, [(F(1), ('dth', m, 2, (1,)))]))
        out.append(("x^%d th2 d_1" % m, 0, [(F(1), ('dth', m, 1, (2,)))]))
        out.append(("x^%d (th1 d_1 - th2 d_2)" % m, 0,
                    [(F(1), ('dth', m, 1, (1,))), (F(-1), ('dth', m, 2, (2,)))]))
        out.append(("x^%d d_1" % m, 1, [(F(1), ('dth', m, 1, ()))]))
        out.append(("x^%d d_2" % m, 1, [(F(1), ('dth', m, 2, ()))]))
    if lam == 0 and not derived:
        # th1 th2 d_x is EVEN (two odd factors); the parity field here is used only for
        # the sdim printed by radical_svect; with it the sdim reads (7|8), as
        # reported by analyse_span.
        out.append(("th1 th2 d_x", 0, [(F(1), ('dx', 0, None, (1, 2)))]))
    return out

def check_svect_div(lam, M=3):
    """the (lambda-twisted) divergence of every listed basis element vanishes:
    Div_lambda(X) = Div(X) + lambda * (d_x-part coefficient)'/x ... we instead
    verify closure under the bracket, which is what matters here."""
    pass

def radical_svect(lam, M, derived=False, guard=2):
    i_dx, i_dth = make_i(2)

    def img_tok(tok):
        kind, m, j, T = tok
        return i_dx(m, T) if kind == 'dx' else i_dth(m, j, T)

    def img(el):
        f = {}
        for c, tok in el[2]:
            f = add(f, smul(c, img_tok(tok)))
        return f

    rows = svect_basis(lam, M + guard, derived)
    cols = svect_basis(lam, M, derived)
    imr = [img(e) for e in rows]
    cmat = {}
    for j, eg in enumerate(cols):
        sg = d_t(img(eg))
        for i, f in enumerate(imr):
            v = L_res(4, mul(f, sg))
            if v:
                cmat[(i, j)] = v
    r, ns = rank_and_radical(rows, cols, cmat)
    tag = "svect^L'(1|2)" if derived else ("svect^L_%s(1|2)" % lam)
    print("-" * 72)
    print("%s, cocycle (3) :  window |m| <= %d,  dim radical = %d" % (tag, M, len(ns)))
    ev = od = 0
    for v in ns:
        nz = [j for j, x in enumerate(v) if x]
        ps = {cols[j][1] for j in nz}
        if len(ps) == 1:
            (ev, od) = (ev + 1, od) if ps == {0} else (ev, od + 1)
        print("     %s" % " + ".join(
            ("%s*%s" % (v[j], cols[j][0]) if v[j] != 1 else cols[j][0]) for j in nz))
    print("   sdim = (%d|%d)" % (ev, od))
    return ns, cols

# ======================================================= structure of a span

def analyse_span(vecs, n, names=None, tag=""):
    """vecs: generating functions in k^L(1|2n) spanning a subspace; report
    closure under the contact bracket, sdim, [g,g], the centre, and whether
    [odd,odd] spans the even part."""
    keys = sorted({k for f in vecs for k in f})
    idx = {k: i for i, k in enumerate(keys)}

    def vec(f):
        v = [F(0)] * len(keys)
        for k, c in f.items():
            if k not in idx:
                return None
            v[idx[k]] = F(c)
        return v

    def rref(rows):
        rows = [r[:] for r in rows]
        m, nn, r, piv = len(rows), len(keys), 0, []
        for j in range(nn):
            p = next((i for i in range(r, m) if rows[i][j] != 0), None)
            if p is None:
                continue
            rows[r], rows[p] = rows[p], rows[r]
            iv = F(1) / rows[r][j]
            rows[r] = [x * iv for x in rows[r]]
            for i in range(m):
                if i != r and rows[i][j] != 0:
                    f0 = rows[i][j]
                    rows[i] = [a - f0 * b for a, b in zip(rows[i], rows[r])]
            piv.append(j)
            r += 1
        return rows[:r], piv

    B = [vec(f) for f in vecs]
    R, piv = rref(B)

    def inspan(f):
        v = vec(f)
        if v is None:
            return False
        for row, j in zip(R, piv):
            if v[j] != 0:
                c = v[j]
                v = [a - c * b for a, b in zip(v, row)]
        return all(x == 0 for x in v)

    closed = all(inspan(L_bracket_h(n, f, g)) for f in vecs for g in vecs)
    ev = [f for f in vecs if parity(f) == 0]
    od = [f for f in vecs if parity(f) == 1]
    der = [L_bracket_h(n, f, g) for f in vecs for g in vecs]
    der = [d for d in der if d]
    dr, _ = rref([v for v in (vec(d) for d in der) if v is not None])
    oo = [L_bracket_h(n, f, g) for f in od for g in od]
    oo = [d for d in oo if d]
    oor, _ = rref([v for v in (vec(d) for d in oo) if v is not None])
    # centre
    cen = 0
    for f in vecs:
        if all(not L_bracket_h(n, f, g) for g in vecs):
            cen += 1
    print("   %s: sdim = (%d|%d), closed = %s, dim[g,g] = %d, dim[g_1,g_1] = %d, "
          "# basis vectors that are central = %d"
          % (tag, len(ev), len(od), closed, len(dr), len(oor), cen))

# ==================================================================== driver

if __name__ == "__main__":
    print("=" * 72)
    print("(0) control: k^L'(1|4), cocycle (1), s_4 = Delta_IS")
    m1 = report_kL4("s_4 = Delta_IS (D_1^{-1} first)", Delta_IS, A=4)
    check_closed(m1, "s_4")
    print("   -- the literal reading of (58), D_1^{-1} applied last:")
    m1b = report_kL4("s_4 literal", Delta_IS_literal, A=4)
    check_closed(m1b, "s_4 literal")
    print("=" * 72)
    print("(1) k^L'(1|4), cocycle (2), s_{4^0} = (2-E)/t")
    m2 = report_kL4("s_{4^0} = (2-E)/t", s40, A=4)
    check_closed(m2, "s40")
    check_loop_o4(m2)
    print("=" * 72)
    print("(2) k^L'(1|4), cocycle (3), s_{4'} = d_t")
    m3 = report_kL4("s_{4'} = d_t", s4p, A=4)
    check_closed(m3, "s4p")
    check_po04(m3)

    print("=" * 72)
    print("(3) vect^L(1|1) inside k^L(1|2), s = D_xi D_eta d_t")
    D1_2, Dth_2, _ = L_ops(2)
    def s_v11(f):
        # the ns(2) cocycle in the (xi,eta)-realisation: the ANTISYMMETRISED
        # product 1/2 (D_xi D_eta - D_eta D_xi) d_t = (D_xi D_eta - d_t) d_t.
        # The plain product D_xi D_eta d_t is NOT a cocycle (checked).
        return add(D_xi(1)(D_eta(1)(d_t(f))), smul(-1, d_t(d_t(f))))
    check_i_hom(1, 3)
    radical_vect(1, 4, s_v11, lambda f: L_res(2, f), "vect^L(1|1)")

    print("=" * 72)
    print("(4) vect^L(1|2) inside k^L'(1|4), cocycle (3): s = d_t")
    check_i_hom(2, 2)
    radical_vect(2, 3, d_t, lambda f: L_res(4, f), "vect^L(1|2)")

    print("=" * 72)
    print("(5) svect^L_lambda(1|2) and svect^L'(1|2)")
    for lam, der in ((F(1, 3), False), (F(2, 5), False), (F(0), False), (F(0), True)):
        ns, cols = radical_svect(lam, 3, der)
        i_dx, i_dth = make_i(2)

        def img_el(el):
            f = {}
            for c, tok in el[2]:
                kind, m, j, T = tok
                f = add(f, smul(c, i_dx(m, T) if kind == 'dx' else i_dth(m, j, T)))
            return f
        vecs = []
        for v in ns:
            f = {}
            for j, x in enumerate(v):
                if x:
                    f = add(f, smul(x, img_el(cols[j])))
            vecs.append(f)
        analyse_span(vecs, 2, tag="structure")

    print("=" * 72)
    print("(3') structure of the vect^L(1|1) answer")
    ns, cols = radical_vect(1, 4, s_v11, lambda f: L_res(2, f), "vect^L(1|1)")
    i_dx1, i_dth1 = make_i(1)
    vecs = []
    for v in ns:
        f = {}
        for j, x in enumerate(v):
            if x:
                kind, m, jj, T = cols[j]
                f = add(f, smul(x, i_dx1(m, T) if kind == 'dx' else i_dth1(m, jj, T)))
        vecs.append(f)
    analyse_span(vecs, 1, tag="structure")
