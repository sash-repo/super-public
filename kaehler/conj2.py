r"""Twisted Dolbeault algebras in the flat model (Theorems 1.8.2 and 3.2.1).

a) one even J and even h: does the action of hei(0|4) +) sp(2) on the Dolbeault
   algebra become an osp(2|2)-action on twisted forms?
b) three even J's (hyper-Kaehler): does hei(0|8) +) sp(4) become osp(2|4)?

The twist is Bernstein's: with alpha the connection 1-form, d alpha = hbar*omega
(hbar even and non-zero, scaled to 1),

    D^+ := d + (1/2) alpha,      D^- := [X^-, D^+],

and its J-rotated partner D_c^+ := J D^+ J^{-1}.  This rotation rotates the
coefficients as well, so its curvature vanishes; the right partner is fixed in
conj2d.py / conj2e.py.  The module also provides the helpers (span, sdim, centre,
derived algebra) used by the other scripts; importing it runs the computation.

Run (from this folder):  ../venv/bin/python conj2.py
"""
from fractions import Fraction
from flatmodel import Model, Echelon, closure, d_operator, \
    form_to_multiplication, form_to_convolution

half = Fraction(1, 2)


def span(M, elts):
    ech = {0: Echelon(), 1: Echelon()}
    for e in elts:
        if e:
            ech[M.parity(e)].add(e)
    return ech


def sd(M, elts):
    ech = span(M, elts)
    return (len(ech[0]), len(ech[1]))


def contains(M, ech, e):
    return True if not e else not ech[M.parity(e)].reduce(e)[0]


def reduce_basis(M, elts):
    ech = {0: Echelon(), 1: Echelon()}
    out = []
    for e in elts:
        if e and ech[M.parity(e)].add(e):
            out.append(e)
    return out


def centre(M, basis):
    """dimension of the centre of the span of `basis`, per parity."""
    red = reduce_basis(M, basis)
    dims = {}
    for par in (0, 1):
        vecs = [e for e in red if M.parity(e) == par]
        if not vecs:
            dims[par] = 0
            continue
        rows = []
        for g in red:
            brs = [M.br(g, v) for v in vecs]
            mons = sorted({m for br in brs for m in br})
            for m in mons:
                rows.append([br.get(m, Fraction(0)) for br in brs])
        dims[par] = len(vecs) - rank(rows, len(vecs))
    return dims[0], dims[1]


def rank(rows, ncol):
    mat = [r[:] for r in rows]
    r0 = 0
    for col in range(ncol):
        piv = None
        for r in range(r0, len(mat)):
            if mat[r][col] != 0:
                piv = r
                break
        if piv is None:
            continue
        mat[r0], mat[piv] = mat[piv], mat[r0]
        pv = mat[r0][col]
        mat[r0] = [v / pv for v in mat[r0]]
        for r in range(len(mat)):
            if r != r0 and mat[r][col] != 0:
                f = mat[r][col]
                mat[r] = [a - f * b for a, b in zip(mat[r], mat[r0])]
        r0 += 1
    return r0


def derived(M, basis):
    out = []
    for i in range(len(basis)):
        for j in range(i + 1):
            b = M.br(basis[i], basis[j])
            if b:
                out.append(b)
    if not out:
        return [], (0, 0)
    return closure(M, out)


# =====================================================================
# 2a)  Kaehler: one even J, even h, on C^{2n} with coordinates q_i, p_i
# =====================================================================
print('===== 2a) Kaehler, twisted Dolbeault algebra =====')
for n in (1, 2):
    N = 2 * n
    M = Model([0] * N)
    q = list(range(n))
    p = list(range(n, N))
    c, b, x, t = (lambda a: M.gen('c', a)), (lambda a: M.gen('b', a)), \
                 (lambda a: M.gen('x', a)), (lambda a: M.gen('t', a))
    d = d_operator(M)
    # the Kaehler form omega = sum q_hat p_hat and its dual bivector
    Xp = form_to_multiplication(M, [(1, (q[i], p[i])) for i in range(n)])
    Xm = form_to_convolution(M, [(1, (q[i], p[i])) for i in range(n)])
    H = M.br(Xp, Xm)
    # J : dq -> dp, dp -> -dq  (and the same on the vector fields)
    dc = M.add(*[M.add(M.mul(c(p[i]), t(q[i])),
                       M.smul(-1, M.mul(c(q[i]), t(p[i])))) for i in range(n)])
    delta = M.br(d, Xm)
    deltac = M.br(dc, Xm)
    Lap = M.br(d, delta)
    print('--- C^{%d|0}' % N)
    print('  d_c^2 = 0 :', not M.br(dc, dc), '; [d,d_c] = 0 :', not M.br(d, dc),
          '; [d,delta] = [d_c,delta_c] :',
          not M.add(M.br(d, delta), M.smul(-1, M.br(dc, deltac))))
    print('  untwisted closure <X^+,X^-,d,d_c> : sdim',
          closure(M, [Xp, Xm, d, dc])[1], ' (expected (4|4))')

    # the connection form alpha = sum (q_hat p - p_hat q); d alpha = -2 omega
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    # its J-rotation: q_hat -> p_hat, p_hat -> -q_hat, q -> p, p -> -q
    alphac = M.add(*[M.add(M.smul(-1, M.mul(c(p[i]), x(q[i]))),
                           M.smul(-1, M.mul(c(q[i]), x(p[i])))) for i in range(n)])
    print('  [d,alpha] = -2 X^+ :',
          not M.add(M.br(d, alpha), M.smul(2, Xp)),
          '; [d_c,alpha_c] = ? ', '-2X^+' if not M.add(M.br(dc, alphac), M.smul(2, Xp))
          else ('+2X^+' if not M.add(M.br(dc, alphac), M.smul(-2, Xp)) else 'other'))

    Dp = M.add(d, M.smul(-half, alpha))
    Dcp = M.add(dc, M.smul(-half, alphac))
    for label, gens in (('<D^+, X^+, X^->', [Dp, Xp, Xm]),
                        ('<D^+, D_c^+, X^+, X^->', [Dp, Dcp, Xp, Xm])):
        basis, s = closure(M, gens, limit=400)
        print('  %-24s sdim %s ; centre %s ; derived %s'
              % (label, s, centre(M, basis), derived(M, basis)[1]))
    basis, s = closure(M, [Dp, Dcp, Xp, Xm], limit=400)
    ech = span(M, basis)
    print('    Laplacian in it :', contains(M, ech, Lap),
          '; d in it :', contains(M, ech, d),
          '; delta in it :', contains(M, ech, delta))
    # the four odd elements and the even part
    Dm = M.br(Xm, Dp)
    Dcm = M.br(Xm, Dcp)
    print('    [D^+,D^+] = -2*(-X^+)? ', M.br(Dp, Dp) and
          not M.add(M.br(Dp, Dp), M.smul(-1, Xp)),
          '; [D^+,D_c^+] = 0 :', not M.br(Dp, Dcp))
    print('    sdim <D^+,D^-,D_c^+,D_c^-> :', closure(M, [Dp, Dm, Dcp, Dcm])[1])
    print()

# =====================================================================
# 2b)  hyper-Kaehler: three even J's on C^{4n}, two twists
# =====================================================================
print('===== 2b) hyper-Kaehler, twisted =====')
M = Model([0, 0, 0, 0])
c, b, x, t = (lambda a: M.gen('c', a)), (lambda a: M.gen('b', a)), \
             (lambda a: M.gen('x', a)), (lambda a: M.gen('t', a))
d = d_operator(M)
wi = [(1, (0, 1)), (1, (2, 3))]
wj = [(1, (0, 2)), (-1, (1, 3))]
wk = [(1, (0, 3)), (1, (1, 2))]
E = [form_to_multiplication(M, w) for w in (wi, wj, wk)]
F = [form_to_convolution(M, w) for w in (wi, wj, wk)]
print('  untwisted <E_l,F_l,d> : sdim', closure(M, E + F + [d])[1],
      ' (expected (11|8))')


def potential(w):
    """a 1-form alpha with d alpha = -2 * (the 2-form w): sum w_ab (x_a_hat x_b
    -> (1/1) (c_a x_b - c_b x_a))."""
    terms = []
    for co, (a, bb) in w:
        terms.append(M.smul(co, M.mul(c(a), x(bb))))
        terms.append(M.smul(-co, M.mul(c(bb), x(a))))
    return M.add(*terms)


for names, ws in (('omega_i, omega_j', (wi, wj)),):
    a1, a2 = (potential(w) for w in ws)
    E1 = form_to_multiplication(M, ws[0])
    E2 = form_to_multiplication(M, ws[1])
    print('  [d,alpha_1] = -2 E_i :', not M.add(M.br(d, a1), M.smul(2, E1)),
          '; [d,alpha_2] = -2 E_j :', not M.add(M.br(d, a2), M.smul(2, E2)))
    D1 = M.add(d, M.smul(-half, a1))
    D2 = M.add(d, M.smul(-half, a2))
    F1, F2 = form_to_convolution(M, ws[0]), form_to_convolution(M, ws[1])
    D1m = M.br(F1, D1)
    D2m = M.br(F2, D2)
    for label, gens in (('<D_1^+, D_1^->', [D1, D1m]),
                        ('<D_1^+, D_2^+, F_i, F_j>', [D1, D2, F1, F2]),
                        ('<D_s^+, D_s^-, E_l, F_l>', [D1, D2, D1m, D2m] + E + F)):
        try:
            basis, s = closure(M, gens, limit=600)
            print('  %-26s sdim %s ; centre %s' % (label, s, centre(M, basis)))
        except RuntimeError as exc:
            print('  %-26s %s' % (label, exc))
