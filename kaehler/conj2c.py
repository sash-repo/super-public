r"""Identification of twisted algebras by structure invariants: simplicity, the
even part, the odd part as a g_ev-module.  By Kac's classification, a simple Lie
superalgebra of sdim (11|8) with g_ev = sp(4) + C is osp(2|4) (Theorem 3.2.1);
one of sdim (4|4) with g_ev = sl(2) + C is osp(2|2) = sl(1|2).

Run (from this folder):  ../venv/bin/python conj2c.py
"""
from fractions import Fraction
from flatmodel import Model, Echelon, closure, d_operator, \
    form_to_multiplication, form_to_convolution
from conj2 import span, sd, contains, centre, derived, reduce_basis, rank

half = Fraction(1, 2)


def ideal(M, basis, seeds):
    ech = {0: Echelon(), 1: Echelon()}
    cur = []
    for e in seeds:
        if e and ech[M.parity(e)].add(e):
            cur.append(e)
    i = 0
    while i < len(cur):
        for v in basis:
            br = M.br(cur[i], v)
            if br and ech[M.parity(br)].add(br):
                cur.append(br)
        i += 1
    return (len(ech[0]), len(ech[1]))


def is_simple(M, basis):
    red = reduce_basis(M, basis)
    total = sd(M, red)
    for u in red:
        if ideal(M, red, [u]) != total:
            return False, u
    return True, None


def report(M, basis, name):
    red = reduce_basis(M, basis)
    total = sd(M, red)
    ev = [e for e in red if M.parity(e) == 0]
    od = [e for e in red if M.parity(e) == 1]
    print('  %s : sdim %s' % (name, total))
    print('     centre of g          :', centre(M, red))
    print('     g perfect            :', derived(M, red)[1] == total)
    simple, _ = is_simple(M, red)
    print('     g simple             :', simple)
    print('     centre of g_ev in g_ev:', centre(M, ev))
    dev = derived(M, ev)
    print('     [g_ev,g_ev] sdim     :', dev[1])
    print('     [g_od,g_od] = g_ev   :',
          sd(M, [M.br(u, v) for u in od for v in od]) == (total[0], 0))
    return total


# =====================================================================
# 2a)
# =====================================================================
print('===== 2a) Kaehler, twisted: the (4|4) algebra =====')
for n in (1, 2):
    N = 2 * n
    M = Model([0] * N)
    q, p = list(range(n)), list(range(n, N))
    c, x, t = (lambda a: M.gen('c', a)), (lambda a: M.gen('x', a)), \
              (lambda a: M.gen('t', a))
    d = d_operator(M)
    Xp = form_to_multiplication(M, [(1, (q[i], p[i])) for i in range(n)])
    Xm = form_to_convolution(M, [(1, (q[i], p[i])) for i in range(n)])
    H = M.br(Xp, Xm)
    dc = M.add(*[M.add(M.mul(c(p[i]), t(q[i])),
                       M.smul(-1, M.mul(c(q[i]), t(p[i])))) for i in range(n)])
    delta = M.br(d, Xm)
    Lap = M.br(d, delta)
    alpha = M.add(*[M.add(M.mul(c(q[i]), x(p[i])),
                          M.smul(-1, M.mul(c(p[i]), x(q[i])))) for i in range(n)])
    alphac = M.add(*[M.add(M.smul(-1, M.mul(c(p[i]), x(q[i]))),
                           M.smul(-1, M.mul(c(q[i]), x(p[i])))) for i in range(n)])
    Dp, Dcp = M.add(d, M.smul(-half, alpha)), M.add(dc, M.smul(-half, alphac))
    print('--- C^{%d|0}' % N)
    b0, s0 = closure(M, [Xp, Xm, d, dc], limit=400)
    print('  untwisted <X^+,X^-,d,d_c>: sdim', s0, '; centre', centre(M, b0),
          '; simple :', is_simple(M, b0)[0])
    basis, s = closure(M, [Dp, Dcp, Xp, Xm], limit=400)
    report(M, basis, '<D^+, D_c^+, X^+, X^->')
    ech = span(M, basis)
    print('     Delta = [d,delta] in g :', contains(M, ech, Lap),
          '; sdim <X^+,X^-,H,Delta> :', sd(M, [Xp, Xm, H, Lap]))
    # which even element is central in g_ev?
    ev = [e for e in reduce_basis(M, basis) if M.parity(e) == 0]
    for e in ev:
        if all(not M.br(e, u) for u in ev):
            print('     an element central in g_ev found; is it Delta ? ',
                  not M.add(e, M.smul(-1, Lap)),
                  '; is it Delta + a*H ? ',
                  sd(M, [e, Lap, H]) == (2, 0))
    print()

# =====================================================================
# 2b)
# =====================================================================
print('===== 2b) hyper-Kaehler, twisted =====')
M = Model([0, 0, 0, 0])
c, x = (lambda a: M.gen('c', a)), (lambda a: M.gen('x', a))
d = d_operator(M)
wi = [(1, (0, 1)), (1, (2, 3))]
wj = [(1, (0, 2)), (-1, (1, 3))]
wk = [(1, (0, 3)), (1, (1, 2))]
E = [form_to_multiplication(M, w) for w in (wi, wj, wk)]
F = [form_to_convolution(M, w) for w in (wi, wj, wk)]


def potential(w):
    terms = []
    for co, (a, bb) in w:
        terms.append(M.smul(co, M.mul(c(a), x(bb))))
        terms.append(M.smul(-co, M.mul(c(bb), x(a))))
    return M.add(*terms)


a1, a2 = potential(wi), potential(wj)
b0, s0 = closure(M, E + F + [d], limit=800)
print('  untwisted <E_l,F_l,d> : sdim', s0, '; centre', centre(M, b0),
      '; simple :', is_simple(M, b0)[0])
D1 = M.add(d, M.smul(-half, a1))
report(M, closure(M, [D1] + E + F, limit=800)[0], 'one twist: <D_1^+, E_l, F_l>')
D2 = M.add(d, M.smul(-half * 2, a2))
report(M, closure(M, [D1, D2] + E + F, limit=800)[0],
       'two twists (hbar_2 = 2 hbar_1): <D_1^+, D_2^+, E_l, F_l>')
D1m, D2m = M.br(F[0], D1), M.br(F[1], D2)
report(M, closure(M, [D1, D2, D1m, D2m], limit=800)[0],
       'eq.(sqrtX): <D_1^+, D_1^-, D_2^+, D_2^->')
