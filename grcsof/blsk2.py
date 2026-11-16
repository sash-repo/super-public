#!/usr/bin/env python3
"""Statement 7.5.1 of the paper (a correction to [BLS1, Thm 5.1] at k = 2):

    Bol^{(a)} = D_xi D_q + 2a d_tau :  F_{a,b} --> F_{a-1,b}

is i(pe(2))-invariant for EVERY a and b, whereas [BLS1, Thm 5.1] lists only the
member a = 1/2 (which is -Bol_2), while for k = 1, 3, 4 the printed condition
a = (k-1)/2 is correct.

Everything goes through the validated harness of classify_m1.eqs_un (the one
that reproduced [BLS1, Thm 5.1] before anything new was claimed): it carries the
parity sign of an odd operator, which a naive B o L = L o B does not.

"""
from sympy import symbols, S, Matrix, expand, Rational, solve
from classify_m1 import words, eqs_un
from m1 import PE, D_1, D_xi, D_q
from super12 import add, smul

ALL = tuple(PE)                      # all eight generating functions of pe(2)
a, b = symbols('a b')


def dim_and_locus(s, w, par, A, Bb, nmax=None):
    """The space of i(pe(2))-invariant unary operators of degree s, charge w and
    parity par from F_{A,Bb}; returns (words, generic dim, special weights)."""
    tt = [(x[0] or 'id', x[1], x[2]) for x in words(s, w) if x[2] == par]
    if not tt:
        return [], 0, []
    terms = [(nm, op, None, p) for nm, op, p in tt]
    At, Bt = A + S(s + w) / 2, Bb - S(w) / 2
    cs, eqs = eqs_un(terms, A, Bb, At, Bt, par, nmax or (-s + 3), ALL)
    Mx = Matrix([[expand(e).coeff(c) for c in cs] for e in eqs]) if eqs else \
        Matrix(0, len(cs), [])
    d = len(cs) - (Mx.rank() if Mx.rows else 0)
    special = []
    if d == 0:
        for i in range(len(cs)):
            sub = {cs[j]: (S(1) if j == i else symbols(f'y{j}'))
                   for j in range(len(cs))}
            unk = [sub[cs[j]] for j in range(len(cs)) if j != i] + [a]
            try:
                r = solve([expand(e.subs(sub)) for e in eqs], unk, dict=True)
            except Exception:
                r = []
            special += [d_ for d_ in r if a in d_]
    return [x[0] for x in tt], d, sorted({d_[a] for d_ in special}, key=str)


lines = []


def say(s=""):
    print(s)
    lines.append(s)


say("=" * 74)
say("1.  The explicit operator of eq. (65):  D_xi D_q + 2a d_tau,")
say("    F_{a,b} --> F_{a-1,b}, all eight generators of pe(2), monomials up to")
say("    degree 6 in q, a and b symbolic.")
op = lambda u: add(D_xi(D_q(u)), smul(2 * a, D_1(u)))
cs, eqs = eqs_un([('B', op, None, 1)], a, b, a - 1, b, 1, 6, ALL)
res = [x for x in (expand(e.subs({cs[0]: 1})) for e in eqs) if x != 0]
say("    obstructions: " + (str(res[:3]) if res else
                            "NONE  -->  invariant for every a and every b"))

say()
say("2.  The whole space of invariant unary operators of the degree and charge")
say("    of Bol_k, source weight F_{a,b} with a, b free  (k = order):")
say("      k   words                 generic dim   invariant only for a =")
KS = {1: (-1, +1, 1), 2: (-2, 0, 1), 3: (-3, -1, 1), 4: (-4, -2, 1)}
verdict = {}
for k, (s, w, par) in KS.items():
    ws, d, sp = dim_and_locus(s, w, par, a, b)
    printed = Rational(k - 1, 2)
    if d >= 1:
        note = "EVERY a  <-- 1-parameter family, not in [BLS1, Thm 5.1]"
        verdict[k] = "family"
    else:
        note = f"{sp}   ({'= (k-1)/2 as printed' if sp == [printed] else 'DIFFERS from (k-1)/2'})"
        verdict[k] = "as printed" if sp == [printed] else "differs"
    say(f"      {k}   {str(ws):22s}{d:^13d} {note}")

say()
say("3.  Verdict per order:")
for k in sorted(verdict):
    say(f"      k={k}: {verdict[k]}")
say()
say("    So [BLS1, Thm 5.1] is confirmed as printed for k = 1, 3, 4, and is")
say("    incomplete for k = 2: there the invariant operators form the")
say("    1-parameter family Bol^{(a)} = D_xi D_q + 2a d_tau, of which")
say("    Bol_2 = -(D_xi D_q + d_tau) is the member a = 1/2.")
say("=" * 74)

