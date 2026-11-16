#!/usr/bin/env python3
"""Which entries of [GLS, Secs. 11 and 12] can have a 1-DIMENSIONAL fibre?

This is the analogue, for the two other W-regradings, of the dominance test of
Subsection 6.7.2 of the paper -- and here it is much sharper, because a 1-dimensional
module is finite-dimensional for free, so no dominance question arises at all.

Over a 1-dimensional fibre every m_i with i > 1 is ZERO: the m_i are obtained
from the highest weight vector m1 by lowering operators of g_0, and those act
by 0 on a 1-dimensional module.  So an entry of [GLS] survives at such
a fibre if and only if

  (a) its source weight lambda, for some values of its parameters, lies on the
      line of the characters of g_0, and
  (b) the coefficient of the pure-m1 part of its singular vector is then
      non-zero, and its own side conditions hold.

The line of characters is computed in regrade.py, not read off [GLS]:

  r = 3xi : g_0 is PERFECT, so the only 1-dimensional module is the trivial
            one, lambda = 0;
  r = 3eta: the characters form the line  lam*(1,1,1,1)  in the basis
            (K_t, K_{xi_i eta_i}) of h, i.e. lam*(1,-1,-1,-1) in [GLS]'s
            basis, by the dictionary below.

The dictionary is not guessed either: it is over-determined by the weight
SHIFTS of the entries themselves, which are computable on our side as the
weight of the g_{-1}-part of the printed singular vector.  See dict_check().

Usage:  python3 glsreg.py            the whole audit
"""
import sympy as sp

k, l, m, lam = sp.symbols('k l m lam')

# ---------------------------------------------------------------- the lists
#
# Each entry: (name, degree, source weight, m1-coefficient of the vector,
#              extra conditions that must hold, comment on the m1-part).
# The m1-coefficient is the coefficient with which the pure-m1 monomial (the
# one carrying no m_i, i > 1) enters; None means the vector has no pure-m1
# part at all, so the entry dies at once over a 1-dimensional fibre.

SEC11 = [   # g = kas(1|6; 3xi)
    ('1a', 1, (k, l, m, sp.Symbol('n')), sp.Integer(1), [], 'xi1xi2xi3 m1'),
    ('1b', 1, (k, l - 1, -l - 1, k), None, [], 'xi1xi2 m1 is m1-pure'),
    ('1c', 1, (k, l, k - 1, -l - 1), -(k + l), [sp.Ne(k + l, 0)],
     '-(k+l) xi2xi3 m1'),
    ('1d', 1, (k + 1, k - 1, l, -l), -(-1 + k - l) * (k + l),
     [sp.Ne(k + l, 0), sp.Ne(k, l + 1)], '-(k-l-1)(k+l) xi1xi3 m1'),
    ('1e', 1, (k + 1, k - 1, 0, 0), 2 * (1 - k), [sp.Ne(k, 1)],
     '2(1-k) xi1xi3 m1'),
    ('2a', 2, (k, l, m, -2 - m), sp.Integer(1), [], 'xi1xi2.xi1xi2xi3 m1'),
    ('2b', 2, (k, l, m, -1 - l), -1 - l - m,
     [sp.Ne(l + m, -1)], '(-1-l2-l3) xi2xi3.xi1xi2xi3 m1'),
    ('2c', 2, (k, l, m, -m), -(l - m) * (1 + l + m),
     [sp.Ne(l, m), sp.Ne(l + m, -1)], '-(l2-l3)(1+l2+l3) xi1xi3.xi1xi2xi3 m1'),
    ('2d', 2, (k, l, 0, 0), l, [sp.Ne(l, 0)], 'l xi1xi3.xi1xi2xi3 m1'),
    ('2e', 2, (k, -k, k - 3, k - 1), -2, [], '-2 xi2.xi1xi2xi3 m1'),
    ('2f', 2, (k, -k, l, l), -4 * (k - l), [sp.Ne(k, l)],
     '-4(k-l) xi2.xi1xi2xi3 m1'),
    ('2g', 2, (1 + k, l, -k, 1 + l),
     (k - l) * (1 + k - l) * (k + l), [], '(k-l)(1+k-l)(k+l) xi1.xi1xi2xi3 m1'),
    ('2h', 2, (k + 2, k - 2, k, -k), -4 * k * (-1 + 2 * k), [],
     '-4k(2k-1) xi3.xi1xi2xi3 m1'),
    ('2i', 2, (-l + 1, k, k, l + 1), -2 * (-1 + k - l) * (k + l), [],
     '-2(k-l-1)(k+l) xi3.xi1xi2xi3 m1'),
]

SEC12 = [   # g = kas(1|6; 3eta)
    ('1a', 1, (k, l, m, -m), sp.Integer(1), [], 'eta1eta3 m1'),
    ('1b', 1, (k, l, m, -l - 1), -l + m, [], '(-l+m) eta2eta3 m1'),
    ('1c', 1, (k, l, -l - 2, m), (1 + l - m) * (2 + l + m), [],
     '(1+l-m)(2+l+m) eta1eta2 m1'),
    ('1d', 1, (k + 3, -k - 2, k, k - 1), 2 * k, [], '2k eta1 m1'),
    ('1e', 1, (k + 3, k - 1, -k - 1, k - 1), -4 * k ** 2, [], '-4k^2 eta2 m1'),
    ('1f', 1, (4, 0, -1, -1), sp.Integer(2), [], '2 eta2 m1'),
    ('1g', 1, (4, 0, 0, 0), sp.Integer(6), [], '6 . 1 . m1'),
    ('2a', 2, (k, l, m, 2 - m), sp.Integer(1), [], '(eta1eta3)^2 m1'),
    ('2b', 2, (k, l, l + 2, -l - 1), sp.Integer(2), [],
     '2 eta1eta3.eta2eta3 m1'),
    ('2c', 2, (k, l - 2, -l, l + 1), -2 * (1 + 2 * l), [sp.Ne(l, -sp.Rational(1, 2))],
     '-2(1+2l) eta1eta2.eta1eta3 m1'),
    ('2d', 2, (k, l, m, -l), (-1 + l - m) * (l - m), [],
     '(l-m-1)(l-m) (eta2eta3)^2 m1'),
    ('2e', 2, (k, l - 1, -l - 1, -l + 1), -4 * l * (-1 + 2 * l), [],
     '-4l(2l-1) eta1eta2.eta2eta3 m1'),
    ('2f', 2, (k, l, -l - 1, m),
     (l - m) * (1 + l - m) * (1 + l + m) * (2 + l + m),
     [sp.Ne(m, l), sp.Ne(m, l + 1), sp.Ne(m, -l - 1), sp.Ne(m, -l - 2)],
     '(l-m)(1+l-m)(1+l+m)(2+l+m) (eta1eta2)^2 m1'),
    ('2g', 2, (4, -3, 1, 0), sp.Integer(2), [], '2 eta1.eta1eta3 m1'),
]

# the line of the 1-dimensional fibres, in [GLS]'s coordinates
LINE = {'3xi': (0, 0, 0, 0), '3eta': (lam, -lam, -lam, -lam)}


def audit(name, entries, line):
    print(f"=== [GLS, Sec. {'11' if name == '3xi' else '12'}], "
          f"g = kas(1|6; {name}) " + "=" * 20)
    print(f"    the 1-dimensional fibres are the weights {line}")
    for nm, deg, src, c1, cond, note in entries:
        eqs = [sp.Eq(a, b) for a, b in zip(src, line)]
        sols = sp.solve(eqs, dict=True)
        if not sols:
            print(f"  {nm}) deg {deg}: source never meets the line "
                  f"-- no 1-dimensional fibre")
            continue
        for s in sols:
            if c1 is None:
                print(f"  {nm}) deg {deg}: meets the line, but the vector has "
                      f"no pure-m1 part -- vanishes")
                continue
            cval = sp.simplify(sp.sympify(c1).subs(s))
            lv = sp.simplify(sp.sympify(line[0]).subs(s))
            bad = [c for c in cond if sp.simplify(sp.sympify(c).subs(s))
                   is sp.false]
            if cval == 0:
                print(f"  {nm}) deg {deg}: meets the line at lam1 = {lv}, but "
                      f"the m1-coefficient [{note}] vanishes there")
            elif bad:
                print(f"  {nm}) deg {deg}: meets the line at lam1 = {lv}, but "
                      f"its own side condition fails: {bad}")
            else:
                print(f"  ** {nm}) deg {deg}: SURVIVES at lam1 = {lv}, "
                      f"m1-coefficient {cval}   [{note}]")
    print()


def dict_check():
    """the dictionary [GLS]-weight = A(our weight), over-determined.

    A is read off the weight SHIFTS: for each entry, the shift printed by
    [GLS] must equal A applied to the h-weight (in our basis) of the
    g_{-1}-part of the printed singular vector."""
    ours = {   # our h-weights in the basis (K_t, K_{xi_i eta_i})
        '1': (-2, 0, 0, 0),
        'eta1': (-1, 1, 0, 0), 'eta2': (-1, 0, 1, 0), 'eta3': (-1, 0, 0, 1),
        'eta1eta2': (0, 1, 1, 0), 'eta1eta3': (0, 1, 0, 1),
        'eta2eta3': (0, 0, 1, 1),
        'xi1eta2': (0, -1, 1, 0), 'xi3eta1': (0, 1, 0, -1),
    }
    A = sp.Matrix([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, -1]])
    tests = [   # (entry, our word(s) whose weights add up, [GLS] shift)
        ('12 1g', ['1'], (-2, 0, 0, 0)),
        ('12 1d', ['eta1'], (-1, 0, -1, 0)),
        ('12 1c', ['eta1eta2'], (0, -1, -1, 0)),
        ('12 1b', ['eta1eta3', 'xi1eta2'], (0, -1, 0, -1)),
        ('12 1a', ['eta1eta3'], (0, 0, -1, -1)),
        ('12 2a', ['eta1eta3', 'eta1eta3'], (0, 0, -2, -2)),
    ]
    print("=== the dictionary, over-determined " + "=" * 30)
    print("    [GLS](lam1,lam2,lam3,lam4) = (our_0, -our_2, -our_1, -our_3)")
    ok = True
    for nm, ws, shift in tests:
        v = sp.Matrix([sum(ours[w][i] for w in ws) for i in range(4)])
        got = tuple((A * v).T)
        good = got == tuple(sp.Integer(x) for x in shift)
        ok &= good
        print(f"  {nm}: our shift {tuple(v.T)} -> {got}, "
              f"[GLS] prints {shift}   {'OK' if good else 'MISMATCH'}")
    print(f"  all six agree: {ok}")
    print("  (the first coordinate of [GLS, Sec. 12, 1a)] reads k --> l, "
          "which no\n   constant shift can do; every other entry of Sec. 12 "
          "has it as k --> k.)")
    print()


if __name__ == '__main__':
    dict_check()
    audit('3xi', SEC11, LINE['3xi'])
    audit('3eta', SEC12, LINE['3eta'])
