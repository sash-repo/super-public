#!/usr/bin/env python3
"""
run_tables.py -- validation of sv.py against the classical ADE singularities
(Milnor numbers), and the census of sdim O_f and of dim Nt_f = dim(m/m I_f)_odd
for the normal forms of the Serganova-Vaintrob tables of simple
supersingularities (Math. Scand. 64 (1989) 251-284).

Column meaning:
  sdim O_f = m_0|m_1   the Milnor number of the germ 
  layer                dim (O / m I_f)_odd -- the space the extra datum g of
                       f + theta*g lives in (first-layer theorem)
"""

from fractions import Fraction
import sv


def show(name, p, q, terms, D, expect=None):
    r = sv.stable(p, q, terms, D)
    if r["sdim"] is None:
        print(f"{name:28s} p|q={p}|{q}  UNSTABLE at D={D}: {r['unstable']}")
        return
    m0, m1 = r["sdim"]
    ok = ""
    if expect is not None:
        ok = "  OK" if (m0, m1) == expect else f"  MISMATCH (expected {expect[0]}|{expect[1]})"
    print(f"{name:28s} p|q={p}|{q}   sdim O_f = {m0}|{m1}   layer = {r['layer']}{ok}")
    return r


print("=" * 78)
print("A. validation against the classical case")
print("=" * 78)

# classical A_n : x^{n+1} in O^{1|0}, Milnor number n
for n in (2, 3, 4, 5):
    show(f"A_{n} = x^{n+1}", 1, 0, [(1, (n + 1,), ())], D=n + 4, expect=(n, 0))

# classical D_n : x^{n-1} + x y^2, Milnor number n
for n in (4, 5, 6):
    show(f"D_{n} = x^{n-1}+xy^2", 2, 0,
         [(1, (n - 1, 0), ()), (1, (1, 2), ())], D=n + 3, expect=(n, 0))

# classical E_6, E_7, E_8
show("E_6 = x^3+y^4", 2, 0, [(1, (3, 0), ()), (1, (0, 4), ())], D=8, expect=(6, 0))
show("E_7 = x^3+xy^3", 2, 0, [(1, (3, 0), ()), (1, (1, 3), ())], D=8, expect=(7, 0))
show("E_8 = x^3+y^5", 2, 0, [(1, (3, 0), ()), (1, (0, 5), ())], D=9, expect=(8, 0))

print()
print("=" * 78)
print("B. the Serganova-Vaintrob table of simple supersingularities: the odd layer")
print("=" * 78)

# ---- type I: p = 0, purely odd germs
I = [
    ("I1  xi1234",            4, [(1, (), (0, 1, 2, 3))]),
    ("I2  1234+1256",         6, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5))]),
    ("I3  1234+1256+3456",    6, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5)),
                                  (1, (), (2, 3, 4, 5))]),
    ("I4  xi123456",          6, [(1, (), (0, 1, 2, 3, 4, 5))]),
    ("I5  1234+1567",         7, [(1, (), (0, 1, 2, 3)), (1, (), (0, 4, 5, 6))]),
    ("I6  1234+1256+1357",    7, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5)),
                                  (1, (), (0, 2, 4, 6))]),
    ("I8  ..+2457",           7, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5)),
                                  (1, (), (0, 2, 4, 6)), (1, (), (1, 3, 4, 6))]),
    ("I9  ..+2467",           7, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5)),
                                  (1, (), (0, 2, 4, 6)), (1, (), (1, 3, 5, 6))]),
    ("I10 ..+2367+3567",      7, [(1, (), (0, 1, 2, 3)), (1, (), (0, 1, 4, 5)),
                                  (1, (), (0, 2, 4, 6)), (1, (), (1, 2, 5, 6)),
                                  (1, (), (2, 4, 5, 6))]),
]
for name, q, terms in I:
    show(name, 0, q, terms, D=1)

print()
# ---- type II(n,m) : x^n + x^m xi_1 xi_2, corank 1|2
for n in (3, 4, 5):
    for m in range(1, n):
        show(f"II({n},{m})", 1, 2, [(1, (n,), ()), (1, (m,), (0, 1))], D=n + 4)

print()
# ---- type III_1(n,m), III_2(n), III_3, III_4, III_5 : corank 1|4
for n in (3, 4, 5):
    for m in range(1, n):
        show(f"III_1({n},{m})", 1, 4,
             [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3))], D=n + 4)
for n in (3, 4, 5):
    show(f"III_2({n})", 1, 4, [(1, (n,), ()), (1, (1,), (0, 1))], D=n + 4)
show("III_3  x^3+xi1234", 1, 4, [(1, (3,), ()), (1, (0,), (0, 1, 2, 3))], D=7)
show("III_4  x^3+x xi1234", 1, 4, [(1, (3,), ()), (1, (1,), (0, 1, 2, 3))], D=7)
show("III_5  x^3", 1, 4, [(1, (3,), ())], D=7)

print()
# ---- type IV : corank 1|5
for n in (3, 4, 5):
    for m in range(2, n):
        show(f"IV_1({n},{m})", 1, 5,
             [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3)),
              (1, (0,), (0, 2, 3, 4))], D=n + 4)
for n in (3, 4, 5):
    for m in range(1, n):
        show(f"IV_2({n},{m})", 1, 5,
             [(1, (n,), ()), (1, (1,), (0, 1)), (1, (m,), (2, 3))], D=n + 4)
for n in (3, 4, 5):
    show(f"IV_3({n})", 1, 5, [(1, (n,), ()), (1, (1,), (0, 1))], D=n + 4)
show("IV_5  x^3+x xi1234", 1, 5, [(1, (3,), ()), (1, (1,), (0, 1, 2, 3))], D=7)
show("IV_6  x^3", 1, 5, [(1, (3,), ())], D=7)

print()
# ---- type V(n) : corank 1|6
for n in (3, 4):
    show(f"V({n})", 1, 6,
         [(1, (n,), ()), (1, (1,), (0, 1)), (1, (1,), (2, 3)), (1, (1,), (4, 5))],
         D=n + 3)

print()
# ---- type VII : corank 2|3
for n in (3, 4):
    show(f"VII_1({n})", 2, 3,
         [(1, (2, 1), ()), (1, (0, n), ()), (1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))],
         D=n + 3)
show("VII_2", 2, 3,
     [(1, (3, 0), ()), (1, (0, 4), ()), (1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))], D=8)
show("VII_3", 2, 3,
     [(1, (3, 0), ()), (1, (1, 3), ()), (1, (1, 0), (0, 1)), (1, (0, 1), (1, 2))], D=8)
