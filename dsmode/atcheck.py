#!/usr/bin/env python3
"""
Check the atypicality tables of Subsections 6.3.1 (brj(2;3), p = 3) and 6.3.2
(brj(2;5), p = 5) of the text: the columns (lambda+rho, mu) and
(lambda+rho, mu)_p, mu running over the two roots isotropic over R.

Everything is done over Q with the *integer* Gram matrix B lifted from the
Cartan matrix A = DB, i.e. exactly the "form over R" of the text.  The last
part prints (a, a) for every odd positive root, over R and mod p.

Usage:  python3 atcheck.py
"""
from fractions import Fraction as F

B = [[0, -2], [-2, 1]]          # same B in both subsections


def ip(u, v):
    return sum(u[i] * v[j] * B[i][j] for i in range(2) for j in range(2))


def rho(ev, od):
    s_ev = [sum(a[i] for a in ev) for i in range(2)]
    s_od = [sum(a[i] for a in od) for i in range(2)]
    return [F(s_ev[i] - s_od[i], 2) for i in range(2)]


def run(title, p, ev, od, pi, iso, table):
    r = rho(ev, od)
    print(f"--- {title}  (p={p}),  rho = {r}")
    bad = 0
    for lam_c, printed_ip, printed_mod in table:
        lam = [sum(F(lam_c[k]) * pi[k][i] for k in range(2)) for i in range(2)]
        lr = [lam[i] + r[i] for i in range(2)]
        vals = [ip(lr, mu) for mu in iso]
        mods = [int(v) % p if v.denominator == 1 else None for v in vals]
        ok = (vals == [F(x) for x in printed_ip]) and (mods == list(printed_mod))
        if not ok:
            bad += 1
            print(f"   lambda={lam_c}: computed {vals} mod p {mods} "
                  f"| printed {printed_ip} / {printed_mod}   <<< MISMATCH")
    print(f"   {len(table)} rows checked, {bad} mismatches\n")


# ---- 6.3.1  brj(2;3),  p=3 ----------------------------------------------
ev3 = [[1, 1], [0, 2], [1, 3], [2, 4]]
od3 = [[1, 0], [0, 1], [1, 2], [1, 4]]
pi3 = [[F(-1, 2), F(-1)], [F(-1, 2), F(0)]]
iso3 = [[1, 0], [1, 4]]
tab3 = [((0, 1), (-3, 3), (0, 0)),
        ((0, 2), (-3, 7), (0, 1)),
        ((1, 0), (-1, 1), (2, 1)),
        ((1, 1), (-1, 5), (2, 2)),
        ((1, 2), (-1, 9), (2, 0)),
        ((2, 0), (1, 3), (1, 0)),
        ((2, 1), (1, 7), (1, 1)),
        ((2, 2), (1, 11), (1, 2))]
run("brj(2;3)", 3, ev3, od3, pi3, iso3, tab3)

# ---- 6.3.2  brj(2;5),  p=5 ----------------------------------------------
ev5 = [[1, 1], [0, 2], [1, 3], [2, 4]]
od5 = [[1, 0], [0, 1], [1, 2], [2, 3], [1, 4], [2, 5]]
# The Cartan matrix is the same as in 6.3.1, so pi_1 = -1/2 alpha_1 - alpha_2:
pi5 = [[F(-1, 2), F(-1)], [F(-1, 2), F(0)]]
iso5 = [[1, 0], [1, 4]]
tab5 = [((0, 1), (5, 11), (0, 1)), ((0, 2), (5, 15), (0, 0)),
        ((0, 3), (5, 19), (0, 4)), ((0, 4), (5, 23), (0, 3)),
        ((1, 0), (7, 9), (2, 4)), ((1, 1), (7, 13), (2, 3)),
        ((1, 2), (7, 17), (2, 2)), ((1, 3), (7, 21), (2, 1)),
        ((1, 4), (7, 25), (2, 0)),
        ((2, 0), (9, 11), (4, 1)), ((2, 1), (9, 15), (4, 0)),
        ((2, 2), (9, 19), (4, 4)), ((2, 3), (9, 23), (4, 3)),
        ((2, 4), (9, 27), (4, 2)),
        ((3, 0), (11, 13), (1, 3)), ((3, 1), (11, 17), (1, 2)),
        ((3, 2), (11, 21), (1, 1)), ((3, 3), (11, 25), (1, 0)),
        ((3, 4), (11, 29), (1, 4)),
        ((4, 0), (13, 15), (3, 0)), ((4, 1), (13, 19), (3, 4)),
        ((4, 2), (13, 23), (3, 3)), ((4, 3), (13, 27), (3, 2)),
        ((4, 4), (13, 31), (3, 1))]
run("brj(2;5)", 5, ev5, od5, pi5, iso5, tab5)

# ---- isotropy of the odd roots, over R vs. mod p ----------------------
for name, p, od in (("brj(2;3)", 3, od3), ("brj(2;5)", 5, od5)):
    print(f"{name}: odd positive roots, (a,a) over R  /  mod {p}")
    for a in od:
        v = ip(a, a)
        print(f"   {a[0]}a1+{a[1]}a2 : {v:>4}  /  {v % p}"
              + ("   <- isotropic over R" if v == 0 else
                 ("   <- isotropic only mod p" if v % p == 0 else "")))
    print()
