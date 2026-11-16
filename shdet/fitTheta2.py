"""
The corrected form of the table (70) of the text.

Same support as the printed table -- the same list of non-zero named pairs and
the same dependence on the index -- but every index pair carries its OWN
scalar, since the printed table's uniform alpha/beta split is what is in
question:

  c(L_m, L_n)        = A       * m(m^2-1) d0
  c(G^i_m, G^i_n)    = a_i     * m(m+1)   d1        i = 1..4
  c(T^ij_m, T^ij_n)  = b_{ij}  * m        d0        6 pairs
  c(F^i_m, F^i_n)    = f_i                d1        i = 1..4
  c(S_m, S_n)        = s       * 1/m      d0
  c(L_m, S_n)        = (p + q m)          d0
  c(G^i_m, F^i_n)    = (u_i + v_i m)      d1        i = 1..4
  c(T^ij_m, T^kl_n)  = w_{ij}  * m        d0        {ij},{kl} complementary

29 unknowns.  The 2-cocycle identity is imposed on every triple in a window.
"""

from fractions import Fraction as Q
from itertools import product, combinations
import sys

from stringy import mono, parity
from nsTheta import bracket, monomials, weight, name_of, ODD, ODD_NAMES

TPAIRS = [(i, j) for i, j in combinations(ODD, 2)]          # 6
COMPL = [((1, 2), (3, 4)), ((1, 3), (2, 4)), ((1, 4), (2, 3))]

# ---- index layout
IDX = {}
def _reg(name):
    IDX[name] = len(IDX)

_reg("A")
for i in ODD:
    _reg("a%d" % i)
for t in TPAIRS:
    _reg("b%d%d" % t)
for i in ODD:
    _reg("f%d" % i)
_reg("s")
_reg("p")
_reg("q")
for i in ODD:
    _reg("u%d" % i)
for i in ODD:
    _reg("v%d" % i)
for t, _ in COMPL:
    _reg("w%d%d" % t)

N = len(IDX)


def zerov():
    return [Q(0)] * N


def entry(nx, ny):
    (n1, m), (n2, n) = nx, ny
    d0 = (m + n == 0)
    d1 = (m + n + 1 == 0)
    v = zerov()
    if n1 == "L" and n2 == "L":
        if d0:
            v[IDX["A"]] = Q(m * (m * m - 1))
        return v
    if n1 == n2 and n1.startswith("G"):
        if d1:
            v[IDX["a%s" % n1[1]]] = Q(m * (m + 1))
        return v
    if n1 == n2 and n1.startswith("T"):
        if d0:
            v[IDX["b%s%s" % (n1[1], n1[2])]] = Q(m)
        return v
    if n1 == n2 and n1.startswith("F"):
        if d1:
            v[IDX["f%s" % n1[1]]] = Q(1)
        return v
    if n1 == "S" and n2 == "S":
        if d0 and m != 0:
            v[IDX["s"]] = Q(1, m)
        return v
    if n1 == "L" and n2 == "S":
        if d0:
            v[IDX["p"]], v[IDX["q"]] = Q(1), Q(m)
        return v
    if n1[0] == "G" and n2[0] == "F" and n1[1] == n2[1]:
        if d1:
            v[IDX["u%s" % n1[1]]], v[IDX["v%s" % n1[1]]] = Q(1), Q(m)
        return v
    if n1.startswith("T") and n2.startswith("T") and n1 != n2:
        s1 = (int(n1[1]), int(n1[2]))
        s2 = (int(n2[1]), int(n2[2]))
        if set(s1) | set(s2) == set(ODD) and d0:
            for x, y in COMPL:
                if (s1, s2) == (x, y):
                    v[IDX["w%d%d" % x]] = Q(m)
                    return v
                if (s1, s2) == (y, x):
                    v[IDX["w%d%d" % x]] = Q(-m)   # forced by skew-symmetry
                    return v
        return v
    return None


def c_mono(k1, k2):
    nx, ny = name_of(k1), name_of(k2)
    s = nx[2] * ny[2]
    ax, ay = (nx[0], nx[1]), (ny[0], ny[1])
    v = entry(ax, ay)
    if v is None:
        w = entry(ay, ax)
        if w is None:
            return zerov()
        sg = -((-1) ** (int(ODD_NAMES(ax)) * int(ODD_NAMES(ay))))
        v = [sg * x for x in w]
    return [s * x for x in v]


def c_fun(f, g):
    out = zerov()
    for k1, c1 in f.items():
        for k2, c2 in g.items():
            v = c_mono(k1, k2)
            for i in range(N):
                if v[i]:
                    out[i] += c1 * c2 * v[i]
    return out


def rows(keys, kappa=Q(1), drop_S0=True):
    W = {}
    for k in keys:
        W.setdefault(weight(k), []).append(k)
    seen = set()
    for kf, kg in product(keys, repeat=2):
        for kh in W.get(-(weight(kf) + weight(kg)), ()):
            if drop_S0 and any(name_of(k)[:2] == ("S", 0) for k in (kf, kg, kh)):
                continue
            f, g, h = mono(*kf), mono(*kg), mono(*kh)
            lhs = c_fun(f, bracket(g, h, kappa))
            r1 = c_fun(bracket(f, g, kappa), h)
            sgn = (-1) ** (parity(f) * parity(g))
            r2 = c_fun(g, bracket(f, h, kappa))
            row = tuple(lhs[i] - r1[i] - sgn * r2[i] for i in range(N))
            if any(row) and row not in seen:
                seen.add(row)
                yield row


def solve(rowlist):
    piv = {}
    for row in rowlist:
        row = {i: v for i, v in enumerate(row) if v}
        while row:
            p = min(row)
            if p in piv:
                c = row[p]
                for i, v in piv[p].items():
                    row[i] = row.get(i, 0) - c * v
                row = {i: v for i, v in row.items() if v}
            else:
                piv[p] = {i: Q(v) / row[p] for i, v in row.items()}
                break
    free = [j for j in range(N) if j not in piv]
    basis = []
    for j in free:
        v = [Q(0)] * N
        v[j] = Q(1)
        for p in sorted(piv, reverse=True):
            s = sum(piv[p][i] * v[i] for i in piv[p] if i != p)
            v[p] = -s
        basis.append(v)
    return free, basis


REV = {v: k for k, v in IDX.items()}


def show(v):
    return "  ".join("%s=%s" % (REV[i], v[i]) for i in range(N) if v[i])


if __name__ == "__main__":
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    kappa = Q(int(sys.argv[2])) if len(sys.argv) > 2 else Q(1)
    keys = monomials(range(-B, B + 1))
    rl = list(rows(keys, kappa))
    print("window |a| <= %d, kappa = %s : %d equations on %d unknowns"
          % (B, kappa, len(rl), N))
    free, basis = solve(rl)
    print("solution space dimension %d; free: %s"
          % (len(basis), [REV[j] for j in free]))
    for v in basis:
        print("   " + show(v))
