"""Which natural degree-5 expressions are the NEW identities?

Builds C_n (the identities that follow from lower degrees), then for each
candidate expression reports
    * whether it is an identity at all,
    * whether it is new (not in C_n),
    * the S_n-module it generates modulo C_n.
"""

import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exactlin import rref as ex_rref, rank as ex_rank  # noqa: E402

import volid
from volid import Space
from videal import (MSpace, identities, consequences, relabel_tree, msubst,
                    mmul, mcomm, mcirc, mjac, mjbar, madd, msmul, var,
                    mevaluate)
from vchar import module_of, decompose, CHARS

P = 2147483647


def build(n, p=P):
    ms = MSpace(n)
    lower = {k: identities(k, MSpace(k)) for k in range(3, n)}
    rows, seen = [], set()
    for f in consequences(lower, n):
        v = ms.vec(f)
        if not any(v):
            continue
        key = tuple(v)
        if key in seen:
            continue
        seen.add(key)
        rows.append([int(c) % p for c in v])
        if len(rows) > 4000:
            R, _ = ex_rref(rows, p=p)
            rows = [r for r in R if any(r)]
    R, piv = ex_rref(rows, p=p)
    R = [r for r in R if any(r)]
    return ms, R, list(piv)[:len(R)]


def reduce_mod(v, basis, pivots, p):
    v = [int(c) % p for c in v]
    for r, c in zip(basis, pivots):
        if v[c]:
            f = v[c] * pow(r[c], p - 2, p) % p
            v = [(a - f * b) % p for a, b in zip(v, r)]
    return v


def is_identity(f):
    """Exact check, over Q, that the expression vanishes identically."""
    return not mevaluate(f)


def orbit(f, n):
    """The S_n-orbit of an expression."""
    import itertools
    out = []
    for s in itertools.permutations(range(1, n + 1)):
        sub = {i + 1: s[i] for i in range(n)}
        out.append(msubst(f, sub))
    return out


def analyse(name, f, ms, basis, pivots, n, p=P):
    ident = is_identity(f)
    v = ms.vec(f)
    red = reduce_mod(v, basis, pivots, p)
    new = any(red)
    line = "%-46s identity: %-5s new: %-5s" % (name, "yes" if ident else "NO",
                                               "yes" if new else "no")
    if ident and new:
        orb = [ms.vec(g) for g in orbit(f, n)]
        rows = basis + [[int(c) % p for c in w] for w in orb]
        dec, char = module_of(rows, ms, n, p)
        dec0, char0 = module_of(basis, ms, n, p)
        d = {k: dec.get(k, 0) - dec0.get(k, 0) for k in set(dec) | set(dec0)}
        d = {k: m for k, m in d.items() if m}
        dim = sum(m * len(CHARS[n]["irr"]) * 0 + m * IRRDIM[n][k]
                  for k, m in d.items())
        line += "  generates dim %-3d = %s" % (
            dim, " + ".join("%d.%s" % (m, k) if m > 1 else k
                            for k, m in sorted(d.items())))
    print(line, flush=True)
    return ident, new


IRRDIM = {
    4: {"[4]": 1, "[3,1]": 3, "[2,2]": 2, "[2,1,1]": 3, "[1^4]": 1},
    5: {"[5]": 1, "[4,1]": 4, "[3,2]": 5, "[3,1,1]": 6, "[2,2,1]": 5,
        "[2,1,1,1]": 4, "[1^5]": 1},
}


# ----------------------------------------------------------- the candidates ---

def candidates5():
    a, b, c, d, e = (var(i) for i in range(1, 6))
    out = []
    j = mjbar(a, b, c)
    J = mjac(a, b, c)

    def der_circ(u, x, y):
        """[u, x o y] - [u,x] o y - [u,y] o x   -- Baranov (iv)/(v)"""
        return madd(mcomm(u, mcirc(x, y)),
                    msmul(-1, mcirc(mcomm(u, x), y)),
                    msmul(-1, mcirc(mcomm(u, y), x)))

    def der_prod(u, x, y):
        """u(xy) - (ux)y - x(uy)   -- Volichenko (c)"""
        return madd(mmul(u, mmul(x, y)),
                    msmul(-1, mmul(mmul(u, x), y)),
                    msmul(-1, mmul(x, mmul(u, y))))

    out.append(("(v)   [j(a,b,c), d o e] = [j,d] o e + [j,e] o d",
                der_circ(j, d, e)))
    out.append(("(v-J) the same with the Jacobi element J(a,b,c)",
                der_circ(J, d, e)))
    out.append(("(c-j) j(a,b,c)(de) = (j d)e + d(j e)",
                der_prod(j, d, e)))
    out.append(("(c-J) J(a,b,c)(de) = (J d)e + d(J e)",
                der_prod(J, d, e)))
    out.append(("      j(a,b,c) o (d o e) = 0", mcirc(j, mcirc(d, e))))
    out.append(("      j(a o b, c, d) o e = 0",
                mcirc(mjbar(mcirc(a, b), c, d), e)))
    out.append(("      j(ab, c, d) o e = 0",
                mcirc(mjbar(mmul(a, b), c, d), e)))
    out.append(("      J(ab, c, d) o e = 0",
                mcirc(mjac(mmul(a, b), c, d), e)))
    out.append(("(iv') [a o b, c o d] o e = 0",
                mcirc(mcomm(mcirc(a, b), mcirc(c, d)), e)))
    out.append(("      [[a o b, c], d o e] - ...",
                der_circ(mcomm(mcirc(a, b), c), d, e)))
    out.append(("      j(a,b,c o d) o e", mcirc(mjbar(a, b, mcirc(c, d)), e)))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    ms, basis, piv = build(n)
    print("C_%d built: dim %d   [%.1fs]" % (n, len(basis), time.time() - t0))
    for name, f in candidates5():
        analyse(name, f, ms, basis, piv, n)
