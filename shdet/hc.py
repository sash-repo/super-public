"""
The leading term of HC(Omega) for k^M(1|7) -- eq. (48) of the text --
and the differences HC(Omega)(lambda) - HC(Omega)(lambda-mu) for each of the
four simple roots.

The eight elements H_{ijk} (i,j,k in {0,1}) are coordinates on h_ev; the
maximal torus t is spanned by H_000, H_100, H_010, H_001, so a weight mu in
t^* shifts exactly those four coordinates and leaves H_011, H_101, H_110,
H_111 alone.  Everything below is exact polynomial arithmetic over Z.
"""

from itertools import product
from fractions import Fraction as Q

IJK = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
NAME = {t: "H_%d%d%d" % t for t in IJK}
TORUS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]


# ---- polynomials in the eight H's: dict {sorted tuple of variables: coeff}

def pmul(p, q):
    out = {}
    for a, ca in p.items():
        for b, cb in q.items():
            k = tuple(sorted(a + b))
            out[k] = out.get(k, 0) + ca * cb
    return {k: v for k, v in out.items() if v}


def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def pscal(c, p):
    return {k: c * v for k, v in p.items() if c * v}


def var(t):
    return {(t,): 1}


def one():
    return {(): 1}


def show(p):
    if not p:
        return "0"
    terms = []
    for k in sorted(p, key=lambda k: (len(k), k)):
        c = p[k]
        body = "".join(NAME[t] for t in k) if k else "1"
        if c == 1 and k:
            terms.append(body)
        elif c == -1 and k:
            terms.append("-" + body)
        else:
            terms.append("%s%s" % (c, body if k else ""))
    s = " + ".join(terms).replace("+ -", "- ")
    return s


# ---- HC_cubic, built from the definition rather than copied from the text
# the sum runs over i,j,k,i',j',k' in {0,1} with i+i', j+j', k+k' <= 1, and the
# summand is H_{1-i,1-j,1-k} H_{i+i',j+j',k+k'} H_{1-i',1-j',1-k'}

def hc_cubic():
    out = {}
    for i, j, k, ip, jp, kp in product((0, 1), repeat=6):
        if i + ip > 1 or j + jp > 1 or k + kp > 1:
            continue
        term = pmul(pmul(var((1 - i, 1 - j, 1 - k)),
                         var((i + ip, j + jp, k + kp))),
                    var((1 - ip, 1 - jp, 1 - kp)))
        out = padd(out, term)
    return out


def printed_cubic():
    """eq. (48) exactly as printed in the text."""
    H = var
    return padd(
        pscal(3, pmul(pmul(H((1, 1, 1)), H((1, 1, 1))), H((0, 0, 0)))),
        pscal(6, padd(
            pmul(pmul(H((1, 1, 1)), H((1, 0, 0))), H((0, 1, 1))),
            pmul(pmul(H((1, 1, 1)), H((0, 1, 0))), H((1, 0, 1))),
            pmul(pmul(H((1, 1, 1)), H((0, 0, 1))), H((1, 1, 0))),
            pmul(pmul(H((0, 1, 1)), H((1, 0, 1))), H((1, 1, 0))))))


def shift(p, mu):
    """substitute H_t -> H_t - mu[t] for t in the torus (and leave the rest)."""
    out = {}
    for k, c in p.items():
        # expand the product of (H_t - mu_t)
        cur = {(): c}
        for t in k:
            if t in mu:
                cur = pmul(cur, padd(var(t), {(): -mu[t]}))
            else:
                cur = pmul(cur, var(t))
        out = padd(out, cur)
    return out


SIMPLE = {
    "(-1,1,1,1)": {(0, 0, 0): -1, (1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1},
    "(0,1,-1,0)": {(1, 0, 0): 1, (0, 1, 0): -1},
    "(0,0,1,-1)": {(0, 1, 0): 1, (0, 0, 1): -1},
    "(0,0,1,1)": {(0, 1, 0): 1, (0, 0, 1): 1},
    "(1,-1,-1,-1)": {(0, 0, 0): 1, (1, 0, 0): -1, (0, 1, 0): -1, (0, 0, 1): -1},
}


if __name__ == "__main__":
    built = hc_cubic()
    printed = printed_cubic()
    print("(48) rebuilt from its own defining sum:")
    print("   ", show(built))
    print("(48) as printed in the text:")
    print("   ", show(printed))
    print("   identical:", built == printed)
    print()
    for nm, mu in SIMPLE.items():
        d = padd(built, pscal(-1, shift(built, mu)))
        print("mu = %-14s HC(l) - HC(l-mu) = %s" % (nm, show(d)))
    print()
    print("the text's claim for mu = (1,-1,-1,-1):")
    print("    H_111(3H_111 - 6H_100 - 6H_010 - 6H_001)")
    claim = pmul(var((1, 1, 1)),
                 padd(pscal(3, var((1, 1, 1))),
                      pscal(-6, var((1, 0, 0))),
                      pscal(-6, var((0, 1, 0))),
                      pscal(-6, var((0, 0, 1)))))
    truth = padd(built, pscal(-1, shift(built, SIMPLE["(1,-1,-1,-1)"])))
    print("    equals the computed difference:", claim == truth)
    fixed = pmul(var((1, 1, 1)),
                 padd(pscal(3, var((1, 1, 1))),
                      pscal(-6, var((0, 1, 1))),
                      pscal(-6, var((1, 0, 1))),
                      pscal(-6, var((1, 1, 0)))))
    print("    with H_100,H_010,H_001 replaced by H_011,H_101,H_110:",
          fixed == truth)
