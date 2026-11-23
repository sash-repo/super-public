"""
Problem 5.22.1 for g = vect^L(1|2): is H^1(g; g (x) g) = 0?  Proposition 5.22.5, Lemma 5.22.6
and Remark 5.22.7.

Remark 5.22.2 reduces H^*(g; M) to the cochains of degree 0 by means of the grading element
z = 2t d_t + theta_1 d_{theta_1} + theta_2 d_{theta_2}.  The superalgebra vect^L(1|2) has, more
than that, the three-dimensional maximal toral subalgebra

       t = < t d_t,  theta_1 d_{theta_1},  theta_2 d_{theta_2} >  (formula (40)),

all of whose elements lie in g.  Cartan's homotopy formula applies to each of them, so
H^*(g; M) is computed by the cochains of WEIGHT 0 for the whole of t, and the weight-0
component of g is t itself:

       g_0 = t,   dim g_0 = 3,

with every root space of dimension at most 3.  A weight-0 1-cocycle then vanishes on t when
M^g = 0, whence H^1(g; M) = H^1(g, t; M) (formula (42)).  In a weight space of g (x) g only
the splitting of the t-weight between the two factors is free, so the cocycle condition is a
recursion in one integer variable.

Printed: part 1) t is a torus acting by the weights (41); part 2) g_0 = t and the root-space
dimensions; part 3) (g (x) g)^g has no element of weight 0 in the window |a| <= 3; part 4)
dim Z^1 and dim B^1 of the relative complex over the window |a| <= N (all cocycle equations
inside the window).  Part 4 is run for N = 1 here; the windows |a| <= 1 and |a| <= 2 of
Remark 5.22.7 are done by vect2b.py, which needs far less memory.

Everything is exact rational arithmetic.

Run:  python3 vect2.py
"""

import itertools
import os
import sys
from fractions import Fraction as Q

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "lib"), HERE]

from yb import Vect
from exactlin import rank, nullspace, backend

FAILED = []


def say(s=""):
    print(s, flush=True)


def hdr(s):
    say("\n" + "=" * 74)
    say(s)
    say("=" * 74)


def check(name, ok):
    say("  %-64s %s" % (name, "ok" if ok else "FAILED"))
    if not ok:
        FAILED.append(name)
    return ok


# ------------------------------------------------------------------ the algebra
G = Vect(2)                      # vect^L(1|2), keys (mu, a, S)
ODD = (1, 2)


def weight(key):
    """the weight of a basis element for the torus t = <t d_t, th_1 d_1, th_2 d_2>.

    [t d_t,          t^a th_S d_mu] = (a - [mu = t]) * ...
    [th_i d_{th_i},  t^a th_S d_mu] = ([i in S] - [mu = i]) * ...
    """
    mu, a, S = key
    m = a - (1 if mu == 't' else 0)
    n = tuple((1 if i in S else 0) - (1 if mu == i else 0) for i in ODD)
    return (m,) + n


def mono(key, c=1):
    return {key: c}


TORUS = [('t', 1, ()), (1, 0, (1,)), (2, 0, (2,))]


def basis(N):
    """the basis of g with Laurent exponents |a| <= N"""
    return G.keys(range(-N, N + 1))


# ============================================================ 1) the torus
def part1(N=4):
    hdr("1) t = < t d_t, th_1 d_{th_1}, th_2 d_{th_2} > is a torus in g,"
        "\n   and it acts on the basis element of weight w by w.")
    B = basis(N)
    say("  window |a| <= %d, dim of the window = %d" % (N, len(B)))

    ok = True
    for h in TORUS:
        check("%-22s lies in g (it is a basis element)" % str(h), h in B)
    for j, h in enumerate(TORUS):
        good = True
        for k in B:
            br = G.brk(mono(h), mono(k))
            w = weight(k)[j]
            want = {k: w} if w else {}
            if {a: b for a, b in br.items() if b} != want:
                good = False
                say("    MISMATCH on %s: %s" % (str(k), br))
        ok &= check("ad(%s) is diagonal with eigenvalue w_%d" % (str(h), j + 1), good)

    # the torus is abelian
    ab = all(not G.brk(mono(h1), mono(h2)) for h1 in TORUS for h2 in TORUS)
    ok &= check("t is abelian", ab)
    return ok


# ================================================= 2) g_0 = t, and the roots
def part2(N=6):
    hdr("2) the weight-0 component of g is t itself, and the root spaces")
    B = basis(N)
    zero = [k for k in B if weight(k) == (0, 0, 0)]
    say("  weight-0 basis elements: %s" % sorted(map(str, zero)))
    ok = check("g_0 = t, of dim 3",
              sorted(map(str, zero)) == sorted(map(str, TORUS)))

    bysize = {}
    for k in B:
        w = weight(k)
        if w != (0, 0, 0):
            bysize.setdefault(w, []).append(k)
    dims = {}
    for w, ks in bysize.items():
        dims[len(ks)] = dims.get(len(ks), 0) + 1
    say("  root-space dimensions over the window (dim: how many roots):")
    for d in sorted(dims):
        say("      dim %d : %d roots" % (d, dims[d]))
    ok &= check("every root space is of dim <= 3", max(dims) <= 3)

    # which roots have dim 3 / 2 / 1 -- a description, not just a count
    ex = {d: [] for d in dims}
    for w, ks in sorted(bysize.items()):
        if len(ex[len(ks)]) < 3:
            ex[len(ks)].append((w, sorted(map(str, ks))))
    for d in sorted(ex):
        say("    sample roots of dim %d:" % d)
        for w, ks in ex[d]:
            say("        %-14s %s" % (str(w), ks))
    return ok


# =============================== 3) the module g (x) g and its invariants
def tensor_weight(p):
    (k1, k2) = p
    a, b = weight(k1), weight(k2)
    return tuple(x + y for x, y in zip(a, b))


def act(x, p):
    """x . (u (x) v) = [x,u] (x) v + (-1)^{p(x)p(u)} u (x) [x,v], as a dict on pairs"""
    u, v = p
    out = {}
    for k, c in G.brk(mono(x), mono(u)).items():
        out[(k, v)] = out.get((k, v), 0) + c
    sgn = (-1) ** (G.par(x) * G.par(u))
    for k, c in G.brk(mono(x), mono(v)).items():
        out[(u, k)] = out.get((u, k), 0) + sgn * c
    return {k: c for k, c in out.items() if c}


def part3(N=3):
    hdr("3) the invariants of g (x) g")
    B = basis(N)
    pairs = [(u, v) for u in B for v in B if tensor_weight((u, v)) == (0, 0, 0)]
    idx = {p: i for i, p in enumerate(pairs)}
    say("  window |a| <= %d : dim (g (x) g)_0 inside the window = %d"
        % (N, len(pairs)))

    # The TARGET of r |--> (x.r)_x is not the weight-0 space but the whole of
    # g (x) g, one weight space per root: x.(u (x) v) has the weight of x
    # ADDED.  Indexing the target by the weight-0 pairs would make the matrix
    # empty and the test vacuous.
    inwin = set(B)
    rows = {}
    for x in B:
        for p in pairs:
            for k, c in act(x, p).items():
                if k[0] not in inwin or k[1] not in inwin:
                    continue                       # the window truncation
                rows.setdefault((x, k), {})[idx[p]] = \
                    rows.setdefault((x, k), {}).get(idx[p], 0) + c
    rows = [[Q(r.get(j, 0)) for j in range(len(pairs))] for r in rows.values()]
    if rows:
        r = rank(rows)
        inv = len(pairs) - r
    else:
        inv = len(pairs)
    say("  rank of the action = %d, so the space of invariants of weight 0"
        % (r if rows else 0))
    say("  inside the window is of dimension %d" % inv)
    return check("(g (x) g)^g has no element of weight 0 in the window", inv == 0)


# ======================= 4) H^1 of the RELATIVE complex, over a window
def part4(N=2):
    hdr("4) H^1(g, t; g (x) g) over a window: Z^1 and B^1")
    B = basis(N)
    roots = [k for k in B if weight(k) != (0, 0, 0)]
    pairs = [(u, v) for u in B for v in B]
    bywt = {}
    for p in pairs:
        bywt.setdefault(tensor_weight(p), []).append(p)

    # ---- the unknowns: phi(x) in M_{weight(x)}, for every root vector x
    cols, colof = [], {}
    for x in roots:
        for p in bywt.get(weight(x), []):
            colof[(x, p)] = len(cols)
            cols.append((x, p))
    say("  window |a| <= %d: dim g = %d, root vectors = %d, unknowns = %d"
        % (N, len(B), len(roots), len(cols)))

    def phi_of(x):
        """the formal value of phi on a basis element (0 on the torus)"""
        if weight(x) == (0, 0, 0):
            return {}
        return {p: colof[(x, p)] for p in bywt.get(weight(x), [])}

    # ---- the cocycle equations, imposed only where everything stays inside
    eqs = {}
    nrow = 0
    inside = set(B)
    for i, x in enumerate(B):
        for y in B[i:]:
            br = G.brk(mono(x), mono(y))
            if any(k not in inside for k in br):
                continue                               # truncation: skip
            # d phi (x,y) = x.phi(y) -(-1)^{pp} y.phi(x) - phi([x,y])
            acc = {}

            def addv(target, c):
                acc[target] = acc.get(target, 0) + c

            for p, col in phi_of(y).items():
                for k, c in act(x, p).items():
                    addv((k, col), c)
            sgn = -((-1) ** (G.par(x) * G.par(y)))
            for p, col in phi_of(x).items():
                for k, c in act(y, p).items():
                    addv((k, col), sgn * c)
            for k, c in br.items():
                for p, col in phi_of(k).items():
                    addv((p, col), -c)
            # split into one equation per target basis vector of M
            byt = {}
            for (tgt, col), c in acc.items():
                if c:
                    byt.setdefault(tgt, {})[col] = byt.setdefault(tgt, {}).get(col, 0) + c
            for tgt, row in byt.items():
                if row:
                    eqs[nrow] = row
                    nrow += 1
    say("  cocycle equations kept (both sides inside the window): %d" % nrow)

    # ---- rank of the system  =>  dim Z^1 inside the window
    if nrow:
        mat = [[Q(eqs[r].get(j, 0)) for j in range(len(cols))] for r in sorted(eqs)]
        rk = rank(mat)
    else:
        rk = 0
    z1 = len(cols) - rk
    say("  rank = %d,  so dim Z^1 (window, truncated conditions) = %d" % (rk, z1))

    # ---- the coboundaries: r in M of weight 0, phi = d r,  phi(x) = x.r
    M0 = bywt.get((0, 0, 0), [])
    say("  dim M_0 inside the window = %d" % len(M0))
    brows = []
    for p0 in M0:
        vec = [Q(0)] * len(cols)
        skip = False
        for x in roots:
            for k, c in act(x, p0).items():
                key = (x, k)
                if key not in colof:
                    skip = True                       # value left the window
                    break
                vec[colof[key]] += c
            if skip:
                break
        if not skip:
            brows.append(vec)
    b1 = rank(brows) if brows else 0
    say("  coboundaries d r with r and d r inside the window: %d of %d,"
        % (len(brows), len(M0)))
    say("  their rank = dim B^1 (window) = %d" % b1)
    say("")
    say("  dim Z^1 - dim B^1 (window) = %d" % (z1 - b1))
    say("  NB: the truncation makes Z^1 an OVER-estimate (conditions crossing")
    say("      the window edge are dropped) and B^1 an UNDER-estimate, so a")
    say("      difference 0 means Z^1 = B^1 over this window; it does not prove")
    say("      H^1 = 0 for g.")
    return z1, b1


def main():
    say("exact-linear-algebra backend: %s" % backend())
    part1(4)
    part2(6)
    part3(3)
    for N in (1,):
        part4(N)
    hdr("SUMMARY")
    if FAILED:
        say("FAILED: %s" % FAILED)
    else:
        say("every structural check passed")


if __name__ == "__main__":
    main()
