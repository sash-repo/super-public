#!/usr/bin/env python3
"""
Exact arithmetic over F_2 for the Lie algebra vect(3; N) of vector fields with
divided-power coefficients, characteristic 2, and for the sl(3)-module structure
of its degree-1 component.

Purpose: reconstruct the Weisfeiler grading of the Eick
algebras 15:7, 15:8, 16:3, 16:4 stated in the bestiary of simple modular Lie algebras,
    g_{-1} = V (dim 3),  g_0 = sl(V) (dim 8),  dim g_1 = 4 (resp. 5),
by classifying the sl(3)-submodules W of vect(3;1)_1 of dimension 4 and 5, and
by testing which of them are candidate degree-1 components of a Z-graded
15- resp. 16-dimensional Lie algebra (i.e. [W,W] = 0 and W^{[2]} = 0).

Basis conventions.  O(3; 1) = K[x_1,x_2,x_3]/(x_i^2), monomials indexed by a
bitmask r in {0,1}^3.  A vector field is a dict {(r, i): 1} meaning x^r d_i.
Everything is over F_2, so coefficients are just membership in a set.
"""

from itertools import combinations

NVARS = 3
MONS = list(range(1 << NVARS))          # bitmask of exponents, each in {0,1}


def deg(r):
    return bin(r).count("1")


# ---------------------------------------------------------------- vector fields
# A field is a frozenset of pairs (r, i) -- the terms present with coefficient 1.

def fadd(*fields):
    acc = set()
    for f in fields:
        acc ^= set(f)
    return frozenset(acc)


def mon_mul(r, s):
    """x^(r) * x^(s) in O(3;1): zero if the supports overlap."""
    if r & s:
        return None
    return r | s


def d(r, i):
    """partial_i applied to x^(r)."""
    if not (r >> i) & 1:
        return None
    return r & ~(1 << i)


def bracket(f, g):
    """[f, g] = f(g) + g(f) for vector fields over F_2."""
    out = set()
    for (r, i) in f:
        for (s, j) in g:
            # x^r d_i (x^s d_j) = (d_i x^s) x^r d_j
            ds = d(s, i)
            if ds is not None:
                m = mon_mul(r, ds)
                if m is not None:
                    out ^= {(m, j)}
            # + x^s d_j (x^r d_i)
            dr = d(r, j)
            if dr is not None:
                m = mon_mul(s, dr)
                if m is not None:
                    out ^= {(m, i)}
    return frozenset(out)


def square(f):
    """x -> x^[2] = the square of the operator; for p=2, [f,f]=0 so use f o f."""
    out = set()
    for (r, i) in f:
        for (s, j) in f:
            ds = d(s, i)
            if ds is not None:
                m = mon_mul(r, ds)
                if m is not None:
                    out ^= {(m, j)}
    return frozenset(out)


def divergence(f):
    """Div(sum f_i d_i) = sum d_i(f_i), an element of O(3;1)."""
    out = set()
    for (r, i) in f:
        dr = d(r, i)
        if dr is not None:
            out ^= {dr}
    return frozenset(out)


def show(f):
    if not f:
        return "0"
    names = []
    for (r, i) in sorted(f):
        mon = "".join("x%d" % (k + 1) for k in range(NVARS) if (r >> k) & 1) or "1"
        names.append("%s d%d" % (mon, i + 1))
    return " + ".join(names)


# ---------------------------------------------------------------- linear algebra
def to_vec(f, basis_index):
    v = 0
    for t in f:
        v |= 1 << basis_index[t]
    return v


def rref(vectors):
    """Canonical reduced row echelon form of a list of F_2 bit vectors."""
    rows = []                                  # pivot bit -> row, kept as dict
    piv = {}
    for v in vectors:
        for p in sorted(piv, reverse=True):
            if (v >> p) & 1:
                v ^= piv[p]
        if v:
            p = v.bit_length() - 1
            # back-substitute into the existing rows
            for q in list(piv):
                if (piv[q] >> p) & 1:
                    piv[q] ^= v
            piv[p] = v
    rows = [piv[p] for p in sorted(piv, reverse=True)]
    return rows


def reduce_vec(v, rows):
    for r in rows:
        v = min(v, v ^ r)
    return v


def span(vectors):
    return tuple(rref(list(vectors)))


def in_span(v, rows):
    return reduce_vec(v, rows) == 0


def all_elements(rows):
    els = [0]
    for r in rows:
        els += [e ^ r for e in els]
    return els


# ---------------------------------------------------------------- graded pieces
def graded_basis(k):
    """Basis of vect(3;1)_k: monomials of degree k+1 times a derivative."""
    return [(r, i) for r in MONS for i in range(NVARS) if deg(r) == k + 1]


G_m1 = graded_basis(-1)      # d_i           : 3
G_0 = graded_basis(0)        # x_a d_b       : 9
G_1 = graded_basis(1)        # x_a x_b d_c   : 9
G_2 = graded_basis(2)        # x1x2x3 d_c    : 3

IDX1 = {t: n for n, t in enumerate(G_1)}
IDX2 = {t: n for n, t in enumerate(G_2)}


def sl3_basis():
    """sl(3) inside gl(3) = span(x_a d_b), p=2: trace-zero, dim 8."""
    out = []
    for a in range(NVARS):
        for b in range(NVARS):
            if a != b:
                out.append(frozenset({(1 << a, b)}))
    # diagonal trace-zero: E11+E22, E22+E33
    out.append(frozenset({(1 << 0, 0), (1 << 1, 1)}))
    out.append(frozenset({(1 << 1, 1), (1 << 2, 2)}))
    return out


SL3 = sl3_basis()


def act_matrix_rows():
    """action of each sl(3) generator on G_1, as a map on bit vectors."""
    mats = []
    for X in SL3:
        col = {}
        for t in G_1:
            img = bracket(X, frozenset({t}))
            assert all(deg(r) == 2 for (r, i) in img), show(img)
            col[t] = to_vec(img, IDX1)
        mats.append(col)
    return mats


MATS = act_matrix_rows()


def apply_gen(v, col):
    out = 0
    for t, n in IDX1.items():
        if (v >> n) & 1:
            out ^= col[t]
    return out


def submodule_generated(vs):
    """Closure under the sl(3)-action; linearity => acting on a basis suffices."""
    rows = rref([v for v in vs if v])
    queue = list(rows)
    while queue:
        v = queue.pop()
        for col in MATS:
            w = apply_gen(v, col)
            if w and not in_span(w, rows):
                rows = rref(rows + [w])
                queue.append(w)
    return tuple(rows)


def all_submodules(maxdim=None):
    """Every submodule is a sum of cyclic ones, so close the cyclic ones under
    sums; if maxdim is given, only submodules of dimension <= maxdim are kept
    (legitimate, since a summand of such a submodule is again of dim <= maxdim)."""
    cyc = set()
    for v in range(1, 1 << len(G_1)):
        s = submodule_generated([v])
        if maxdim is None or len(s) <= maxdim:
            cyc.add(s)
    subs = set(cyc)
    subs.add(())
    frontier = set(subs)
    while frontier:
        new = set()
        for a in frontier:
            for b in cyc:
                s = tuple(rref(list(a) + list(b)))
                if (maxdim is None or len(s) <= maxdim) and s not in subs:
                    new.add(s)
        subs |= new
        frontier = new
    return subs


def vec_to_field(v):
    terms = set()
    for t, n in IDX1.items():
        if (v >> n) & 1:
            terms.add(t)
    return frozenset(terms)


def main():
    print("dim vect(3;1)_k :", [len(graded_basis(k)) for k in (-1, 0, 1, 2)],
          " total", 3 + 9 + 9 + 3)
    print("dim sl(3) =", len(SL3))

    subs = all_submodules()
    by_dim = {}
    for s in subs:
        by_dim.setdefault(len(s), []).append(s)
    print("\nsl(3)-submodules of vect(3;1)_1 (dim 9), count by dimension:")
    for k in sorted(by_dim):
        print("   dim %d : %d submodule(s)" % (k, len(by_dim[k])))

    # divergence-free part
    div0 = [to_vec(frozenset({t}), IDX1) for t in G_1 if not divergence(frozenset({t}))]
    svect1 = rref([to_vec(f, IDX1) for f in
                   [frozenset({t}) for t in G_1] if not divergence(f)])
    # general element of svect_1: solve Div = 0 on the 9-dim space
    rows = []
    for v in range(1, 1 << len(G_1)):
        if not divergence(vec_to_field(v)):
            rows.append(v)
    svect1 = rref(rows)
    print("\ndim svect(3;1)_1 =", len(svect1))
    print("   basis:", [show(vec_to_field(v)) for v in svect1])

    for target in (4, 5):
        print("\n=== submodules of dimension %d ===" % target)
        for s in by_dim.get(target, []):
            W = [vec_to_field(v) for v in s]
            ab = all(not bracket(a, b) for a in W for b in W)
            sq = all(not square(a) for a in
                     [vec_to_field(v) for v in all_elements(list(s)) if v])
            insv = all(in_span(v, svect1) for v in s)
            print("  W = %s" % [show(w) for w in W])
            print("     [W,W]=0 : %s   W^[2]=0 : %s   W < svect_1 : %s"
                  % (ab, sq, insv))


if __name__ == "__main__":
    main()
