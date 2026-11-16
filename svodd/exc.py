#!/usr/bin/env python3
"""Audit of the three exceptional rows 17)-19) of Table 1.5 of Chapter 1 of "Seminar on Supersymmetries", vol. 2:
ag(2) = G(3), ab(3) = F(4), osp_a(2|4) = D(2,1;a).

Same reading as chars.py: the marks define the Z-grading in which a marked simple
root has degree 1 and an unmarked one degree 0; the centralizer of the grading
element is

    Z_h = g_0 = h + sum of g_alpha over the roots alpha whose support avoids
                every marked node,

so  sdim Z_h = ( rank g + #{even roots of degree 0} | #{odd roots of degree 0} ).

Unlike the classical series, the exceptional root systems have no shuffle
description of their simple systems, so the bases are enumerated honestly: a
positive system is cut out by a linear functional not vanishing on any root, and
the base is the set of positive roots that are not a sum of two positive ones.
Every chamber of the arrangement is met by sweeping integer functionals; the
sweep is repeated at two ranges and the answers compared.

The bilinear form is carried along so that a node can be called even, odd
isotropic (crossed) or odd non-isotropic (black), and so that the bonds of the
drawn diagram can be compared with the computed ones.

Exact arithmetic throughout (Fraction); no floating point anywhere.
"""
import itertools
from fractions import Fraction as F


def V(*xs):
    return tuple(F(x) for x in xs)


# --------------------------------------------------------------- the algebras
def ag2():
    """G(3) = ag(2), rank 3, sdim (17|14).
    Coordinates (e1,e2,d) with e3 = -e1-e2.
    Even roots: G_2 = {+-e_i} U {e_i-e_j}, and +-2d.
    Odd roots:  +-d and +-e_i+-d.
    Form: (e_i,e_j) = delta_ij - 1/3, (d,d) = -2/3, e _|_ d.  Then d is
    non-isotropic (2d is a root) and every e_i+-d is isotropic."""
    e = [V(1, 0, 0), V(0, 1, 0), V(-1, -1, 0)]
    d = V(0, 0, 1)

    def lin(*terms):
        out = [F(0)] * 3
        for c, v in terms:
            for k in range(3):
                out[k] += c * v[k]
        return tuple(out)

    roots = []
    for i in range(3):
        for s in (1, -1):
            roots.append((lin((s, e[i])), 0))
    for i in range(3):
        for j in range(3):
            if i != j:
                roots.append((lin((1, e[i]), (-1, e[j])), 0))
    for s in (1, -1):
        roots.append((lin((2 * s, d)), 0))
    for s in (1, -1):
        roots.append((lin((s, d)), 1))
    for i in range(3):
        for s1 in (1, -1):
            for s2 in (1, -1):
                roots.append((lin((s1, e[i]), (s2, d)), 1))
    G = [[F(2, 3), F(-1, 3), F(0)],
         [F(-1, 3), F(2, 3), F(0)],
         [F(0), F(0), F(-2, 3)]]
    return "ag(2)", roots, G, 3, (17, 14)


def ab3():
    """F(4) = ab(3), rank 4, sdim (24|16).
    Coordinates (e1,e2,e3,d).
    Even roots: B_3 = {+-e_i, +-e_i+-e_j}, and +-d.
    Odd roots:  (+-e1+-e2+-e3+-d)/2.
    Form: (e_i,e_j) = delta_ij, (d,d) = -3, e _|_ d; every odd root isotropic."""
    roots = []
    for i in range(3):
        for s in (1, -1):
            v = [0, 0, 0, 0]
            v[i] = s
            roots.append((V(*v), 0))
    for i in range(3):
        for j in range(i + 1, 3):
            for s1 in (1, -1):
                for s2 in (1, -1):
                    v = [0, 0, 0, 0]
                    v[i], v[j] = s1, s2
                    roots.append((V(*v), 0))
    for s in (1, -1):
        roots.append((V(0, 0, 0, s), 0))
    for ss in itertools.product((1, -1), repeat=4):
        roots.append((tuple(F(s, 2) for s in ss), 1))
    G = [[F(1), F(0), F(0), F(0)],
         [F(0), F(1), F(0), F(0)],
         [F(0), F(0), F(1), F(0)],
         [F(0), F(0), F(0), F(-3)]]
    return "ab(3)", roots, G, 4, (24, 16)


def dab(a=F(1, 3)):
    """D(2,1;a) = osp_a(2|4) of the table, rank 3, sdim (9|8).
    Coordinates (e1,e2,e3).  Even roots +-2e_i, odd roots +-e1+-e2+-e3.
    Form: diag(-(1+a), 1, a)/2, so that every odd root is isotropic.
    The combinatorics does not depend on a; a is kept generic."""
    roots = []
    for i in range(3):
        for s in (1, -1):
            v = [0, 0, 0]
            v[i] = 2 * s
            roots.append((V(*v), 0))
    for ss in itertools.product((1, -1), repeat=3):
        roots.append((V(*ss), 1))
    G = [[(F(-1) - a) / 2, F(0), F(0)],
         [F(0), F(1, 2), F(0)],
         [F(0), F(0), a / 2]]
    return "osp_a(2|4)", roots, G, 3, (9, 8)


# --------------------------------------------------------------- linear algebra
def form(G, u, v):
    n = len(G)
    return sum(u[i] * G[i][j] * v[j] for i in range(n) for j in range(n))


def solve(A, r):
    rows = [row[:] for row in A]
    piv, i = [], 0
    for col in range(r):
        sel = None
        for k in range(i, len(rows)):
            if rows[k][col]:
                sel = k
                break
        if sel is None:
            continue
        rows[i], rows[sel] = rows[sel], rows[i]
        f = rows[i][col]
        rows[i] = [x / f for x in rows[i]]
        for k in range(len(rows)):
            if k != i and rows[k][col]:
                g = rows[k][col]
                rows[k] = [x - g * y for x, y in zip(rows[k], rows[i])]
        piv.append(col)
        i += 1
    for k in range(i, len(rows)):
        if rows[k][r] != 0:
            return None
    c = [F(0)] * r
    for k, col in enumerate(piv):
        c[col] = rows[k][r]
    return c


def expand(roots, simple, rank):
    r = len(simple)
    out = []
    for (v, p) in roots:
        A = [[simple[k][i] for k in range(r)] + [v[i]] for i in range(rank)]
        c = solve(A, r)
        if c is None:
            return None
        out.append((tuple(c), p))
    return out


# --------------------------------------------------------------- bases
def positive_systems(roots, rank, N):
    seen = {}
    for f in itertools.product(range(-N, N + 1), repeat=rank):
        if not any(f):
            continue
        vals = [sum(F(f[i]) * r[0][i] for i in range(rank)) for r in roots]
        if any(v == 0 for v in vals):
            continue
        pos = frozenset(r[0] for r, v in zip(roots, vals) if v > 0)
        seen.setdefault(pos, f)
    return seen


def base_of(pos):
    S = set(pos)
    simple = []
    for a in pos:
        if any(tuple(x - y for x, y in zip(a, b)) in S for b in pos):
            continue
        simple.append(a)
    return sorted(simple)


def all_bases(roots, rank, N=8):
    out = {}
    for p in positive_systems(roots, rank, N):
        S = base_of(p)
        out.setdefault(frozenset(S), S)
    return out


# --------------------------------------------------------------- the audit
def nodetype(alpha, p, G):
    if p == 0:
        return "o"                                    # white: even
    return "x" if form(G, alpha, alpha) == 0 else "b"  # crossed / black


def sdim_g0(exp, rank, marks):
    ev = od = 0
    r = len(marks)
    for (c, p) in exp:
        if any(marks[k] and c[k] != 0 for k in range(r)):
            continue
        if p:
            od += 1
        else:
            ev += 1
    return (rank + ev, od)


def bondlabel(G, a, b):
    """A crude but sufficient bond invariant: the pair of Cartan entries when
    defined, else the scalar product normalised by a fixed root length."""
    bij = form(G, a, b)
    if bij == 0:
        return None
    na, nb = form(G, a, a), form(G, b, b)
    return (bij, na, nb)


def report(alg, printed, types_wanted, marks_wanted, note=""):
    name, roots, G, rank, sd = alg
    par = {r[0]: r[1] for r in roots}
    ev = sum(1 for r in roots if r[1] == 0)
    od = sum(1 for r in roots if r[1] == 1)
    print("=" * 74)
    print(f"{name}:  {ev} even + {od} odd roots, rank {rank}  ->  sdim g = "
          f"({rank + ev}|{od})   [expected {sd}]")
    assert (rank + ev, od) == sd, "root system does not reproduce sdim g"

    bases = {}
    for N in (6, 8):
        b = all_bases(roots, rank, N)
        bases = b if len(b) >= len(bases) else bases
        print(f"    sweep N={N}: {len(b)} bases")
    print(f"    bases used: {len(bases)}")

    shapes = {}
    hits_exact, hits_total = [], []
    for key, S in bases.items():
        if len(S) != rank:
            print(f"    !! a 'base' of {len(S)} elements -- skipped")
            continue
        exp = expand(roots, S, rank)
        if exp is None:
            continue
        types = tuple(nodetype(a, par[a], G) for a in S)
        shapes[types] = shapes.get(types, 0) + 1
        for bits in range(1 << rank):
            marks = tuple((bits >> k) & 1 for k in range(rank))
            g0 = sdim_g0(exp, rank, marks)
            if g0 == printed:
                hits_exact.append((types, marks, g0))
            elif g0[0] + g0[1] == printed[0] + printed[1]:
                hits_total.append((types, marks, g0))
    print("    node-type patterns of the bases (sorted):")
    for t, c in sorted(shapes.items()):
        print(f"        {''.join(t)}   x{c}")
    print(f"    (base, marks) with sdim g_0 exactly {printed}: {len(hits_exact)}")
    seen = set()
    for t, m, g0 in hits_exact:
        k = (''.join(t), m)
        if k not in seen:
            seen.add(k)
            print(f"        {''.join(t)}  marks {m}")

    # --- what the DRAWN characteristic gives
    print(f"    -- the characteristic as drawn: types {''.join(types_wanted)}, "
          f"marks {marks_wanted} --")
    got = {}
    for key, S in bases.items():
        if len(S) != rank:
            continue
        types = [nodetype(a, par[a], G) for a in S]
        exp = expand(roots, S, rank)
        if exp is None:
            continue
        for perm in itertools.permutations(range(rank)):
            if [types[perm[k]] for k in range(rank)] != list(types_wanted):
                continue
            marks = [0] * rank
            for k in range(rank):
                marks[perm[k]] = marks_wanted[k]
            g0 = sdim_g0(exp, rank, marks)
            ordered = [S[perm[k]] for k in range(rank)]
            bl = tuple(sorted(
                (i, j, str(bondlabel(G, ordered[i], ordered[j])))
                for i in range(rank) for j in range(i + 1, rank)
                if bondlabel(G, ordered[i], ordered[j]) is not None))
            got.setdefault(g0, set()).add(bl)
    if not got:
        print("        no base of the drawn node-type pattern exists")
    for g0 in sorted(got):
        verdict = ("MATCHES the printed number" if g0 == printed else
                   ("total matches, splitting does not"
                    if g0[0] + g0[1] == printed[0] + printed[1] else "no"))
        print(f"        sdim g_0 = {g0}   [{verdict}]   "
              f"{len(got[g0])} bond pattern(s)")
        for bl in sorted(got[g0])[:3]:
            print(f"            bonds: {bl}")
    if note:
        print(f"    note: {note}")
    return bases, par, G, rank, roots


if __name__ == "__main__":
    # row 17): ag(2).  Drawn: two crossed nodes, both marked, and one white node,
    # unmarked.  Printed value 5.
    report(ag2(), (5, 0), ("x", "x", "o"), (1, 1, 0))

    # row 18): ab(3).  Drawn: a white node (marked), two crossed nodes (unmarked)
    # and a white node (marked).  Printed value 10.
    report(ab3(), (10, 0), ("o", "x", "x", "o"), (1, 0, 0, 1))

    # row 19): osp_a(2|4).  Drawn: three crossed nodes, all marked.  Printed 3.
    report(dab(), (3, 0), ("x", "x", "x"), (1, 1, 1))
