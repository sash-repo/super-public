#!/usr/bin/env python3
"""Integrability of the 5 cocycles spanning H^2(g;g), g = k^(1)(3;1), and the
Weisfeiler filtrations of the resulting deforms."""
from itertools import combinations, product
from k3 import P, rref, nullspace
from coh import Complex
from deform import (cocycles, cval, cbil, nr_square, deform, is_simple, cls,
                    weisfeiler, filtration, span_add, contains, unit, g, cx)

print("\n" + "=" * 70)
print("A. integrability of the individual cocycles")
print("=" * 70)

# the obstruction [c,c] lives in C^3; is it a coboundary?
src3 = cx.cochains(3)
pos3 = {c: i for i, c in enumerate(src3)}
N3 = len(src3)
d2 = cx.diff(2)
src2 = cx.cochains(2)


def obstruction(c1, c2):
    """[c1,c2]_NR as a coordinate vector on C^3."""
    out = [0] * N3
    n = g.n
    for i, j, k in combinations(range(n), 3):
        tot = [0] * n
        for a, b, d in ((i, j, k), (j, k, i), (k, i, j)):
            for (u, v) in ((c1, c2), (c2, c1)):
                t = cbil(v, cval(u, a, b), unit(n, d))
                tot = [(x + y) % P for x, y in zip(tot, t)]
        for m, co in enumerate(tot):
            if co:
                out[pos3[((i, j, k), m)]] = co
    return out


def is_coboundary(vec3):
    """is the C^3-vector a coboundary d(C^2)?  Everything is done inside the
    single weight block the vector lives in, which keeps the matrices small."""
    if not any(vec3):
        return True
    ws = {cx.weight(*src3[i]) for i, x in enumerate(vec3) if x}
    assert len(ws) == 1, f"the obstruction is not homogeneous: {ws}"
    w = ws.pop()
    tgt_idx = [i for i, c in enumerate(src3) if cx.weight(*c) == w]
    col_of = {t: k for k, t in enumerate(tgt_idx)}
    rows = []
    for s in src2:
        if cx.weight(*s) != w:
            continue
        col = d2.get(s, {})
        if col:
            r = [0] * len(tgt_idx)
            for tgt, c in col.items():
                r[col_of[pos3[tgt]]] = c
            rows.append(r)
    v = [vec3[i] for i in tgt_idx]
    if not rows:
        return not any(v)
    red = rref(rows, len(tgt_idx))[0]
    piv = [next(i for i, x in enumerate(r) if x) for r in red]
    for i, pc in enumerate(piv):
        if v[pc]:
            f = v[pc]
            v = [(a - f * b) % P for a, b in zip(v, red[i])]
    return not any(v)


names = [f"c^{n}_{w}" for w, n, c in cocycles]
for a in range(len(cocycles)):
    ob = obstruction(cocycles[a][2], cocycles[a][2])
    lin = not any(ob)
    print(f"  {names[a]:>18}: [c,c] = 0 ? {'YES (deformation linear in the '
                                          'parameter)' if lin else 'no'}"
          + ("" if lin else f"; [c,c] a coboundary (unobstructed) ? "
                            f"{'YES' if is_coboundary(ob) else 'NO'}"))

print("\n  which directions in each weight plane are linear?")
for w in ((-3, -3), (-3, 3)):
    idx = [i for i, (ww, n, c) in enumerate(cocycles) if ww == w]
    print(f"   weight {w}: c = a*{names[idx[0]]} + b*{names[idx[1]]}")
    for a, b in product(range(P), repeat=2):
        if not (a or b):
            continue
        c = {}
        for co, i in ((a, idx[0]), (b, idx[1])):
            if not co:
                continue
            for S, v in cocycles[i][2].items():
                c.setdefault(S, [0] * g.n)
                c[S] = [(x + co * y) % P for x, y in zip(c[S], v)]
        ob = obstruction(c, c)
        print(f"      (a,b) = ({a},{b}): [c,c] = 0 ? {'yes' if not any(ob) else 'no'}")

print("\n" + "=" * 70)
print("B. the deforms and their Weisfeiler filtrations")
print("=" * 70)
print("  (only subalgebras graded for the (Z/3)^2-grading that survives the")
print("   deformation are searched; the maximal torus lies in each of them)")


def report(lam, label):
    alg = deform(lam)
    if not alg.check_jacobi()[0]:
        print(f"\n  {label}: NOT a Lie algebra (Jacobi fails) -- skipped")
        return
    simple = is_simple(alg)[0]
    ws = weisfeiler(alg, maxcodim=3)
    print(f"\n  {label}: Lie algebra, simple = {simple}, "
          f"{len(ws)} graded subalgebras of codim <= 3")
    seen = set()
    for c, L0 in ws:
        dims = tuple(filtration(alg, L0))
        if (c, dims) in seen:
            continue
        seen.add((c, dims))
        print(f"      codim {c}: dim L_0 = {dims[0]}, filtration {list(dims)}"
              f"  =>  dim L_(-1)/L_0 = {dims[1]-dims[0]}, depth {len(dims)-1}")


report([0] * 5, "k^(1)(3;1) itself")
for i in range(5):
    lam = [0] * 5
    lam[i] = 1
    report(lam, f"deform along {names[i]}")
