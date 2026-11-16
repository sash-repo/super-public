#!/usr/bin/env python3
"""The filtered deforms svect_h(3;1) of svect(3;1) for h = exp(x_i^{(3)}) and
h = 1 + x^{(2)}x^{(2)}x^{(2)}: are they Lie algebras, and are they deforms?"""
from k3 import P, rref, nullspace
from vect3 import BASIS, POS, EXP, build_vect, oder, omul, subalg_from_rows

vect = build_vect()
n = vect.n
mons = {m: k for k, m in enumerate(EXP)}


def condition_rows(hlog):
    """rows of the linear map D -> div D + D(hlog), for hlog given as a dict
    of formal divided-power monomials with entries possibly = 3 (i.e. x_i^{(3)});
    D(x_i^{(3)}) = D_i x_i^{(2)}."""
    rows = []
    for k in range(n):
        m, i = BASIS[k]
        val = {}
        d = oder(m, i)
        if d is not None:                       # divergence
            val[d] = (val.get(d, 0) + 1) % P
        for hm, hc in hlog.items():             # D(hlog)
            dh = oder(hm, i)
            if dh is None:
                continue
            mm, co = omul(m, dh)
            if co:
                val[mm] = (val.get(mm, 0) + hc * co) % P
        r = [0] * len(EXP)
        for mm, c in val.items():
            if mm in mons:
                r[mons[mm]] = c % P
        rows.append(r)
    return rows


def stabiliser(hlog):
    rows = condition_rows(hlog)
    M = [[rows[k][c] for k in range(n)] for c in range(len(EXP))]
    ker = nullspace(M, n)
    return rref(ker, n)[0]


def closed(alg_rows):
    for a in alg_rows:
        for b in alg_rows:
            br = vect.bracket(a, b)
            if any(br) and len(rref(alg_rows + [br], n)[0]) > len(alg_rows):
                return False
    return True


def gr_dims(rows):
    """dimensions of the associated graded for the standard filtration
    (deg x^(a) d_i = |a| - 1); L_k = span of vectors of degree >= k."""
    deg = [sum(m) - 1 for (m, i) in BASIS]
    out = {}
    for k in sorted(set(deg)):
        # dim of the image of L_k in L_k / L_{k+1}
        sel = [i for i in range(n) if deg[i] >= k]
        selh = [i for i in range(n) if deg[i] >= k + 1]
        A = [[r[i] for i in sel] for r in rows]
        B = [[r[i] for i in selh] for r in rows]
        # vectors of the algebra lying in L_k, modulo those in L_{k+1}
        def inL(k_):
            idx = [i for i in range(n) if deg[i] < k_]
            M = [[r[i] for i in idx] for r in rows]
            if not idx:
                return len(rows)
            return len(nullspace([[M[j][c] for j in range(len(rows))]
                                  for c in range(len(idx))], len(rows)))
        out[k] = inL(k) - inL(k + 1)
    return out


svect = stabiliser({})
print("dim svect(3;1) =", len(svect), " closed:", closed(svect))
print("  graded dims:", gr_dims(svect))
for name, hlog in (("exp(x_1^{(3)})", {(3, 0, 0): 1}),
                   ("exp(x_2^{(3)})", {(0, 3, 0): 1}),
                   ("exp(x_3^{(3)})", {(0, 0, 3): 1}),
                   ("1 + x_1^{(2)}x_2^{(2)}x_3^{(2)}  (log = -bar x + ...)",
                    {(2, 2, 2): 1})):
    rows = stabiliser(hlog)
    print(f"\nsvect_h, h = {name}: dim {len(rows)}, closed under the bracket:"
          f" {closed(rows)}")
    print("  dims of the associated graded:", gr_dims(rows))
    same = len(rref(svect + rows, n)[0]) == len(svect)
    print("  equal to svect(3;1) as a subspace of vect(3;1):", same)
