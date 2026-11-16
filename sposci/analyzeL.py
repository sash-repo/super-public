"""
Quotient the computed space of weight-0 2-cocycles of k^L(1|4) by the
coboundaries and compare the result, entry by entry, with equation 273.
"""

from fractions import Fraction as Q
from itertools import product

from stringy import mono, parity
from kL4 import (bracket, monomials, weight, by_weight, name_of, table_raw,
                 par_of_name, ZERO)
from solveL import Unknowns, reduce_rows, nullspace, eqs, par

B = 3
A = 2 * B + 1
S0 = (-1, (1, 2, 3, 4))
keysA = [k for k in monomials(range(-A, A + 1)) if k != S0]
keysB = [k for k in monomials(range(-B, B + 1)) if k != S0]

U = Unknowns()
WA = by_weight(keysA)
for k1 in keysB:
    for k2 in WA.get(-weight(k1), ()):
        U.get(k1, k2)
nB = len(U.keys)
rows = list(eqs(U, keysB))
n = len(U.keys)
print("B=%d  unknowns=%d  equations=%d" % (B, n, len(rows)))
piv = reduce_rows(rows)
free, basis = nullspace(piv, n)
sols = [tuple(v[:nB]) for v in basis]
piv2 = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in sols])
print("dim (weight-0 cocycles restricted to the window) =", len(piv2))

# ------------------------------------------------------------ coboundaries
# weight-0 elements of the algebra: L_0 = K_t, T^{..}_0 = K_{M}, (S_0 dropped)
w0 = [k for k in keysB if weight(k) == 0]
print("weight-0 elements (support of a weight-0 cochain):", [name_of(k) for k in w0])

def coboundary(phi_key):
    """c(x,y) = phi([x,y]) with phi dual to the basis vector phi_key."""
    v = [Q(0)] * nB
    for i, (k1, k2) in enumerate(U.keys):
        br = bracket(mono(*k1), mono(*k2))
        v[i] = Q(br.get(phi_key, 0))
    return v

cob = [coboundary(k) for k in w0]
pivC = reduce_rows([{i: x for i, x in enumerate(v) if x} for v in cob])
print("dim of coboundaries =", len(pivC))

def reduce_mod(pivots, vec):
    row = {i: x for i, x in enumerate(vec) if x}
    while row:
        p = min(row)
        if p in pivots:
            c = row[p]
            for i, val in pivots[p].items():
                row[i] = row.get(i, 0) - c * val
            row = {i: x for i, x in row.items() if x}
        else:
            break
    return row

def in_span(pivots, vec):
    return not reduce_mod(pivots, vec)

# --------------------------------------------------- the table's 3 cocycles
def table_vec(which):
    """which in {0,1,2} = alpha, beta, gamma component of equation 273."""
    v = [Q(0)] * nB
    for i, (k1, k2) in enumerate(U.keys):
        v[i] = c_mono_tab(k1, k2)[which]
    return v

from kL4 import c_mono as c_mono_tab

names = ("alpha", "beta", "gamma")
tvecs = [table_vec(j) for j in range(3)]
print()
for j, nm in enumerate(names):
    print("table's %-5s part is a cocycle: %s" % (nm, in_span(piv2, tvecs[j])))

# ---------------------------------------------- correct the alpha cocycle
# Look for v in the cocycle space that agrees with the table everywhere except
# possibly on the pairs (G,Gbar), (F,Fbar), (S,S), (T,T) -- i.e. determine
# the whole alpha-cocycle from its L-L entries.
def pretty(vec, skip_zero=True):
    out = {}
    for i in range(nB):
        if vec[i] or not skip_zero:
            k1, k2 = U.keys[i]
            out[(name_of(k1), name_of(k2))] = vec[i]
    return out

# basis of the cocycle space
sp = [ [Q(0)]*nB for _ in piv2 ]
bas = []
for p in sorted(piv2):
    v = [Q(0)] * nB
    for i, x in piv2[p].items():
        v[i] = x
    bas.append(v)

def solve_in_span(vectors, target, extra_constraints=None):
    """find coefficients so that sum c_i v_i = target (exactly), if possible"""
    m = len(vectors)
    # solve least-squares exactly by Gaussian elimination on the transpose
    rows = []
    for i in range(nB):
        rows.append(([v[i] for v in vectors], target[i]))
    # elimination
    piv_cols = {}
    aug = [list(r[0]) + [r[1]] for r in rows]
    r = 0
    for c in range(m):
        pr = None
        for i in range(r, len(aug)):
            if aug[i][c]:
                pr = i
                break
        if pr is None:
            continue
        aug[r], aug[pr] = aug[pr], aug[r]
        pivv = aug[r][c]
        aug[r] = [x / pivv for x in aug[r]]
        for i in range(len(aug)):
            if i != r and aug[i][c]:
                f = aug[i][c]
                aug[i] = [a - f * b for a, b in zip(aug[i], aug[r])]
        piv_cols[c] = r
        r += 1
    for i in range(len(aug)):
        if all(x == 0 for x in aug[i][:m]) and aug[i][m] != 0:
            return None
    sol = [Q(0)] * m
    for c, rr in piv_cols.items():
        sol[c] = aug[rr][m]
    return sol

# The alpha cocycle: pin it down by requiring the L-L entries m(m^2-1) and
# no beta/gamma admixture (the beta, gamma cocycles vanish on (L,L)); then
# kill the coboundary freedom by demanding the entries the table gives for
# T^{xi_i eta_i} pairs stay diagonal.
LL = {}
for i, (k1, k2) in enumerate(U.keys):
    n1, n2 = name_of(k1), name_of(k2)
    if n1[0] == "L" and n2[0] == "L":
        LL[i] = Q(n1[1] * (n1[1] ** 2 - 1))

print()
print("--- what the cocycle space says about the entries of equation 273 ---")
# express every basis vector in named form and look at the entry types present
def entry_types(vec):
    tp = {}
    for i in range(nB):
        if vec[i]:
            n1, n2 = name_of(U.keys[i][0]), name_of(U.keys[i][1])
            tp.setdefault((n1[0], n2[0]), []).append((n1[1], n2[1], vec[i]))
    return tp

for bi, v in enumerate(bas):
    tp = entry_types(v)
    print(" basis vector %d: entry types %s" % (bi, sorted(tp)))

print()
print("--- the alpha-type cocycle, normalised to c(L_m,L_n)=m(m^2-1) ---")
v = None
for b in bas:
    if ('S', 'S') in entry_types(b):
        v = b
        break
tp = entry_types(v)
m0, n0, val = [e for e in tp[('L', 'L')] if e[0] not in (0, 1, -1)][0]
lam = val / Q(m0 * (m0 ** 2 - 1))
for t in sorted(tp):
    ents = sorted(e for e in tp[t] if -3 <= e[0] <= 3)
    print("  c(%-6s,%-6s): %s" % (t[0], t[1],
          ", ".join("m=%s,n=%s: %s" % (a, b, c / lam) for a, b, c in ents)))
