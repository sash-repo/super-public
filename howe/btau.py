"""
btau.py -- what does the condition  d f / d tau = 0  cut out of m(n)?

Claim: the condition df/dtau = 0 on m(n) defines b(n), the central extension of le(n),
and not le(n) itself.  Checked with mn.py / le.py for n = 2,3,4:

  (1) the tau-free generating functions are closed under [,]_M, and there the M-bracket
      IS the Buttin antibracket -- so the subalgebra is C[x,xi] with the antibracket;
  (2) the map f |-> 'f into W(n|n+1) is INJECTIVE on them: in particular '1 = 2D != 0,
      so no constant is killed by the realization and nothing is "modulo the constants";
  (3) C.1 is exactly the center of that subalgebra, it sits in M-degree -2, and as
      a vector field it is ODD (p_M(f) = p(f)+1);
  (4) 1 = [x_i, -xi_i]_M lies in the derived algebra, so the central extension
          0 --> C.1 --> b(n) --> le(n) --> 0
      does NOT split;
  (5) le(n) = { D_f } is the quotient: dim b(n)_j = dim le(n)_j for every j >= -1, and
      b(n) has the one extra dimension in degree -2 that le(n) lacks.

So the set is b(n) = B(n) = b_0(n) (Kochetkov's central extension of Le(n)), and
le(n) = b(n)/C.1.

Run:  python3 btau.py
"""

from fractions import Fraction as F

import le
import mn
from mn import one, tau, mono, mbr, Mvf, monos, layer, mdeg, par_m


def vf_eq(u, v):
    """equality of two MVFs"""
    gs = set(u.comp) | set(v.comp)
    return all(le.padd(u.comp.get(g, {}), v.comp.get(g, {}), -1) == {} for g in gs)


def vf_zero(u):
    return all(not c for c in u.comp.values())


def taufree(f, n):
    return all(n not in S for (_a, S) in f)


def basis_taufree(n, dmax):
    """all monomials in x,xi free of tau, of deg_{x,xi} <= dmax"""
    out = []
    for d in range(dmax + 1):
        out += monos(n, d, 0)
    return out


def flatten(vecs, n):
    """represent a list of polynomials as rows over the union of their monomials"""
    cols = sorted({m for v in vecs for m in v})
    idx = {m: i for i, m in enumerate(cols)}
    rows = [[F(0)] * len(cols) for _ in vecs]
    for r, v in enumerate(vecs):
        for m, c in v.items():
            rows[r][idx[m]] = c
    return rows, len(cols)


def vf_rows(fs, n):
    """each 'f as a sparse row {(component, monomial): coeff}"""
    rows = []
    for f in fs:
        u = Mvf(f, n)
        rows.append({(g, m): v for g, c in u.comp.items() for m, v in c.items() if v})
    return rows


def le_rows(fs, n):
    """each D_f as a sparse row -- the le(n) realization, which kills the constants"""
    rows = []
    for f in fs:
        u = le.D(f, n).comp
        rows.append({(g, m): v for g, c in u.items() for m, v in c.items() if v})
    return rows


def rank(rows):
    """rank of a list of sparse rows"""
    rows = [r for r in rows if r]
    return len(le.rref(rows, 0)[0]) if rows else 0


def check(n, dmax=4):
    print(f"\n=== n = {n}  (deg_{{x,xi}} <= {dmax}) ===")
    B = basis_taufree(n, dmax)
    print(f"    tau-free monomials: {len(B)}")

    # (1) closure + the bracket is the antibracket -------------------------------
    bad_cl = bad_ab = 0
    for f in B:
        for g in B:
            br = mbr(f, g, n)
            if not taufree(br, n):
                bad_cl += 1
            if le.padd(br, le.abr(f, g, n), -1) != {}:
                bad_ab += 1
    print(f"(1) closed under [,]_M: {'YES' if not bad_cl else f'NO ({bad_cl})'};"
          f"  [,]_M = antibracket on them: {'YES' if not bad_ab else f'NO ({bad_ab})'}"
          f"   ({len(B)**2} pairs)")

    # (2) f |-> 'f is injective on the tau-free functions -------------------------
    o = one(n)
    vf1 = Mvf(o, n)
    D_comp = {2 * n: {((0,) * n, ()): F(2)}}
    print(f"(2) '1 = 2D ? {'YES' if vf_eq(vf1, mn.MVF(n, D_comp, 1)) else 'NO'}"
          f"   ('1 = 0 ? {'YES' if vf_zero(vf1) else 'NO'})")
    r = rank(vf_rows(B, n))
    print(f"    rank of f |-> 'f on the {len(B)} of them: {r}"
          f"  ->  kernel {len(B) - r}-dimensional"
          f"  ({'injective' if r == len(B) else 'NOT injective'})")

    r_le = rank(le_rows(B, n))
    print(f"    rank of f |-> D_f (the le(n) realization): {r_le}"
          f"  ->  kernel {len(B) - r_le}-dimensional (the constants)")
    print(f"    so the tau-free part of m(n) is {r - r_le} dimension(s) bigger than le(n)")

    # (3) the center of the tau-free subalgebra, exactly ------------------------
    #     solve [sum_f c_f f, g] = 0 for every g of the basis, degree by degree
    cen = []
    for j in range(-2, dmax - 1):
        bj = [f for f in layer(n, j) if taufree(f, n)]
        if not bj:
            continue
        eqs = []
        for g in B:
            row = {}                      # one equation per output monomial
            cols_ = {}
            for k, f in enumerate(bj):
                for m, v in mbr(f, g, n).items():
                    cols_.setdefault(m, {})[k] = cols_.get(m, {}).get(k, F(0)) + v
            for m, r in cols_.items():
                eqs.append({k: v for k, v in r.items() if v})
        ker = le.nullspace(eqs, list(range(len(bj)))) if eqs else \
            [{k: F(1)} for k in range(len(bj))]
        for sol in ker:
            z = {}
            for k, v in sol.items():
                z = le.padd(z, le.pscal(bj[k], v))
            cen.append((j, z))
    print(f"(3) center of the tau-free subalgebra: {len(cen)}-dimensional,"
          f" spanned by {[le_str(z, n) for _j, z in cen]}"
          f"  in M-degree {[j for j, _z in cen]},"
          f"  parity {[('odd' if par_m(z) else 'even') for _j, z in cen]}"
          f"  ->  sdim ({sum(1 for _j, z in cen if not par_m(z))}|"
          f"{sum(1 for _j, z in cen if par_m(z))})")

    # (4) is 1 in the derived algebra? ------------------------------------------
    wit = None
    for i in range(n):
        xi_ = mono(n, [1 if k == i else 0 for k in range(n)], [])
        xii = mono(n, [0] * n, [i])
        br = mbr(xi_, xii, n)
        if br and all(sum(a) == 0 and not S for (a, S) in br):
            wit = (i, br)
            break
    if wit:
        i, br = wit
        c = list(br.values())[0]
        print(f"(4) [x_{i+1}, xi_{i+1}]_M = {c} . 1  ->  1 in [b,b]:"
              f" the extension does NOT split")
    else:
        print("(4) 1 not obtained as a bracket of two generators -- CHECK")

    # (5) dimensions degree by degree -------------------------------------------
    print("(5)  j   dim b(n)_j   dim le(n)_j")
    for j in range(-2, dmax - 1):
        bj = [f for f in layer(n, j) if taufree(f, n)]
        rb = rank(vf_rows(bj, n)) if bj else 0
        rl = rank(le_rows(bj, n)) if bj else 0
        print(f"    {j:2d}     {rb:3d}          {rl:3d}"
              + ("     <-- the odd center C.1 = C.2D" if j == -2 else ""))


def le_str(f, n):
    """a readable name for a monomial"""
    out = []
    for (a, S), c in f.items():
        s = ""
        for i in range(n):
            if a[i]:
                s += f"x{i+1}" + (f"^{a[i]}" if a[i] > 1 else "")
        for i in S:
            s += "tau" if i == n else f"xi{i+1}"
        out.append(s or "1")
    return ",".join(out)


if __name__ == "__main__":
    for n in (2, 3, 4):
        check(n, dmax=4 if n < 4 else 3)
    print("\nConclusion: { f in m(n) | df/dtau = 0 } = C[x,xi] with the antibracket = b(n),")
    print("a non-split central extension of le(n) by the ODD center C.'1 = C.2D of M-degree -2;")
    print("le(n) = b(n)/C.1.  The condition defines b(n), not le(n).")
