#!/usr/bin/env python3
"""Independent check of Theorem 7.3 of the paper (the GRC brackets over
pe(2) in m(1)).

What is tested, with SYMBOLIC weights l1,n1,l2,n2 and against ALL EIGHT generating
functions of pe(2):

  (a) Bol_k := D_q (-D_xi)^{k-1} + (k-1)(-D_xi)^{k-2} d_tau  is invariant
      F_{a,b} -> F_{a+1-k, b-1+k/2}  exactly for a = (k-1)/2, b free;
  (b) the multiplication F_{a,b} x F_{a',b'} -> F_{a+a',b+b'} is invariant
      for all four weights;
  (c) J^k_l (phi,psi) = Bol_k(phi) psi  is invariant exactly for l1 = (k-1)/2,
      the other three weights free, with target F_{l2+(1-k)/2, n1+n2+k/2-1};
  (d) J^k_r (phi,psi) = (-1)^{p(phi)} phi Bol_k(psi)  symmetrically;
  (e) both at once exactly on l1 = l2 = (k-1)/2, target F_{0, n1+n2+k/2-1}.

Run:  python check_bj.py [KMAX]        # default KMAX = 4
"""
import sys
from sympy import symbols, S, expand, Matrix, solve, simplify
from super12 import add, smul, mul, mono, parity
from m1 import M, D_1, D_xi, D_q, L, PE, Delta_m

l1, n1, l2, n2 = symbols('lambda_1 nu_1 lambda_2 nu_2')
aa, bb = symbols('a b')

# test monomials q^a tau^e xi^f
TEST = [(a, e, f) for a in range(0, 8) for e in (0, 1) for f in (0, 1)]


def compose(*ops):
    def r(u):
        for o in reversed(ops):
            u = o(u)
        return u
    return r


def minus_Dxi(u):
    return smul(-1, D_xi(u))


def Bol(k):
    """Bol_k = D_q (-D_xi)^{k-1} + (k-1) (-D_xi)^{k-2} d_tau, see eq. (62) of the paper,
    in the usual composition order: D_q is applied LAST.  (The order matters:
    [D_xi, D_q] = -2 d_tau, so applying D_q first gives another operator.)"""
    def r(u):
        out = u
        for _ in range(k - 1):
            out = minus_Dxi(out)
        out = D_q(out)
        if k >= 2:
            w = D_1(u)
            for _ in range(k - 2):
                w = minus_Dxi(w)
            out = add(out, smul(k - 1, w))
        return out
    return r


def Bol_printed(k):
    """the three operators printed explicitly in eq. (62), for k = 1, 2, 3."""
    if k == 1:
        return D_q
    if k == 2:
        return lambda u: smul(-1, add(D_xi(D_q(u)), D_1(u)))
    if k == 3:
        return lambda u: add(D_xi(D_xi(D_q(u))), smul(2, D_xi(D_1(u))))
    return None


# Bol_k is odd for every k (D_q odd, D_xi even, d_tau odd).
P_BOL = 1


def unary_defect(op, pop, A, B, At, Bt, fields, tests):
    """collect the non-zero coefficients of  L_t(op u) - (-1)^{p(f)p(op)} op(L_s u)."""
    eqs = set()
    for nm in fields:
        f = PE[nm]
        pf = (parity(f) + 1) % 2          # parity of the vector field M_f
        s = -1 if (pf and pop) else 1
        Ls, Lt = L(f, A, B), L(f, At, Bt)
        for m in tests:
            u = mono(*m)
            res = add(Lt(op(u)), smul(-s, op(Ls(u))))
            for v in res.values():
                e = expand(v)
                if e:
                    eqs.add(e)
    return sorted(eqs, key=str)


def binary_defect(B2, pB, A1, B1, A2, B2w, At, Bt, fields, tests):
    """B2(x,y,px) is the bilinear operator; pB its parity."""
    eqs = set()
    for nm in fields:
        f = PE[nm]
        pf = (parity(f) + 1) % 2
        sB = -1 if (pf and pB) else 1
        Lf1, Lf2, Lft = L(f, A1, B1), L(f, A2, B2w), L(f, At, Bt)
        for m1 in tests:
            u = mono(*m1)
            pu = (m1[1] + m1[2]) % 2
            Lu = Lf1(u)
            for m2 in tests:
                v = mono(*m2)
                Lv = Lf2(v)
                s2 = -1 if (pf and pu) else 1
                res = add(Lft(B2(u, v, pu)),
                          smul(-sB, add(B2(Lu, v, (pu + pf) % 2),
                                        smul(s2, B2(u, Lv, pu)))))
                for val in res.values():
                    e = expand(val)
                    if e:
                        eqs.add(e)
    return sorted(eqs, key=str)


def locus(eqs, unknowns):
    """the common zero locus of the equations, as a list of solution dicts."""
    if not eqs:
        return 'ALL WEIGHTS'
    return solve(eqs, unknowns, dict=True)


def main():
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    ALL8 = tuple(PE)
    ok = True

    print("=" * 78)
    print("check_bj.py -- Theorem 7.3, tested against all 8 generating functions")
    print("=" * 78)

    # ---------- (b) the multiplication ----------
    def MULT(x, y, px):
        return mul(x, y)
    eqs = binary_defect(MULT, 0, l1, n1, l2, n2, l1 + l2, n1 + n2,
                        ALL8, TEST[:12])
    print(f"\n(b) multiplication F_(l1,n1) x F_(l2,n2) -> F_(l1+l2, n1+n2): "
          f"{len(eqs)} defect equations -> {locus(eqs, [l1, n1, l2, n2])}")
    if eqs:
        ok = False
        print("    FAIL: the multiplication is not invariant?!", eqs[:3])

    for k in range(1, kmax + 1):
        Bk = Bol(k)
        print("\n" + "-" * 78)
        print(f"k = {k}")

        # ---------- (a0) the general formula against the printed examples ----
        Bp = Bol_printed(k)
        if Bp is not None:
            bad = [m for m in TEST
                   if any(expand(v) for v in
                          add(Bk(mono(*m)), smul(-1, Bp(mono(*m)))).values())]
            print(f"(a0) general formula = the printed Bol_{k} on all "
                  f"{len(TEST)} test monomials: "
                  f"{'YES' if not bad else 'NO -- ' + str(bad[:3])}")
            if bad:
                ok = False

        # ---------- (a) the unary Bol operator ----------
        At, Bt = aa + 1 - k, bb - 1 + S(k) / 2
        eqs = unary_defect(Bk, P_BOL, aa, bb, At, Bt, ALL8, TEST)
        sol = locus(eqs, [aa, bb])
        print(f"(a) Bol_{k}: F_(a,b) -> F_({At},{Bt}); {len(eqs)} equations; "
              f"locus {sol}")
        want = [{aa: S(k - 1) / 2}]
        if sol != want:
            ok = False
            print(f"    MISMATCH: expected a = (k-1)/2 = {S(k-1)/2}, b free")

        # ---------- (c) J^k_l = Bol_k(phi) psi ----------
        def JL(x, y, px, Bk=Bk):
            return mul(Bk(x), y)
        Atl, Btl = l2 + S(1 - k) / 2, n1 + n2 + S(k) / 2 - 1
        eqs = binary_defect(JL, P_BOL, l1, n1, l2, n2, Atl, Btl,
                            ALL8, TEST[:12])
        sol = locus(eqs, [l1, n1, l2, n2])
        print(f"(c) J^{k}_l = Bol_{k}(phi).psi -> F_({Atl},{Btl}); "
              f"{len(eqs)} equations; locus {sol}")
        if sol != [{l1: S(k - 1) / 2}]:
            ok = False
            print(f"    MISMATCH: expected lambda_1 = {S(k-1)/2}, the rest free")

        # ---------- (d) J^k_r = (-1)^{p(phi)} phi Bol_k(psi) ----------
        def JR(x, y, px, Bk=Bk):
            return smul(-1 if px else 1, mul(x, Bk(y)))
        Atr, Btr = l1 + S(1 - k) / 2, n1 + n2 + S(k) / 2 - 1
        eqs = binary_defect(JR, P_BOL, l1, n1, l2, n2, Atr, Btr,
                            ALL8, TEST[:12])
        sol = locus(eqs, [l1, n1, l2, n2])
        print(f"(d) J^{k}_r = (-1)^p(phi) phi.Bol_{k}(psi) -> F_({Atr},{Btr}); "
              f"{len(eqs)} equations; locus {sol}")
        if sol != [{l2: S(k - 1) / 2}]:
            ok = False
            print(f"    MISMATCH: expected lambda_2 = {S(k-1)/2}, the rest free")

        # ---------- (e) both at once ----------
        sub = {l1: S(k - 1) / 2, l2: S(k - 1) / 2}
        At0, Bt0 = S(0), n1 + n2 + S(k) / 2 - 1
        for name, Bop in (('J_l', JL), ('J_r', JR)):
            eqs = binary_defect(Bop, P_BOL, sub[l1], n1, sub[l2], n2, At0, Bt0,
                                ALL8, TEST[:12])
            status = 'invariant' if not eqs else f'NOT invariant ({len(eqs)} eqs)'
            print(f"(e) at l1=l2={S(k-1)/2}: {name}: "
                  f"F_({sub[l1]},n1) x F_({sub[l2]},n2) -> F_(0,{Bt0}): {status}")
            if eqs:
                ok = False

    print("\n" + "=" * 78)
    print("VERDICT:", "Theorem 7.3 CONFIRMED as printed" if ok
          else "SOMETHING FAILED -- see above")
    print("=" * 78)


if __name__ == '__main__':
    main()
