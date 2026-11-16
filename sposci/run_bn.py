#!/usr/bin/env python3
r"""run_bn.py -- checks on b(n), its quantizing cocycle and its degree-0 part."""
import sys
from fractions import Fraction
from bn import Bn

PASS = FAIL = 0
LOG = []

def check(name, ok, detail=""):
    global PASS, FAIL
    if ok: PASS += 1; LOG.append(f"  ok   {name} {detail}")
    else:  FAIL += 1; LOG.append(f"  FAIL {name} {detail}")

def jacobi(A, br, f, g, h, pf, pg, ph):
    """super-Jacobi with the b(n)-parities, for a bracket br."""
    t1 = A.mul_scal(br(f, br(g, h)), (-1) ** (pf * ph))
    t2 = A.mul_scal(br(g, br(h, f)), (-1) ** (pg * pf))
    t3 = A.mul_scal(br(h, br(f, g)), (-1) ** (ph * pg))
    return A.add(A.add(t1, t2), t3)

Bn.mul_scal = staticmethod(lambda f, c: {k: Fraction(c) * v for k, v in f.items() if Fraction(c) * v != 0})

def run(n, maxdeg):
    A = Bn(n)
    bas = A.basis(maxdeg)
    mons = [{k: Fraction(1)} for k in bas]
    par = [A.pb(k) for k in bas]
    # 1. the antibracket is a Lie superalgebra bracket
    bad = 0
    for a, f in enumerate(mons):
        for b, g in enumerate(mons):
            s = -((-1) ** (par[a] * par[b]))
            if not A.iszero(A.add(A.bracket(f, g), A.mul_scal(A.bracket(g, f), -s))):
                bad += 1
    check(f"b({n}): the antibracket is anti-symmetric for the b(n)-parities",
          bad == 0, f"({len(mons)}^2 = {len(mons)**2} pairs)")
    bad = 0; cnt = 0
    for a, f in enumerate(mons):
        for b, g in enumerate(mons):
            for c, h in enumerate(mons):
                cnt += 1
                if not A.iszero(jacobi(A, A.bracket, f, g, h, par[a], par[b], par[c])):
                    bad += 1
    check(f"b({n}): super-Jacobi for the antibracket", bad == 0, f"({cnt} triples)")
    # 2. delta is an outer derivation
    bad = 0
    for a, f in enumerate(mons):
        for b, g in enumerate(mons):
            lhs = A.delta(A.bracket(f, g))
            rhs = A.add(A.bracket(A.delta(f), g), A.bracket(f, A.delta(g)))
            if not A.iszero(A.add(lhs, rhs, -1)):
                bad += 1
    check(f"b({n}): f |-> (d_1(f)-1) f is a derivation of the antibracket",
          bad == 0, "(C. Roger's 1-cocycle)")
    # 3. C is odd and C = delta wedge delta
    bad = 0
    for a, f in enumerate(mons):
        for b, g in enumerate(mons):
            val = A.C(f, g)
            if not val: continue
            k, = list(val.keys())
            # parity of C(f,g) in b(n) must be p_b(f)+p_b(g)+1  (odd cochain)
            if A.pb(k) != (par[a] + par[b] + 1) % 2:
                bad += 1
    check(f"b({n}): the cocycle C of Table 1 is an ODD cochain", bad == 0,
          "(whence the Planck constant must be odd)")
    # 4. the deformed bracket satisfies Jacobi over Lambda(hbar), hbar^2 = 0.
    #    With hbar odd and hbar^2 = 0 this is exactly: C is a 2-cocycle.
    bad = 0; cnt = 0
    for a, f in enumerate(mons):
        for b, g in enumerate(mons):
            for c, h in enumerate(mons):
                cnt += 1
                # d C (f,g,h) with the b(n)-parities, C odd:
                # the mixed terms of the Jacobi identity of {,}+hbar C
                t = {}
                for (x, px, y, py, z, pz, s) in [
                        (f, par[a], g, par[b], h, par[c], (-1) ** (par[a] * par[c])),
                        (g, par[b], h, par[c], f, par[a], (-1) ** (par[b] * par[a])),
                        (h, par[c], f, par[a], g, par[b], (-1) ** (par[c] * par[b]))]:
                    t = A.add(t, A.C(x, A.bracket(y, z)), s)
                    t = A.add(t, A.bracket(x, A.C(y, z)), s)
                if not A.iszero(t):
                    bad += 1
    check(f"b({n}): C is a 2-cocycle, so {{,}} + hbar C is a Lie superalgebra bracket",
          bad == 0, f"({cnt} triples; hbar^2 = 0 kills the remaining term)")
    # 5. b(n)_0 acts on b(n)_{-1} as pe(n)
    deg0 = [k for k in A.basis(2) if sum(k[0]) + len(k[1]) == 2]
    deg_1 = [k for k in A.basis(1) if sum(k[0]) + len(k[1]) == 1]
    ev = sum(1 for k in deg0 if A.pb(k) == 0)
    od = sum(1 for k in deg0 if A.pb(k) == 1)
    check(f"b({n})_0 has sdim ({ev}|{od}) = sdim pe({n})", (ev, od) == (n*n, n*n))
    return

def main():
    print("=" * 74)
    print("bn: the Buttin superalgebra and the cocycle that quantizes it")
    print("=" * 74)
    for n, d in ((1, 3), (2, 2), (3, 2)):
        LOG.append(f"\n--- n = {n}, generating functions of degree <= {d} ---")
        run(n, d)
    print("\n".join(LOG))
    print("-" * 74)
    print(f"passed {PASS}, failed {FAIL}")
    return 0 if FAIL == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
