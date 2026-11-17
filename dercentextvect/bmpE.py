"""The derivations E, F of Subsec. 5.15 (from [BMP] = arXiv:2209.14822, proof of
Thm 3.4) on g = H(2;(1,n))^(2), p = 3: is each a derivation, and is it induced by
an element of Der O?  (Proposition 5.16.)

Usage: bmpE.py [n ...]; default n = 1, 2."""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
sys.path.insert(0, HERE)
from dervect import Super, poisson, derived, normalizer, Echelon

def run(n, fh):
    p = 3
    S = Super(p, 2, 0, (1, n))
    L1, fields = poisson(S, 'h', 1)
    L = derived(S, L1)
    EL = Echelon([S.vec(X) for X in L], p)
    def H(a, b):
        return fields.get(((a, b), ()), {})
    top = (2, 3 ** n - 1)
    basis = [(a, b) for a in range(3) for b in range(3 ** n)
             if (a, b) not in [(0, 0), top]]
    assert len(basis) == EL.dim
    def lin(img):          # img: dict (a,b)->field image; extend linearly via basis
        return img
    E = {(a, b): (H(0, b + 1) if a == 2 and b + 1 < 3 ** n else {}) for (a, b) in basis}
    F = {(a, b): (H(2, b - 1) if a == 0 and b >= 1 else {}) for (a, b) in basis}
    def isder(M):
        bad = 0
        for u in basis:
            for v in basis:
                X, Y = H(*u), H(*v)
                Z = S.bracket(X, Y)
                # express Z in basis H(a,b): Z = H({f,g}); compute M(Z) via coords
                lhs = apply(M, Z)
                rhs = S.add(S.bracket(M[u], Y), S.bracket(X, M[v]))
                d = S.add(lhs, {k: (-c) % p for k, c in rhs.items()})
                if any(c % p for c in d.values()):
                    bad += 1
        return bad
    # coordinates of a field in the basis H(a,b)
    vecs = [S.vec(H(*u)) for u in basis]
    import exactlin
    def apply(M, Z):
        z = S.vec(Z)
        if not any(x % p for x in z):
            return {}
        sol = exactlin.solve([[vecs[j][i] for j in range(len(basis))]
                              for i in range(len(z))], z, p)
        out = {}
        for c, u in zip(sol, basis):
            if int(c) % p:
                for k, w in M[u].items():
                    out = S.add(out, {k: (int(c) * w) % p})
        return out
    nz = normalizer(S, L)
    # restrictions of normalizer elements to L, as matrices on basis -> vectors
    def mat(M):
        return sum((S.vec(M[u]) for u in basis), [])
    Nmats = [mat({u: S.bracket(D, H(*u)) for u in basis}) for D in nz]
    rN = Echelon(Nmats, p).dim
    for name, M in (('E', E), ('F', F)):
        bad = isder(M)
        r2 = Echelon(Nmats + [mat(M)], p).dim
        s = (f"n={n}: {name} derivation? bad pairs={bad} of {len(basis)**2}; "
             f"in N_DerO|_g? {'yes' if r2 == rN else 'NO'} (rank {rN}->{r2})")
        print(s, flush=True); fh.write(s + '\n')

with open(os.path.join(HERE, 'bmpE.log'), 'w') as fh:
    for n in map(int, sys.argv[1:] or ['1', '2']):
        run(n, fh)
