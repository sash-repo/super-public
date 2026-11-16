"""Express every invariant map found by the brute-force census (all
osp(2|2)-invariant linear maps F_{a,mu} -> F_{b,nu}, weights in F_p) in the basis d_t^i, D_xi D_eta d_t^i (even), D_xi d_t^i, D_eta d_t^i (odd).
Fails loudly if some invariant map is not in that span (the commutant claim).
Usage: python3 explicit2.py p N > explicit-p-N.txt    (then closed2.py)"""
import sys, itertools
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from exactlin import nullspace
from osp2 import Alg, osp22, structure, reps, equivariant
p, N = int(sys.argv[1]), int(sys.argv[2])
A = Alg(p, N); gens = osp22(A); Ks, sc = structure(A, gens); P = A.P
DD = A.mul(A.DXI, A.DETA)
c = lambda f: [(x - y) % p for x, y in zip(A.apply(A.Dt, f), A.apply(DD, f))]
pw = [A.I]
for i in range(1, P): pw.append(A.mul(pw[-1], A.Dt))
basis = {0: [(f"dt^{i}", pw[i]) for i in range(P)] + [(f"DxDe dt^{i}", A.mul(DD, pw[i])) for i in range(P)],
         1: [(f"Dx dt^{i}", A.mul(A.DXI, pw[i])) for i in range(P)] + [(f"De dt^{i}", A.mul(A.DETA, pw[i])) for i in range(P)]}
W = list(itertools.product(range(p), repeat=2))
R = {w: reps(A, gens, Ks, c, *w) for w in W}
def express(T, par):
    cols = [[x for r in M for x in r] for _, M in basis[par]]
    tv = [x for r in T for x in r]
    ns = nullspace([list(r) for r in zip(*(cols + [tv]))], p)
    sol = [v for v in ns if v[-1] % p]
    assert sol, "NOT IN THE ANSATZ SPAN"
    v = sol[0]; inv = pow(v[-1], p-2, p)
    return {basis[par][k][0]: (-v[k]*inv) % p for k in range(len(cols)) if v[k] % p}
for wa in W:
    for wb in W:
        for par in (0, 1):
            ns, unk = equivariant(A, R[wa], R[wb], par)
            if not ns or (wa == wb and par == 0 and len(ns) == 1): continue
            for v in ns:
                T = A.zero()
                for (i, j), x in zip(unk, v): T[i][j] = x
                print(f"F_{wa} -> F_{wb} {'odd' if par else 'even'}:", express(T, par), flush=True)
