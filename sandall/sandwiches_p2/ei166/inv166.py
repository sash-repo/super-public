import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from p166 import prolong, algebra, TAB, lebedev, lie_closure
from css import rref
def dens(lam):
    # g_5..g_12 of realizVF at lambda in F_2, as matrices: row k = image of d_k under ad(field) -> [X, d_k] = sum M[k][j] d_j
    # field sum c_{kj} y_k d_j  -> M[k][j] = c_{kj}
    l, l1 = lam % 2, (lam + 1) % 2
    F = [ {(0,1):1,(2,3):1}, {(0,2):1,(1,3):1}, {(0,0):l,(1,1):l1,(2,2):l,(3,3):l1}, {(1,2):1}, {(2,1):1},
          {(0,0):l,(1,1):l,(2,2):l1,(3,3):l1}, {(2,0):l,(3,1):l1}, {(1,0):l,(3,2):l1} ]
    mats = []
    for f in F:
        M = [[0]*4 for _ in range(4)]
        for (k, j), c in f.items():
            M[k][j] = c
        mats.append(M)
    return mats
cases = []
for row in TAB.splitlines():
    N, gens = lebedev(row)
    els, d = lie_closure(gens)
    cases.append(("N%d" % N, els))
for lam in (0, 1):
    m = dens(lam); els, d = lie_closure(m)
    cases.append(("F_%d (g0 dim %d)" % (lam, d), els))
if __name__ == "__main__":
    for name, els in cases:
      comps = prolong(els, (1, 1, 1, 1))
      A = algebra(comps, (1, 1, 1, 1))
      ok, msg = A.check_jacobi()
      print(name, [len(c) for c in comps], msg, "derived", len(A.derived()), "der", A.dim_der(),
            "nil2", A.num_2nilps() + 1, "sandw", len(rref(A.sandwiches())), flush=True)
