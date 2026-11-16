from crit import *
import os; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, '..', '..', 'lib'), _HERE]
from exactlin import solve

def lam_from_labels(n, m, word, labvals):
    rows = []
    for k in range(n+m):
        e = [F(0)]*(n+m); e[k] = F(1)
        rows.append([v for _, v in labels(n, m, word, e)])
    # labels are linear: L = M lam with M[j][k] = rows[k][j]
    M = [[rows[k][j] for k in range(n+m)] for j in range(n+m)]
    x = solve(M, list(labvals))
    return tuple(F(v) for v in x)

def randlabs(n, m, word):
    kinds = [k for k, _ in labels(n, m, word, tuple(F(0) for _ in range(n+m)))]
    out = []
    for k in kinds:
        if k == 'e': out.append(F(random.choice([0, 0, 1, 1, 2, 3, -1])))
        else: out.append(F(random.choice([0, 0, 0, 1, 1, 2, 3, -1, -2, 4])) + random.choice([0]*6+[F(1,2)]))
    return out

def run(types, variant, trials=400, sizes=((1,2),(2,2),(3,2),(1,3),(2,3),(3,3),(4,2),(2,4))):
    tot = mis = posv = 0; ex = []
    for n, m in sizes:
        for t in types:
            for w in words(n, m, t):
                B = base_of_word(n, m, w)
                for _ in range(trials):
                    lam = lam_from_labels(n, m, w, randlabs(n, m, w))
                    g = ground_truth(n, m, B, lam)
                    c = criterion(n, m, w, lam, variant)
                    tot += 1; posv += g
                    if c != g:
                        mis += 1
                        if len(ex) < 5: ex.append((n, m, w, lam, g, c, labels(n, m, w, lam)))
    print(types, variant, "cases", tot, "dominant", posv, "mismatch", mis)
    for e in ex: print("  ", e)

if __name__ == '__main__':
    random.seed(2)
    run(['III'], 'ours', 200)
    run(['I'], 'S2', 200)
    run(['I'], 'ours', 200)
