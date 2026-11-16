"""End diagram of the third shape: ... x=eps_{m-1}, D1=(delta..), y=s eps_m, D2=(delta..), 2delta."""
from samp import *
import crit

def endII(n, m, word, lam):
    labs = labels(n, m, word, lam)
    N = len(labs)
    jx = [k for k, w in enumerate(word) if w[0] == 'e' and w[1] == m-2][0]
    jy = [k for k, w in enumerate(word) if w[0] == 'e' and w[1] == m-1][0]
    s = jy - jx - 1                     # |D1|
    t = len(word) - 1 - jy              # |D2| >= 1
    z = labs[N-1][1]                    # label of 2 delta
    g = labs[jy][1]                     # odd node (y, first delta of D2)
    h = [labs[jy+j][1] for j in range(1, t)]   # even nodes inside D2
    # step A: y crosses D2
    u = [g]
    for j in range(1, t): u.append(u[-1] - h[j-1] - nz(u[-1]))
    hp = [h[j-1] + nz(u[j-1]) - nz(u[j]) for j in range(1, t)]   # new even labels in D2
    Tm = u[-1]                                   # triangle node delta - y
    Tp = 2*z + 2*nz(u[-1]) - u[-1]               # triangle node delta + y
    # step B: x crosses D1, the junction, D2
    if s >= 1:
        a1 = labs[jx][1]
        e = [labs[jx+j][1] for j in range(1, s)]
        f = labs[jy-1][1]
        J = f - g - nz(g)
        c = cross(a1, e + [J] + hp)
    else:
        a1 = labs[jx][1]                         # even node x - y
        c = cross(a1 + g + nz(g), hp)
    return [c - Tp - nz(c)]

def criterionII(n, m, word, lam):
    labs = labels(n, m, word, lam)
    if any(k == 'e' and not zplus(v) for k, v in labs): return False
    odd = [k for k in range(len(labs)) if labs[k][0] == 'o']
    conds = segment_conditions(labs, odd) + endII(n, m, word, lam)
    return all(zplus(x) for x in conds)

if __name__ == '__main__':
    random.seed(5)
    cnt = {}; shown = 0
    for n, m in ((1,2),(2,2),(3,2),(1,3),(2,3),(3,3),(4,2),(2,4),(4,3),(3,4)):
        for w in words(n, m, 'II'):
            B = base_of_word(n, m, w)
            for _ in range(150):
                lam = lam_from_labels(n, m, w, randlabs(n, m, w))
                g = ground_truth(n, m, B, lam); c = criterionII(n, m, w, lam)
                cnt[(g, c)] = cnt.get((g, c), 0) + 1
                if g != c and shown < 5:
                    shown += 1; print(n, m, w, lam, g, c, labels(n, m, w, lam))
    print(cnt)
