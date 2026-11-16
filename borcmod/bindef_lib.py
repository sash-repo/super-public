"""Defect of a bilinear map T (dict (z,u,v) -> coefficient) under K_{t theta}, numerically over F_p."""
def make_defect(S, g, p, n):
    from osp1 import rep
    cache = {}
    def X(lam):
        if lam not in cache: cache[lam] = rep(S, g, lam)[4][0]
        return cache[lam]
    def defect_num(T, a, b, c, pi):
        out = {}
        def addto(k, w): out[k] = (out.get(k, 0) + w) % p
        s = -1 if pi else 1
        Xa, Xb, Xc = X(a), X(b), X(c)
        for (z, u, v), w in T.items():
            for y in range(n):
                if Xc[y][z]: addto((y, u, v), Xc[y][z]*w)
            for u2 in range(n):
                if Xa[u][u2]: addto((z, u2, v), -s*Xa[u][u2]*w)
            for v2 in range(n):
                if Xb[v][v2]: addto((z, u, v2), -s*(-1 if (u % 2) else 1)*Xb[v][v2]*w)
        return {k: w for k, w in out.items() if w}
    return defect_num
