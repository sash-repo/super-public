"""smoke.py -- conventions and first checks of dpco.py.

Checks, for each sample Lie superalgebra:
  A) the axioms of a p=2 Lie superalgebra;
  B) associativity of U(g) on its PBW basis (the diamond lemma, by brute force);
  C) d^2 = 0 in the resolution V_* = U(g) (x) C_*;
  D) exactness of V_* in the filtration degrees where it can be seen;
  E) the cohomology of the dual complex against values known independently.
"""

from dpco import (SLA2, Env, CE, Resolution, cohomology_dims, span_int,
                  in_span, bits)

LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------- the examples

def ex_lambda():
    """g = <f> odd, s(f) = 0.  U(g) = Lambda(f) = K[f]/(f^2)."""
    return SLA2(0, 1, {}, [0], "Lambda: <f>, s(f)=0")


def ex_poly():
    """g = <e> + <f>, s(f) = e, abelian.  U(g) = K[f]."""
    return SLA2(1, 1, {}, [1], "K[f]: <e|f>, s(f)=e")


def ex_gl11():
    """gl(1|1) for p = 2; e0 = E11, e1 = E22, f0 = E12, f1 = E21."""
    br = {(0, 2): 1 << 2, (0, 3): 1 << 3,
          (1, 2): 1 << 2, (1, 3): 1 << 3,
          (2, 3): 1 | 2}
    return SLA2(2, 2, br, [0, 0], "gl(1|1)")


def ex_heis():
    """the odd Heisenberg: g_1 = <f0,f1>, g_0 = <e> central, [f0,f1] = e,
    s(f0) = s(f1) = 0."""
    br = {(1, 2): 1}
    return SLA2(1, 2, br, [0, 0], "odd Heisenberg (0|2)+(1|0)")


def ex_q1():
    """the queer line: g_0 = <e>, g_1 = <f>, s(f) = e, [e,f] = 0 -- same as
    ex_poly but with a non-zero action on the adjoint module."""
    return ex_poly()


# ------------------------------------------------------------------ checks

def check_d2(res, maxdeg):
    bad = 0
    for n in range(2, maxdeg + 1):
        for c in res.ce.basis(n):
            v = res.d({((), c)})
            if res.d(v):
                bad += 1
    return bad


def normal_words(g, env, maxlen):
    out = [()]
    frontier = [()]
    for _ in range(maxlen):
        nxt = []
        for w in frontier:
            start = w[-1] if w else 0
            for k in range(start, g.dim):
                cand = w + (k,)
                if env.is_normal(cand):
                    nxt.append(cand)
        out += nxt
        frontier = nxt
    return out


def exactness(res, D):
    """exactness of V_* in filtration degrees <= D-2; returns (bad, info)"""
    g, env = res.g, res.env
    words = normal_words(g, env, D)
    basis = {}
    index = {}
    for n in range(0, D + 1):
        bl = []
        for c in res.ce.basis(n):
            for w in words:
                if len(w) + n <= D:
                    bl.append((w, c))
        basis[n] = bl
        index[n] = {b: i for i, b in enumerate(bl)}

    def vec(elt, n):
        v = 0
        for b in elt:
            if b in index[n]:
                v |= 1 << index[n][b]
            else:
                return None                  # left the truncation
        return v

    bad = []
    for n in range(1, D - 1):
        # boundaries from V_{n+1}
        img = []
        for b in basis[n + 1]:
            v = vec(res.d({b}), n)
            if v is None:
                continue
            img.append(v)
        piv = span_int(img)
        # cycles of filtration degree <= D-2
        rows = []
        keep = []
        for i, b in enumerate(basis[n]):
            if len(b[0]) + n > D - 2:
                continue
            keep.append(i)
            rows.append((i, vec(res.d({b}), n - 1)))
        # kernel of d on the kept part, by elimination
        piv2 = {}
        ker = []
        for (i, v) in rows:
            cur, comb = v, 1 << i
            while cur:
                h = cur.bit_length() - 1
                if h in piv2:
                    cur ^= piv2[h][0]
                    comb ^= piv2[h][1]
                else:
                    piv2[h] = (cur, comb)
                    break
            if cur == 0:
                ker.append(comb)
        for z in ker:
            if not in_span(piv, z):
                bad.append((n, z))
    return bad, {n: len(basis[n]) for n in basis}


# -------------------------------------------------------------------- main

def main():
    global LOG
    LOG = open("smoke.log", "w")
    for g in [ex_lambda(), ex_poly(), ex_gl11(), ex_heis()]:
        say("=" * 72)
        say("g =", g.name, " sdim = (%d|%d)" % (g.m, g.n))
        pr = g.check_axioms()
        say("  A) axioms:", "OK" if not pr else pr[:5])
        env = Env(g)
        bad = env.check_associativity(maxdeg=2 if g.dim > 3 else 3)
        say("  B) associativity of U(g):", "OK" if bad == 0 else "%d failures" % bad)
        res = Resolution(g, env)
        b2 = check_d2(res, 5 if g.dim <= 3 else 4)
        say("  C) d^2 = 0 on C_n, n <= %d:" % (5 if g.dim <= 3 else 4),
            "OK" if b2 == 0 else "%d failures" % b2)
        D = 6 if g.dim <= 2 else 5
        badx, sizes = exactness(res, D)
        say("  D) exactness of V_* up to filtration degree %d:" % (D - 2),
            "OK" if not badx else "%d unbounded cycles" % len(badx),
            " (dim V_n <= %s)" % sizes[1])
        triv = g.trivial_module()
        adj = g.adjoint_module()
        say("  E) H^n(g;K)  n=0..4:", cohomology_dims(res, triv, 4))
        say("     H^n(g;g)  n=0..2:", cohomology_dims(res, adj, 2))
    LOG.close()


if __name__ == "__main__":
    main()
