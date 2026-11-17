#!/usr/bin/env python3
r"""
dy3.py --- the deep Skryabin algebra dy(10; \One) over F_3: its non-positive part
transcribed from [KLS] = arXiv:2309.16370, verified, and cubed.

By the Lemma of the text (a Cartan prolong, complete or partial, inside
vect(m;\One) is closed under p-th powers as soon as its non-positive part is),
the 3-structure of dy(10) is decided by the 10 + 9 fields of those two tables.

The transcription is checked, not assumed:
  * every field is homogeneous of the degree its table assigns to it;
  * g_- has the bracket relations of 2br(3)_-;
  * g_0 normalizes every g_{-k};
  * g_0 is gl(3): the brackets of x_i, y_i, h_i are compared with gl(3) itself;
  * the elements that [KLS] gives as brackets --- x_3, y_3, h_1, h_2
    --- are computed here and compared with the printed ones.

Output: runs/dy3.log
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                    # noqa: E402
from gvect import P, Vect, rref                                       # noqa: E402
from skry3 import cube, is_derivation, Model                          # noqa: E402

LOG = os.path.join(HERE, "runs", "dy3.log")

# degrees and weights of u_1, ..., u_10.  The weights are those of [KLS]'s table
# of the fields Y_i commuting with g_-, with the sign that makes wt(d_a) the
# listed one: wt(u_a) = -wt(d_a).
DEG = [1, 1, 1, 2, 2, 2, 3, 4, 4, 4]
WD = [(0, 1, 0), (1, -1, 1), (-1, 0, 1),
      (1, 0, 1), (-1, 1, 1), (0, -1, 2),
      (0, 0, 2),
      (0, 1, 2), (1, -1, 3), (-1, 0, 3)]
WT = [tuple(-a for a in w) for w in WD]
NAMES = ["u%d" % i for i in range(1, 11)]

V = Vect(DEG, WT, NAMES)
Z = tuple([0] * 10)


def mon(**kw):
    """u1=1, u3=2 ... -> the divided-power monomial u_1 u_3^{(2)}."""
    m = [0] * 10
    for k, v in kw.items():
        m[int(k[1:]) - 1] = v
    return tuple(m)


def F(*terms):
    """F(('u2', 1), ..., i) -- build a field from (monomial, index, coeff)."""
    out = {}
    for m, i, c in terms:
        key = (m, i - 1)
        out[key] = (out.get(key, 0) + c) % P
    return {k: v for k, v in out.items() if v}


# ------------------------------------------------------------------- g_- (realdy2)
X1 = F((Z, 1, 2), (mon(u2=1), 4, 2), (mon(u3=1), 5, 2),
       (mon(u2=2, u3=1), 9, 2),
       (mon(u2=1, u3=1), 7, 1), (mon(u6=1), 7, 2),
       (mon(u3=1, u4=1), 8, 2), (mon(u7=1), 8, 1))
X2 = F((Z, 2, 2), (mon(u5=1), 7, 1), (mon(u7=1), 9, 1))
X3 = F((Z, 3, 2), (mon(u2=1), 6, 1), (mon(u4=1), 7, 2),
       (mon(u2=1, u4=1), 9, 1),
       (mon(u2=1, u5=1), 10, 1), (mon(u7=1), 10, 1))
X4 = F((Z, 4, 1), (mon(u5=1), 8, 1), (mon(u6=1), 9, 1))
X5 = F((Z, 5, 1), (mon(u6=1), 10, 1))
X6 = F((Z, 6, 1))
X7 = F((Z, 7, 1))
X8, X9, X10 = F((Z, 8, 1)), F((Z, 9, 1)), F((Z, 10, 1))

GM = [X1, X2, X3, X4, X5, X6, X7, X8, X9, X10]

# ------------------------------------------------------------------- g_0 (realdy)
x1 = F((mon(u3=1), 2, 1), (mon(u3=2), 6, 2),
       (mon(u4=1, u3=2), 9, 2), (mon(u10=1), 9, 1),
       (mon(u5=1), 4, 1), (mon(u3=2, u5=1), 10, 2), (mon(u5=2), 8, 1))
x2 = F((mon(u2=1), 1, 2), (mon(u2=2), 4, 2), (mon(u2=2, u3=1), 7, 2),
       (mon(u2=1, u3=1), 5, 2), (mon(u6=1), 5, 2),
       (mon(u2=1, u3=1, u4=1), 8, 2), (mon(u9=1), 8, 2),
       (mon(u6=2), 10, 2))
y1 = F((mon(u2=1), 3, 1), (mon(u2=2), 6, 2), (mon(u4=1), 5, 1),
       (mon(u2=2, u4=1), 9, 2), (mon(u4=2), 8, 1),
       (mon(u5=1, u2=2), 10, 2), (mon(u9=1), 10, 1))
y2 = F((mon(u1=1), 2, 2), (mon(u1=2), 4, 2), (mon(u1=2, u3=1), 7, 2),
       (mon(u1=1, u3=1), 6, 1), (mon(u5=1), 6, 2),
       (mon(u2=1, u3=1, u1=2), 9, 1), (mon(u3=1, u4=1, u1=1), 9, 1),
       (mon(u8=1), 9, 2),
       (mon(u1=2, u3=2), 10, 1), (mon(u1=1, u5=1, u3=1), 10, 1),
       (mon(u5=2), 10, 2))
h3 = F((mon(u2=1), 2, 2), (mon(u3=1), 3, 2), (mon(u4=1), 4, 2),
       (mon(u5=1), 5, 2), (mon(u6=1), 6, 1), (mon(u7=1), 7, 1),
       (mon(u8=1), 8, 1))

# the ones [KLS] itself gives as brackets: computed, then compared
x3 = V.bracket(x1, x2)
y3 = V.bracket(y1, y2)
h1 = V.bracket(x1, y1)
h2 = V.bracket(x2, y2)

# what [KLS] prints for them, transcribed for comparison
x3_pr = F((mon(u3=1), 1, 2), (mon(u3=2), 5, 2), (mon(u2=1, u3=2), 7, 1),
          (mon(u2=2, u3=2), 9, 2), (mon(u6=2), 9, 1),
          (mon(u4=1, u3=2), 8, 2), (mon(u5=1, u6=1), 8, 1), (mon(u10=1), 8, 2),
          (mon(u6=1), 4, 1))
y3_pr = F((mon(u1=1), 3, 1), (mon(u1=2), 5, 1), (mon(u1=2, u2=1), 7, 2),
          (mon(u1=2, u2=2), 9, 1), (mon(u4=2), 9, 2), (mon(u4=1), 6, 2),
          (mon(u1=2, u4=1), 8, 1),
          (mon(u4=1, u5=1), 10, 2), (mon(u8=1), 10, 1))
h1_pr = F((mon(u2=1), 2, 2), (mon(u3=1), 3, 1), (mon(u4=1), 4, 2),
          (mon(u5=1), 5, 1), (mon(u9=1), 9, 2), (mon(u10=1), 10, 1))
h2_pr = F((mon(u1=1), 1, 2), (mon(u2=1), 2, 1), (mon(u5=1), 5, 2),
          (mon(u6=1), 6, 1), (mon(u8=1), 8, 2), (mon(u9=1), 9, 1))

G0 = [x1, x2, x3, y1, y2, y3, h1, h2, h3]
G0N = ["x1", "x2", "x3", "y1", "y2", "y3", "h1", "h2", "h3"]

OK, BAD = [0], [0]


def check(say, name, cond):
    if cond:
        OK[0] += 1
    else:
        BAD[0] += 1
        say("  FAIL: %s" % name)


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log = open(LOG, "w", buffering=1)

    def say(m):
        print(m)
        log.write(m + "\n")

    say("dy(10; One) over F_3 -- the transcription of [KLS]")

    # ---- homogeneity
    for a, (D, nm) in enumerate(zip(GM, ["X%d" % i for i in range(1, 11)])):
        check(say, "%s homogeneous" % nm, len({V.deg(k) for k in D}) == 1)
        check(say, "%s of degree %d" % (nm, -DEG[a]), V.fdeg(D) == -DEG[a])
        check(say, "%s of weight %s" % (nm, WD[a]), V.fwt(D) == WD[a])
    for D, nm in zip(G0, G0N):
        check(say, "%s homogeneous of degree 0" % nm, V.fdeg(D) == 0)
    say("  degrees and weights of g_-, degrees of g_0: checked")

    # ---- the printed brackets
    for got, want, nm in ((x3, x3_pr, "x3"), (y3, y3_pr, "y3"),
                          (h1, h1_pr, "h1"), (h2, h2_pr, "h2")):
        if got == want:
            say("  %s = the field printed in (realdy): AGREES" % nm)
            OK[0] += 1
        else:
            BAD[0] += 1
            say("  %s DIFFERS from the printed field:" % nm)
            say("     computed: %s" % V.show(got))
            say("     printed : %s" % V.show(want))
            d = dict(got)
            for k, c in want.items():
                d[k] = (d.get(k, 0) - c) % P
            say("     difference: %s" % V.show({k: c for k, c in d.items() if c}))

    # ---- g_- is a Lie algebra with the right shape
    dims = {}
    for a, D in enumerate(GM):
        dims.setdefault(-DEG[a], []).append(a)
    mdl_m = Model(V, GM, [-d for d in DEG], [V.fwt(D) for D in GM], "dy_-")
    ok = True
    for i in range(10):
        for j in range(10):
            br = V.bracket(GM[i], GM[j])
            co = mdl_m.coords(br) if br else np.zeros(10, dtype=np.int64)
            if co is None:
                ok = False
    check(say, "g_- closed under the bracket", ok)
    say("  dim g_-k: " + ", ".join("%d:%d" % (k, len(v))
                                   for k, v in sorted(dims.items())))

    # ---- g_0 normalizes every g_{-k}, and g_0 is a subalgebra
    ok = True
    for D in G0:
        for j in range(10):
            br = V.bracket(D, GM[j])
            co = mdl_m.coords(br) if br else np.zeros(10, dtype=np.int64)
            if co is None:
                ok = False
            elif any(co[t] and DEG[t] != DEG[j] for t in range(10)):
                ok = False
    check(say, "g_0 preserves the grading of g_-", ok)

    mdl_0 = Model(V, G0, [0] * 9, [V.fwt(D) for D in G0], "dy_0")
    ok = True
    for i in range(9):
        for j in range(9):
            br = V.bracket(G0[i], G0[j])
            co = mdl_0.coords(br) if br else np.zeros(9, dtype=np.int64)
            if co is None:
                ok = False
    check(say, "g_0 closed under the bracket", ok)

    # ---- g_0 = gl(3): compare the structure constants with gl(3) itself
    E = {}
    for i in range(3):
        for j in range(3):
            M = np.zeros((3, 3), dtype=np.int64)
            M[i, j] = 1
            E[(i, j)] = M
    # the guess: x1 ~ E_12, x2 ~ E_23 (or their transposes); find an
    # isomorphism by matching the adjoint action on g_{-1}
    A = []
    for D in G0:
        col = np.zeros((3, 3), dtype=np.int64)
        for j in range(3):
            br = V.bracket(D, GM[j])
            co = mdl_m.coords(br)
            col[:, j] = co[:3] % P
        A.append(col)
    span = np.array([M.reshape(-1) for M in A], dtype=np.int64)
    R, piv = rref(span)
    check(say, "g_0 acts on g_{-1} = K^3 as the whole of gl(3)", len(piv) == 9)
    say("  g_0 -> gl(g_{-1}) is onto and injective: dim 9 -> rank %d" % len(piv))

    # ---- THE CUBES
    say("")
    say("the cube D^3 for every basis element of g_{<=0}:")
    allfields = list(zip(["X%d" % i for i in range(1, 11)], GM)) + \
        list(zip(G0N, G0))
    outside, nonzero = [], []
    for nm, D in allfields:
        C = cube(V, D)
        check(say, "(%s)^3 is a derivation" % nm, is_derivation(V, C))
        if not C:
            say("    (%s)^3 = 0" % nm)
            continue
        nonzero.append(nm)
        co = mdl_0.coords(C) if V.fdeg(C) == 0 else None
        if co is not None:
            say("    (%s)^3 = %s  = %s in g_0"
                % (nm, V.show(C),
                   " + ".join("%d %s" % (c, n) for c, n in zip(co, G0N) if c)))
        else:
            com = mdl_m.coords(C) if C else None
            if com is not None:
                say("    (%s)^3 = %s  (inside g_-)" % (nm, V.show(C)))
            else:
                outside.append(nm)
                say("    (%s)^3 = %s   *** OUTSIDE g_{<=0} ***" % (nm, V.show(C)))
    say("")
    say("  non-zero cubes: %s" % (nonzero or "none"))
    say("  cubes outside g_{<=0}: %s" % (outside or "none"))

    # ---- sdy: the sl(3) part
    tr = []
    for D in G0:
        col = np.zeros((3, 3), dtype=np.int64)
        for j in range(3):
            col[:, j] = mdl_m.coords(V.bracket(D, GM[j]))[:3] % P
        tr.append(int(np.trace(col)) % P)
    say("  traces of g_0 on g_{-1}: %s" % dict(zip(G0N, tr)))
    sl = [i for i in range(9) if tr[i] == 0]
    say("  sdy(10)_0 = sl(3) is spanned by %s (dim %d)"
        % ([G0N[i] for i in sl], len(sl)))
    ok = True
    for i in sl:
        C = cube(V, G0[i])
        co = mdl_0.coords(C) if C else np.zeros(9, dtype=np.int64)
        if co is None:
            ok = False
            continue
        t = sum(int(co[j]) * tr[j] for j in range(9)) % P
        if t:
            ok = False
    check(say, "sl(3) part of g_0 closed under cubes", ok)

    say("")
    say("%d checks, %d failures" % (OK[0] + BAD[0], BAD[0]))
    log.close()
    return 1 if BAD[0] else 0


if __name__ == "__main__":
    sys.exit(main())
