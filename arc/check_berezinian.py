"""
Manin/[GO, eq.(1.5)] claim  ber( d(tilde x, tilde theta)/d(x,theta) ) = (Cx+D+delta theta)^{-1};
   Theorem 3.5.3 of the text says this is false.  Where does the factor
   (Cx + D + delta theta)^{-1} come from, then?

We compute, for a C-point of SOSp(1|2) (the parametrization of the text) acting by
   tilde x = (Ax+B+gamma th)/(Cx+D+delta th),  tilde th = (alpha x+beta+E th)/(Cx+D+delta th),
the following:
   * ber of the Jacobi matrix (both layouts);
   * the contact multiplier m_G  (G^*(dx + th dth) = m_G (dx + th dth));
   * D_th(tilde th)  for  D_th = d_th + th d_x  and  Delta_th = d_th - th d_x;
   * the superconformality condition.
Grassmann algebra: odd generators alpha, beta, theta; the defining relations of the C-point are
solved as  gamma = alpha B - beta A,  delta = alpha D - beta C,  E = 1 - alpha beta,
AD - BC = 1 + alpha beta   (so D = (1 + BC + alpha beta)/A).
"""

import sympy as sp
from itertools import combinations

A, B, C, x = sp.symbols('A B C x')
GENS = ('al', 'be', 'th')      # odd generators


def mono(S, c=1):
    S = tuple(S)
    S, s = _sort(S)
    return {} if s == 0 else {S: sp.expand(c * s)}

def _sort(S):
    S = list(S); s = 1
    for i in range(len(S)):
        for j in range(len(S) - 1 - i):
            if S[j] == S[j + 1]:
                return (), 0
            if GENS.index(S[j]) > GENS.index(S[j + 1]):
                S[j], S[j + 1] = S[j + 1], S[j]; s = -s
    for i in range(len(S) - 1):
        if S[i] == S[i + 1]:
            return (), 0
    return tuple(S), s

def ad(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = sp.expand(out.get(k, 0) + v)
    return {k: v for k, v in out.items() if sp.simplify(v) != 0}

def sc(c, f):
    return {k: sp.expand(c * v) for k, v in f.items() if sp.expand(c * v) != 0}

def ml(f, g):
    out = {}
    for S1, c1 in f.items():
        for S2, c2 in g.items():
            S, s = _sort(S1 + S2)
            if s:
                out[S] = sp.expand(out.get(S, 0) + s * c1 * c2)
    return {k: sp.simplify(v) for k, v in out.items() if sp.simplify(v) != 0}

def inv(f):
    """inverse of an even invertible element: f = f0 (1 + nu), nu nilpotent"""
    f0 = f.get((), 0)
    if f0 == 0:
        raise ZeroDivisionError
    nu = {k: v / f0 for k, v in f.items() if k != ()}
    out = mono((), 1)
    term = mono((), 1)
    for _ in range(len(GENS)):
        term = sc(-1, ml(term, nu))
        if not term:
            break
        out = ad(out, term)
    return sc(1 / f0, out)

def dodd(j, f):
    """left derivative"""
    out = {}
    for S, c in f.items():
        if j in S:
            p = S.index(j)
            T = S[:p] + S[p + 1:]
            out[T] = sp.expand(out.get(T, 0) + (-1) ** p * c)
    return {k: v for k, v in out.items() if sp.simplify(v) != 0}

def dx(f):
    return {S: sp.diff(c, x) for S, c in f.items() if sp.diff(c, x) != 0}

def show(f):
    if not f:
        return "0"
    parts = []
    for S, c in sorted(f.items(), key=lambda kv: (len(kv[0]), kv[0])):
        s = "*".join(S)
        parts.append("(%s)%s" % (sp.simplify(sp.factor(c)), ("*" + s) if s else ""))
    return " + ".join(parts)

def eq(f, g):
    return not ad(f, sc(-1, g))

# ------------------------------------------------------------ the C-point

al = mono(('al',))
be = mono(('be',))
th = mono(('th',))
albe = ml(al, be)

Dd = ad(sc((1 + B * C) / A, mono(())), sc(1 / A, albe))       # D = (1+BC+al be)/A
Ee = ad(mono((), 1), sc(-1, albe))                            # E = 1 - al be
ga = ad(sc(B, al), sc(-A, be))                                # gamma = al B - be A
de = ad(ml(al, Dd), sc(-C, be))                               # delta  = al D - be C

def check_relations():
    print("relations of the C-point:")
    AD_BC = ad(sc(A, Dd), sc(-B * C, mono(())))
    print("   AD - BC - al*be - 1 = ", show(ad(AD_BC, sc(-1, albe), sc(-1, mono(())))))
    r2 = ad(sc(A, de), sc(-C, ga), sc(-1, ml(Ee, al)))
    print("   A d - C g - E al    = ", show(r2))
    r3 = ad(sc(B, de), sc(-1, sc(1, ml(Dd, ga))), sc(-1, ml(Ee, be)))
    print("   B d - D g - E be    = ", show(r3))
    r4 = ad(ml(Ee, Ee), sc(2, ml(ga, de)), sc(-1, mono(())))
    print("   E^2 + 2 g d - 1     = ", show(r4))

# ------------------------------------------------------- the transformation

den = ad(sc(C * x, mono(())), Dd, ml(de, th))          # C x + D + delta theta
deninv = inv(den)
xt = ml(ad(sc(A * x, mono(())), sc(B, mono(())), ml(ga, th)), deninv)
tht = ml(ad(sc(x, al), be, ml(Ee, th)), deninv)

def berezinian(R, S, T, U):
    """ber [[R,S],[T,U]] with R,U even, S,T odd  =  (R - S U^{-1} T) U^{-1}"""
    Ui = inv(U)
    return ml(ad(R, sc(-1, ml(S, ml(Ui, T)))), Ui)

if __name__ == "__main__":
    check_relations()
    print()
    print("tilde x     =", show(xt))
    print("tilde theta =", show(tht))
    print()
    R = dx(xt); T = dodd('th', xt); S = dx(tht); U = dodd('th', tht)
    print("d(tilde x)/dx      =", show(R))
    print("d(tilde x)/dth     =", show(T))
    print("d(tilde th)/dx     =", show(S))
    print("d(tilde th)/dth    =", show(U))
    print()
    b1 = berezinian(R, S, T, U)          # layout of eq. (80) of the text
    b2 = berezinian(R, T, S, U)          # the usual Jacobi matrix
    print("ber (layout of eq.(80), i.e. [[dx~/dx, dth~/dx],[dx~/dth, dth~/dth]]) =", show(b1))
    print("ber (usual Jacobi matrix   [[dx~/dx, dx~/dth],[dth~/dx, dth~/dth]])   =", show(b2))
    print()
    print("(Cx+D+delta th)      =", show(den))
    print("(Cx+D+delta th)^{-1} =", show(deninv))
    print("   ber(VB layout) = (Cx+D+d th)^{-1} ?", eq(b1, deninv))
    print("   ber(usual)     = (Cx+D+d th)^{-1} ?", eq(b2, deninv))
    print()
    # the contact/superconformal structure
    for nm, sgn in (("D_th = d_th + th d_x", 1), ("Delta_th = d_th - th d_x", -1)):
        Dth = lambda f: ad(dodd('th', f), sc(sgn, ml(th, dx(f))))
        dtht = Dth(tht)
        print("%s :" % nm)
        print("   D(tilde th)          =", show(dtht))
        print("   D(tilde x) - tilde th D(tilde th) =",
              show(ad(Dth(xt), sc(-1, ml(tht, dtht)))))
        print("   D(tilde th)^2        =", show(ml(dtht, dtht)))
        print("   D(tilde th) = (Cx+D+delta th)^{-1} ?", eq(dtht, deninv))
        print("   ber(usual) = D(tilde th) ?", eq(b2, dtht))
        print("   ber(VB layout) = D(tilde th) ?", eq(b1, dtht))
    # multiplier: G^*(dx + th dth) = m (dx + th dth); with p(dz) = p(z)+1 the
    # dx-component is  d_x(x~) + th~ d_x(th~) ... computed by pulling back
    m1 = ad(dx(xt), ml(tht, dx(tht)))
    m2 = ad(dx(xt), sc(-1, ml(tht, dx(tht))))
    print()
    print("multiplier candidates:  d_x(x~) + th~ d_x(th~) =", show(m1))
    print("                        d_x(x~) - th~ d_x(th~) =", show(m2))
    print("   equals (Cx+D+delta th)^{-2} ?", eq(m1, ml(deninv, deninv)), eq(m2, ml(deninv, deninv)))

# ---------------------------------------------------------- robustness checks

def dodd_right(j, f):
    """right derivative d/d th acting from the right"""
    out = {}
    for S, c in f.items():
        if j in S:
            p = len(S) - 1 - S.index(j)
            T = tuple(k for k in S if k != j)
            out[T] = sp.expand(out.get(T, 0) + (-1) ** p * c)
    return {k: v for k, v in out.items() if sp.simplify(v) != 0}

def extra():
    print()
    print("=" * 60)
    print("robustness: right derivatives d/dtheta")
    R = dx(xt); T = dodd_right('th', xt); S = dx(tht); U = dodd_right('th', tht)
    b1 = berezinian(R, S, T, U)
    b2 = berezinian(R, T, S, U)
    print("   ber(VB layout, right der.) = (Cx+D+d th)^{-1} ?", eq(b1, deninv))
    print("   ber(usual,     right der.) = (Cx+D+d th)^{-1} ?", eq(b2, deninv))
    print()
    print("the quantity V of the proof of Theorem 3.5.3:")
    V = ad(ml(ad(ml(Ee, sc(C, mono(()))), ml(al, de)), sc(x, mono(()))),
           ml(Ee, Dd), ml(be, de))
    print("   V = (EC + al*de) x + (ED + be*de) =", show(V))
    print("   V = (1 - 2 al be)(Cx + D) ?",
          eq(V, ml(ad(mono((), 1), sc(-2, albe)), ad(sc(C * x, mono(())), Dd))))
    Vi = inv(V)
    claimed = ad(Vi, ml(ad(ml(al, Dd), sc(-C, be)), ml(th, ml(Vi, Vi))))
    print("   the text's answer 1/V + (alD-beC)th/V^2 =", show(claimed))
    print("   correct berezinian                       =", show(b1 if False else berezinian(dx(xt), dx(tht), dodd('th', xt), dodd('th', tht))))
    print("   they agree ?", eq(claimed, berezinian(dx(xt), dx(tht), dodd('th', xt), dodd('th', tht))))

extra()
