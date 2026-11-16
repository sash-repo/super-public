"""Semi-infinite forms over the distinguished stringy Lie superalgebras:
the bundles of Vaintrob's open problem 1), and the analogues of Mumford's
formula they yield.

The problem (Vaintrob, LNP 375 (1991) 263-268, Concluding remarks, item 1)).  For
a~distinguished g the central extension acts in the space of a rank 1 bundle
over the moduli superspace of g-curves -- the DETERMINANT bundle for witt and
k^L(1|1) -- and one wants analogues of Mumford's formula

        det_j = (det)^{6j^2 - 6j + 1}.

Vaintrob announces the construction, with the bundles built as bundles of
SEMI-INFINITE FORMS, for all the distinguished algebras; it was not published.

What is reconstructed here
--------------------------
Let V be a Z-graded g-module with finite-dimensional components, polarized as
V = V_+ + V_- by a cut in the degree.  The space Lambda^{oo/2}V of
semi-infinite forms carries a PROJECTIVE action of g, i.e. an honest action of
a central extension; the cocycle is the Japanese one,

    gamma(x,y) = str(x_{+-} y_{-+}) - (-1)^{p(x)p(y)} str(y_{+-} x_{-+}),

the supertrace because the odd part of V contributes symmetric (bosonic)
"forms" -- the beta-gamma system against the b-c one, which is what turns the
sign.  Both traces are finite.  The class of gamma does not depend on the cut.

Taking for V the module F_a of weighted densities of weight a and restricting
gamma to the Witt subalgebra L_m of g gives a number C_g(a), the central charge
of the ghost system of weight a.  Mumford's exponent is the RATIO of two such:
det_a = (det_{a_0})^{C_g(a)/C_g(a_0)}.  For g = witt, a_0 = 1, this IS Mumford:
C(a) = -2(6a^2-6a+1) and C(1) = -2.

Calibration (the engine is told none of these)
----------------------------------------------
    witt       C(2) = -26,  C(1) = -2,  C(1/2) = 1
    k^L(1|1)   C(3/2) = -15   (the N=1 superghosts: -26 + 11)
    k^L(1|2)   C(a) linear in a  (the N=2 ghost system, c = -6)

Run:  ../venv/bin/python ghost.py
"""

from fractions import Fraction as Q
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]

import yb
from stringy import add, mul, smul, mono, d_t, d_odd, parity


# =============================================================== the modules

class Densities:
    """F_a: the weighted densities of weight a over a stringy g.

    An element is a function g = sum c_{k,S} t^k theta^S, standing for
    g * alpha^a (contact) or g * (dt)^a (witt).  The action is

        rho_a(f)(g) = K_f(g) + kappa * a * D(f) * g,

    where K_f is the VECTOR FIELD (a derivation of the product) and D(f) is the
    contact multiplier: L_{K_f} alpha = kappa D(f) alpha.  Both are read off
    the bracket of yb.py rather than typed in again:

        K_f(g) = {f, g} + kappa * D(f) * g,      kappa = 2 (contact), 1 (witt),

    which is the definition of the contact bracket solved for its first term.
    NB: the term dropped here is kappa D(f) g and NOT D(f) nabla(g) with
    nabla = 2 - E: the difference, D(f) E(g), is the rotation of the odd
    indeterminates, and it is the whole of the super effect.  Without it the
    even and the odd halves of F_a are isomorphic Virasoro modules with equal
    weights, the supertrace cancels them against each other, and every central
    charge below comes out 0.

    Hence rho_a(f) = {f, .} + kappa (1 + a) D(f), so that a = -1 is the ADJOINT
    module -- as it must be, a vector field being a density of weight -1.
    """

    def __init__(self, alg, a, kappa=2):
        self.alg = alg
        self.a = Q(a)
        self.kappa = kappa

    # --- the two ingredients, taken from the algebra itself
    def nabla(self, g):
        if hasattr(self.alg, "nabla"):
            return self.alg.nabla(g)
        return dict(g)                              # witt: nabla = id

    def D(self, f):
        if hasattr(self.alg, "D"):
            return self.alg.D(f)                    # Ramond: d_t - zeta/2t d_zeta
        return d_t(f)

    def K(self, f, g):
        """The vector field K_f applied to the FUNCTION g."""
        return add(self.alg.brk(f, g), smul(self.kappa, mul(self.D(f), g)))

    def act(self, f, g):
        """rho_a(f) g = {f,g} + kappa (1+a) D(f) g."""
        out = add(self.alg.brk(f, g),
                  smul(self.kappa * (1 + self.a), mul(self.D(f), g)))
        return {k: v for k, v in out.items() if v}

    # --- the grading: deg t = 2, deg theta = 1 (deg zeta = 0 for Ramond)
    def deg(self, key):
        return self.alg.deg(key) + 2

    def basis(self, lo, hi):
        """All monomials of module degree in [lo, hi]."""
        out = []
        span = range(-(abs(lo) + abs(hi) + 4), abs(lo) + abs(hi) + 5)
        for k in self.alg.keys(span):
            if lo <= self.deg(k) <= hi:
                out.append(k)
        return out

    def par(self, key):
        return self.alg.par(key)


# ============================================================ the Virasoro

def virasoro(alg, m, kappa):
    """L_m as a generating function, normalized so that [L_m,L_n]=(n-m)L_{m+n}."""
    return mono(m + 1, (), Q(1, kappa))


class DensitiesK12(Densities):
    """F_{a,b} over k^L(1|2): the SECOND parameter, which no other member has.

    k(1|2)_0 is commutative of dimension 2, so k(1|2) carries two divergences,
    and beside alpha_1 there is the symbol beta = [d xi (d eta)^{-1}] of
    a~pseudodifferential form.  With

        Kt_xi(f)  = s (-1)^{p(f)} (d_eta - xi d_t)(f),
        Kt_eta(f) = s (-1)^{p(f)} (d_xi  - eta d_t)(f),
        Div_2     = Kt_eta Kt_xi - d_t,

    the Lie derivative on alpha_1^a beta^b is multiplication by
    2a d_t(f) + (-1)^{p(f)} b Div_2(K_f).  The overall sign s of the two
    operators is a~convention of the frame; it is not guessed here but SCANNED,
    the one that makes rho a representation being kept.
    """

    def __init__(self, alg, a, b, kappa=2, s=1):
        Densities.__init__(self, alg, a, kappa)
        self.b = Q(b)
        self.s = s

    def _Kt(self, i, j, f):
        """s (-1)^{p(f)} (d_{theta_j} - theta_i d_t) f."""
        return smul(self.s * (-1) ** parity(f),
                    add(d_odd(j, f), smul(-1, mul(mono(0, (i,)), d_t(f)))))

    def div2(self, f):
        return add(self._Kt(2, 1, self._Kt(1, 2, f)), smul(-1, d_t(f)))

    def act(self, f, g):
        mult = smul(self.kappa * (1 + self.a), d_t(f))
        if self.b:
            mult = add(mult, smul(self.b * (-1) ** parity(f), self.div2(f)))
        out = add(self.alg.brk(f, g), mul(mult, g))
        return {k: v for k, v in out.items() if v}


def check_rep_mod(mod, alg, span=range(-2, 3)):
    ks = alg.keys(span)
    bad = tested = 0
    for kf in ks:
        f = {kf: Q(1)}
        for kg in ks:
            g = {kg: Q(1)}
            sg = (-1) ** (parity(f) * parity(g))
            for kh in ks:
                h = {kh: Q(1)}
                tested += 1
                lhs = add(mod.act(f, mod.act(g, h)),
                          smul(-sg, mod.act(g, mod.act(f, h))))
                if any(v for v in add(lhs, smul(-1, mod.act(alg.brk(f, g), h))).values()):
                    bad += 1
    return bad, tested


def check_rep(alg, a, kappa, span=range(-2, 3)):
    """rho_a is a representation: [rho(f),rho(g)] = rho({f,g}).  Returns
    (failures, tested).  This is what licenses the whole family F_a."""
    mod = Densities(alg, a, kappa)
    ks = alg.keys(span)
    bad = tested = 0
    for kf in ks:
        f = {kf: Q(1)}
        for kg in ks:
            g = {kg: Q(1)}
            s = (-1) ** (parity(f) * parity(g))
            for kh in ks:
                h = {kh: Q(1)}
                tested += 1
                lhs = add(mod.act(f, mod.act(g, h)),
                          smul(-s, mod.act(g, mod.act(f, h))))
                rhs = mod.act(alg.brk(f, g), h)
                if any(v for v in add(lhs, smul(-1, rhs)).values()):
                    bad += 1
    return bad, tested


def check_virasoro(alg, kappa, rng=range(-3, 4)):
    bad = 0
    for m in rng:
        for n in rng:
            lhs = alg.brk(virasoro(alg, m, kappa), virasoro(alg, n, kappa))
            rhs = smul(n - m, virasoro(alg, m + n, kappa))
            if any(v for v in add(lhs, smul(-1, rhs)).values()):
                bad += 1
    return bad


# ======================================================== the ghost cocycle

def gamma(mod, x, y, cut, pad=8):
    """The Japanese cocycle of the semi-infinite forms Lambda^{oo/2}(mod).

    gamma(x,y) = str(x_{+-} y_{-+}) - (-1)^{p(x)p(y)} str(y_{+-} x_{-+}),
    V_+ = the basis vectors of module degree >= cut.
    """
    bas = mod.basis(cut - pad, cut + pad)
    idx = set(bas)

    def matrix(f):
        out = {}
        for u in bas:
            im = mod.act(f, {u: Q(1)})
            out[u] = {w: c for w, c in im.items() if c}
        return out

    X, Y = matrix(x), matrix(y)
    px, py = parity(x), parity(y)

    def half(A, B):
        """str(A_{+-} B_{-+}) -- B takes V_+ to V_-, then A back to V_+."""
        s = Q(0)
        for u in bas:
            if mod.deg(u) < cut:
                continue
            acc = Q(0)
            for w, cw in B.get(u, {}).items():
                if w not in idx or mod.deg(w) >= cut:
                    continue                        # project onto V_-
                acc += cw * A.get(w, {}).get(u, Q(0))
            if acc:
                s += Q((-1) ** mod.par(u)) * acc
        return s

    return half(X, Y) - Q((-1) ** (px * py)) * half(Y, X)


def central_charge(alg, a, kappa=2, cut=0, ms=(2, 3, 4)):
    """C in gamma(L_m, L_{-m}) = C/12 (m^3-m) + b m; returns (C, consistent)."""
    mod = Densities(alg, a, kappa)
    g1 = gamma(mod, virasoro(alg, 1, kappa), virasoro(alg, -1, kappa), cut)
    b = g1                                          # (m^3-m) vanishes at m=1
    vals = []
    for m in ms:
        gm = gamma(mod, virasoro(alg, m, kappa), virasoro(alg, -m, kappa), cut,
                   pad=4 * m + 6)
        vals.append(Q(12, m ** 3 - m) * (gm - b * m))
    ok = all(v == vals[0] for v in vals)
    return vals[0], ok, vals


# =================================================================== report

def fit_quadratic(pts):
    """The quadratic through three (a, C) points, as (A, B, Cc) for Aa^2+Ba+Cc."""
    (a1, c1), (a2, c2), (a3, c3) = pts[:3]
    from exactlin import solve
    M = [[a1 * a1, a1, Q(1)], [a2 * a2, a2, Q(1)], [a3 * a3, a3, Q(1)]]
    return solve(M, [c1, c2, c3])


if __name__ == "__main__":
    from exactlin import solve

    def say(s=""):
        print(s, flush=True)

    WEIGHTS = [Q(-1), Q(-1, 2), Q(0), Q(1, 2), Q(1), Q(3, 2), Q(2), Q(5, 2), Q(3)]
    CASES = [("witt", yb.Witt(), 1),
             ("k^L(1|1)", yb.Contact(1), 2),
             ("k^L(1|2)", yb.Contact(2), 2),
             ("k^L(1|3)", yb.Contact(3), 2),
             ("k^M(1|1)", yb.Ramond(1), 2),
             ("k^M(1|2)", yb.Ramond(2), 2)]

    say("Semi-infinite forms and the central charge C_g(a)")
    say("=" * 70)

    for name, alg, kappa in CASES:
        bad = check_virasoro(alg, kappa)
        say("\n%s   [L_m,L_n]=(n-m)L_{m+n}: %s" % (name, "OK" if not bad
                                                   else "FAILS in %d" % bad))
        rb, rt = 0, 0
        for a in (Q(-1), Q(1, 2), Q(3, 2)):
            b, t = check_rep(alg, a, kappa)
            rb, rt = rb + b, rt + t
        say("   rho_a is a representation: %s  (%d triples, a = -1, 1/2, 3/2)"
            % ("OK" if not rb else "FAILS in %d" % rb, rt))
        pts = []
        for a in WEIGHTS:
            try:
                C, ok, vals = central_charge(alg, a, kappa)
            except Exception as e:
                say("   a=%-5s ERROR %s: %s" % (a, type(e).__name__, e))
                continue
            pts.append((a, C))
            say("   a=%-5s C=%-8s  %s" % (a, C, "" if ok else "INCONSISTENT %s" % vals))
        if len(pts) >= 3:
            A, B, Cc = fit_quadratic(pts)
            bad = [p for p in pts if A * p[0] ** 2 + B * p[0] + Cc != p[1]]
            say("   fit: C(a) = %s a^2 + %s a + %s   %s"
                % (A, B, Cc, "(exact on all %d weights)" % len(pts) if not bad
                   else "NOT quadratic: %s" % bad))
    say("")

    # ------------------------------------------------- k^L(1|2): F_{a,b}
    say("")
    say("=" * 70)
    say("k^L(1|2): the second parameter b  (alpha_1^a beta^b)")
    alg = yb.Contact(2)
    good = None
    for s in (1, -1):
        bad, tot = check_rep_mod(DensitiesK12(alg, Q(1, 2), Q(1), 2, s), alg)
        say("   sign s=%+d: rho_{a,b} a representation: %s  (%d triples)"
            % (s, "OK" if not bad else "FAILS in %d" % bad, tot))
        if not bad and good is None:
            good = s
    if good is None:
        say("   neither sign gives a representation -- the b-family is not built")
    else:
        for b in (Q(0), Q(1, 2), Q(1), Q(2), Q(-1)):
            row = []
            for a in (Q(-1), Q(0), Q(1, 2), Q(1), Q(3, 2)):
                mod = DensitiesK12(alg, a, b, 2, good)
                g1 = gamma(mod, virasoro(alg, 1, 2), virasoro(alg, -1, 2), 0)
                vals = []
                for m in (2, 3, 4):
                    gm = gamma(mod, virasoro(alg, m, 2), virasoro(alg, -m, 2),
                               0, pad=4 * m + 6)
                    vals.append(Q(12, m ** 3 - m) * (gm - g1 * m))
                row.append(str(vals[0]) if all(v == vals[0] for v in vals)
                           else "INCONS")
            say("   b=%-5s C(a) for a=-1,0,1/2,1,3/2:  %s" % (b, "  ".join(row)))
