#!/usr/bin/env python3
"""The SECOND normal shape of a non-alternating symplectic form at p = 2.

The census of the non-alternating forms (run1.py, run3.py, survey.py) leaves one
thing open: the second of the two normal shapes was known only as a machine output
for each m separately, with no closed form for general m.

Here we test the candidate

    omega_2 = alpha^(2) + omega',   alpha = dx_1 + x_1 dx_2,   i.e.
    omega_2 = omega_1 + x_1 dx_1 dx_2,   omega_1 = (dx_1)^(2) + omega'

(recall x_1^2 = 0 in a divided power algebra, so alpha^(2) = (dx_1)^(2) +
x_1 dx_1dx_2 exactly), and:

  1) closedness and non-degeneracy of omega_2 for every m and every shearing
     vector tested -- both are proved in the text, this is the check;
  2) the invariant (the dimensions of the derived series of h(omega)) against
     the two classes of the census, for 1 <= m <= 5;
  3) for m = 2, 3 an EXACT orbit test: omega_2 is not in the orbit of omega_1
     and its orbit is the other one (exhaustive over Aut O(m;1));
  4) the (2,4)-structure question for the second shape in an ODD number of
     indeterminates: Span{H_f}, the 2-structure, the Z/2-grading deg x_1 = 1,
     the map [4] and the additivity rule -- the same tests nonalt276.py runs
     for the standard shape omega_I.

Run:  python3 shape2.py        
"""
import sys, os, time, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from nalt2 import (Alg, d2, lie, act, bracket, gram, det, preserving_algebra,
                   reduce_basis, span_dim, derived_series, is_simple,
                   pullback, g2_basis, alternating, closed_space)
from exactlin import solve, rank                          # noqa: E402
from nonalt276 import square, in_span, ad_inner, xdeg     # noqa: E402
from run1 import key_of, stab_gens, enumerate_forms, show_form  # noqa: E402

LOG = None


def out(s=""):
    print(s, flush=True)
    if LOG is not None:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------------ the forms

def omega_std(alg):
    """the standard non-alternating shape (dx_1)^(2) + omega'.

    m odd  = 2n+1:  omega' = sum_{1<=i<=n} dx_{2i} dx_{2i+1}   (1-based)
    m even = 2n  :  omega' = dx_1dx_2 + sum_{2<=i<=n} dx_{2i-1} dx_{2i}
    In both cases the Gram matrix is constant and non-degenerate.
    """
    m = alg.m
    om = {('b', 0): alg.one}
    start = 1 if m % 2 else 0                    # 0-based index of the 1st pair
    i = start
    while i + 1 < m:
        om[('a', i, i + 1)] = alg.one
        i += 2
    return om


def omega_two(alg):
    """the candidate second shape alpha^(2) + omega'.

    m odd :  alpha = dx_1 + x_1 dx_2      (x_1 is the extra indeterminate)
    m even:  alpha = dx_1 + x_2 dx_2      (x_2 is the omega'-partner of x_1)
    In both cases alpha^(2) = (dx_1)^(2) + f dx_1dx_2 with f = x_1 resp. x_2,
    since the square of an f with zero constant term vanishes.
    """
    om = dict(omega_std(alg))
    k = ('a', 0, 1)
    om[k] = om.get(k, 0) ^ alg.gen(0 if alg.m % 2 else 1)
    if om[k] == 0:
        del om[k]
    return om


# ------------------------------------------------- Hamiltonian fields, general

def ham_solver(alg, om):
    """returns f -> H_f, defined by iota_{H_f} omega = df, for any omega.

    The map D |-> G(omega).D is F_2-linear on vect(m;u) = O^m; we build its
    matrix once and solve for each f.
    """
    m, dim = alg.m, alg.dim
    G = gram(alg, om)
    cols = []
    for i in range(m):
        for t in range(dim):
            D = [0] * m
            D[i] = 1 << t
            w = [0] * m
            for r in range(m):
                for c in range(m):
                    w[r] ^= alg.mul(G[r][c], D[c])
            v = []
            for r in range(m):
                v.extend((w[r] >> s) & 1 for s in range(dim))
            cols.append(v)
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]

    def H(f):
        b = []
        for r in range(m):
            g = alg.partial(r, f)
            b.extend((g >> s) & 1 for s in range(dim))
        z = solve(M, b, p=2)
        if z is None:
            return None
        D = [0] * m
        for c, coef in enumerate(z):
            if int(coef) % 2:
                i, t = divmod(c, dim)
                D[i] ^= 1 << t
        return D

    return H


# ---------------------------------------------------------------- 1)-2) basics

def basics():
    out("=" * 78)
    out("1)-2)  the candidate omega_2 = alpha^(2) + omega' "
        "(alpha = dx_1 + x_1dx_2 for m odd, dx_1 + x_2dx_2 for m even)")
    out("=" * 78)
    out("  m | closed | non-deg | dim h | derived series of h(omega)")
    for m in range(1, 6):
        alg = Alg(m, (1,) * m)
        for name, om in (("omega_1", omega_std(alg)), ("omega_2", omega_two(alg))):
            if m == 1 and name == "omega_2":
                continue
            cl = not d2(alg, om)
            nd = alg.const_term(det(alg, gram(alg, om))) == 1
            h = preserving_algebra(alg, om)
            ds = derived_series(alg, h, steps=4)
            out("  %d | %s | %-7s | %-7s | %-5d | %s"
                % (m, name, cl, nd, len(h), ds))
    # a shearing vector other than One, for the two proved statements
    out()
    out("  the same two statements for u != One (m = 3, u = (2,1,1)):")
    alg = Alg(3, (2, 1, 1))
    for name, om in (("omega_1", omega_std(alg)), ("omega_2", omega_two(alg))):
        cl = not d2(alg, om)
        nd = alg.const_term(det(alg, gram(alg, om))) == 1
        out("    %s: dim O = %d, closed %s, non-degenerate %s"
            % (name, alg.dim, cl, nd))


# ------------------------------------------------------- 3) the exact orbit test

def orbits(m):
    """exhaustive Aut-orbit test for m = 2, 3: which class is omega_2 in?"""
    alg = Alg(m, (1,) * m)
    b2 = g2_basis(m)
    Z = closed_space(alg)
    forms = enumerate_forms(alg, Z, fix_b=tuple([1] + [0] * (m - 1)))
    nd = [om for om in forms
          if alg.const_term(det(alg, gram(alg, om))) == 1]
    keys = set(key_of(alg, om, b2) for om in nd)
    gens = stab_gens(alg, tuple([1] + [0] * (m - 1)))

    # linearize the generators
    imgs = []
    nb = len(b2) * alg.dim
    for phi in gens:
        img = []
        for s in range(nb):
            k, t = divmod(s, alg.dim)
            img.append(key_of(alg, pullback(alg, phi, {b2[k]: 1 << t}), b2))
        imgs.append(img)

    def apply_lin(img, key):
        r, s = 0, 0
        while key:
            if key & 1:
                r ^= img[s]
            key >>= 1
            s += 1
        return r

    def orbit_of(om):
        start = key_of(alg, om, b2)
        seen = {start}
        stack = [start]
        while stack:
            k = stack.pop()
            for img in imgs:
                k2 = apply_lin(img, k)
                if k2 not in seen:
                    seen.add(k2)
                    stack.append(k2)
        return seen

    o1 = orbit_of(omega_std(alg))
    o2 = orbit_of(omega_two(alg))
    k2 = key_of(alg, omega_two(alg), b2)
    out("  m = %d: |non-degenerate forms with diagonal (1,0,..,0)| = %d"
        % (m, len(keys)))
    out("     |orbit(omega_1)| = %d, |orbit(omega_2)| = %d, they are %s, "
        "together %s the whole set"
        % (len(o1), len(o2), "disjoint" if not (o1 & o2) else "NOT DISJOINT",
           "exhaust" if (o1 | o2) == keys else "DO NOT exhaust"))
    out("     omega_2 in orbit(omega_1): %s   (must be False)" % (k2 in o1))


# ------------------------------------------ 4) the (2,4)-structure, second shape

def two_four(m, name, om, samples=60):
    alg = Alg(m, (1,) * m)
    H = ham_solver(alg, om)
    fields = {}
    for t in range(alg.dim):
        f = 1 << t
        D = H(f)
        if D is None:
            out("  %s: H_f does not exist for a basis f -- form degenerate?" % name)
            return
        fields[f] = D
    bad = sum(1 for D in fields.values() if lie(alg, D, om))
    hb = reduce_basis(alg, list(fields.values()))
    pres = preserving_algebra(alg, om)
    out("  --- %s, m = %d ---" % (name, m))
    out("     H_f preserve omega: %d failures;  dim Span{H_f} = %d;  "
        "dim h(omega) = %d" % (bad, len(hb), len(pres)))
    nosq = [D for D in hb if not in_span(alg, hb, square(alg, D))]
    notinner = [D for D in hb if not ad_inner(alg, hb, square(alg, D))]
    out("     basis fields with D^2 outside Span{H_f}: %d of %d;  "
        "with (ad_D)^2 not inner: %d" % (len(nosq), len(hb), len(notinner)))
    # the functions carrying the obstruction
    obstr = [f for f, D in fields.items()
             if D != [0] * m and not in_span(alg, hb, square(alg, D))]
    out("     generating functions with a non-inner square: %d  [%s]"
        % (len(obstr), ", ".join(sorted(fname(alg, f) for f in obstr))))
    if len(hb) <= 9:
        out("     Span{H_f} simple over F_2: %s;  dim of its derived algebra: %d"
            % (is_simple(alg, hb), len(reduce_basis(alg,
               [bracket(alg, X, Y) for X in hb for Y in hb]))))
    # the Z/2-grading deg x_1 = 1
    plus, minus = [], []
    for f, D in fields.items():
        if D == [0] * m:
            continue
        d = xdeg(alg, f)
        if d == {0}:
            plus.append(D)
        elif d == {1}:
            minus.append(D)
    plus, minus = reduce_basis(alg, plus), reduce_basis(alg, minus)
    out("     grading: dim g_+ = %d, dim g_- = %d (of %d)"
        % (len(plus), len(minus), len(hb)))
    okplus = all(in_span(alg, plus, square(alg, D)) for D in plus)
    if not okplus:
        badp = [f for f, D in fields.items()
                if D != [0] * m and xdeg(alg, f) == {0}
                and not in_span(alg, plus, square(alg, D))]
        out("     g_+ elements whose square leaves g_+: %s"
            % ", ".join(sorted(fname(alg, f) for f in badp)))
    four = [square(alg, square(alg, D)) for D in minus]
    ok4 = all(in_span(alg, plus, D4) for D4 in four)
    out("     g_+ closed under the square: %s;  D^4 in g_+ for D in g_-: %s;  "
        "all D^4 = 0: %s" % (okplus, ok4, all(D4 == [0] * m for D4 in four)))
    # the additivity rule on g_-
    import random
    random.seed(277)
    if len(minus) <= 6:
        rng = range(1 << len(minus))
    else:
        rng = list(range(len(minus) + 1)) + \
            [random.getrandbits(len(minus)) for _ in range(samples)]
    els = []
    for bits in rng:
        D = [0] * m
        for i in range(len(minus)):
            if (bits >> i) & 1:
                for t in range(m):
                    D[t] ^= minus[i][t]
        els.append(D)
    fails = pairs = 0
    br = lambda A, B: bracket(alg, A, B)
    for X in els:
        for Y in els:
            pairs += 1
            lhs = square(alg, square(alg, [X[t] ^ Y[t] for t in range(m)]))
            X4, Y4 = square(alg, square(alg, X)), square(alg, square(alg, Y))
            s = br(X, br(X, br(X, Y)))
            s2 = br(Y, br(Y, br(Y, X)))
            s3 = br(X, br(Y, br(X, Y)))
            s4 = square(alg, br(X, Y))
            s = [s[t] ^ s2[t] ^ s3[t] ^ s4[t] for t in range(m)]
            if lhs != [X4[t] ^ Y4[t] ^ s[t] for t in range(m)]:
                fails += 1
    out("     additivity rule (add4) on %d pairs of g_-: %d failures"
        % (pairs, fails))


def sweep_alpha():
    """all closed perturbations omega_1 + f dx_1dx_d with f a single x_k.

    d(f dx_1 dx_d) = sum_{j not in {1,d}} (partial_j f) dx_j dx_1 dx_d, so for
    f = x_k the form stays closed exactly when k = 1 or k = d; these are the
    forms alpha^(2) + omega' with alpha = dx_1 + x_k dx_d.  Which of them is in
    the second class?
    """
    out("=" * 78)
    out("2b)  the closed perturbations alpha = dx_1 + x_k dx_d, and their class")
    out("=" * 78)
    for m in range(2, 6):
        alg = Alg(m, (1,) * m)
        base = omega_std(alg)
        h0 = derived_series(alg, preserving_algebra(alg, base), steps=4)
        out("  m = %d: omega_1 = %s -> %s" % (m, show_form(alg, base), h0))
        for d in range(1, m):
            for k in (0, d):
                om = dict(base)
                key = ('a', 0, d)
                om[key] = om.get(key, 0) ^ alg.gen(k)
                if om[key] == 0:
                    del om[key]
                cl = not d2(alg, om)
                nd = alg.const_term(det(alg, gram(alg, om))) == 1
                ds = derived_series(alg, preserving_algebra(alg, om), steps=4) \
                    if (cl and nd) else None
                out("     alpha = dx_1 + x_%d dx_%d : closed %s, non-deg %s, %s%s"
                    % (k + 1, d + 1, cl, nd, ds,
                       "   <== second class" if ds is not None and ds != h0 else ""))


def gradings(m):
    """for the second shape: is there ANY Z/2-grading deg x_i = 1 giving a
    2-structure on g_+ (and hence a chance of a (2,4)-structure)?"""
    alg = Alg(m, (1,) * m)
    om = omega_two(alg)
    H = ham_solver(alg, om)
    fields = {1 << t: H(1 << t) for t in range(alg.dim)}
    hb = reduce_basis(alg, list(fields.values()))
    out("  m = %d, omega_2: dim Span{H_f} = %d" % (m, len(hb)))
    for i in range(m):
        plus, minus = [], []
        ok = True
        for f, D in fields.items():
            if D == [0] * m:
                continue
            degs = set()
            for t in range(alg.dim):
                if (f >> t) & 1:
                    degs.add(alg.mons[t][i] % 2)
            if degs == {0}:
                plus.append(D)
            elif degs == {1}:
                minus.append(D)
            else:
                ok = False
        plus, minus = reduce_basis(alg, plus), reduce_basis(alg, minus)
        if not ok or len(plus) + len(minus) != len(hb):
            out("     deg x_%d = 1 : not a grading of Span{H_f}" % (i + 1))
            continue
        # a Z/2-grading of the algebra?
        grad = all(in_span(alg, plus, bracket(alg, X, Y)) for X in plus for Y in plus) \
            and all(in_span(alg, minus, bracket(alg, X, Y)) for X in plus for Y in minus) \
            and all(in_span(alg, plus, bracket(alg, X, Y)) for X in minus for Y in minus)
        okplus = all(in_span(alg, plus, square(alg, D)) for D in plus)
        ok4 = all(in_span(alg, plus, square(alg, square(alg, D))) for D in minus)
        out("     deg x_%d = 1 : grading %s, dim g_+ = %d, g_- = %d, "
            "g_+ closed under the square %s, D^4 in g_+ %s"
            % (i + 1, grad, len(plus), len(minus), okplus, ok4))


def dual_field(alg, om):
    """W with iota_W omega = delta, delta = sum sqrt(b_i) dx_i (constants).

    W is canonically attached to omega (the diagonal is an invariant of the
    form by the closedness theorem), hence so is its square W^2 -- and the
    square of a vector field is again a vector field at p = 2.
    """
    m, dim = alg.m, alg.dim
    G = gram(alg, om)
    cols = []
    for i in range(m):
        for t in range(dim):
            D = [0] * m
            D[i] = 1 << t
            w = [0] * m
            for r in range(m):
                for c in range(m):
                    w[r] ^= alg.mul(G[r][c], D[c])
            v = []
            for r in range(m):
                v.extend((w[r] >> s) & 1 for s in range(dim))
            cols.append(v)
    M = [[cols[c][r] for c in range(len(cols))] for r in range(len(cols[0]))]
    b = []
    for r in range(m):
        # sqrt of a constant over F_2 is itself
        g = om.get(('b', r), 0)
        g = alg.one if alg.const_term(g) else 0
        b.extend((g >> s) & 1 for s in range(dim))
    z = solve(M, b, p=2)
    if z is None:
        return None
    W = [0] * m
    for c, coef in enumerate(z):
        if int(coef) % 2:
            i, t = divmod(c, dim)
            W[i] ^= 1 << t
    return W


def fname(alg, f):
    """name of a basis function of O(m;1)."""
    terms = []
    for t in range(alg.dim):
        if (f >> t) & 1:
            a = alg.mons[t]
            terms.append("1" if not any(a) else
                         "".join("x%d" % (j + 1) for j, e in enumerate(a) for _ in range(e)))
    return "+".join(terms)


def show_field(alg, D):
    out_s = []
    for i, f in enumerate(D):
        if f:
            terms = []
            for t in range(alg.dim):
                if (f >> t) & 1:
                    a = alg.mons[t]
                    terms.append("1" if not any(a) else
                                 "".join("x%d%s" % (j + 1, "" if e == 1 else "^(%d)" % e)
                                         for j, e in enumerate(a) if e))
            out_s.append("(%s)d_%d" % ("+".join(terms), i + 1))
    return " + ".join(out_s) if out_s else "0"


def invariant():
    """the canonical field W and its square as the invariant separating the
    two classes."""
    out("=" * 78)
    out("6)  the canonical field W (iota_W omega = delta) and the invariant W^2")
    out("=" * 78)
    for m in range(1, 6):
        alg = Alg(m, (1,) * m)
        for name, om in (("omega_1", omega_std(alg)), ("omega_2", omega_two(alg))):
            if m == 1 and name == "omega_2":
                continue
            W = dual_field(alg, om)
            W2 = square(alg, W)
            out("  m = %d  %s:  W = %s,  L_W omega = %s,  W^2 = %s"
                % (m, name, show_field(alg, W),
                   "0" if not lie(alg, W, om) else "NOT 0", show_field(alg, W2)))
    out()
    out("  contingency: derived-series class vs (W^2 = 0), random closed "
        "non-degenerate non-alternating forms")
    import random
    random.seed(277)
    for m, ntry in ((3, 200), (4, 60), (5, 25)):
        alg = Alg(m, (1,) * m)
        Z = closed_space(alg)
        h1 = derived_series(alg, preserving_algebra(alg, omega_std(alg)), steps=4)
        table = {}
        seen = 0
        while seen < ntry:
            om = {}
            for z in Z:
                if random.getrandbits(1):
                    for k, v in z.items():
                        om[k] = om.get(k, 0) ^ v
            om = {k: v for k, v in om.items() if v}
            if alternating(om):
                continue
            if alg.const_term(det(alg, gram(alg, om))) != 1:
                continue
            seen += 1
            ds = derived_series(alg, preserving_algebra(alg, om), steps=4)
            W = dual_field(alg, om)
            key = ("class 1" if ds == h1 else "class 2",
                   "W^2 = 0" if square(alg, W) == [0] * m else "W^2 != 0")
            table[key] = table.get(key, 0) + 1
        out("    m = %d, %d samples: %s" % (m, ntry, sorted(table.items())))


def main():
    t0 = time.time()
    basics()
    out()
    sweep_alpha()
    out()
    out("=" * 78)
    out("3)  the exact orbit test")
    out("=" * 78)
    for m in (2, 3):
        orbits(m)
    out()
    out("=" * 78)
    out("4)  the (2,4)-structure question for BOTH shapes, m odd")
    out("=" * 78)
    for m in (3, 5):
        alg = Alg(m, (1,) * m)
        two_four(m, "omega_1 (standard)", omega_std(alg))
        two_four(m, "omega_2 (second shape)", omega_two(alg))
    out()
    out("=" * 78)
    out("5)  is there any Z/2-grading at all for the second shape?")
    out("=" * 78)
    for m in (3, 5):
        gradings(m)
    out()
    invariant()
    out()
    out("[%.1f s]" % (time.time() - t0))


if __name__ == "__main__":
    LOG = open("shape2.log", "w")
    main()
    LOG.close()
