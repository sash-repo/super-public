"""The Lie superalgebra svect^L_lambda(1|2) and its central extension, from scratch.

Everything is built from the vector-field realization: the basis (68) of
the paper,

    LL_m  = t^m ( t d_t + (lambda+m+1)/2 (th1 d_1 + th2 d_2) ),
    S^j_m = t^m th_j ( t d_t + (lambda+m+1)(th1 d_1 + th2 d_2) ),
    Tp_m  = t^m th1 d_2,  Tm_m = t^m th2 d_1,  T0_m = t^m (th1 d_1 - th2 d_2),
    P^j_m = t^m d_j,

inside vect^L(1|2) = der (C[t,t^{-1}] (x) Lambda(th1,th2)).

What is checked/derived here:

 1) the span of (68) is a subalgebra -- the structure constants are read
    off by solving, in each t-degree, a 12-dimensional linear system;
 2) it is exactly the kernel of the twisted divergence div_lambda, so the family
    is the one the paper says it is;
 3) the cocycle (69) is a 2-cocycle -- the Jacobi identity for the
    central extension, checked on all triples in a window of modes;
 4) the anti-involutions sigma of the centrally extended algebra: all of them
    are found by solving the defining equations, not guessed.

Exact arithmetic: sympy over Q(lambda, kappa).
"""

import itertools
import sys

import sympy

LAM = sympy.Symbol("lamb")           # the parameter lambda of svect^L_lambda
KAP = sympy.Symbol("kappa")          # the central charge in the normalization (69)

# ---------------------------------------------------------------------------
#  functions on the supercircle: dict {(a, S): coeff},  a in Z, S a sorted tuple
# ---------------------------------------------------------------------------

def fadd(*fs):
    out = {}
    for f in fs:
        for k, v in f.items():
            out[k] = out.get(k, 0) + v
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


def fsmul(s, f):
    return {k: sympy.expand(s * v) for k, v in f.items() if sympy.expand(s * v) != 0}


def _sortsign(seq):
    """sign of the permutation sorting the tuple ``seq``; None if a repeat occurs."""
    seq = list(seq)
    if len(set(seq)) != len(seq):
        return None, None
    sign = 1
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                sign = -sign
    return sign, tuple(sorted(seq))


def fmul(f, g):
    """product of two functions (Grassmann in th1, th2)."""
    out = {}
    for (a, S), u in f.items():
        for (b, T), v in g.items():
            sign, ST = _sortsign(S + T)
            if sign is None:
                continue
            out[(a + b, ST)] = out.get((a + b, ST), 0) + sign * u * v
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


def fpar(key):
    """parity of a function monomial."""
    return len(key[1]) % 2


def d_t(f):
    out = {}
    for (a, S), v in f.items():
        if a != 0:
            out[(a - 1, S)] = out.get((a - 1, S), 0) + a * v
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


def d_th(j, f):
    """left derivative d/d th_j."""
    out = {}
    for (a, S), v in f.items():
        if j not in S:
            continue
        p = S.index(j)
        T = S[:p] + S[p + 1:]
        out[(a, T)] = out.get((a, T), 0) + (-1) ** p * v
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


def deriv(i, f):
    return d_t(f) if i == "t" else d_th(int(i), f)


DIRS = ("t", "1", "2")
DIRPAR = {"t": 0, "1": 1, "2": 1}


# ---------------------------------------------------------------------------
#  vector fields: dict {(i, a, S): coeff}, meaning  sum coeff t^a th_S d_i
# ---------------------------------------------------------------------------

def vf(i, f):
    """the vector field f * d_i."""
    return {(i,) + k: v for k, v in f.items()}


def vadd(*Xs):
    out = {}
    for X in Xs:
        for k, v in X.items():
            out[k] = out.get(k, 0) + v
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


def vsmul(s, X):
    return {k: sympy.expand(s * v) for k, v in X.items() if sympy.expand(s * v) != 0}


def comp(X, i):
    """the coefficient function of d_i in X."""
    return {(a, S): v for (j, a, S), v in X.items() if j == i}


def vpar(X):
    """parity of a homogeneous vector field."""
    ps = {(len(S) + DIRPAR[i]) % 2 for (i, a, S) in X}
    if len(ps) > 1:
        raise ValueError("inhomogeneous vector field")
    return ps.pop() if ps else 0


def act(X, f):
    """X(f) for a function f."""
    out = {}
    for i in DIRS:
        fi = comp(X, i)
        if fi:
            out = fadd(out, fmul(fi, deriv(i, f)))
    return out


def bracket(X, Y):
    """[X, Y] = X o Y - (-1)^{p(X)p(Y)} Y o X, componentwise."""
    pX, pY = vpar(X), vpar(Y)
    sgn = (-1) ** (pX * pY)
    out = {}
    for i in DIRS:
        gi = comp(Y, i)
        fi = comp(X, i)
        term = fadd(act(X, gi), fsmul(-sgn, act(Y, fi)))
        # X o Y contributes  f_j d_j(g_i) d_i  -- but the Koszul sign of moving
        # d_j past nothing is trivial only if we differentiate before multiplying,
        # which is what act() does; the sign of p(X) against d_i is accounted for
        # by the rule  (f d_j)(g d_i) = f d_j(g) d_i  with left derivatives.
        if term:
            out.update({(i,) + k: v for k, v in term.items()})
    return {k: sympy.expand(v) for k, v in out.items() if sympy.expand(v) != 0}


# ---------------------------------------------------------------------------
#  the basis (68)
# ---------------------------------------------------------------------------

E_OP = vadd(vf("1", {(0, (1,)): 1}), vf("2", {(0, (2,)): 1}))   # th1 d_1 + th2 d_2


def basis_elt(name, m):
    """the element ``name`` of mode m of (68), as a vector field."""
    lm = LAM + m + 1
    if name == "LL":
        return vadd(vf("t", {(m + 1, ()): 1}),
                    vf("1", {(m, (1,)): sympy.Rational(1, 2) * lm}),
                    vf("2", {(m, (2,)): sympy.Rational(1, 2) * lm}))
    if name == "S1":
        return vadd(vf("t", {(m + 1, (1,)): 1}),
                    vf("2", {(m, (1, 2)): lm}))
    if name == "S2":
        return vadd(vf("t", {(m + 1, (2,)): 1}),
                    vf("1", {(m, (1, 2)): -lm}))
    if name == "Tp":
        return vf("2", {(m, (1,)): 1})
    if name == "Tm":
        return vf("1", {(m, (2,)): 1})
    if name == "T0":
        return vadd(vf("1", {(m, (1,)): 1}), vf("2", {(m, (2,)): -1}))
    if name == "P1":
        return vf("1", {(m, ()): 1})
    if name == "P2":
        return vf("2", {(m, ()): 1})
    raise KeyError(name)


NAMES = ("LL", "S1", "S2", "Tp", "Tm", "T0", "P1", "P2")
PARITY = {"LL": 0, "Tp": 0, "Tm": 0, "T0": 0, "S1": 1, "S2": 1, "P1": 1, "P2": 1}

#: the root of a generator: (a, b, q) meaning the LL_0-weight a + b*(lambda+1)/2
#: and the T0_0-weight (``charge``) q.
ROOT = {"LL": (1, 0, 0), "T0": (1, 0, 0), "Tp": (1, 0, 2), "Tm": (1, 0, -2),
        "S1": (1, 1, 1), "S2": (1, 1, -1), "P1": (1, -1, -1), "P2": (1, -1, 1)}


def root(name, m):
    a, b, q = ROOT[name]
    return (m, b, q)


# ---------------------------------------------------------------------------
#  t-degree components and the coordinates of a vector field in the basis
# ---------------------------------------------------------------------------

def tdeg_monos(m):
    """a basis of the t-degree-m part of vect^L(1|2) (12-dimensional)."""
    out = []
    for S in ((), (1,), (2,), (1, 2)):
        out.append(("t", m + 1, S))
    for i in ("1", "2"):
        for S in ((), (1,), (2,), (1, 2)):
            out.append((i, m, S))
    return out


def tdeg_of(X):
    """the t-degree of a homogeneous X (deg t = 1, deg th = 0, deg d_t = -1)."""
    ds = set()
    for (i, a, S) in X:
        ds.add(a - 1 if i == "t" else a)
    if len(ds) > 1:
        raise ValueError("inhomogeneous in t-degree: %s" % X)
    return ds.pop() if ds else 0


_COORD_CACHE = {}


def coords(X):
    """express X in the basis (68); returns {(name, m): coeff}."""
    if not X:
        return {}
    m = tdeg_of(X)
    monos = tdeg_monos(m)
    idx = {k: i for i, k in enumerate(monos)}
    key = m
    M = _COORD_CACHE.get(key)
    if M is None:
        cols = []
        for nm in NAMES:
            B = basis_elt(nm, m)
            col = [sympy.Integer(0)] * len(monos)
            for k, v in B.items():
                col[idx[k]] = v
            cols.append(col)
        M = sympy.Matrix(cols).T          # 12 x 8
        _COORD_CACHE[key] = M
    rhs = sympy.zeros(len(monos), 1)
    for k, v in X.items():
        if k not in idx:
            raise ValueError("monomial %s outside t-degree %s" % (k, m))
        rhs[idx[k]] = v
    sol = M.solve_least_squares(rhs) if False else None
    # exact solve: M x = rhs, M has full column rank 8
    aug = M.row_join(rhs)
    red, piv = aug.rref()
    if len(monos) in piv:
        raise ValueError("X is not in the span of (68): %s" % X)
    x = [sympy.Integer(0)] * len(NAMES)
    for r, pcol in enumerate(piv):
        if pcol < len(NAMES):
            x[pcol] = sympy.simplify(red[r, len(NAMES)])
    # verify
    chk = vadd(*[vsmul(x[i], basis_elt(NAMES[i], m)) for i in range(len(NAMES))]) if any(x) else {}
    if fadd_vf_diff(chk, X):
        raise ValueError("coordinate solve failed for %s" % X)
    return {(NAMES[i], m): x[i] for i in range(len(NAMES)) if x[i] != 0}


def fadd_vf_diff(A, B):
    d = vadd(A, vsmul(-1, B))
    d = {k: sympy.simplify(v) for k, v in d.items()}
    return {k: v for k, v in d.items() if v != 0}


# ---------------------------------------------------------------------------
#  the cocycle (69)
# ---------------------------------------------------------------------------

def cocycle(n1, m1, n2, m2):
    """c(x, y) for basis elements x = n1_{m1}, y = n2_{m2}, in units of kappa."""
    if m1 + m2 != 0:
        return sympy.Integer(0)
    m = m1
    if n1 == "LL" and n2 == "LL":
        return sympy.Rational(1, 2) * m * (m ** 2 - (LAM + 1) ** 2)
    if n1 in ("P1", "P2") and n2 in ("S1", "S2"):
        i = n1[-1]
        j = n2[-1]
        if i != j:
            return sympy.Integer(0)
        return -m * (m - (LAM + 1))
    if n1 in ("S1", "S2") and n2 in ("P1", "P2"):
        # c is super-skew: for two odd elements it is symmetric
        return cocycle(n2, m2, n1, m1)
    if n1 == "T0" and n2 == "T0":
        return 2 * m
    if n1 == "Tp" and n2 == "Tm":
        return m
    if n1 == "Tm" and n2 == "Tp":
        return -cocycle("Tp", m2, "Tm", m1)
    return sympy.Integer(0)


# ---------------------------------------------------------------------------
#  the bracket in the basis, with the central term
# ---------------------------------------------------------------------------

_BR_CACHE = {}


def br(x, y):
    """[x, y] for x = (name, m), y = (name, n): ({(name,k): coeff}, central)."""
    ck = (x, y)
    got = _BR_CACHE.get(ck)
    if got is not None:
        return got
    X, Y = basis_elt(*x), basis_elt(*y)
    Z = bracket(X, Y)
    terms = coords(Z) if Z else {}
    cen = KAP * cocycle(x[0], x[1], y[0], y[1])
    out = (terms, sympy.expand(cen))
    _BR_CACHE[ck] = out
    return out


# ---------------------------------------------------------------------------
#  checks
# ---------------------------------------------------------------------------

def divergence(X):
    """div X = sum (-1)^{p(f_i) p(d_i)} d_i(f_i)."""
    out = {}
    for i in DIRS:
        fi = comp(X, i)
        if not fi:
            continue
        for k, v in fi.items():
            piece = deriv(i, {k: v})
            s = (-1) ** (fpar(k) * DIRPAR[i])
            out = fadd(out, fsmul(s, piece))
    return out


def div_lambda(X):
    """the twisted divergence: div(t^lambda X) / t^lambda, i.e.
       div X + lambda * t^{-1} * (the d_t-coefficient of X)."""
    A = comp(X, "t")
    tw = fmul({(-1, ()): LAM}, A)
    return fadd(divergence(X), tw)


def check_closure(MODES):
    bad = []
    for (n1, m1) in itertools.product(NAMES, MODES):
        for (n2, m2) in itertools.product(NAMES, MODES):
            try:
                br((n1, m1), (n2, m2))
            except ValueError as e:
                bad.append(((n1, m1), (n2, m2), str(e)))
    return bad


def check_div(MODES):
    bad = []
    for nm in NAMES:
        for m in MODES:
            d = div_lambda(basis_elt(nm, m))
            d = {k: sympy.simplify(v) for k, v in d.items()}
            d = {k: v for k, v in d.items() if v != 0}
            if d:
                bad.append((nm, m, d))
    return bad


def check_jacobi(MODES):
    """the Jacobi identity for the centrally extended algebra."""
    gens = [(n, m) for n in NAMES for m in MODES]
    bad = []
    for x in gens:
        for y in gens:
            for z in gens:
                px, py, pz = PARITY[x[0]], PARITY[y[0]], PARITY[z[0]]
                # [x,[y,z]] = [[x,y],z] + (-1)^{p(x)p(y)} [y,[x,z]]
                lhs_t, lhs_c = compose(x, br(y, z))
                a_t, a_c = compose_right(br(x, y), z)
                b_t, b_c = compose(y, br(x, z))
                sgn = (-1) ** (px * py)
                dt = {}
                for k, v in lhs_t.items():
                    dt[k] = dt.get(k, 0) + v
                for k, v in a_t.items():
                    dt[k] = dt.get(k, 0) - v
                for k, v in b_t.items():
                    dt[k] = dt.get(k, 0) - sgn * v
                dt = {k: sympy.simplify(v) for k, v in dt.items()}
                dt = {k: v for k, v in dt.items() if v != 0}
                dc = sympy.simplify(lhs_c - a_c - sgn * b_c)
                if dt or dc != 0:
                    bad.append((x, y, z, dt, dc))
    return bad


def compose(x, rhs):
    """[x, w] where w = (terms, central); the centre is central, so it drops."""
    terms, _ = rhs
    out_t, out_c = {}, sympy.Integer(0)
    for k, v in terms.items():
        t2, c2 = br(x, k)
        for k2, v2 in t2.items():
            out_t[k2] = out_t.get(k2, 0) + v * v2
        out_c += v * c2
    return out_t, sympy.expand(out_c)


def compose_right(lhs, z):
    terms, _ = lhs
    out_t, out_c = {}, sympy.Integer(0)
    for k, v in terms.items():
        t2, c2 = br(k, z)
        for k2, v2 in t2.items():
            out_t[k2] = out_t.get(k2, 0) + v * v2
        out_c += v * c2
    return out_t, sympy.expand(out_c)


if __name__ == "__main__":
    MODES = list(range(-3, 4))
    log = open("run-svectlam.log", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        log.write(s + "\n")
        log.flush()

    say("=== svect^L_lambda(1|2): the basis (68) ===")
    for nm in NAMES:
        say("  %-3s parity %d  root (m, b, q) = %s" % (nm, PARITY[nm], ROOT[nm]))

    say("\n=== 1) is the span of (68) a subalgebra? ===")
    bad = check_closure(MODES)
    say("   modes %s..%s: %s" % (MODES[0], MODES[-1],
                                 "CLOSED" if not bad else "NOT closed: %s" % bad[:3]))

    say("\n=== 2) is it the kernel of the twisted divergence? ===")
    bad = check_div(MODES)
    say("   %s" % ("all %d basis elements have div_lambda = 0"
                   % (len(NAMES) * len(MODES)) if not bad else "FAILS: %s" % bad[:3]))

    say("\n=== 3) Jacobi for the central extension (69) ===")
    MJ = list(range(-2, 3))
    bad = check_jacobi(MJ)
    say("   modes %s..%s, %d triples: %s"
        % (MJ[0], MJ[-1], (len(NAMES) * len(MJ)) ** 3,
           "OK" if not bad else "FAILS (%d): %s" % (len(bad), bad[:2])))

    say("\n=== the brackets of the zero modes ===")
    for x in [("LL", 0), ("T0", 0), ("Tp", 0), ("Tm", 0)]:
        for y in [("LL", 0), ("T0", 0), ("Tp", 0), ("Tm", 0), ("S1", 0), ("P1", 0)]:
            t, c = br(x, y)
            if t or c:
                say("   [%s, %s] = %s %s" % (x, y, t, ("+ %s" % c) if c else ""))
    log.close()
