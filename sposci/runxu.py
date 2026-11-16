"""Checks on Xu's conformal representations -- see xuconf.py for the model.

Usage: python3 runxu.py [DEG]      (default DEG=3)

For each (n, m) [and each of the two families, osp(2n+2|2m) and osp(2n+3|2m)]:

  FORM  the matrices Xu writes down preserve a nondegenerate even supersymmetric
        form, unique up to a scalar, and there are dim osp(...) of them;
  HOM   [pi_c(X), pi_c(Y)] = pi_c([X,Y]) on every monomial of degree <= DEG,
        as an identity in c;
  VECT  pi_0(X) is a derivation of A for every X, i.e., a vector field;
  MULT  pi_c(X) - pi_0(X) is the multiplication by the function h_X := pi_c(X)1;
  DIV   h_X = lambda Div(pi_0(X)) with lambda = c/(sdim A), sdim A = 2n-2m
        [resp. 2n+1-2m] -- i.e., A is the module F_lambda of weighted densities.
        When sdim A = 0 the divergence vanishes identically while h is not 0:
        the c-family is then NOT a density twist.
"""
import sys, os
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "lib"))
from xuconf import (Xu, Op, in_osp, pnum, psubs, norm, eadd, escale,   # noqa
                    ZERO, ONE, CVAR, Fraction as _F)
from exactlin import nullspace, rank, backend                          # noqa

DEG = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ FORM
def check_form(X):
    mat, N = X.mat, X.mat.N
    rows = []
    for M, _ in X.pairs:
        pX = mat.parity(M)
        for u in range(N):
            for v in range(N):
                row = [Fraction(0)] * (N * N)
                for i in range(N):
                    a = M.get((i, u), Fraction(0))
                    if a:
                        row[i * N + v] += a
                sgn = -1 if (pX and mat.par(u)) else 1
                for i in range(N):
                    b = M.get((i, v), Fraction(0))
                    if b:
                        row[u * N + i] += sgn * b
                if any(row):
                    rows.append(row)
    ns = nullspace(rows)
    dimforms = len(ns)
    inosp = all(in_osp(mat, M, X.F) for M, _ in X.pairs)
    d = X.dim_osp()
    return dimforms, inosp, len(X.pairs), d, sum(d)


# ------------------------------------------------------------------- HOM
def check_hom(X, deg):
    mat = X.mat
    monos = [m for d in range(deg + 1) for m in X.alg.monomials(d)]
    bad = 0
    for a, (MA, OA) in enumerate(X.pairs):
        for b, (MB, OB) in enumerate(X.pairs):
            if b < a:
                continue
            Mbr = mat.bracket(MA, MB)
            coords = X.expand(Mbr) if Mbr else [Fraction(0)] * len(X.pairs)
            if coords is None:
                log("   !! bracket leaves the span:", a, b)
                bad += 1
                continue
            lhs = OA.bracket(OB)
            rhs = X.pi(coords)
            for mono in monos:
                if norm(eadd(lhs(mono), escale(rhs(mono), pnum(-1)))):
                    log("   !! [%d,%d] differs on %s" % (a, b, str(mono)))
                    bad += 1
                    break
    return bad


# ------------------------------------------------------- VECT / MULT / DIV
def mono_mul(alg, m1, m2):
    (a1, J1), (a2, J2) = m1, m2
    if set(J1) & set(J2):
        return None, 0
    sgn = 0
    for p in J2:                                   # move each th of J2 left
        sgn += sum(1 for q in J1 if q > p)
    a = tuple(u + v for u, v in zip(a1, a2))
    return (a, tuple(sorted(J1 + J2))), (-1) ** sgn


def at_c(elt, val):
    """substitute c = val in an element"""
    return {k: (psubs(v, val),) for k, v in elt.items() if psubs(v, val)}


def is_derivation(alg, op, deg):
    """op(fg) = op(f)g + (-1)^{p(op)p(f)} f op(g) for monomials f, g"""
    monos = [m for d in range(deg + 1) for m in alg.monomials(d)]
    for f in monos:
        for g in monos:
            fg, s = mono_mul(alg, f, g)
            if fg is None:
                continue
            lhs = escale(op(fg), pnum(s))
            rhs = {}
            for mono, coef in op(f).items():
                h, s2 = mono_mul(alg, mono, g)
                if h:
                    rhs = eadd(rhs, escale({h: coef}, pnum(s2)))
            sg = -1 if (op.parity and alg.parity(f)) else 1
            for mono, coef in op(g).items():
                h, s2 = mono_mul(alg, f, mono)
                if h:
                    rhs = eadd(rhs, escale({h: coef}, pnum(s2 * sg)))
            if norm(eadd(lhs, escale(rhs, pnum(-1)))):
                return False
    return True


def pi_zero(op, X):
    """the c=0 part of an operator: substitute c=0 in every coefficient"""
    return Op(lambda m: {k: (v[0],) for k, v in op(m).items() if v and v[0]},
              op.parity, X.alg)


def divergence(X, op):
    """Div of the vector field op = sum f_r d_{x_r} + sum g_s d_{th_s},

           Div = sum_r d_{x_r}(f_r) + sum_s (-1)^{p(g_s)} d_{th_s}(g_s),

    where f_r = op(x_r) and g_s = op(th_s), op being a derivation.
    """
    alg = X.alg
    out = {}
    for r in range(alg.nx):
        xr = ((0,) * r + (1,) + (0,) * (alg.nx - r - 1), ())
        out = eadd(out, alg.d_x(r)(op(xr)))
    for s in range(alg.nth):
        ths = ((0,) * alg.nx, (s,))
        for mono, coef in op(ths).items():
            sg = -1 if alg.parity(mono) else 1
            out = eadd(out, escale(escale(alg.d_th(s)(mono), coef), pnum(sg)))
    return norm(out)


def check_vect_mult_div(X, deg):
    alg = X.alg
    one = ((0,) * alg.nx, ())
    monos = [m for d in range(deg + 1) for m in alg.monomials(d)]
    sdim = alg.nx - alg.nth
    bad_vect = bad_mult = bad_div = 0
    hs = []
    for M, op in X.pairs:
        op0 = pi_zero(op, X)
        if not is_derivation(alg, op0, min(deg, 2)):
            bad_vect += 1
        h = op(one)                                    # h_X = pi_c(X) 1
        h0 = op0(one)
        assert not norm(h0), "pi_0(X)1 must vanish for a vector field"
        # pi_c(X) - pi_0(X) = multiplication by h?
        for f in monos:
            lhs = eadd(op(f), escale(op0(f), pnum(-1)))
            rhs = {}
            for mono, coef in h.items():
                g, s = mono_mul(alg, mono, f)
                if g:
                    rhs = eadd(rhs, escale({g: coef}, pnum(s)))
            if norm(eadd(lhs, escale(rhs, pnum(-1)))):
                bad_mult += 1
                break
        d = divergence(X, op0)
        hs.append((h, d))
        if sdim:
            want = escale(d, (Fraction(0), Fraction(1, sdim)))   # (c/sdim)*Div
            if norm(eadd(h, escale(want, pnum(-1)))):
                bad_div += 1
        else:
            if norm(d):
                bad_div += 1                            # Div must vanish
    nz_h = sum(1 for h, d in hs if norm(h))
    nz_d = sum(1 for h, d in hs if norm(d))
    return bad_vect, bad_mult, bad_div, nz_h, nz_d, sdim


# ------------------------------------------------- finite-dimensional pieces
def A_ell(X, ell):
    """a spanning set of Xu's A_[ell] = A_0 + sum_{i<=ell} (A_i + eta^i A_{ell-i})"""
    alg = X.alg
    one = ((0,) * alg.nx, ())
    out = [{one: ONE}]
    for i in range(1, ell + 1):
        for m in alg.monomials(i):
            out.append({m: ONE})
    # eta^i A_{ell-i}
    etas = [{one: ONE}]
    for i in range(1, ell + 1):
        etas.append(X.eta(etas[-1]))
    for i in range(1, ell + 1):
        for m in alg.monomials(ell - i):
            e = {}
            for mono, coef in etas[i].items():
                g, s = mono_mul(alg, mono, m)
                if g:
                    e = eadd(e, escale({g: coef}, pnum(s)))
            if norm(e):
                out.append(e)
    return out


def check_findim(X, ell):
    """at c = -ell, is A_[ell] a submodule, and of what superdimension?"""
    alg = X.alg
    gens = A_ell(X, ell)
    gens = [at_c(g, -ell) for g in gens]
    keys = sorted({k for g in gens for k in g})
    idx = {k: i for i, k in enumerate(keys)}

    def vec(e, ks=None):
        ks = ks if ks is not None else keys
        return [e.get(k, (Fraction(0),))[0] if e.get(k) else Fraction(0)
                for k in ks]

    rows = [vec(g) for g in gens]
    dim = rank(rows)
    # superdimension of the span
    ev = [g for g in gens if all(alg.parity(k) == 0 for k in g)]
    od = [g for g in gens if all(alg.parity(k) == 1 for k in g)]
    sd = (rank([vec(g) for g in ev]) if ev else 0,
          rank([vec(g) for g in od]) if od else 0)
    # invariance
    bad = 0
    for M, op in X.pairs:
        for g in gens:
            img = {}
            for mono, coef in g.items():
                img = eadd(img, escale(op(mono), coef))
            img = at_c(img, -ell)
            if not img:
                continue
            allk = sorted(set(keys) | set(img))
            if rank([vec(h, allk) for h in gens] + [vec(img, allk)]) > dim:
                bad += 1
                break
    return dim, sd, bad


# ------------------------------------------------------------------- COCY
def check_coboundary(X, deg):
    """is h_X = pi_0(X) f for one f in A of degree <= deg?

    If it were, pi_c would be pi_0 twisted by multiplication by an invertible
    function, i.e., the c-family would be trivial.  We solve the linear system
    for the coefficients of f at c = 1.
    """
    alg = X.alg
    one = ((0,) * alg.nx, ())
    fmonos = [m for d in range(deg + 1) for m in alg.monomials(d)]
    rows, rhs = [], []
    tgt = {}                                        # target monomials
    cols = {}
    for M, op in X.pairs:
        op0 = pi_zero(op, X)
        h = at_c(op(one), 1)
        img = {}
        for k, fm in enumerate(fmonos):
            img[k] = op0(fm)
        keys = sorted({kk for v in img.values() for kk in v} | set(h))
        for kk in keys:
            rows.append([img[k].get(kk, (Fraction(0),))[0] if img[k].get(kk)
                         else Fraction(0) for k in range(len(fmonos))])
            rhs.append(h.get(kk, (Fraction(0),))[0] if h.get(kk)
                       else Fraction(0))
    from exactlin import solve as _solve
    sol = _solve(rows, rhs)
    return sol is not None


# ---------------------------------------------------------------------- run
def report(n, m, odd, deg):
    tag = "osp(%d|%d) on C^{%d|%d}" % (2 * n + 2 + (1 if odd else 0), 2 * m,
                                       2 * n + (1 if odd else 0), 2 * m)
    log("=" * 72)
    log("n=%d m=%d  %s   (monomials up to degree %d)" % (n, m, tag, deg))
    X = Xu(n, m, odd)
    dimforms, inosp, npairs, d, dtot = check_form(X)
    log("  FORM  invariant forms: %d   all in osp: %s   basis %d = dim osp (%d|%d)=%d : %s"
        % (dimforms, inosp, npairs, d[0], d[1], dtot, npairs == dtot))
    bad = check_hom(X, deg)
    log("  HOM   failures over all %d pairs: %d"
        % (len(X.pairs) * (len(X.pairs) + 1) // 2, bad))
    bv, bm, bd, nzh, nzd, sdim = check_vect_mult_div(X, deg)
    log("  VECT  pi_0(X) not a derivation: %d" % bv)
    log("  MULT  pi_c(X)-pi_0(X) not a multiplication: %d" % bm)
    if sdim:
        log("  DIV   h_X != (c/%d) Div pi_0(X): %d   [nonzero h: %d, nonzero Div: %d]"
            % (sdim, bd, nzh, nzd))
    else:
        log("  DIV   sdim A = 0: Div nonzero on %d elements (want 0), "
            "h nonzero on %d  ==> the c-family is NOT a density twist"
            % (nzd, nzh))
        cob = check_coboundary(X, 2)
        log("  COCY  h a coboundary X |-> pi_0(X)f, deg f <= 2: %s "
            "==> the family is %s" % (cob, "trivial" if cob else "a genuine "
                                      "deformation"))
    return X


if __name__ == "__main__":
    log("backend:", backend())
    cases = [(1, 1, False), (2, 1, False), (1, 2, False), (2, 2, False),
             (1, 1, True), (2, 1, True)]
    for n, m, odd in cases:
        X = report(n, m, odd, DEG)
        if not odd:
            lo = min(2 * (n - m), 0) - 1
            for ell in (1, 2):
                if -ell <= lo:
                    dim, sd, bad = check_findim(X, ell)
                    log("  FIN   c=%d: dim A_[%d] = %d, sdim = (%d|%d), "
                        "non-invariant elements: %d" % (-ell, ell, dim,
                                                        sd[0], sd[1], bad))
    log("=" * 72)
