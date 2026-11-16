"""Does [GMO]'s construction of a symmetry survive outside dimension one?

[GMO] = Gargoubi--Mathonet--Ovsienko, arXiv:math-ph/0506044, build every non-scalar
symmetry of a module of differential operators on the line as

        A  |-->  B( pi(A), - ),

where B is an invariant BILINEAR operator of Grozman's list and pi an invariant
projection of the module of operators onto a module of tensor fields.  Their two
richest examples are B = the Grozman operator: their eqs. (5.4) and Section 5.4.1.

The construction is not one-dimensional.  For B: A_1 (x) A_2 --> A_3 of order d and
M := D^{<=W}(A_1 --> A_3) the operators of (weighted) order <= W, the composite
B o sigma_W, sigma_W the principal symbol followed by the projection onto an
isotypic piece, is an invariant endomorphism of M as soon as

   (L)   A_2 occurs in the symbol layer  gr^W M,

and it squares to zero as soon as W > d, so that End M contains [GMO]'s algebra a.
What has to be decided in each geometry is (L).  This engine decides it:

 * the CONTACT case, k(2n+1), modules T(k,lam)=S^k D^* (x) L^lam.  With
   Lam^2 D = L^{-1} (BN's own normalization of varpi), D = D^* (x) L^{-1} and
   T/D = L^{-1}, one has

     gr^W M = (+)_{r+2s=W} S^{k_1}D (x) S^{k_3}D (x) S^r D (x) L^{lam_3-lam_1+k_3-s},

   and a one-row piece S^m D of the triple product costs L^{-c}, c=(k_1+k_3+r-m)/2.

 * the SYMPLECTIC case, h(2m): no L at all (omega^m trivializes the volume), so (L)
   is a pure multiplicity question.

The sp(2n)-multiplicities are exact: weight multiplicities by convolution and
Racah's alternation over the Weyl group of C_n,

     mult(V_mu in X) = sum_{w in W} det(w) * dim X_{mu+rho-w(rho)}.

Self-tests: python3 layers.py --test
Census:     python3 layers.py --census
"""

import sys
from collections import defaultdict
from itertools import permutations, product


# ---------------------------------------------------------------------------
# sp(2n): weights in Z^n, the module V of weights +-eps_i.
# ---------------------------------------------------------------------------

def weyl_group(n):
    """The signed permutations of C_n, with their determinants."""
    out = []
    for perm in permutations(range(n)):
        sgn_perm = 1
        seen = [False] * n
        for i in range(n):                      # sign of the permutation
            if seen[i]:
                continue
            j, length = i, 0
            while not seen[j]:
                seen[j] = True
                j = perm[j]
                length += 1
            if length % 2 == 0:
                sgn_perm = -sgn_perm
        for signs in product((1, -1), repeat=n):
            det = sgn_perm
            for s in signs:
                if s == -1:
                    det = -det
            out.append((perm, signs, det))
    return out


def w_apply(w, v):
    perm, signs, _ = w
    return tuple(signs[i] * v[perm[i]] for i in range(len(v)))


def rho(n):
    return tuple(n - i for i in range(n))


def wt_V(n):
    """Weight multiplicities of the tautological module."""
    m = defaultdict(int)
    for i in range(n):
        for s in (1, -1):
            v = [0] * n
            v[i] = s
            m[tuple(v)] += 1
    return dict(m)


def wt_sym(n, a):
    """Weight multiplicities of S^a V: monomials of degree a in the 2n weight vectors."""
    basis = []
    for i in range(n):
        for s in (1, -1):
            v = [0] * n
            v[i] = s
            basis.append(tuple(v))
    m = defaultdict(int)

    def rec(idx, left, acc):
        if idx == len(basis):
            if left == 0:
                m[tuple(acc)] += 1
            return
        for t in range(left + 1):
            rec(idx + 1, left - t,
                [acc[i] + t * basis[idx][i] for i in range(n)])
    rec(0, a, [0] * n)
    return dict(m)


def wt_tensor(A, B, n):
    m = defaultdict(int)
    for u, cu in A.items():
        for v, cv in B.items():
            m[tuple(u[i] + v[i] for i in range(n))] += cu * cv
    return dict(m)


def mult(weights, mu, n, W=None):
    """Racah: the multiplicity of the irreducible with highest weight mu."""
    if W is None:
        W = weyl_group(n)
    r = rho(n)
    total = 0
    for w in W:
        wr = w_apply(w, r)
        key = tuple(mu[i] + r[i] - wr[i] for i in range(n))
        total += w[2] * weights.get(key, 0)
    return total


def triple_one_row(n, a, b, c, m, W=None):
    """Multiplicity of S^m V in S^a V (x) S^b V (x) S^c V over sp(2n)."""
    X = wt_tensor(wt_tensor(wt_sym(n, a), wt_sym(n, b), n), wt_sym(n, c), n)
    mu = tuple([m] + [0] * (n - 1))
    return mult(X, mu, n, W)


def pair_one_row(n, a, b, m, W=None):
    X = wt_tensor(wt_sym(n, a), wt_sym(n, b), n)
    mu = tuple([m] + [0] * (n - 1))
    return mult(X, mu, n, W)


def dim_sym(n, a):
    """dim S^a V for V of dimension 2n."""
    num, den = 1, 1
    for t in range(a):
        num *= 2 * n + t
        den *= t + 1
    return num // den


# ---------------------------------------------------------------------------
# The contact case.
# ---------------------------------------------------------------------------

def contact_layers(n, k1, lam1, k3, lam3, k2, lam2, Wmax=24):
    """The weighted orders W at which T(k2,lam2) occurs in gr^W D(T(k1,lam1)->T(k3,lam3)).

    Returns a list of (W, r, s, c, multiplicity).
    """
    G = weyl_group(n)
    out = []
    for W in range(Wmax + 1):
        for s in range(W // 2 + 1):
            r = W - 2 * s
            # the one-row piece S^{k2}D of S^{k1}D (x) S^{k3}D (x) S^r D
            tot = k1 + k3 + r
            if (tot - k2) % 2 or k2 > tot:
                continue
            c = (tot - k2) // 2
            # the L-power of that piece, written in the T(k,lam) normalization:
            #   piece = S^{k2}D (x) L^{lam3-lam1+k3-s-c} = S^{k2}D^* (x) L^{...-k2},
            #   since D = D^* (x) L^{-1} gives S^{k2}D = S^{k2}D^* (x) L^{-k2}
            lam = lam3 - lam1 + k3 - s - c - k2
            if lam != lam2:
                continue
            mlt = triple_one_row(n, k1, k3, r, k2, G)
            if mlt:
                out.append((W, r, s, c, mlt))
    return out


# ---------------------------------------------------------------------------
# The symplectic case: no L, so a pure multiplicity question.
# ---------------------------------------------------------------------------

def symplectic_layers(m, k1, k3, k2, Wmax=8):
    """The orders W at which S^{k2}W occurs in gr^W D(S^{k1} -> S^{k3}) over sp(2m)."""
    G = weyl_group(m)
    out = []
    for W in range(Wmax + 1):
        mlt = triple_one_row(m, k1, k3, W, k2, G)
        if mlt:
            out.append((W, mlt))
    return out


# ---------------------------------------------------------------------------
# Self-tests.
# ---------------------------------------------------------------------------

def test():
    ok = 0

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            print(f"FAIL: {name}")
            raise SystemExit(1)

    # dim V^{(a,0,..,0)} = dim S^a V for sp(2n): the symmetric powers are irreducible.
    for n in (1, 2, 3):
        G = weyl_group(n)
        chk(f"|W(C_{n})| = 2^n n!", len(G) == 2 ** n * _fact(n))
        for a in range(0, 5):
            w = wt_sym(n, a)
            chk(f"S^{a}V irreducible, n={n}",
                mult(w, tuple([a] + [0] * (n - 1)), n, G) == 1)
            chk(f"S^{a}V has no lower one-row piece, n={n}",
                all(mult(w, tuple([b] + [0] * (n - 1)), n, G) == 0
                    for b in range(0, a) if (a - b) % 2 == 0))
            chk(f"dim S^{a}V, n={n}", sum(w.values()) == dim_sym(n, a))

    # V (x) V = S^2 V + Lam^2_0 V + trivial: one-row multiplicities 1 (for m=2) and
    # 1 (for m=0), and nothing else.
    for n in (1, 2, 3):
        G = weyl_group(n)
        chk(f"V(x)V one-row m=2, n={n}", pair_one_row(n, 1, 1, 2, G) == 1)
        chk(f"V(x)V one-row m=0, n={n}", pair_one_row(n, 1, 1, 0, G) == 1)

    # S^2 V (x) S^2 V for n=1 is 3(x)3 = 5+3+1.
    G1 = weyl_group(1)
    chk("sl(2): S^2(x)S^2 -> S^4", pair_one_row(1, 2, 2, 4, G1) == 1)
    chk("sl(2): S^2(x)S^2 -> S^2", pair_one_row(1, 2, 2, 2, G1) == 1)
    chk("sl(2): S^2(x)S^2 -> S^0", pair_one_row(1, 2, 2, 0, G1) == 1)
    chk("sl(2): S^2(x)S^2 -> S^6 is zero", pair_one_row(1, 2, 2, 6, G1) == 0)

    # Clebsch-Gordan: for n=1 every pair product is multiplicity-free and exhausts
    # the dimension.
    for a in range(0, 5):
        for b in range(0, 5):
            tot = sum(pair_one_row(1, a, b, mm, G1) * (mm + 1)
                      for mm in range(0, a + b + 1))
            chk(f"sl(2) CG dimension a={a} b={b}", tot == (a + 1) * (b + 1))

    # The adjoint of sp(2m) is S^2 V, and its multiplicity in S^2V (x) S^2V is 1
    # (from the bracket; sp has no invariant symmetric cubic).
    for n in (1, 2, 3):
        chk(f"adjoint in g(x)g, n={n}",
            pair_one_row(n, 2, 2, 2, weyl_group(n)) == 1)

    print(f"layers.py: {ok} checks pass")


def _fact(n):
    r = 1
    for t in range(2, n + 1):
        r *= t
    return r


# ---------------------------------------------------------------------------
# The census.
# ---------------------------------------------------------------------------

def census():
    from fractions import Fraction as Q
    lam = Q(1, 7)          # a generic weight; the answer will be uniform in it

    print("CONTACT, n=1 (a contact 3-manifold), the operator kGz:")
    print("  kGz: T(2,lam) (x) T(2,-lam) --> F_3 = T(0,3), weighted order 2.")
    print("  M = D^{<=W}( T(2,-lam) --> F_3 ); where does T(2,lam) sit in gr^W M?")
    for W, r, s, c, mlt in contact_layers(1, 2, -lam, 0, Q(3), 2, lam, Wmax=30):
        print(f"      W={W:3d}   r={r} s={s} c={c}   multiplicity {mlt}")
    print()
    print("  the same with lam symbolic-free: a second generic weight, as a control")
    lam2 = Q(-5, 3)
    for W, r, s, c, mlt in contact_layers(1, 2, -lam2, 0, Q(3), 2, lam2, Wmax=30):
        print(f"      W={W:3d}   r={r} s={s} c={c}   multiplicity {mlt}")
    print()
    print("  and in dimension 5 (n=2), where kGz does not exist, the same layer:")
    for W, r, s, c, mlt in contact_layers(2, 2, -lam, 0, Q(3), 2, lam, Wmax=20):
        print(f"      W={W:3d}   r={r} s={s} c={c}   multiplicity {mlt}")
    print()

    print("SYMPLECTIC, the operator hGz: metrics (x) metrics --> metrics, order 2.")
    print("  M = D^{<=W}( metrics --> metrics ); where does S^2W sit in gr^W M?")
    for m in (1, 2, 3):
        rows = symplectic_layers(m, 2, 2, 2, Wmax=8)
        txt = ", ".join(f"W={W} (mult {mlt})" for W, mlt in rows)
        print(f"      m={m}:  {txt}")
    print()
    print("  the one-dimensional model [GMO] work in, for comparison: their sigma")
    print("  lands in F_{mu-lam-k}, and the layer condition is the weight equation")
    print("  mu-lam-k = -2/3 with (lam,mu)=(-2/3,5/3), i.e. k=3: their Section 5.4.1.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        test()
    elif "--census" in sys.argv:
        census()
    else:
        test()
        print()
        census()
