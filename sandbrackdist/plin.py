#!/usr/bin/env python3
"""
plin.py -- the p-th powers of the LINEAR generating functions in the deformed
Hamiltonian algebras h_omega(2k; N), omega of Skryabin's 1st type.

The phenomenon:
in h(2k;N) every "translation" D_{u_i} is p-nilpotent, whereas in the deform
(u_i)^{[p]} is one of the EXTRA generating functions of Lemma ssSpanH -- a
function outside O(F).  The text records it for 2k = 4 only (eq. (pu2)); this
script lifts it to a closed formula for every k and every shape of table (3.3).

Conventions are skrpb.py's.  The only new ingredient: a generating function is
allowed to leave O(2k;N), so it is represented in the divided power algebra
Obar = O(2k; N+1); the corresponding vector field must still preserve O(2k;N),
and that is checked on every monomial, not assumed.
"""
import itertools
import sys

from skrpb import DP, Ham, J_k_0, C_k, J_kr


class Ext:
    """The pair (h_omega(2k;N), Obar) -- small algebra + big function space."""

    def __init__(self, p, N, B, eps=1):
        self.p, self.N, self.B, self.eps = p, tuple(N), B, eps
        self.k = len(N) // 2
        self.H = Ham(p, N, B, eps, off=1)          # the true algebra
        self.A = self.H.A                           # O(2k;N)
        self.Ab = DP(p, [Ni + 1 for Ni in N])       # Obar = O(2k;N+1)
        k = self.k
        # xbar_i = u_i^{(p^{N_i}-1)} -- the SAME monomial, embedded in Obar
        self.xbar = [self.Ab.var(i, p ** N[i] - 1) for i in range(2 * k)]
        # S, G, Omega recomputed inside Obar (xbar_i^2 = 0 there too, by Kummer)
        self.S = [[self.Ab.scal(self.H.B[i][j],
                                self.Ab.mul(self.xbar[i], self.xbar[j + k]))
                   for j in range(k)] for i in range(k)]
        self.G = self._inv()
        self.Omega = self._omega()

    def _inv(self):
        Ab, k, p = self.Ab, self.k, self.p
        G = [[Ab.one() if i == j else Ab.zero() for j in range(k)] for i in range(k)]
        term = [[Ab.one() if i == j else Ab.zero() for j in range(k)] for i in range(k)]
        sign = 1
        for _ in range(2 * k + 2):
            new = [[Ab.zero() for _ in range(k)] for _ in range(k)]
            nz = False
            for i in range(k):
                for j in range(k):
                    acc = Ab.zero()
                    for t in range(k):
                        if term[i][t] and self.S[t][j]:
                            acc = Ab.add(acc, Ab.mul(term[i][t], self.S[t][j]))
                    new[i][j] = acc
                    if acc:
                        nz = True
            if not nz:
                break
            term = new
            sign = (-sign * self.eps) % p
            for i in range(k):
                for j in range(k):
                    if term[i][j]:
                        G[i][j] = Ab.add(G[i][j], Ab.scal(sign, term[i][j]))
        return G

    def _omega(self):
        Ab, k = self.Ab, self.k
        n = 2 * k
        Om = [[Ab.zero() for _ in range(n)] for _ in range(n)]
        for i in range(k):
            Om[i][i + k] = Ab.one()
            Om[i + k][i] = Ab.scal(-1, Ab.one())
        for i in range(k):
            for j in range(k):
                if self.H.B[i][j]:
                    s = Ab.scal(self.eps * self.H.B[i][j],
                                Ab.mul(self.xbar[i], self.xbar[j + k]))
                    Om[i][j + k] = Ab.add(Om[i][j + k], s)
                    Om[j + k][i] = Ab.sub(Om[j + k][i], s)
        return Om

    # ---- the vector field of a generating function a in Obar, applied in Obar
    def Dbar(self, a, g):
        Ab, k = self.Ab, self.k
        r = Ab.zero()
        for i in range(k):
            dia = Ab.diff(i + k, a)
            for j in range(k):
                Gij = self.G[i][j]
                if not Gij:
                    continue
                t = Ab.sub(Ab.mul(dia, Ab.diff(j, g)),
                           Ab.mul(Ab.diff(j, a), Ab.diff(i + k, g)))
                if t:
                    r = Ab.add(r, Ab.mul(Gij, t))
        return r

    def in_small(self, f):
        return all(all(e[i] < self.A.top[i] for i in range(2 * self.k)) for e in f)

    # ---- (ad_f)^p as a derivation of the SMALL algebra, then its generator
    def p_power_ext(self, f):
        """Return (a, ok): a in Obar with Dbar_a = (ad_f)^p on O(2k;N)."""
        A, Ab, n = self.A, self.Ab, 2 * self.k
        c = [self.H.ad_pow(f, A.var(i), self.p) for i in range(n)]   # in O(2k;N)
        dh = []
        for j in range(n):
            acc = Ab.zero()
            for i in range(n):
                if c[i] and self.Omega[i][j]:
                    acc = Ab.add(acc, Ab.mul(c[i], self.Omega[i][j]))
            dh.append(acc)
        a, rem = Ab.zero(), list(dh)
        for j in range(n):
            part, _ = Ab.integrate(j, rem[j])
            if not part:
                continue
            a = Ab.add(a, part)
            for s in range(n):
                rem[s] = Ab.sub(rem[s], Ab.diff(s, part))
        ok = all(not Ab.sub(Ab.diff(j, a), dh[j]) for j in range(n))
        return a, ok

    def verify(self, f, a):
        """Check Dbar_a = (ad_f)^p on EVERY monomial of O(2k;N), and that the
        result stays inside O(2k;N)."""
        A = self.A
        allm = list(itertools.product(*[range(t) for t in A.top]))
        for e in allm:
            g = A.mono(e)
            lhs = self.Dbar(a, g)
            rhs = self.H.ad_pow(f, g, self.p)
            if not self.in_small(lhs):
                return False, "leaves O(2k;N) on %s" % (e,)
            if self.Ab.sub(lhs, rhs):
                return False, "differs on %s" % (e,)
        return True, "ok"


def tex(Ab, f, N, p):
    """Print with p^{N_i} exponents shown symbolically where they are top."""
    if not f:
        return "0"
    parts = []
    for e in sorted(f, reverse=True):
        c = f[e]
        s = ""
        for i, a in enumerate(e):
            if a == 0:
                continue
            top = p ** N[i]
            if a == top:
                s += "u_{%d}^{(p^{N_%d})}" % (i + 1, i + 1)
            elif a == top - p:
                s += "u_{%d}^{(p^{N_%d}-p)}" % (i + 1, i + 1)
            elif a == top - 1:
                s += "u_{%d}^{(p^{N_%d}-1)}" % (i + 1, i + 1)
            elif a == 1:
                s += "u_{%d}" % (i + 1)
            else:
                s += "u_{%d}^{(%d)}" % (i + 1, a)
        cc = c if c <= p // 2 else c - p
        parts.append(("%d" % cc if s == "" else
                      (s if cc == 1 else ("-" + s if cc == -1 else "%d%s" % (cc, s)))))
    out = parts[0]
    for t in parts[1:]:
        out += (" " + t) if t.startswith("-") else (" + " + t)
    return out


def run(name, p, N, B, quiet=False):
    E = Ext(p, N, B)
    k = E.k
    out = []
    for i in range(2 * k):
        f = E.A.var(i)
        a, ok = E.p_power_ext(f)
        good, msg = E.verify(f, a)
        out.append((i, a, ok and good, msg))
        if not quiet:
            print("    (u_%d)^[%d] = %-58s  %s"
                  % (i + 1, p, tex(E.Ab, a, N, p), "verified" if (ok and good) else msg))
    return E, out


# ---------------------------------------------------------------- the census

def dp_deriv(A, i, s, g):
    """The divided power derivation d_i^{(p^s)}: u_i^{(a)} |-> u_i^{(a-p^s)}."""
    r = {}
    q = A.p ** s
    for e, c in g.items():
        if e[i] < q:
            continue
        e2 = list(e)
        e2[i] -= q
        r[tuple(e2)] = c
    return r


def linear_pmap_one(p, N, B):
    """un = One: return the matrix M with (u_j)^{[p]} = sum_i M[i][j] u_i^{(p)}."""
    E = Ext(p, N, B)
    n = 2 * E.k
    M = [[0] * n for _ in range(n)]
    for j in range(n):
        a, ok = E.p_power_ext(E.A.var(j))
        good, msg = E.verify(E.A.var(j), a)
        assert ok and good, (j, msg)
        for e, c in a.items():
            src = [i for i in range(n) if e[i]]
            assert len(src) == 1 and e[src[0]] == p ** N[src[0]], (j, e)
            M[src[0]][j] = c
    return M


def check_formula_one(p, k, B, name):
    """(u_j)^{[p]} = -sum_i alpha_ij u_i^{(p)},  (u_~i)^{[p]} = -sum_j alpha_ij u_~j^{(p)}."""
    N = [1] * (2 * k)
    M = linear_pmap_one(p, N, B)
    pred = [[0] * (2 * k) for _ in range(2 * k)]
    for i in range(k):
        for j in range(k):
            if B[i][j]:
                pred[i][j] = (-B[i][j]) % p          # (u_j)^[p] gets -alpha_ij u_i^{(p)}
                pred[j + k][i + k] = (-B[i][j]) % p  # (u_~i)^[p] gets -alpha_ij u_~j^{(p)}
    ok = all(M[i][j] % p == pred[i][j] % p for i in range(2 * k) for j in range(2 * k))
    print("    %-26s p=%d k=%d : %s" % (name, p, k, "MATCHES" if ok else "FAILS"))
    if not ok:
        print("       got ", M)
        print("       pred", pred)
    return ok


def check_notrestricted(p, N, B, name):
    """un != One: is (ad_{u_i})^p a special derivation (in W(2k;un))?  And if
    not, is it  sum_j c_j d_j  -  eps alpha_ij d_{~j}^{(p)}  for the arrows?"""
    E = Ext(p, N, B)
    A, H, k = E.A, E.H, E.k
    n = 2 * k
    allm = list(itertools.product(*[range(t) for t in A.top]))
    res = []
    for i in range(n):
        f = A.var(i)
        c = [H.ad_pow(f, A.var(j), p) for j in range(n)]
        special, matched = True, None
        for e in allm:
            g = A.mono(e)
            lhs = H.ad_pow(f, g, p)
            rhs = A.zero()
            for j in range(n):
                if c[j]:
                    rhs = A.add(rhs, A.mul(c[j], A.diff(j, g)))
            if A.sub(lhs, rhs):
                special = False
                break
        if not special:
            # try  sum c_j d_j + sum_t lam_t d_t^{(p)}   with constants lam_t
            lam = [0] * n
            for t in range(n):
                if N[t] < 2:
                    continue
                probe = A.mono(tuple(p if s == t else 0 for s in range(n)))
                val = H.ad_pow(f, probe, p)
                lam[t] = A.const_term(val)
            matched = True
            for e in allm:
                g = A.mono(e)
                lhs = H.ad_pow(f, g, p)
                rhs = A.zero()
                for j in range(n):
                    if c[j]:
                        rhs = A.add(rhs, A.mul(c[j], A.diff(j, g)))
                for t in range(n):
                    if lam[t]:
                        rhs = A.add(rhs, A.scal(lam[t], dp_deriv(A, t, 1, g)))
                if A.sub(lhs, rhs):
                    matched = False
                    break
            res.append((i, special, matched, lam))
        else:
            res.append((i, special, None, None))
    print("    %-24s p=%d N=%s" % (name, p, list(N)))
    for i, special, matched, lam in res:
        if special:
            print("      (ad_{u_%d})^%d : in W(2k;un)" % (i + 1, p))
        else:
            print("      (ad_{u_%d})^%d : NOT in W(2k;un); = vect + sum lam_t d_t^{(p)}, "
                  "lam=%s : %s" % (i + 1, p, lam, "verified" if matched else "NO"))
    return res


if __name__ == "__main__":
    print("=" * 78)
    print("A.  un = One : the p-map on the linear generating functions")
    print("    claim  (u_j)^[p] = -sum_i alpha_ij u_i^{(p)},")
    print("           (u_~i)^[p] = -sum_j alpha_ij u_~j^{(p)}        (i,j <= k)")
    print("    each case verified on ALL p^{2k} monomials of O(2k;One).")
    print("=" * 78)
    nA = 0
    okA = True
    for p in (3, 5, 7):
        for k in (1, 2, 3):
            if p == 7 and k == 3:
                continue
            shapes = [("J_%d(0)" % k, J_k_0(k)), ("C_%d" % k, C_k(k))]
            for lam in (0, 1, 2):
                shapes.append(("J_{%d,1}(%d)" % (k, lam), J_kr(k, 1, lam)))
                if k % 2 == 0:
                    shapes.append(("J_{%d,2}(%d)" % (k, lam), J_kr(k, 2, lam)))
            for nm, B in shapes:
                okA &= check_formula_one(p, k, B, nm)
                nA += 1
    print("  --> %d cases, all match: %s" % (nA, okA))

    print()
    print("=" * 78)
    print("B.  un != One : (ad_{u_i})^p is not even a vector field")
    print("=" * 78)
    for p, N, B, nm in [(3, [2, 1, 1, 2], [[0, 0], [0, 0]], "h(4;un) UNDEFORMED"),
                        (3, [2, 1, 1, 2], J_k_0(2), "om_{1,0}(4;un)"),
                        (5, [2, 1, 1, 2], J_k_0(2), "om_{1,0}(4;un)"),
                        (3, [2, 1, 1, 2], C_k(2), "om_{1,C}(4;un)"),
                        (3, [1, 2, 2, 1], J_k_0(2), "om_{1,0}(4;un')"),
                        (3, [2, 2], J_kr(1, 1, 2), "om_{1,1,2}(2;un)"),
                        (5, [2, 2], J_kr(1, 1, 2), "om_{1,1,2}(2;un)")]:
        check_notrestricted(p, N, B, nm)
    print("  --> the d^{(p)}-part is the SAME in the deform and in h(2k;un):")
    print("      it is the non-restrictedness of W(2k;un), not of the deform.")

    print()
    print("=" * 78)
    print("C.  the contrast at un = One : translations in h(2k;One) are p-nilpotent")
    print("=" * 78)
    import itertools as _it
    for p, N in [(3, [1] * 4), (5, [1] * 4), (3, [1] * 6)]:
        H0 = Ham(p, N, None, 1, off=1)
        A0 = H0.A
        nz = []
        for i in range(len(N)):
            f = A0.var(i)
            if any(H0.ad_pow(f, A0.mono(e), p)
                   for e in _it.product(*[range(t) for t in A0.top])):
                nz.append(i + 1)
        print("    h(%d;One) p=%d : translations with (ad_{u_i})^p != 0 : %s"
              % (len(N), p, nz if nz else "none"))
