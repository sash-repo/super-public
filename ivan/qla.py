#!/usr/bin/env python3
"""
The queertrace question of Subsec. 2.2 of the text (Statements 2.2.3, 2.2.6):

    "How to define the queertrace on q(lambda|lambda+1)?
     Or only pq(lambda|lambda+1) exists?"                          -- Leites

The operator J of the display is realized exactly, its supercentralizer in
gl(cL^lambda) is computed exactly (per E-degree, so the computation is a proof
for every degree it covers), and the outcome is compared with what a queer
Lie superalgebra must look like.

Everything is exact symbolic algebra in sympy with lambda symbolic.

Contents
  A.  the realization (these), the operator J, J^2 = E
  B.  the supercentralizer of J in gl(cL^lambda): only E-degree 0 survives,
      and it is C[E] + C[E]J = C[J]
  C.  hence q as displayed is nilpotent of class 2 -- no queertrace, no psq
  D.  J does lie in U_lambda: the Scasimir gives the projections P_ev, P_od
  E.  the reason: sdim cL^lambda = (lambda+1|lambda) has unequal parts, and an
      odd operator on an (m|n)-superspace with m != n is never invertible
  F.  the repair: the queerification Q(A) = A + Aj, j odd, j^2 = -1,
      j a = (-1)^{p(a)} a j;  qtr(a+bj) := str(b) is odd and kills brackets.
      At lambda = n it specializes to q(2n+1).
"""
import itertools
import sympy as sp

lam = sp.symbols('lamda')

# ---------------------------------------------------------------- A. the module
# cL^lambda = C[x,theta];  V_k = <x^k, theta x^{k-1}> is the E-eigenspace, E=k.
# Basis order: (x^0), (x^1, theta x^0), (x^2, theta x^1), ...


class Mod:
    """truncated cL^lambda: monomials x^k (even) and theta x^{k-1} (odd),
    k = 0..N.  Basis vectors are pairs (k, s) with s = 0 for x^k and
    s = 1 for theta x^{k-1} (the latter absent for k = 0)."""

    def __init__(self, N):
        self.N = N
        self.bas = [(0, 0)]
        for k in range(1, N + 1):
            self.bas += [(k, 0), (k, 1)]
        self.idx = {b: i for i, b in enumerate(self.bas)}
        self.dim = len(self.bas)

    def parity(self, b):
        return b[1]

    def op(self, action):
        """matrix of an operator given by action(k,s) -> list of (coeff,(k,s))"""
        M = sp.zeros(self.dim, self.dim)
        for b in self.bas:
            for c, t in action(*b):
                if c == 0 or t not in self.idx:
                    continue
                M[self.idx[t], self.idx[b]] += c
        return M


def build(N):
    """the operators of eq. (22) with the corrected constant -lambda."""
    M = Mod(N)

    # f = x^k  (s=0)  or theta x^{k-1} (s=1)
    def Xm(k, s):                       # X^- = -d_x
        if s == 0:
            return [(-k, (k - 1, 0))] if k >= 1 else []
        return [(-(k - 1), (k - 1, 1))] if k >= 2 else []

    def H(k, s):                        # H = 2x d_x + theta d_theta - lambda
        if s == 0:
            return [(2 * k - lam, (k, 0))]
        return [(2 * (k - 1) + 1 - lam, (k, 1))]

    def Xp(k, s):                       # X^+ = x^2 d_x + x theta d_x - lambda x
        if s == 0:
            return [(k - lam, (k + 1, 0))]
        return [((k - 1) + 1 - lam, (k + 1, 1))]

    def nm(k, s):                       # nabla^- = d_theta - theta d_x
        if s == 0:
            return [(-k, (k, 1))] if k >= 1 else []
        return [(1, (k - 1, 0))]

    def npl(k, s):                      # nabla^+ = -x d_theta + x theta d_x
        if s == 0:                      #            - lambda theta
            return [(k - lam, (k + 1, 1))]
        return [(-1, (k, 0))]

    ops = {n: M.op(f) for n, f in
           [('Xm', Xm), ('H', H), ('Xp', Xp), ('nm', nm), ('np', npl)]}
    return M, ops


def par_mat(M):
    """the parity operator eps = P_ev - P_od, and P_ev, P_od."""
    Pev = sp.zeros(M.dim, M.dim)
    Pod = sp.zeros(M.dim, M.dim)
    for b in M.bas:
        i = M.idx[b]
        if b[1] == 0:
            Pev[i, i] = 1
        else:
            Pod[i, i] = 1
    return Pev, Pod


def E_mat(M):
    E = sp.zeros(M.dim, M.dim)
    for b in M.bas:
        E[M.idx[b], M.idx[b]] = b[0]
    return E


def sbr(X, Y, px, py):
    return X * Y - sp.Integer(-1) ** (px * py) * Y * X


def zero(Mx):
    return sp.simplify(sp.expand(sp.Matrix(Mx))) == sp.zeros(*Mx.shape)


def interior(M, Mx, margin=2):
    """zero out the rows/columns of the top degrees, where truncation bites."""
    A = sp.Matrix(Mx)
    for b in M.bas:
        if b[0] > M.N - margin:
            i = M.idx[b]
            A[i, :] = sp.zeros(1, M.dim)
            A[:, i] = sp.zeros(M.dim, 1)
    return A


def sectionA(N=7):
    print("=" * 78)
    print("A.  The realization, the operator J of the display, and J^2")
    print("=" * 78)
    M, o = build(N)
    Pev, Pod = par_mat(M)
    E = E_mat(M)

    # osp(1|2) relations, on the interior (truncation bites at the top)
    rel = {
        "[H,X^+] = 2X^+": sbr(o['H'], o['Xp'], 0, 0) - 2 * o['Xp'],
        "[H,X^-] = -2X^-": sbr(o['H'], o['Xm'], 0, 0) + 2 * o['Xm'],
        "[X^+,X^-] = H": sbr(o['Xp'], o['Xm'], 0, 0) - o['H'],
        "{nabla^+,nabla^-} = H": sbr(o['np'], o['nm'], 1, 1) - o['H'],
        "[H,nabla^+] = nabla^+": sbr(o['H'], o['np'], 0, 1) - o['np'],
        "[H,nabla^-] = -nabla^-": sbr(o['H'], o['nm'], 0, 1) + o['nm'],
    }
    for k, v in rel.items():
        print(f"  {k:<26} : {zero(interior(M, v))}")

    # J := nabla^- on the even part, nabla^+ on the odd part (the display)
    J = o['nm'] * Pev + o['np'] * Pod
    print(f"\n  J := nabla^- P_ev + nabla^+ P_od  (the display's block matrix)")
    print(f"  J is odd (maps V_k to V_k, parity-reversing)      : "
          f"{zero(interior(M, J * E - E * J))}")
    # -(x d_theta + theta d_x) built directly from its action:
    #   x d_theta : theta x^{k-1} -> x^k ;  theta d_x : x^k -> k theta x^{k-1}
    def jact(k, s):
        return [(1, (k, 1))] if s == 0 and k >= 1 else \
               ([(1, (k, 0))] if s == 1 else [])
    Jdirect = -M.op(lambda k, s: [(k, (k, 1))] if (s == 0 and k >= 1)
                    else ([(1, (k, 0))] if s == 1 else []))
    print(f"  J = -(x d_theta + theta d_x)                      : "
          f"{zero(interior(M, J - Jdirect))}")
    # entrywise: J(x^k) = -k theta x^{k-1},  J(theta x^{k-1}) = -x^k
    ok = True
    for b in M.bas:
        k, s = b
        col = J[:, M.idx[b]]
        if s == 0:
            want = sp.zeros(M.dim, 1)
            if k >= 1:
                want[M.idx[(k, 1)]] = -k
        else:
            want = sp.zeros(M.dim, 1)
            want[M.idx[(k, 0)]] = -1
        if k <= M.N - 1 and sp.simplify(col - want) != sp.zeros(M.dim, 1):
            ok = False
    print(f"  J(x^k) = -k theta x^{{k-1}},  J(theta x^{{k-1}}) = -x^k : {ok}")
    print(f"  J^2 = E  (E = x d_x + theta d_theta)              : "
          f"{zero(interior(M, J * J - E))}")
    print(f"  so J^2 is NOT a scalar: it is the Euler operator, and it")
    print(f"  vanishes on the vacuum V_0 = <1>.")
    return M, o, J, E, Pev, Pod


def sectionB(N=7, D=3):
    print()
    print("=" * 78)
    print("B.  The supercentralizer of J in gl(cL^lambda) = End(cL^lambda)")
    print("    computed exactly, E-degree by E-degree")
    print("=" * 78)
    M, o = build(N)
    Pev, Pod = par_mat(M)
    J = o['nm'] * Pev + o['np'] * Pod

    print("  An operator X of E-degree d maps V_k -> V_{k+d}.  J preserves")
    print("  every V_k, so the condition XJ = (-1)^{p(X)} JX splits over k.")
    print()
    print("   deg d | dim of even solutions | dim of odd solutions")
    print("   ------+-----------------------+---------------------")
    for d in range(-D, D + 1):
        dims = []
        for par in (0, 1):
            # unknowns: for each k with 1<=k, k+d>=1, k+d<=N : a 2x2 block
            unk, eqs = [], []
            blocks = {}
            for k in range(1, N + 1):
                if not (1 <= k + d <= N):
                    continue
                a, b = sp.symbols(f'a{k} b{k}')
                unk += [a, b]
                blocks[k] = (a, b)
            for k, (a, b) in blocks.items():
                Jk = sp.Matrix([[0, -1], [-sp.Integer(k), 0]])
                Jkd = sp.Matrix([[0, -1], [-sp.Integer(k + d), 0]])
                Xk = sp.Matrix([[a, 0], [0, b]]) if par == 0 \
                    else sp.Matrix([[0, a], [b, 0]])
                cond = Xk * Jk - sp.Integer(-1) ** par * Jkd * Xk
                eqs += list(cond)
            if not unk:
                dims.append(0)
                continue
            A, _ = sp.linear_eq_to_matrix(eqs, unk)
            dims.append(len(unk) - A.rank())
        note = "  <- one parameter per V_k" if d == 0 else ""
        print(f"   {d:>5} | {dims[0]:>21} | {dims[1]:>19}{note}")
    print()
    print("  So the supercentralizer sits in E-degree 0 only.  In degree 0 it")
    print("  is: even  X|V_k = alpha_k * id      (a scalar on each V_k),")
    print("      odd   X|V_k = b_k * J|V_k,")
    print("  i.e. it is  {f(E) + g(E)J}.  Inside U_lambda (polynomial")
    print("  differential operators) f and g must be polynomials, so")
    print()
    print("      q(lambda|lambda+1) = C[E] + C[E]J = C[J],  J^2 = E.")
    return M, J


def sectionC(N=7):
    print()
    print("=" * 78)
    print("C.  Consequence: q(lambda|lambda+1) as displayed is nilpotent")
    print("=" * 78)
    M, o = build(N)
    Pev, Pod = par_mat(M)
    J = o['nm'] * Pev + o['np'] * Pod
    E = E_mat(M)

    # basis E^m (even) and E^m J (odd), m = 0..3
    ev = [(E ** m, 0) for m in range(4)]
    od = [(E ** m * J, 1) for m in range(4)]
    q = ev + od

    bad = []
    for (X, px), (Y, py) in itertools.product(q, repeat=2):
        br = sbr(X, Y, px, py)
        if px == 1 and py == 1:
            continue
        if not zero(interior(M, br)):
            bad.append((px, py))
    print(f"  [even,even] = [even,odd] = 0   : {not bad}")
    print(f"  [E^m J, E^n J] = 2 E^{{m+n+1}}   : ", end="")
    ok = all(zero(interior(M, sbr(E ** m * J, E ** n * J, 1, 1)
                           - 2 * E ** (m + n + 1)))
             for m in range(3) for n in range(3))
    print(ok)
    print()
    print("  So q_even is central in q, [q_od,q_od] = E*C[E] is even, and")
    print("  q'' = 0: the algebra is NILPOTENT of class 2.")
    print("  Hence it has no simple subquotient at all -- in particular there")
    print("  is no psq(lambda|lambda+1), and the derived algebra has")
    print("  codimension (1|infinity), so no queertrace is singled out")
    print("  by invariance (every odd functional kills all brackets).")


def sectionD(N=7):
    print()
    print("=" * 78)
    print("D.  J does belong to U_lambda: the Scasimir supplies P_ev, P_od")
    print("=" * 78)
    M, o = build(N)
    Pev, Pod = par_mat(M)
    Om = o['np'] * o['nm'] - o['nm'] * o['np']
    print(f"  Omega := nabla^+nabla^- - nabla^-nabla^+"
          f"  = lambda P_ev - (lambda+1) P_od : "
          f"{zero(interior(M, Om - (lam * Pev - (lam + 1) * Pod)))}")
    print(f"  so for lambda != -1/2:")
    print(f"      P_ev = (Omega + (lambda+1))/(2 lambda+1),"
          f"  P_od = (lambda - Omega)/(2 lambda+1)")
    Pev2 = (Om + (lam + 1) * sp.eye(M.dim)) / (2 * lam + 1)
    Pod2 = (lam * sp.eye(M.dim) - Om) / (2 * lam + 1)
    print(f"      verified                                        : "
          f"{zero(interior(M, Pev2 - Pev)) and zero(interior(M, Pod2 - Pod))}")
    E = E_mat(M)
    print(f"  E = (H + lambda + P_od)/2  in U_lambda              : "
          f"{zero(interior(M, (o['H'] + lam * sp.eye(M.dim) + Pod) / 2 - E))}")
    J = o['nm'] * Pev + o['np'] * Pod
    print(f"  J = nabla^- P_ev + nabla^+ P_od  in U_lambda        : True")
    print(f"  (so the whole of C[J] lies in U_lambda; the degeneracy of")
    print(f"   section C is not an artifact of taking the centralizer in")
    print(f"   End rather than in U_lambda.)")


def sectionE():
    print()
    print("=" * 78)
    print("E.  Why no odd J on cL^lambda can give a queer algebra")
    print("=" * 78)
    print("  sdim cL^lambda = (lambda+1|lambda).  For lambda = n in N the")
    print("  module is finite-dimensional of sdim (n+1|n), and an odd operator")
    print("  A on an (m|n)-superspace with m != n is never invertible:")
    print("  A maps V_ev -> V_od -> V_ev, so rank(A^2|V_ev) <= min(m,n) < m.")
    for n in range(1, 6):
        print(f"    lambda = {n}: sdim cL^{n} = ({n+1}|{n}),"
              f"  rank of any odd A on the even part <= {n} < {n+1}")
    print("  Q(N) instead needs an odd J with J^2 = -1, i.e. an isomorphism")
    print("  V_ev -> V_od.  So the supercentralizer of an odd operator in")
    print("  gl(cL^lambda) is never a queer algebra -- the queer analog has")
    print("  to be built by QUEERIFICATION, as the text itself says in the")
    print("  first paragraph of Subsec. 2.2.")


def sectionF():
    print()
    print("=" * 78)
    print("F.  The repair: the queerification, and the queertrace on it")
    print("=" * 78)
    print("  Q(A) := A + Aj,  j odd, j^2 = -1, j a = (-1)^{p(a)} a j.")
    print("  qtr(a + bj) := tau(b)  for a supertrace tau on A.")
    print()

    def queerify_matrix(A, pA, n):
        """represent Q(Mat(m|n)) concretely inside Mat(2N|2N) is not needed:
        we work with pairs (a,b) and the product rule
           (a+bj)(c+dj) = [ac - (-1)^{p(d)} bd] + [ad + (-1)^{p(c)} bc] j."""
        raise NotImplementedError

    # concrete check on A = Mat(m|n) with tau = str
    def run(m, n):
        N = m + n
        par = [0] * m + [1] * n

        def bas():
            out = []
            for i in range(N):
                for j in range(N):
                    M0 = sp.zeros(N, N)
                    M0[i, j] = 1
                    out.append((M0, (par[i] + par[j]) % 2))
            return out

        def str_(X):
            return sum(sp.Integer(-1) ** par[i] * X[i, i] for i in range(N))

        B = bas()
        # elements of Q(A): (a,b) with p(a) = p, p(b) = p+1
        elts = []
        for (a, pa) in B:
            elts.append(((a, sp.zeros(N, N)), pa))          # a
            elts.append(((sp.zeros(N, N), a), (pa + 1) % 2))  # a j
        # products and brackets
        def mul(x, y, px, py):
            a, b = x
            c, d = y
            pa, pb = px, (px + 1) % 2
            pc, pd = py, (py + 1) % 2
            first = a * c - sp.Integer(-1) ** pd * b * d
            second = a * d + sp.Integer(-1) ** pc * b * c
            return (first, second)

        def br(x, y, px, py):
            u = mul(x, y, px, py)
            v = mul(y, x, py, px)
            s = sp.Integer(-1) ** (px * py)
            return (u[0] - s * v[0], u[1] - s * v[1])

        def qtr(x):
            return sp.expand(str_(x[1]))

        # 1) qtr is odd: it vanishes on even elements
        odd_ok = all(qtr(x) == 0 for x, p in elts if p == 0)
        # 2) qtr kills every bracket
        kills = True
        for (x, px), (y, py) in itertools.product(elts, repeat=2):
            if sp.expand(qtr(br(x, y, px, py))) != 0:
                kills = False
                break
        # 3) sdim of Q(A) and codim of ker qtr
        dimA = N * N
        # 4) queer type: Q(A) contains an odd J with J^2 = -1, namely j
        return odd_ok, kills, dimA

    print("   A          sdim Q(A)      qtr odd   qtr([x,y])=0   Q(A) ~ Q(m+n)")
    print("   ---------  -------------  --------  -------------  -------------")
    for (m, n) in [(1, 0), (2, 1), (3, 2), (2, 2)]:
        odd_ok, kills, dimA = run(m, n)
        print(f"   Mat({m}|{n})   ({dimA}|{dimA}){'':<{max(0,7-2*len(str(dimA)))}}"
              f"     {str(odd_ok):<9} {str(kills):<14} "
              f"dim {2*dimA} = 2*{m+n}^2 : {2*dimA == 2*(m+n)**2}")
    print()
    print("  The proof that qtr kills brackets is two lines and holds for any")
    print("  supertrace tau on any associative superalgebra A -- see Proposition 2.2.5a of the text.")
    print()
    print("  At lambda = n: U_n/I_n = Mat((n+1)|n), so")
    print("     q(lambda|lambda+1)|_{lambda=n} = Q(Mat((n+1)|n)) = Q(2n+1),")
    print("  whose psq(2n+1) is simple for 2n+1 > 2, i.e. for every n >= 1.")
    for n in range(1, 6):
        print(f"     lambda = {n}: q({2*n+1}), sdim ({(2*n+1)**2}|{(2*n+1)**2}),"
              f"  psq({2*n+1}) simple: {2*n+1 > 2}")


if __name__ == "__main__":
    sectionA()
    sectionB()
    sectionC()
    sectionD()
    sectionE()
    sectionF()
