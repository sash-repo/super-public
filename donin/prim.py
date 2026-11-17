"""S^k(gl(n)) by hand: monomials in the e_{ij}, the adjoint action, and Donin's
vector (3).  Used to settle Proposition 5.1 in a single explicit case."""

from itertools import product
from fractions import Fraction


def mono(pairs):
    """a monomial of S(gl(n)): the sorted multiset of its factors e_{ij}."""
    return tuple(sorted(pairs))


def mul(x, y):
    """product of two elements (dicts monomial -> coefficient)."""
    out = {}
    for a, ca in x.items():
        for b, cb in y.items():
            k = mono(a + b)
            out[k] = out.get(k, 0) + ca * cb
    return {k: v for k, v in out.items() if v}


def e(i, j):
    return {((i, j),): 1}


def add(*xs):
    out = {}
    for x in xs:
        for k, v in x.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v}


def smul(c, x):
    return {k: c * v for k, v in x.items() if c * v}


def delta(m, i, j, n):
    """delta_m e_{ij} = sum e_{i nu_1} e_{nu_1 nu_2} ... e_{nu_{m-1} j}."""
    if m == 1:
        return e(i, j)
    out = {}
    for nus in product(range(1, n + 1), repeat=m - 1):
        idx = (i,) + nus + (j,)
        term = mono(tuple((idx[t], idx[t + 1]) for t in range(m)))
        out[term] = out.get(term, 0) + 1
    return {k: v for k, v in out.items() if v}


def act(a, b, x):
    """the adjoint action of e_{ab} as a derivation of S(gl(n))."""
    out = {}
    for m, c in x.items():
        for t, (i, j) in enumerate(m):
            # [e_ab, e_ij] = delta_{bi} e_aj - delta_{ja} e_ib
            for coeff, new in ((1 if b == i else 0, (a, j)), (-1 if j == a else 0, (i, b))):
                if not coeff:
                    continue
                mm = mono(m[:t] + m[t + 1:] + (new,))
                out[mm] = out.get(mm, 0) + coeff * c
    return {k: v for k, v in out.items() if v}


def is_primitive(x, n):
    return all(not act(a, b, x) for a in range(1, n + 1) for b in range(a + 1, n + 1))


def weight(m, n):
    w = [0] * n
    for (i, j) in m:
        w[i - 1] += 1
        w[j - 1] -= 1
    return tuple(w)


def trace(n):
    return add(*[e(i, i) for i in range(1, n + 1)])


def donin_vector(kappa, irows, jrows, n):
    """Donin's (3): sum over the column stabilizers pi of A and tau of B of
    (-1)^{pi+tau} prod_m delta_{kappa_m} e_{i_{pi(m)} j_{tau(m)}}.

    Here `irows[m]` is the row of the m-th cell in A, `jrows[m]` the row of its
    image in B (already translated into the index that labels it), and A, B are
    assumed to be single columns, so both stabilizers are the full symmetric
    groups on the p cells."""
    from itertools import permutations
    p = len(kappa)

    def sgn(perm):
        s = 1
        for a in range(p):
            for b in range(a + 1, p):
                if perm[a] > perm[b]:
                    s = -s
        return s

    total = {}
    for pi in permutations(range(p)):
        for tau in permutations(range(p)):
            term = {(): 1}
            for m in range(p):
                term = mul(term, delta(kappa[m], irows[pi[m]], jrows[tau[m]], n))
            total = add(total, smul(sgn(pi) * sgn(tau), term))
    return total
