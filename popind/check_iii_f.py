# Item (iii), part F: symmorphy test, parametrised.
# W = < (R1,0), (R2,0), (-R1,u) >, u in H_{R1};  A = [[2,-a],[-b,2]], ab > 4.
from sympy import Matrix, eye, Rational, lcm, symbols, simplify, nsimplify
from sympy.matrices.normalforms import hermite_normal_form


def run(a, b, t, depth=11):
    R1 = Matrix([[-1, a], [0, 1]])
    R2 = Matrix([[1, 0], [b, -1]])
    G = Matrix([[2 * b, -a * b], [-a * b, 2 * a]])
    assert R1.T * G * R1 == G and R2.T * G * R2 == G, "not isometries"
    h = (R1 - eye(2)).nullspace()[0]
    h = h * lcm([x.q for x in h])                 # primitive generator of H_{R1}
    u = t * h
    gens = [(R1, Matrix([0, 0])), (R2, Matrix([0, 0])), (-R1, u)]

    mul = lambda A, B: (A[0] * B[0], A[0] * B[1] + A[1])
    key = lambda g: (tuple(g[0]), tuple(g[1]))
    seen = {key(gens[0]): 1}
    frontier, trans = [(eye(2), Matrix([0, 0]))], set()
    for _ in range(depth):
        new = []
        for g in frontier:
            for hgen in gens:
                x = mul(g, hgen)
                k = key(x)
                if k in seen:
                    continue
                seen[k] = 1
                new.append(x)
                if x[0] == eye(2):
                    trans.add(tuple(x[1]))
        frontier = new
    if not trans:
        return None
    den = lcm([Rational(c).q for tt in trans for c in tt])
    M = Matrix([[int(c * den) for c in tt] for tt in trans]).T
    Lb = hermite_normal_form(M) / den             # columns = basis of Tran W
    # symmorphy: exists w with u_i + (1 - P_i) w in Tran W for all generators
    w1, w2 = symbols('w1 w2')
    w = Matrix([w1, w2])
    conds = [simplify(Lb.inv() * (uu + (eye(2) - P) * w)) for P, uu in gens]
    return h, u, Lb, conds


for (a, b, t) in [(3, 3, 1), (3, 2, Rational(1, 2)), (3, 2, 1), (4, 2, Rational(1, 2))]:
    out = run(a, b, t)
    if out is None:
        print(f"a={a} b={b} t={t}: no translations found"); continue
    h, u, Lb, conds = out
    print(f"\n=== A=(2,-{a};-{b},2), u = {t}*{h.T.tolist()} = {u.T.tolist()} ===")
    print("   Tran W basis (columns):", Lb.tolist(), " covolume index:", abs(Lb.det()))
    for c in conds:
        print("   must be integral:", c.T.tolist())
