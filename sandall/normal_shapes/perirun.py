#!/usr/bin/env python3
"""B) The periplectic shapes: omega_0, the exceptional p=3 shape, the 2nd type.

Sections
  A) the candidate shapes: closed? non-degenerate? exact? class in H^2?
  B) the preserving Lie superalgebras h(omega) and Span{H_f}: superdimensions,
     and the identity  h(omega)/Span{H_f} = H^1
  C) the brackets: the closed formulas and the identities they satisfy
  D) the action of Aut O(n|n;u) on H^2_odd at p = 3
  E) Skryabin's 2nd type, superized: omega_{2,j} = d(exp(q_j^{(p^{u_j})}) alpha_0)

Usage: python3 perirun.py
"""
import sys, os, itertools, random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
from peri import (Omega, omega0, alpha0, omegaXi, omega2exp, exp_v,       # noqa
                  is_closed, is_exact, nondegenerate, vfield_basis, vec,
                  vf_to_vec, cohomology, rank_p, nullspace_p)
from exactlin import rref                                                  # noqa


# ------------------------------------------------------------------ helpers

def partials(alg):
    """the 2n partial derivatives of O(n|n;u), as derivations of Omega^*."""
    out = []
    for i in range(alg.n):
        ev = [{}] * (2 * alg.n); od = [{}] * (2 * alg.n)
        ev = list(ev); ev[i] = alg.one()
        out.append(alg.derivation(ev, od, 0))                    # d/dq_i
    for i in range(alg.n):
        ev = [{}] * (2 * alg.n); od = [{}] * (2 * alg.n)
        od = list(od); od[i] = alg.one()
        out.append(alg.derivation(ev, od, 1))                    # d/dxi_i
    return out


class Ham:
    """f |-> H_f with iota_{H_f} omega = (-1)^{p(f)} df, for a non-degenerate
    omega.  The sign is the normalization that makes {f,g} := H_f(g) the Buttin
    antibracket for omega_0; drop it (sign=False) and the bracket acquires the
    overall factor (-1)^{p(f)}."""

    def __init__(self, alg, om, sign=True):
        self.sign = sign
        self.alg, self.om = alg, om
        B1 = alg.basis(1)
        self.B1, self.i1 = B1, {m: i for i, m in enumerate(B1)}
        self.vfs = vfield_basis(alg)
        cols = [vec(alg, alg.iota(D)(om), self.i1) for D in self.vfs]
        M = [[cols[j][i] for j in range(len(cols))] for i in range(len(B1))]
        aug = [M[i] + [1 if i == j else 0 for j in range(len(B1))]
               for i in range(len(B1))]
        R, piv = rref(aug, alg.p)
        assert piv == list(range(len(cols))), "omega is degenerate"
        self.inv = [[R[i][len(cols) + j] for j in range(len(B1))]
                    for i in range(len(cols))]
        self.d = alg.d()

    def __call__(self, f):
        alg = self.alg
        t = vec(alg, self.d(f), self.i1)
        x = [sum(self.inv[i][j] * t[j] for j in range(len(t))) % alg.p
             for i in range(len(self.inv))]
        fs = [dict() for _ in range(alg.n)]
        gs = [dict() for _ in range(alg.n)]
        for c, D in zip(x, self.vfs):
            c %= alg.p
            if not c:
                continue
            for i in range(alg.n):
                for m, v in D[0][i].items():
                    fs[i][m] = (fs[i].get(m, 0) + c * v) % alg.p
                for m, v in D[1][i].items():
                    gs[i][m] = (gs[i].get(m, 0) + c * v) % alg.p
        fs = tuple({m: v for m, v in y.items() if v} for y in fs)
        gs = tuple({m: v for m, v in y.items() if v} for y in gs)
        ps = set()
        for y in fs:
            ps |= {alg.parity(m) for m in y}
        for y in gs:
            ps |= {(alg.parity(m) + 1) % 2 for m in y}
        par = ps.pop() if len(ps) == 1 else 0
        if self.sign:
            pf = set(alg.parity(m) for m in f)
            if len(pf) == 1 and pf.pop() == 1:
                fs = tuple(alg.smul(-1, y) for y in fs)
                gs = tuple(alg.smul(-1, y) for y in gs)
        return (fs, gs, par)


def h_omega(alg, om, fields=None):
    """{D | L_D omega = 0}: coordinates in the given basis of vector fields."""
    B2 = alg.basis(2)
    i2 = {m: i for i, m in enumerate(B2)}
    vfs = fields if fields is not None else vfield_basis(alg)
    rows = [vec(alg, alg.lie(D)(om), i2) for D in vfs]
    cols = [[rows[j][i] for j in range(len(rows))] for i in range(len(B2))]
    return nullspace_p(cols, alg.p), vfs


def sdim_graded(alg, ns, vfs):
    """(even|odd) dimension of a parity-graded space of fields."""
    par = [D[2] for D in vfs]
    ie = [i for i, q in enumerate(par) if q == 0]
    io = [i for i, q in enumerate(par) if q == 1]
    return (rank_p([[v[i] for i in ie] for v in ns], alg.p),
            rank_p([[v[i] for i in io] for v in ns], alg.p))


def class_coeffs(alg, om):
    """at p = 3: the coordinates of [om] on the basis [xi_j (dxi_j)^{(2)}] of
    H^2_odd, or None if om is exact."""
    B2 = alg.basis(2, par=1)
    i2 = {m: i for i, m in enumerate(B2)}
    d = alg.d()
    B1 = [m for m in alg.basis(1) if alg.parity(m) == 0]
    cob = [vec(alg, d({m: 1}), i2) for m in B1]
    base = rank_p(cob, alg.p)
    zs = [vec(alg, alg.mul(alg.xi(j), alg.dxi(j, 2)), i2) for j in range(alg.n)]
    t = vec(alg, om, i2)
    for a in itertools.product(range(alg.p), repeat=alg.n):
        diff = [(t[i] - sum(a[j] * zs[j][i] for j in range(alg.n))) % alg.p
                for i in range(len(B2))]
        if rank_p(cob + [diff], alg.p) == base:
            return list(a)
    return None


def pullback(alg, F, G, x):
    """Phi^*(x) for the automorphism Phi: q_i -> F_i, xi_i -> G_i.
    Divided powers are taken as g^{(a)} = Phi(g)^a / a!, so every exponent that
    occurs must be < p (true for u = One and for 2-forms)."""
    d = alg.d()
    imgs = list(F) + [d(g) for g in G] + list(G) + [d(f) for f in F]
    out = alg.zero()
    for mon, c in x.items():
        ev, od = mon
        term = alg.smul(c, alg.one())
        for i in range(alg.nev):
            a = ev[i]
            if a:
                assert a < alg.p, "divided power out of reach of the pullback"
                y = alg.one()
                for _ in range(a):
                    y = alg.mul(y, imgs[i])
                fact = 1
                for k in range(2, a + 1):
                    fact = (fact * k) % alg.p
                term = alg.mul(term, alg.smul(pow(fact, alg.p - 2, alg.p), y))
        for j in range(alg.nod):
            if od[j]:
                term = alg.mul(term, imgs[alg.nev + j])
        out = alg.add(out, term)
    return out


def main():
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "perirun.log"), "w")

    def out(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()

    bad = 0

    # ---------------------------------------------------------------- A)
    out("A) the candidate shapes")
    out("   omega_0  = sum dq_i dxi_i ;   omega_Xi = omega_0 + xi_1 (dxi_1)^(2)")
    cases = [(1, 3, (1,)), (1, 3, (2,)), (2, 3, (1, 1)), (2, 3, (1, 2)),
             (3, 3, (1, 1, 1)), (1, 5, (1,)), (2, 5, (1, 1)), (1, 7, (1,))]
    for (n, p, u) in cases:
        alg = Omega(n, p, u)
        shapes = [("omega_0", omega0(alg))]
        if p == 3:
            shapes.append(("omega_Xi", omegaXi(alg)))
            if n > 1:
                shapes.append(("omega_Xi(all)", omegaXi(alg, [1] * n)))
        out(f"\n  n={n} p={p} u={u}")
        for name, om in shapes:
            cl, nd, ex = is_closed(alg, om), nondegenerate(alg, om), is_exact(alg, om)
            cc = class_coeffs(alg, om) if p == 3 else None
            out(f"    {name:13s} closed={cl}  nondeg={nd}  exact={ex}"
                + (f"  [om] = {cc}" if cc is not None else ""))
            if not cl or not nd:
                bad += 1
                out("      <<< NOT a periplectic form")

    # ---------------------------------------------------------------- B)
    out("\nB) the preserving superalgebras;  h(omega)/Span{H_f} = H^1")
    for (n, p, u) in [(1, 3, (1,)), (1, 3, (2,)), (2, 3, (1, 1)),
                      (1, 5, (1,)), (1, 7, (1,)), (2, 5, (1, 1))]:
        alg = Omega(n, p, u)
        fb = alg.fun_basis()
        dO = (len([m for m in fb if alg.parity(m) == 0]),
              len([m for m in fb if alg.parity(m) == 1]))
        h1o = cohomology(alg, 1, par=1)[0]
        h1e = cohomology(alg, 1, par=0)[0]
        out(f"\n  n={n} p={p} u={u}  sdim O = ({dO[0]}|{dO[1]})  "
            f"sdim H^1 = ({h1e}|{h1o})")
        shapes = [("omega_0", omega0(alg))]
        if p == 3:
            shapes.append(("omega_Xi", omegaXi(alg)))
        for name, om in shapes:
            ns, vfs = h_omega(alg, om)
            e, o = sdim_graded(alg, ns, vfs)
            H = Ham(alg, om)
            idx = {m: i for i, m in enumerate(fb)}
            rows_fe = [vf_to_vec(alg, H({m: 1}), idx) for m in fb
                       if alg.parity(m) == 0]
            rows_fo = [vf_to_vec(alg, H({m: 1}), idx) for m in fb
                       if alg.parity(m) == 1]
            # p(H_f) = p(f) + 1
            se, so = rank_p(rows_fo, alg.p), rank_p(rows_fe, alg.p)
            tot_h, tot_s = e + o, se + so
            ok = (tot_h - tot_s == h1e + h1o) and (tot_s == sum(dO) - 1)
            out(f"    {name:10s} sdim h(omega) = ({e}|{o})   "
                f"sdim Span(H_f) = ({se}|{so})   "
                f"quotient = {tot_h - tot_s} (= dim H^1 = {h1e + h1o}) "
                f"{'ok' if ok else '<<< MISMATCH'}")
            if not ok:
                bad += 1

    # ---------------------------------------------------------------- C)
    out("\nC) the brackets:  {f,g} := H_f(g),  iota_{H_f}omega = (-1)^{p(f)}df")
    out("   omega_0 :  {f,g} = sum_i (d_qi f d_xii g + (-1)^{p(f)} d_xii f d_qi g)")
    out("   omega_Xi:  the same MINUS (-1)^{p(f)} xi_1 d_q1 f d_q1 g")
    for (n, p, u) in [(1, 3, (1,)), (2, 3, (1, 1)), (1, 5, (1,)), (1, 7, (1,)),
                      (1, 3, (2,)), (2, 5, (1, 1))]:
        alg = Omega(n, p, u)
        fb = alg.fun_basis()
        par = partials(alg)
        shapes = [("omega_0", omega0(alg))]
        if p == 3:
            shapes.append(("omega_Xi", omegaXi(alg)))
        out(f"\n  n={n} p={p} u={u}   dim O = {len(fb)}")
        for name, om in shapes:
            H = Ham(alg, om)

            def br(f, g):
                return alg.lie(H(f))(g)

            def closed_formula(f, g, pf, deformed):
                r = alg.zero()
                for i in range(n):
                    r = alg.add(r, alg.mul(par[i](f), par[n + i](g)))
                    r = alg.add(r, alg.smul((-1) ** pf,
                                            alg.mul(par[n + i](f), par[i](g))))
                if deformed:
                    r = alg.add(r, alg.smul(-((-1) ** pf),
                                            alg.mul(alg.xi(0),
                                                    alg.mul(par[0](f), par[0](g)))))
                return r

            dfm = (name == "omega_Xi")
            okf = sum(1 for m in fb for m2 in fb
                      if br({m: 1}, {m2: 1})
                      == closed_formula({m: 1}, {m2: 1}, alg.parity(m), dfm))
            tot = len(fb) ** 2
            out(f"    {name:9s} closed formula: {okf}/{tot}"
                + ("" if okf == tot else "   <<< FAILS"))
            if okf != tot:
                bad += 1

            # the identities
            rng = random.Random(5)
            mons = fb if len(fb) <= 12 else [rng.choice(fb) for _ in range(12)]
            f_ham = f_pair = f_sym = f_leib = f_jac = 0
            n_ham = n_pair = n_sym = n_leib = n_jac = 0
            from peri import vf_bracket
            for m in mons:
                f = {m: 1}
                pf = alg.parity(m)
                n_ham += 1
                if alg.lie(H(f))(om):
                    f_ham += 1
                for m2 in mons:
                    g = {m2: 1}
                    pg = alg.parity(m2)
                    n_pair += 1
                    lhs = vf_bracket(alg, H(f), H(g))
                    rhs = H(br(f, g))
                    if not (all(lhs[0][i] == rhs[0][i] for i in range(n))
                            and all(lhs[1][i] == rhs[1][i] for i in range(n))):
                        f_pair += 1
                    n_sym += 1
                    if br(f, g) != alg.smul(-((-1) ** ((pf + 1) * (pg + 1))),
                                            br(g, f)):
                        f_sym += 1
                    for m3 in mons[:6]:
                        h = {m3: 1}
                        n_leib += 1
                        # {f, gh} = {f,g}h + (-1)^{(p(f)+1)p(g)} g {f,h}
                        if br(f, alg.mul(g, h)) != alg.add(
                                alg.mul(br(f, g), h),
                                alg.smul((-1) ** ((pf + 1) * pg),
                                         alg.mul(g, br(f, h)))):
                            f_leib += 1
                        n_jac += 1
                        # {f,{g,h}} = {{f,g},h} + (-1)^{(p(f)+1)(p(g)+1)}{g,{f,h}}
                        if br(f, br(g, h)) != alg.add(
                                br(br(f, g), h),
                                alg.smul((-1) ** ((pf + 1) * (pg + 1)),
                                         br(g, br(f, h)))):
                            f_jac += 1
            out(f"      L_{{H_f}}omega=0 {n_ham - f_ham}/{n_ham};  "
                f"[H_f,H_g]=H_{{f,g}} {n_pair - f_pair}/{n_pair};  "
                f"antisymmetry {n_sym - f_sym}/{n_sym};  "
                f"Leibniz {n_leib - f_leib}/{n_leib};  "
                f"Jacobi {n_jac - f_jac}/{n_jac}")
            if f_ham or f_pair or f_sym or f_leib or f_jac:
                bad += 1
                out("      <<< an identity FAILED")

    # ---------------------------------------------------------------- D)
    out("\nD) the action of Aut O on H^2_odd at p = 3 "
        "(prediction: the Frobenius twist C^{[3]} of the linear part in the xi)")
    for (n, u) in [(1, (1,)), (2, (1, 1)), (3, (1, 1, 1))]:
        alg = Omega(n, 3, u)
        rng = random.Random(11)
        bads = 0
        trials = 0
        for _ in range(40 if n < 3 else 12):
            # a random automorphism: linear part + higher terms
            while True:
                C = [[rng.randrange(3) for _ in range(n)] for _ in range(n)]
                Bm = [[rng.randrange(3) for _ in range(n)] for _ in range(n)]
                if rank_p(C, 3) == n and rank_p(Bm, 3) == n:
                    break
            F, G = [], []
            for i in range(n):
                f = alg.zero()
                for j in range(n):
                    f = alg.add(f, alg.smul(Bm[i][j], alg.q(j)))
                # a quadratic even correction
                if n >= 2 and rng.randrange(2):
                    f = alg.add(f, alg.mul(alg.xi(0), alg.xi(1)))
                F.append(f)
            for i in range(n):
                g = alg.zero()
                for j in range(n):
                    g = alg.add(g, alg.smul(C[i][j], alg.xi(j)))
                if rng.randrange(2):
                    g = alg.add(g, alg.mul(alg.q(0), alg.xi(i)))
                G.append(g)
            for j in range(n):
                z = alg.mul(alg.xi(j), alg.dxi(j, 2))
                pz = pullback(alg, F, G, z)
                if not is_closed(alg, pz):
                    out("      <<< the pullback of a cocycle is not closed")
                    bads += 1
                    continue
                got = class_coeffs(alg, pz)
                pred = [pow(C[j][k], 3, 3) for k in range(n)]
                trials += 1
                if got != pred:
                    bads += 1
                    out(f"      <<< n={n}: got {got}, predicted {pred} "
                        f"(row {j} of C = {C[j]})")
        out(f"    n={n}: {trials - bads}/{trials} pullbacks match the "
            f"Frobenius-twisted prediction")
        if bads:
            bad += 1

    # ---------------------------------------------------------------- E)
    out("\nE) Skryabin's 2nd type superized: omega_{2,j} = d(exp(q_j^{(p^u_j)}) alpha_0)")
    out("   computed in the algebra with the j-th height raised by 1; the fields")
    out("   are those of vect(n|n;u), i.e. with coefficients in the small O.")
    for (n, p, u, j) in [(1, 3, (1,), 0), (1, 5, (1,), 0), (2, 3, (1, 1), 0),
                         (1, 3, (2,), 0), (1, 7, (1,), 0)]:
        ubig = list(u); ubig[j] += 1
        big = Omega(n, p, tuple(ubig))
        small = Omega(n, p, u)
        om2 = omega2exp(big, j)
        om0 = omega0(big)
        # the fields of vect(n|n;u) sitting inside vect over the big algebra
        smallmons = set(m for m in big.fun_basis()
                        if all(m[0][i] < p ** u[i] for i in range(n)))
        fields = [D for D in vfield_basis(big)
                  if all(set(x) <= smallmons for x in D[0] + D[1])]
        ns2, _ = h_omega(big, om2, fields)
        ns0, _ = h_omega(big, om0, fields)
        e2, o2 = sdim_graded(big, ns2, fields)
        e0, o0 = sdim_graded(big, ns0, fields)
        out(f"    n={n} p={p} u={u} j={j}:  closed={is_closed(big, om2)}  "
            f"sdim h_exp = ({e2}|{o2})   sdim h(omega_0) = ({e0}|{o0})   "
            f"{'SAME' if (e2, o2) == (e0, o0) else 'DIFFERENT'}")
        if not is_closed(big, om2):
            bad += 1

    out(f"\n  problems: {bad}")
    log.close()


if __name__ == "__main__":
    main()
