"""Mixed fiber types: the ansatz in which each of the three 2-tensor slots of a
bilinear operator carries its OWN symmetry type.

Subsec. 11.24 of the text treats the configurations

        S^2(W^*) x S^2(W^*) --> Pi S^2(W^*),
        S^2(W^*) x S^2(W^*) --> Lambda^2(W^*)

over a periplectic form, and says what stands in the way: the fiber model of
Section 7 of the text -- (image of a Young symmetrizer) cap (all omega-traces zero) -- has
no verbatim analog over pe(k).  It has none; but for a fiber with TWO boxes it
is not needed, because in the u-encoding of ssuper.py the symmetry type of a
2-tensor IS the parity of the auxiliary variable:

        A~(z,u) = sum_{ij} A_{ij}(z) u^i u^j,   p(u^i) = p(i) + s,

encodes the super-symmetric 2-tensors for s = 0 and, for s = 1, the
super-antisymmetric ones -- the differential 2-forms, which is where omega
lives.  No symmetrizer and no trace condition anywhere.  What ssuper.py cannot
do is use DIFFERENT s for the two arguments and the value, because there one
single family of u's carries the fiber slots of both arguments and the output
slots as well.  Here they are separated into five families:

        u^i (parity p(i)+sA)  the fiber of the first argument,
        w^i (parity p(i)+sB)  the fiber of the second,
        t^i (parity p(i)+sC)  the value,
        r^i, s^i (parity p(i))  the derivative indices of the two arguments,

the last two because a derivative index is super-symmetric with the natural
parities whatever the symmetry type of the fiber is: d_i d_j = (-1)^{p(i)p(j)}
d_j d_i, always.  Pooling them with the fiber indices, as ssuper.py may, is
legitimate only when the fiber is of type s = 0.

A scheme is a perfect matching of the slot list

        a  derivative slots of A  |  2 fiber slots of A
        b  derivative slots of B  |  2 fiber slots of B  |  2 output slots,

a + b = N, exactly as in Subsec. 7.2 of the text.  An (in,in) pair
is a summation with the weight omega^{ij}; an (in,out) pair LEAVES its index
where it is -- that is the whole of it, and it is what keeps the untouched
factor even and the Koszul bookkeeping the same as in sgen.py; an (out,out)
pair inserts the invariant element of the target fiber, which is not assumed to
be omega but computed (invariants(), below): over a periplectic form omega is
an odd 2-form, so it lies in Lambda^2 and not in S^2, and the (out,out) schemes
exist only for the former.

At the very end the surviving indices, wherever they sit, are read off as a
2-tensor and rewritten in the value family -- Psi(), the one place where the
symmetry type changes and hence the one place where the projection onto the
target fiber happens.

The one sign made by hand is sgen.py's: an operator applied to the second
factor is moved past the first one.  Everything else is made by the
supercommutative arithmetic.  test_sfib.py checks the whole against operators
whose invariance is known beforehand -- the identity, (ptr A)B, (ptr B)A,
d(ptr A)^d(ptr B) -- and against every number the pooled engine returns.
"""

from fractions import Fraction

from ssuper import Super


AFIB, BFIB, OUT, ADER, BDER = 0, 1, 2, 3, 4
FREE = (AFIB, BFIB, ADER, BDER)          # where a surviving index can sit


# ----------------------------------------------------------------- the setup


class MSetup:
    """Coordinates z and five families of fiber variables; shifts = (sA,sB,sC).

    form = "even": sdim (2m|n), omega even, h(2m|n), aut(omega) = osp(n|2m).
    form = "odd" : sdim (k|k) with k = m = n, omega odd (Buttin), le(k),
                   aut(omega) = pe(k).
    """

    NFAM = 5

    def __init__(self, m, n, shifts=(0, 0, 0), form="even", odd_coef=1,
                 ham_sign=1):
        self.m, self.n, self.form = m, n, form
        self.shifts = tuple(shifts) + (0, 0)
        if form == "odd":
            assert m == n, "a periplectic form needs sdim (k|k)"
            self.k = m
            self.nc = 2 * m
            self.par = [0] * m + [1] * m
        else:
            self.nc = 2 * m + n
            self.par = [0] * (2 * m) + [1] * n
        self.odd_coef = odd_coef
        self.ham_sign = ham_sign
        self.fpar = [[(p + s) % 2 for p in self.par] for s in self.shifts]

        self.zev, self.zod = {}, {}
        nev = nod = 0
        for i in range(self.nc):
            if self.par[i] == 0:
                self.zev[i] = nev
                nev += 1
            else:
                self.zod[i] = nod
                nod += 1
        self.yev = [dict() for _ in range(self.NFAM)]
        self.yod = [dict() for _ in range(self.NFAM)]
        for f in range(self.NFAM):
            for i in range(self.nc):
                if self.fpar[f][i] == 0:
                    self.yev[f][i] = nev
                    nev += 1
                else:
                    self.yod[f][i] = nod
                    nod += 1
        self.A = Super(nev, nod)

        Om = [[Fraction(0)] * self.nc for _ in range(self.nc)]
        if form == "odd":
            for i in range(self.k):
                Om[i][self.k + i] = Fraction(1)
                Om[self.k + i][i] = Fraction(odd_coef)
        else:
            for i in range(m):
                Om[i][m + i] = Fraction(-1)
                Om[m + i][i] = Fraction(1)
            for a in range(n):
                Om[2 * m + a][2 * m + a] = Fraction(odd_coef)
        self.Om = Om
        self.Omnz = [(i, j, Om[i][j]) for i in range(self.nc)
                     for j in range(self.nc) if Om[i][j]]

    # --- generators
    def z(self, i):
        return (self.A.gen_ev(self.zev[i]) if self.par[i] == 0
                else self.A.gen_od(self.zod[i]))

    def y(self, f, i):
        return (self.A.gen_ev(self.yev[f][i]) if self.fpar[f][i] == 0
                else self.A.gen_od(self.yod[f][i]))

    def dz(self, p, i):
        return (self.A.d_ev(p, self.zev[i]) if self.par[i] == 0
                else self.A.d_od(p, self.zod[i]))

    def dy(self, p, f, i):
        return (self.A.d_ev(p, self.yev[f][i]) if self.fpar[f][i] == 0
                else self.A.d_od(p, self.yod[f][i]))

    # --- the Poisson / Buttin superalgebra (verbatim from ssuper.SSetup)
    def _bracket_homog(self, f, g, pf):
        A = self.A
        m, n = self.m, self.n
        if self.form == "odd":
            k = self.k
            s = Fraction(self.ham_sign) * (-1 if pf else 1)
            terms = []
            for i in range(k):
                terms.append(A.mul(self.dz(f, i), self.dz(g, k + i)))
                terms.append(A.scale(A.mul(self.dz(f, k + i), self.dz(g, i)), s))
            return A.summ(terms)
        terms = []
        for i in range(m):
            terms.append(A.mul(self.dz(f, i), self.dz(g, m + i)))
            terms.append(A.scale(A.mul(self.dz(f, m + i), self.dz(g, i)), -1))
        s = Fraction(self.ham_sign) * (-1 if pf else 1)
        for a in range(n):
            terms.append(A.scale(A.mul(self.dz(f, 2 * m + a),
                                       self.dz(g, 2 * m + a)), s))
        return A.summ(terms)

    def bracket(self, f, g):
        ev, od = self.A.parity_parts(f)
        out = []
        if ev:
            out.append(self._bracket_homog(ev, g, 0))
        if od:
            out.append(self._bracket_homog(od, g, 1))
        return self.A.summ(out)

    def ham(self, f):
        return [self.bracket(f, self.z(k)) for k in range(self.nc)]

    # --- the Lie derivative on a field whose fiber sits in the family fam
    #
    #   L_X T = sum_k X^k d_{z^k}T + eps sum_{i,k} y^i (d_i X^k)(d_{y^k}T),
    #
    #             eps = (-1)^{s p(X)},   s the parity shift of that family.
    #
    # For s = 0 that is the classical formula verbatim, which is what
    # test_ssuper.py finds by scanning all 256 sign patterns.  For s = 1 the
    # extra sign is not a convention: without it the formula is not even a
    # representation of le(k) on the shift-1 fiber -- test_sfib.py scans the
    # same 256 patterns and every admissible one has this bit set -- and then
    # omega is not invariant.  The omission is invisible in ssuper.py, whose
    # own shift-1 mode houses one constant tensor over an EVEN form, where
    # p(X) never enters.
    def lie(self, f, T, fam):
        A = self.A
        X = self.ham(f)
        pX = (A.parity(f) + (1 if self.form == "odd" else 0)) % 2
        eps = -1 if (self.shifts[fam] * pX) % 2 else 1
        terms = [A.mul(X[k], self.dz(T, k)) for k in range(self.nc)]
        for i in range(self.nc):
            for k in range(self.nc):
                dXk = self.dz(X[k], i)
                if not dXk:
                    continue
                dT = self.dy(T, fam, k)
                if not dT:
                    continue
                t = A.mul(self.y(fam, i), A.mul(dXk, dT))
                terms.append(t if eps > 0 else A.scale(t, -1))
        return A.summ(terms)

    def lie_all(self, f, T):
        """The Lie derivative of a value: its indices may sit in any family."""
        A = self.A
        X = self.ham(f)
        pX = (A.parity(f) + (1 if self.form == "odd" else 0)) % 2
        terms = [A.mul(X[k], self.dz(T, k)) for k in range(self.nc)]
        for fam in range(self.NFAM):
            eps = -1 if (self.shifts[fam] * pX) % 2 else 1
            for i in range(self.nc):
                for k in range(self.nc):
                    dXk = self.dz(X[k], i)
                    if not dXk:
                        continue
                    dT = self.dy(T, fam, k)
                    if not dT:
                        continue
                    t = A.mul(self.y(fam, i), A.mul(dXk, dT))
                    terms.append(t if eps > 0 else A.scale(t, -1))
        return A.summ(terms)

    # --- reading a 2-tensor off, knowing where its two indices sit
    def Psi(self, V, fams):
        """Rewrite the value in the family OUT, its two free indices sitting in
        the families f1, f2 (in the order the scheme's output slots come):

            Psi(V) = sum_{i,j} (-1)^{(s_2 + s_C) p(i)} t^i t^j d_{f2,j}d_{f1,i}V,

        s_f the parity shift of the family f.  The exponent is not guessed: it
        is what makes Psi EQUIVARIANT, and test_sfib.py determines it by
        scanning the sixteen sign patterns and the two parities over all eight
        triples (s_1,s_2,s_C) and every pair of families -- exactly one pattern
        survives in each case, up to the overall sign, which changes no span,
        and the surviving one is this, of parity s_1 + s_2.

        The double derivative reads the WHOLE tensor and the t's re-symmetrize
        it in the target's rule; that re-symmetrization is the projection onto
        the target fiber which the ansatz of Section 7 imposes by hand.  With
        one family and matching types it is twice the identity re-encoding,
        which is measured separately.  A scheme is thus scaled by a factor of
        its own, which again changes no span.
        """
        A = self.A
        f1, f2 = fams
        sgn = (self.shifts[f2] + self.shifts[OUT]) % 2
        out = []
        for i in range(self.nc):
            Vi = self.dy(V, f1, i)
            if not Vi:
                continue
            ti = self.y(OUT, i)
            neg = sgn and self.par[i]
            for j in range(self.nc):
                Vij = self.dy(Vi, f2, j)
                if not Vij:
                    continue
                tt = A.mul(ti, self.y(OUT, j))
                if not tt:
                    continue
                t = A.mul(tt, Vij)
                out.append(A.scale(t, -1) if neg else t)
        return A.summ(out)

    def psi_parity(self, fams):
        return (self.shifts[fams[0]] + self.shifts[fams[1]]) % 2

    def reencode(self, p, ren):
        """The same tensor with the families renamed as `ren` says (a dict, or
        a single pair), monomial by monomial in its own canonical order, so
        that every sign is made by Super.mul."""
        A = self.A
        if not isinstance(ren, dict):
            ren = {ren[0]: ren[1]}
        mev, mod = {}, {}
        for ffrom, fto in ren.items():
            for i in range(self.nc):
                src = (self.yev[ffrom].get(i), self.yod[ffrom].get(i))
                if src[0] is not None:
                    mev[src[0]] = self.y(fto, i)
                else:
                    mod[src[1]] = self.y(fto, i)
        out = []
        for (e, m), c in p.items():
            t = A.const(c)
            for g, ex in enumerate(e):
                for _ in range(ex):
                    t = A.mul(t, mev.get(g) or A.gen_ev(g))
                    if not t:
                        break
            if not t:
                continue
            b = 0
            mm = m
            while mm:
                if mm & 1:
                    t = A.mul(t, mod.get(b) or A.gen_od(b))
                    if not t:
                        break
                mm >>= 1
                b += 1
            if t:
                out.append(t)
        return A.summ(out)

    def ptr(self, p, fam):
        """sum omega^{kl} d_{y^k} d_{y^l} p, the trace of a 2-tensor.  Over a
        periplectic form it is the periplectic trace (26) of the text, an odd
        function; over an even one it kills S^2(W^*)."""
        A = self.A
        return A.summ([A.scale(self.dy(self.dy(p, fam, k), fam, l), w)
                       for k, l, w in self.Omnz])

    def dform(self, f, fam=OUT):
        """df = sum_i y^i (d_i f), in the given family."""
        A = self.A
        return A.summ([A.mul(self.y(fam, i), self.dz(f, i))
                       for i in range(self.nc)])

    def U(self, T, fam):
        """One derivative index left free, made in the family fam."""
        A = self.A
        return A.summ([A.mul(self.y(fam, i), self.dz(T, i))
                       for i in range(self.nc)])

    # --- monomials and fields
    def monomials(self, deg):
        out = []
        self._mon(deg, 0, self.A.const(1), out)
        return out

    def _mon(self, deg, start, acc, out):
        if deg == 0:
            out.append(acc)
            return
        for i in range(start, self.nc):
            p = self.A.mul(acc, self.z(i))
            if not p:
                continue
            self._mon(deg - 1, i if self.par[i] == 0 else i + 1, p, out)

    def fiber_monomials(self, fam):
        out = []
        for i in range(self.nc):
            for j in range(i, self.nc):
                p = self.A.mul(self.y(fam, i), self.y(fam, j))
                if p:
                    out.append(((i, j), p))
        return out

    def sdim_fiber(self, fam):
        ev = od = 0
        for (i, j), _ in self.fiber_monomials(fam):
            if (self.fpar[fam][i] + self.fpar[fam][j]) % 2:
                od += 1
            else:
                ev += 1
        return ev, od

    def random_field(self, fam, deg, rng, spread=4, nterms=None):
        """A random EVEN tensor field with fiber in the given family."""
        A = self.A
        mons = [p for d in range(deg + 1) for p in self.monomials(d)]
        out = []
        for (i, j), fib in self.fiber_monomials(fam):
            pf = A.parity(fib)
            pool = [p for p in mons if A.parity(p) == pf]
            if not pool:
                continue
            if nterms is not None and len(pool) > nterms:
                pool = rng.sample(pool, nterms)
            for p in pool:
                c = rng.randint(-spread, spread)
                if c:
                    out.append(A.scale(A.mul(p, fib), c))
        return A.summ(out)

    # --- the invariant constant tensors of a fiber
    def invariants(self, fam):
        """Constant tensors v with L_X v = 0 for every Hamiltonian field X.
        Over a periplectic form omega is one of them -- in the family whose
        symmetry type it has, which is what this decides."""
        import os
        import sys
        _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "..", "lib"), _HERE]
        from exactlin import nullspace

        A = self.A
        basis = [p for _, p in self.fiber_monomials(fam)]
        hams = [p for d in (1, 2, 3) for p in self.monomials(d)]
        rows = []
        for f in hams:
            ds = [self.lie(f, b, fam) for b in basis]
            keys = set()
            for d in ds:
                keys |= set(d)
            for kk in sorted(keys):
                rows.append([d.get(kk, Fraction(0)) for d in ds])
        ns = nullspace(rows) if rows else [
            [Fraction(1) if i == j else Fraction(0) for j in range(len(basis))]
            for i in range(len(basis))]
        out = []
        for v in ns:
            p = A.summ([A.scale(b, c) for c, b in zip(v, basis) if c])
            for part in A.parity_parts(p):
                if part and all(not self.lie(f, part, fam) for f in hams):
                    out.append(part)
        return out


# --------------------------------------------------------------- the schemes


SLOT = {"Au": (0, AFIB), "Az": (0, None), "Bw": (1, BFIB), "Bz": (1, None)}
FREESLOT = {"Au": AFIB, "Az": ADER, "Bw": BFIB, "Bz": BDER}


class MScheme:
    def __init__(self, a, b, kinds, inv=None):
        self.a, self.b, self.kinds, self.inv = a, b, kinds, inv

    def order(self):
        return self.a + self.b

    def __repr__(self):
        s = "|".join(x + "-" + y for x, y in self.kinds)
        return (f"M(d={self.a}{self.b},{s}"
                + (f",inv{self.inv}" if self.inv is not None else "") + ")")


def _matchings(slots):
    if not slots:
        yield ()
        return
    a = slots[0]
    for idx in range(1, len(slots)):
        b = slots[idx]
        rest = slots[1:idx] + slots[idx + 1:]
        for m in _matchings(rest):
            yield ((a, b),) + m


def enumerate_mixed(N, ninv=1):
    """All schemes of order N; ninv = how many invariant elements the target
    fiber has (0 kills every (out,out) scheme)."""
    out, seen = [], set()
    for a in range(N + 1):
        b = N - a
        slots = ["Az"] * a + ["Au"] * 2 + ["Bz"] * b + ["Bw"] * 2 + ["O"] * 2
        if len(slots) % 2:
            return []
        for mm in _matchings(slots):
            key = (a, b, tuple(sorted(tuple(sorted(p)) for p in mm)))
            if key in seen:
                continue
            seen.add(key)
            noo = sum(1 for p in mm if p[0] == "O" and p[1] == "O")
            if noo:
                for r in range(ninv):
                    out.append(MScheme(a, b, mm, inv=r))
            else:
                out.append(MScheme(a, b, mm))
    return out


def scheme_parity(S, sch):
    """The parity of a scheme, from its combinatorial type.

    An (in,in) pair has p(i)+p(j) = 1 for a periplectic omega and p(i) = p(j)
    for an even one, so it contributes [form odd] + sigma + sigma', sigma the
    shift of the slot's family; an (in,out) pair contributes sC + sigma, the
    index changing family; an (out,out) pair contributes [form odd], the parity
    of the invariant element.  Since every input slot occurs exactly once, the
    shifts of the fiber slots add up to 2sA + 2sB = 0, and the total is
    (N+2)/2 mod 2 for a periplectic form and 0 for an even one -- the same for
    all three symmetry types.  Compared with the parity actually measured on
    the value, in srunfib.py.
    """
    sA, sB, sC = S.shifts[:3]
    sh = {"Az": 0, "Au": sA, "Bz": 0, "Bw": sB, "O": sC}
    odd = 1 if S.form == "odd" else 0
    p = 0
    for (k1, k2) in sch.kinds:
        if k1 == "O" and k2 == "O":
            p += odd
        elif k1 == "O" or k2 == "O":
            p += sC + sh[k2 if k1 == "O" else k1]
        else:
            p += odd + sh[k1] + sh[k2]
    return p % 2


# ------------------------------------------------------------- the evaluation


def evaluate(S, sch, Atil, Btil, invs):
    """The value of a scheme on the two fields, a quadratic in t.

    The derivative indices are made first, by U; then every slot is a variable
    of some family, a contraction is a pair of d_y's weighted by omega, and an
    output slot means that its index is simply left alone.  The one sign made
    by hand is that of moving an operator on the second factor past the first.
    """
    A = S.A
    state = [(Fraction(1), Atil, Btil, A.parity(Atil))]
    for (k1, k2) in sch.kinds:
        if k1 == "O" or k2 == "O":
            continue
        new = []
        for (c, P, Q, pP) in state:
            for (i, j, w) in S.Omnz:
                cc, PP, QQ, ppP = c * w, P, Q, pP
                for (kind, idx) in ((k1, i), (k2, j)):
                    fac, fam = SLOT[kind]
                    # what is moved past the first factor is the INDEX, whose
                    # parity is p(idx); the shift of the family is bookkeeping
                    # for the symmetry type, not a property of the index.  With
                    # sA = sB = 0 -- every configuration sgen.py can do -- the
                    # two readings agree, so this is not a change of the
                    # calibrated convention; with a shifted argument they do
                    # not, and only this one makes the engine symmetric under
                    # the interchange of the two arguments (test_sfib.py).
                    pop = S.par[idx]
                    if fac == 0:
                        PP = (S.dz(PP, idx) if fam is None
                              else S.dy(PP, fam, idx))
                        if not PP:
                            break
                        ppP = (ppP + pop) % 2
                    else:
                        if pop and ppP:
                            cc = -cc
                        QQ = (S.dz(QQ, idx) if fam is None
                              else S.dy(QQ, fam, idx))
                        if not QQ:
                            break
                else:
                    if PP and QQ:
                        new.append((cc, PP, QQ, ppP))
        state = new
        if not state:
            return {}
    # the FREE derivative indices are made last: a contraction never lands on
    # one -- the matching has already said so -- and making them first would
    # put their carrier to the left of the contraction, which costs a Koszul
    # sign per index and makes even d(ptr A)^d(ptr B) non-invariant.
    na = sum(1 for (k1, k2) in sch.kinds
             if {k1, k2} == {"Az", "O"})
    nb = sum(1 for (k1, k2) in sch.kinds
             if {k1, k2} == {"Bz", "O"})
    out = []
    for (c, P, Q, _) in state:
        for _ in range(na):
            P = S.U(P, ADER)
        for _ in range(nb):
            Q = S.U(Q, BDER)
        if P and Q:
            out.append(A.scale(A.mul(P, Q), c))
    V = A.summ(out)
    if not V:
        return {}
    if sch.inv is not None:            # an (out,out) pair: V has no free index
        return A.mul(V, invs[sch.inv])
    fams = [FREESLOT[k2 if k1 == "O" else k1] for (k1, k2) in sch.kinds
            if (k1 == "O") != (k2 == "O")]
    return S.Psi(V, fams)
