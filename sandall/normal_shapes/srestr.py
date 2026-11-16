#!/usr/bin/env python3
"""Restricted structures on the (super)algebras preserving the normal shapes.

Restricted structures (both classical and non-classical for p=2) for the
restricted (super)algebras preserving the normal shapes.

The four notions are those of [BLLSq, Sect. 2] = arXiv:1407.1695:

  * p-structure on a Lie algebra: x -> x^[p] with ad_{x^[p]} = (ad_x)^p;
  * p|2p-structure on a Lie superalgebra (the CLASSICAL restrictedness): a
    p-structure on g_ev with [x^[p],y] = (ad_x)^p(y) for y in the WHOLE of g,
    whence [2p]: g_od -> g_ev, x -> (x^2)^[p];
  * 2|2-structure (p=2 only): the 2-structure of F(g), the Lie algebra obtained
    by forgetting the squaring; it exists as soon as a 2|4-structure does, and
    is defined on inhomogeneous elements as well;
  * (2,4)-structure on a Z/2-GRADED Lie algebra g = g_+ (+) g_-: a 2-structure
    on g_+ together with [4]: g_- -> g_+ with [x^[4],y] = (ad_x)^4(y); and its
    super analog, a (2,4)|4-structure, where g_+ and g_- are superspaces, [2]
    is defined on (g_+)_ev and [4] on g_- and on (g_+)_od.

The one thing this engine is careful about, and the reason it exists rather
than a three-line addition to an older one:

    A DERIVATION OF O(m|n;u) IS NOT DETERMINED BY ITS VALUES ON THE
    GENERATORS.

Indeed, Der O(m;u) = (+) O partial_i^{(p^k)}, k < u_i, whereas
vect(m|n;u) = (+) O partial_i is the summand with k = 0; and
partial_i^{(p)} kills every generator.  So "D^p, computed by applying D twice
to each generator" is NOT D^p unless one checks it: the honest test is whether
the candidate field acts as the operator D^p on the whole of O.  Every power
below is taken as an OPERATOR and then tested against its candidate field
(op_power).  Without that test the p-th power of partial_i reads as 0 and every
algebra comes out restricted for every shearing vector.

Everything is exact over F_p.  Output: srestr.log.
"""
import sys, os, itertools, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "lib"))
import sform as S

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- bookkeeping
class Log:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


# ------------------------------------------------------------- linear algebra
class Ech:
    """row echelon over F_p of a set of vector fields, for membership tests."""

    def __init__(self, A, idx, fields=()):
        self.A, self.idx, self.rows = A, idx, []
        for D in fields:
            self.add(D)

    def _vec(self, D):
        return list(S.vf_vec(self.A, D, self.idx))

    def _reduce(self, w):
        p = self.A.p
        for h, r in self.rows:
            if w[h]:
                c = (w[h] * pow(r[h], p - 2, p)) % p if p > 2 else w[h]
                for t in range(len(w)):
                    w[t] = (w[t] - c * r[t]) % p
        return w

    def add(self, D):
        w = self._reduce(self._vec(D))
        nz = [t for t in range(len(w)) if w[t]]
        if nz:
            self.rows.append((nz[0], w))
            self.rows.sort(key=lambda t: t[0])
            return True
        return False

    def __contains__(self, D):
        return not any(self._reduce(self._vec(D)))

    def __len__(self):
        return len(self.rows)


# ------------------------------------------------------------- vector fields
def zero_field(A, par=0):
    return (tuple([{}] * A.m), tuple([{}] * A.n), par)


def add_fields(A, D, E):
    fs = tuple(A.add(D[0][i], E[0][i]) for i in range(A.m))
    gs = tuple(A.add(D[1][j], E[1][j]) for j in range(A.n))
    return (fs, gs, D[2])


def smul_field(A, c, D):
    fs = tuple(A.smul(c, D[0][i]) for i in range(A.m))
    gs = tuple(A.smul(c, D[1][j]) for j in range(A.n))
    return (fs, gs, D[2])


def bracket(A, D, E):
    """[D,E] = D o E - (-1)^{p(D)p(E)} E o D, as a vector field."""
    LD, LE = A.lie(D), A.lie(E)
    s = (-1) ** (D[2] * E[2])
    fs = tuple(A.sub(LD(LE(A.x(i))), A.smul(s, LE(LD(A.x(i)))))
               for i in range(A.m))
    gs = tuple(A.sub(LD(LE(A.xi(j))), A.smul(s, LE(LD(A.xi(j)))))
               for j in range(A.n))
    return (fs, gs, (D[2] + E[2]) % 2)


def op_power(A, D, k):
    """the k-th power of D as a differential operator.

    Returns (E, ok): E is the vector field with the same values on the
    generators, ok says whether the operator D^k IS that vector field, i.e.
    whether the two agree on the whole of O(m|n;u).  When ok is False, D^k is a
    derivation of O outside vect(m|n;u) -- a divided power derivation -- and
    there is no p-th power inside the algebra to begin with.
    """
    L = A.lie(D)

    def Lk(X):
        for _ in range(k):
            X = L(X)
        return X

    fs = tuple(Lk(A.x(i)) for i in range(A.m))
    gs = tuple(Lk(A.xi(j)) for j in range(A.n))
    E = (fs, gs, (k * D[2]) % 2)
    LE = A.lie(E)
    for mm in A.fun_basis():
        if LE({mm: 1}) != Lk({mm: 1}):
            return E, False
    return E, True


def is_inner(A, idx, D, k, basis):
    """is (ad_D)^k the adjoint action of an element of Span(basis)?

    The test is honest: solve for y in Span(basis) with [y,z] = (ad_D)^k(z) for
    every z in the basis.  Used where the k-th power itself leaves the algebra.
    """
    from exactlin import solve
    p = A.p
    L = len(idx) * (A.m + A.n)
    cols, tgt = [], []
    for z in basis:
        w = z
        for _ in range(k):
            w = bracket(A, D, w)
        tv = S.vf_vec(A, w, idx)
        rows = [S.vf_vec(A, bracket(A, y, z), idx) for y in basis]
        for t in range(L):
            cols.append([rows[j][t] for j in range(len(basis))])
            tgt.append(tv[t])
    x = solve(cols, tgt, p)
    return x is not None


# --------------------------------------------------------- Hamiltonian fields
class Ham:
    """iota_{H_f} omega = df, solved once for the whole basis of Omega^1."""

    def __init__(self, A, om):
        from exactlin import solve
        self._solve = solve
        self.A, self.om = A, om
        B1 = A.basis(1)
        self.i1 = {mm: i for i, mm in enumerate(B1)}
        self.fields = S.vfield_basis(A)
        rows = [S.vec(A, A.iota(D)(om), self.i1) for D in self.fields]
        self.T = [[rows[k][i] for k in range(len(rows))]
                  for i in range(len(B1))]
        self.mons = A.fun_basis()
        self.idx = {mm: i for i, mm in enumerate(self.mons)}
        self.cache = {}

    def field(self, f):
        A = self.A
        v = self._solve(self.T, S.vec(A, A.d()(f), self.i1), A.p)
        if v is None:
            return None
        acc = [0] * ((A.m + A.n) * len(self.mons))
        pars = set()
        for k, c in enumerate(v):
            c = int(c) % A.p
            if not c:
                continue
            pars.add(self.fields[k][2])
            w = S.vf_vec(A, self.fields[k], self.idx)
            for t in range(len(acc)):
                acc[t] = (acc[t] + c * w[t]) % A.p
        par = pars.pop() if len(pars) == 1 else 0
        return S.vec_vf(A, acc, self.mons, par)

    def of(self, mon):
        if mon not in self.cache:
            self.cache[mon] = self.field({mon: 1})
        return self.cache[mon]


def ham_basis(A, om):
    """{H_f | f in O}, indexed by the monomial f; the constants give 0."""
    H = Ham(A, om)
    out = []
    for mm in A.fun_basis():
        D = H.of(mm)
        if D is None:
            continue
        if not any(D[0]) and not any(D[1]):
            continue                      # f constant: H_f = 0
        out.append((mm, D))
    return H, out


# ------------------------------------------------------------------ verdicts
def restricted(A, idx, fields, name, log, verbose=True):
    """Is Span(fields) closed under the p-th power of the even part and, for
    p = 2, under the squaring of the odd part?

    Returns a dict with the verdict and the list of the obstructions.
    """
    p = A.p
    ech = Ech(A, idx, fields)
    even = [D for D in fields if D[2] == 0]
    odd = [D for D in fields if D[2] == 1]
    out = {"dim": (len(even), len(odd)), "p": p,
           "bad_ev": [], "bad_od": [], "notfield_ev": [], "notfield_od": []}
    for D in even:
        E, ok = op_power(A, D, p)
        if not ok:
            out["notfield_ev"].append(D)
        elif E not in ech:
            out["bad_ev"].append(D)
    # the odd part: the squaring x -> x^2 (p = 2) resp. the [2p]-map
    for D in odd:
        if p == 2:
            E, ok = op_power(A, D, 2)
            if not ok:
                out["notfield_od"].append(D)
            elif E not in ech:
                out["bad_od"].append(D)
        else:
            E, ok = op_power(A, D, 2 * p)
            if not ok:
                out["notfield_od"].append(D)
            elif E not in ech:
                out["bad_od"].append(D)
    out["ok_ev"] = not (out["bad_ev"] or out["notfield_ev"])
    out["ok_od"] = not (out["bad_od"] or out["notfield_od"])
    out["restricted"] = out["ok_ev"] and out["ok_od"]
    if verbose:
        log(f"      {name}: sdim = {out['dim']},  "
            f"{'RESTRICTED' if out['restricted'] else 'NOT restricted'}"
            f"  (even: {len(out['bad_ev'])} outside, "
            f"{len(out['notfield_ev'])} not a field; "
            f"odd: {len(out['bad_od'])} outside, "
            f"{len(out['notfield_od'])} not a field)")
    return out


def sq_of_odd_in(A, idx, fields):
    """for p = 2: is Span(fields) a Lie SUPERalgebra, i.e. closed under the
    squaring of its odd part?  (The bracket may close while the squaring does
    not: at p = 2 the squaring is not determined by the bracket.)"""
    ech = Ech(A, idx, fields)
    bad = []
    for D in fields:
        if D[2] != 1:
            continue
        E, ok = op_power(A, D, 2)
        if not ok or E not in ech:
            bad.append(D)
    return bad


# ---------------------------------------------------- the Z/2-graded picture
def deg_grading(A, hb, a):
    """the Z/2-grading of Span{H_f} by the parity of the degree of f in the
    a-th indeterminate (a = 0..m+n-1, the x's then the xi's).

    Returns (plus, minus) as lists of fields, or None if it is not a grading of
    the algebra -- the brackets and the squares are both tested.
    """
    plus, minus = [], []
    for mm, D in hb:
        (minus if mm[a] % 2 else plus).append(D)
    return plus, minus


def parity_grading(A, hb):
    """the Z/2-grading of F(g) by the PARITY: g_+ = g_ev, g_- = g_od.

    It is a grading of the Lie algebra F(g) -- the one obtained by forgetting
    the squaring, [BLLSq, Subsec. 2.4] -- whenever the bracket closes, and it
    is available even when the squaring leaves the algebra, which is exactly
    the case of interest here.
    """
    return ([D for _, D in hb if D[2] == 0], [D for _, D in hb if D[2] == 1])


def br_grading(A, idx, plus, minus):
    """is g_+ (+) g_- a Z/2-grading of the Lie algebra, i.e. does the bracket
    respect it?  The squaring is a separate question, asked by sq_graded."""
    Ep, Em = Ech(A, idx, plus), Ech(A, idx, minus)
    for X, Y, tgt in ((plus, plus, Ep), (plus, minus, Em), (minus, minus, Ep)):
        for D in X:
            for E in Y:
                if bracket(A, D, E) not in tgt:
                    return False
    return True


def sq_graded(A, idx, plus, minus):
    """does the squaring of the odd part land in g_+, as a superalgebra with a
    Z/2-grading requires?  False also when a square leaves the algebra."""
    Ep = Ech(A, idx, plus)
    for D in plus + minus:
        if D[2] != 1:
            continue
        E, ok = op_power(A, D, 2)
        if not ok or E not in Ep:
            return False
    return True


def is_grading(A, idx, plus, minus):
    return (br_grading(A, idx, plus, minus)
            and sq_graded(A, idx, plus, minus))


def elements(A, fields):
    """every element of the F_2-span of `fields` (p = 2 only, small spans)."""
    assert A.p == 2
    out = [zero_field(A, 0)]
    for D in fields:
        out = out + [add_fields(A, X, D) for X in out]
    return out


def struct_24(A, idx, plus, minus, exhaustive=True, forget=False):
    """the (2,4)-structures of [BLLSq, Subsecs. 2.6, 2.6.2] on g = g_+ (+) g_-.

    forget = False: the SUPER version, a (2,4)|4-structure -- a 2-structure on
      (g_+)_ev, and [4] on g_- and on (g_+)_od, landing in g_+.  It presupposes
      that g is a Lie superalgebra, i.e. that the squaring lands inside.

    forget = True: the plain (2,4)-structure on the Z/2-graded Lie algebra
      F(g) of [BLLSq, Subsec. 2.4], where the squaring is forgotten -- [2] on
      the whole of g_+ and [4] on the whole of g_-.  This is the notion
      available when the squaring itself leaves the algebra.

    Returns a dict of verdicts and counts.
    """
    Ep = Ech(A, idx, plus)
    out = {"plus_closed": True, "four_in_plus": True, "add4": 0, "n4": 0,
           "nonzero4": 0, "bad_plus": [], "bad_four": [], "pool": ""}
    for D in plus:
        if forget or D[2] == 0:
            E, ok = op_power(A, D, 2)
            if not ok or E not in Ep:
                out["plus_closed"] = False
                out["bad_plus"].append(D)
        else:
            E, ok = op_power(A, D, 4)
            if not ok or E not in Ep:
                out["four_in_plus"] = False
                out["bad_four"].append(D)
    # the pool the [4]-map and the rule (add4) are tested on: the WHOLE of g_-
    # when it is small enough, otherwise the basis together with its pairwise
    # sums -- a sample, and the log says which was used (no silent cap).
    dm = len(Ech(A, idx, minus))
    if exhaustive and dm <= 5:
        pool = elements(A, minus)
        out["pool"] = f"all {2 ** dm} elements of g_-"
    else:
        pool = list(minus) + [add_fields(A, minus[i], minus[j])
                              for i in range(len(minus))
                              for j in range(i + 1, len(minus))]
        out["pool"] = f"{len(pool)} elements of g_- (basis and pairwise sums)"
    for D in pool:
        E, ok = op_power(A, D, 4)
        out["n4"] += 1
        if not ok or E not in Ep:
            out["four_in_plus"] = False
            out["bad_four"].append(D)
        elif any(E[0]) or any(E[1]):
            out["nonzero4"] += 1
    # the additivity rule (add4) of [BLLS]:
    #   (x+y)^[4] = x^[4] + y^[4] + [x,[x,[x,y]]] + [y,[y,[y,x]]]
    #                             + [x,[y,[x,y]]] + [x,y]^[2]
    bad = 0
    tested = 0
    for D in pool:
        for E in pool:
            X, okx = op_power(A, D, 4)
            Y, oky = op_power(A, E, 4)
            SUM, oks = op_power(A, add_fields(A, D, E), 4)
            if not (okx and oky and oks):
                continue
            c = bracket(A, D, E)
            t1 = bracket(A, D, bracket(A, D, bracket(A, D, E)))
            t2 = bracket(A, E, bracket(A, E, bracket(A, E, D)))
            t3 = bracket(A, D, bracket(A, E, c))
            t4, okc = op_power(A, c, 2)
            if not okc:
                continue
            rhs = X
            for T in (Y, t1, t2, t3, t4):
                rhs = add_fields(A, rhs, T)
            tested += 1
            if S.vf_vec(A, rhs, idx) != S.vf_vec(A, SUM, idx):
                bad += 1
    out["add4"] = bad
    out["add4_tested"] = tested
    return out


# --------------------------------------------------------------- the runs
def shape_name(A, om):
    """the form in the notation of the text, for the log."""
    gn = ([f"x{i+1}" for i in range(A.m)] + [f"e{j+1}" for j in range(A.n)])
    terms = []
    for mon in sorted(om):
        fun = "".join(gn[a] + (f"^({mon[a]})" if mon[a] > 1 else "")
                      for a in range(A.m + A.n) if mon[a])
        form = []
        for k in range(A.m + A.n):
            e = mon[(A.idx if k < A.m else A.idxi - A.m) + k]
            if e == 2:
                form.append(f"(d{gn[k]})^(2)")
            elif e == 1:
                form.append(f"d{gn[k]}")
        terms.append((fun + " " if fun else "") + "".join(form))
    return " + ".join(terms) if terms else "0"


def fun_name(A, mon):
    gn = ([f"x{i+1}" for i in range(A.m)] + [f"e{j+1}" for j in range(A.n)])
    s = "".join(gn[a] + (f"^({mon[a]})" if mon[a] > 1 else "")
                for a in range(A.m + A.n) if mon[a])
    return s or "1"


def analyse(A, om, log, label, exhaustive=True):
    """the whole verdict for one shape: h(omega), Span{H_f}, and the gradings."""
    assert not A.d()(om), f"the form is not closed: {label}"
    mons = A.fun_basis()
    idx = {mm: i for i, mm in enumerate(mons)}
    ev, od, bas = S.preserving(A, om)
    log(f"    {label}:  omega = {shape_name(A, om)}")
    rh = restricted(A, idx, bas, "h(omega)", log)
    H, hb = ham_basis(A, om)
    hf = [D for _, D in hb]
    rham = restricted(A, idx, hf, "Span{H_f}", log)
    byvec = {tuple(S.vf_vec(A, D, idx)): mm for mm, D in hb}

    def named(Ds):
        out = []
        for D in Ds:
            out.append(fun_name(A, byvec.get(tuple(S.vf_vec(A, D, idx)), ())))
        return out

    obstr = named(rham["bad_ev"] + rham["notfield_ev"])
    obstr_od = named(rham["bad_od"] + rham["notfield_od"])
    if obstr or obstr_od:
        log(f"        obstruction at f = {', '.join(obstr)}"
            + (f"   (odd part: {', '.join(obstr_od)})" if obstr_od else ""))
        for D in (rham["bad_ev"] + rham["bad_od"])[:3]:
            E, ok = op_power(A, D, A.p)
            log(f"        the square of H_{named([D])[0]} is "
                f"{shape_field(A, E)}, and it is "
                f"{'inner' if is_inner(A, idx, D, A.p, hf) else 'NOT inner'} "
                f"in Span(H_f)")
    out = {"h": rh, "ham": rham, "grad": {}, "superalg": not bool(
        sq_of_odd_in(A, idx, hf))}
    if A.p == 2 and hf and not out["superalg"]:
        log("        Span{H_f} is NOT closed under the squaring: it is a "
            "subalgebra of F(h(omega)), not a sub-superalgebra")
    if A.p == 2 and not rham["restricted"]:
        cands = [("the parity", None)] + [(f"deg {fun_name(A, unit_mon(A, a))}",
                                           a) for a in range(A.m + A.n)]
        for tag, a in cands:
            if a is None:
                plus, minus = parity_grading(A, hb)
            else:
                plus, minus = deg_grading(A, hb, a)
            if not minus or not plus:
                continue
            if not br_grading(A, idx, plus, minus):
                log(f"        grading by {tag}: not a grading of the algebra")
                continue
            sup = sq_graded(A, idx, plus, minus)
            st = struct_24(A, idx, plus, minus, exhaustive, forget=not sup)
            ok = st["plus_closed"] and st["four_in_plus"] and st["add4"] == 0
            kind = "(2,4)|4" if sup else "(2,4) on F(g)"
            log(f"        grading by {tag}: dim g_+ = "
                f"{len(Ech(A, idx, plus))}, dim g_- = "
                f"{len(Ech(A, idx, minus))};  g_+ closed under [2]: "
                f"{st['plus_closed']};  [4] lands in g_+: "
                f"{st['four_in_plus']};  (add4) fails "
                f"{st['add4']}/{st['add4_tested']};  [4] non-zero on "
                f"{st['nonzero4']}/{st['n4']} ({st['pool']})  "
                f"==> {kind}-structure: "
                f"{'YES' if ok else 'no'}"
                + ("" if ok or st["plus_closed"] else
                   "  [only a (2,-)-structure is conceivable, and not even "
                   "that: g_+ is not closed]"))
            out["grad"][tag] = (ok, sup, st)
    return out


def unit_mon(A, a):
    e = [0] * A.N
    e[a if a < A.m else A.ixi + (a - A.m)] = 1
    return tuple(e)


def shape_field(A, D):
    gn = ([f"x{i+1}" for i in range(A.m)] + [f"e{j+1}" for j in range(A.n)])
    parts = []
    for i in range(A.m):
        if D[0][i]:
            parts.append(f"({fun_poly(A, D[0][i])}) d/d{gn[i]}")
    for j in range(A.n):
        if D[1][j]:
            parts.append(f"({fun_poly(A, D[1][j])}) d/d{gn[A.m + j]}")
    return " + ".join(parts) if parts else "0"


def fun_poly(A, f):
    return " + ".join((f"{c} " if c % A.p != 1 else "") + fun_name(A, mm)
                      for mm, c in sorted(f.items()) if c % A.p)


def census_shapes(M):
    """the normal shapes of the census, class by class, for m + n = M, u = 1.

    The classes are those of the census engine; the representative is the
    shortest one in the orbit, i.e. exactly the shape the tables of the text
    print.  The census modules open their own logs when imported, so they are
    saved and put back: this engine must not overwrite another one's record.
    """
    logs = {}
    for nm in ("scensus.log", "shomog.log"):
        pth = os.path.join(HERE, nm)
        if os.path.exists(pth):
            with open(pth, "rb") as f:
                logs[pth] = f.read()
    from scensus import build, elementary_autos, gen_matrix, show
    from shomog import homogeneous_nondeg
    from ssplit import orbit_of
    out = []
    for n in range(0, M + 1):
        m = M - n
        A = S.Om(m, n, 2, (1,) * m, dh=2, dxh=2)
        B2, idx, zbas, gram_bits, G = build(A)
        maps = [gen_matrix(A, phi, B2, idx)
                for phi in elementary_autos(A, graded=True)]
        for par in (0, 1):
            el, _ = homogeneous_nondeg(A, zbas, gram_bits, G, par)
            if not el:
                continue
            left = set(el)
            reps = []
            while left:
                s = next(iter(left))
                orb = orbit_of(s, maps)
                reps.append((min(orb, key=lambda v: (bin(v).count("1"), v)),
                             len(orb)))
                left -= orb
            for rep, size in sorted(reps, key=lambda t: (bin(t[0]).count("1"),
                                                         t[0])):
                out.append((m, n, par, size, show(A, rep, B2), A))
    for pth, data in logs.items():
        with open(pth, "wb") as f:
            f.write(data)
    return out


def run_census(log):
    log("")
    log("=" * 78)
    log("A.  p = 2, u = 1: the restricted structures of the census classes")
    log("=" * 78)
    tally = {"h24": 0, "ham_restr": 0, "ham_notsuper": 0, "g24": 0, "tot": 0}
    rows = []
    for M in (1, 2, 3):
        for (m, n, par, size, om, A) in census_shapes(M):
            log("")
            res = analyse(A, om, log, f"O({m}|{n};1), "
                          f"{'odd' if par else 'even'} class, |orbit| = {size}")
            tally["tot"] += 1
            tally["h24"] += 1 if res["h"]["restricted"] else 0
            tally["ham_restr"] += 1 if res["ham"]["restricted"] else 0
            mons = A.fun_basis()
            idx = {mm: i for i, mm in enumerate(mons)}
            H, hb = ham_basis(A, om)
            if sq_of_odd_in(A, idx, [D for _, D in hb]):
                tally["ham_notsuper"] += 1
            g24 = [t for t, (ok, _, _) in res["grad"].items() if ok]
            tally["g24"] += 1 if g24 else 0
            rows.append((m, n, par, om, A, res, g24))
    log("")
    log(f"  TOTAL: {tally['tot']} classes;  h(omega) restricted (2|4) in "
        f"{tally['h24']};  Span(H_f) restricted in {tally['ham_restr']};  "
        f"Span(H_f) not closed under the squaring in {tally['ham_notsuper']};  "
        f"a (2,4)|4-structure in {tally['g24']}")
    return rows


def omega_I(A):
    """omega_I = sum_a (dg_a)^{(2)}, closed and non-degenerate for every (m|n)
    and every u, by Theorem Tbrev 2) of the text."""
    om = {}
    for i in range(A.m):
        om = A.add(om, A.dx(i, 2))
    for j in range(A.n):
        om = A.add(om, A.dxi(j, 2))
    return om


def omega_alt(A):
    """the alternating standard shape: the x's paired among themselves, the
    xi's among themselves; it exists when m and n are both even."""
    om = {}
    for i in range(A.m // 2):
        om = A.add(om, A.mul(A.dx(2 * i), A.dx(2 * i + 1)))
    for j in range(A.n // 2):
        om = A.add(om, A.mul(A.dxi(2 * j), A.dxi(2 * j + 1)))
    return om


def omega_ev(A):
    """the even shape of the text, eq. (omev): symplectic in the even
    indeterminates, orthogonal in the odd ones.  p > 2."""
    k = A.m // 2
    om = {}
    for i in range(k):
        om = A.add(om, A.mul(A.dx(i), A.dx(k + i)))
    for j in range(A.n):
        om = A.add(om, A.dxi(j, 2))
    return om


def omega_od(A):
    """the periplectic shape sum dx_i dxi_i over O(n|n;u)."""
    om = {}
    for i in range(A.n):
        om = A.add(om, A.mul(A.dx(i), A.dxi(i)))
    return om


def omega_Xi(A):
    """the exceptional periplectic shape of characteristic 3:
    omega_od + xi_1 (dxi_1)^{(2)}."""
    return A.add(omega_od(A), A.mul(A.xi(0), A.dxi(0, 2)))


def run_shear(log):
    log("")
    log("=" * 78)
    log("B.  p = 2, an arbitrary shearing vector")
    log("=" * 78)
    for (m, n) in ((1, 0), (2, 0), (1, 1), (2, 1), (1, 2), (3, 0)):
        for u in ({(1,) * m, (2,) + (1,) * (m - 1)} if m else {()}):
            u = tuple(u)
            if len(u) != m:
                continue
            A = S.Om(m, n, 2, u, dh=2, dxh=2)
            for nm, om in (("omega_I", omega_I(A)),):
                nd, C = S.nondegenerate(A, om)
                if not nd:
                    continue
                log("")
                analyse(A, om, log, f"O({m}|{n};{u}), {nm}")
                mons = A.fun_basis()
                idx = {mm: i for i, mm in enumerate(mons)}
                ev, od, bas = S.preserving(A, om)
                for a in range(m):
                    fs = [{}] * m
                    fs[a] = A.one()
                    D = (tuple(fs), tuple([{}] * n), 0)
                    E, ok = op_power(A, D, 2)
                    if not ok:
                        log(f"        d/dx{a+1} lies in h(omega): "
                            f"{D in Ech(A, idx, bas)};  its square is not a "
                            f"vector field (it is the divided power derivation "
                            f"of order 2), and (ad)^2 is inner in h(omega): "
                            f"{is_inner(A, idx, D, 2, bas)}")


def run_bigp(log):
    log("")
    log("=" * 78)
    log("C.  p > 2: the even shape, the antibracket, and the deformed one")
    log("=" * 78)
    for (m, n, p, u, which) in ((2, 1, 3, (1, 1), "ev"), (2, 2, 3, (1, 1), "ev"),
                                (2, 1, 5, (1, 1), "ev"), (4, 1, 3, (1,) * 4, "ev"),
                                (2, 0, 3, (2, 1), "ev"),
                                (1, 1, 3, (1,), "od"), (2, 2, 3, (1, 1), "od"),
                                (1, 1, 5, (1,), "od"), (1, 1, 3, (2,), "od"),
                                (1, 1, 3, (1,), "Xi"), (2, 2, 3, (1, 1), "Xi"),
                                (1, 1, 3, (2,), "Xi"), (1, 1, 5, (1,), "Xi")):
        A = S.Om(m, n, p, u, dh=2, dxh=2)
        om = {"ev": omega_ev, "od": omega_od, "Xi": omega_Xi}[which](A)
        nd, C = S.nondegenerate(A, om)
        if not nd or A.d()(om):
            log(f"    O({m}|{n};{u}), p = {p}, shape {which}: "
                f"{'degenerate' if not nd else 'NOT CLOSED'}, skipped "
                f"(the exceptional shape is closed for p = 3 only, "
                f"Theorem Texc)")
            continue
        log("")
        analyse(A, om, log, f"O({m}|{n};{u}), p = {p}, shape {which}")
        mons = A.fun_basis()
        idx = {mm: i for i, mm in enumerate(mons)}
        H, hb = ham_basis(A, om)
        hf = [D for _, D in hb]
        odd = [D for _, D in hb if D[2] == 1]
        E2 = Ech(A, idx, [])
        nz = 0
        for D in odd:
            Q, ok = op_power(A, D, 2)
            if any(Q[0]) or any(Q[1]):
                nz += 1
                E2.add(Q)
        # the span of the WHOLE image of the squaring, not of its values on a
        # basis: the squaring is quadratic, so its image spans the basis
        # squares TOGETHER with the polarizations [x,y] = (x+y)^2 - x^2 - y^2
        E2full = Ech(A, idx, [])
        for D in odd:
            Q, ok = op_power(A, D, 2)
            E2full.add(Q)
        for i in range(len(odd)):
            for j in range(i + 1, len(odd)):
                E2full.add(bracket(A, odd[i], odd[j]))
        log(f"      the squaring of the odd part of Span(H_f): non-zero on "
            f"{nz} of {len(odd)} basis elements; the span of the basis "
            f"squares is of dimension {len(E2)}, that of the whole image "
            f"(with the polarizations) {len(E2full)}")
        E4 = Ech(A, idx, [])
        nz4 = 0
        for mm, D in hb:
            if D[2] != 1:
                continue
            Q, ok = op_power(A, D, 2 * p)
            if not ok:
                continue
            if any(Q[0]) or any(Q[1]):
                nz4 += 1
                E4.add(Q)
        log(f"      the [2p]-map on the odd part of Span(H_f): non-zero on "
            f"{nz4} basis elements, its span is of dimension {len(E4)}")


# ------------------------------------------------- the obstruction map sigma
def sigma_map(A, om, log=None):
    """the obstruction to restrictedness of Span{H_f}, as a map into H^1.

    For f in O/K put sigma(f) := the class of iota_{(H_f)^p} omega in H^1.  It
    is defined because (H_f)^p lies in h(omega), which is Pi^{p(omega)+1} Z^1
    by Prop. Ph; it vanishes exactly when (H_f)^p is Hamiltonian.  Since
    (x+y)^[p] - x^[p] - y^[p] is a sum of Lie polynomials in x, y -- brackets,
    hence Hamiltonian fields -- sigma is ADDITIVE, and sigma(cf) = c^p sigma(f).

    So the whole obstruction is one semilinear map O/K -> H^1, and Span{H_f} is
    restricted if and only if sigma = 0.  Returns (rank, dim H^1, additivity
    failures, equivalence check).
    """
    from exactlin import rank as _rank
    p = A.p
    B1 = A.basis(1)
    i1 = {mm: i for i, mm in enumerate(B1)}
    # the exact 1-forms, in the full basis of Omega^1 (both parities: for
    # p > 2 the p-th power of H_f has the parity of H_f, so both occur)
    dd = A.d()
    cob = [list(S.vec(A, dd({mm: 1}), i1)) for mm in A.fun_basis()]
    H, hb = ham_basis(A, om)
    mons = A.fun_basis()
    idx = {mm: i for i, mm in enumerate(mons)}
    hf = [D for _, D in hb]
    ech = Ech(A, idx, hf)
    base = _rank(cob, p) if cob else 0
    rows, nonham = [], 0
    for mm, D in hb:
        E, ok = op_power(A, D, p if D[2] == 0 else 2)
        if not ok:
            rows.append(None)
            continue
        if E not in ech:
            nonham += 1
        v = S.vec(A, A.iota(E)(om), i1)
        rows.append(list(v))
    good = [r for r in rows if r is not None]
    # the rank of sigma = dim of (image + exact) / exact
    rk = (_rank(cob + good, p) - base) if good else 0
    # additivity, on all pairs of the basis of Span{H_f}
    bad = 0
    for a in range(len(hb)):
        for b in range(a + 1, len(hb)):
            D, E = hb[a][1], hb[b][1]
            if D[2] != E[2]:
                continue                       # only homogeneous sums
            k = p if D[2] == 0 else 2
            X, okx = op_power(A, D, k)
            Y, oky = op_power(A, E, k)
            Z, okz = op_power(A, add_fields(A, D, E), k)
            if not (okx and oky and okz):
                continue
            W = bracket(A, D, E)
            if p == 2:
                cand = add_fields(A, add_fields(A, X, Y), W)
                if S.vf_vec(A, cand, idx) != S.vf_vec(A, Z, idx):
                    bad += 1
            else:
                # sigma is additive even though the [p]-map is not: the
                # correction is a sum of brackets, hence Hamiltonian, hence
                # of class zero.  Test that the CLASSES add.
                vx = S.vec(A, A.iota(X)(om), i1)
                vy = S.vec(A, A.iota(Y)(om), i1)
                vz = S.vec(A, A.iota(Z)(om), i1)
                s = [(vx[t] + vy[t] - vz[t]) % p for t in range(len(vz))]
                if _rank(cob + [s], p) != base:
                    bad += 1
    if log:
        log(f"      sigma: rank {rk} of dim H^1 = "
            f"{A.m + A.n};  (H_f)^{p} non-Hamiltonian for {nonham} basis "
            f"elements;  additivity fails on {bad} pairs;  "
            f"Span(H_f) restricted iff sigma = 0: "
            f"{'consistent' if (rk == 0) == (nonham == 0) else 'MISMATCH'}")
    return rk, bad, nonham


def run_sigma(log):
    log("")
    log("=" * 78)
    log("D.  the obstruction map sigma: O/K -> H^1")
    log("=" * 78)
    for M in (1, 2, 3):
        for (m, n, par, size, om, A) in census_shapes(M):
            log(f"    O({m}|{n};1), {'odd' if par else 'even'}:  "
                f"omega = {shape_name(A, om)}")
            sigma_map(A, om, log)
    for (m, n, p, u, which) in ((2, 1, 3, (1, 1), "ev"), (1, 1, 3, (1,), "od"),
                                (1, 1, 3, (1,), "Xi"), (2, 2, 3, (1, 1), "od"),
                                (2, 2, 3, (1, 1), "Xi"),
                                (2, 1, 5, (1, 1), "ev"), (1, 1, 5, (1,), "od")):
        A = S.Om(m, n, p, u, dh=2, dxh=2)
        om = {"ev": omega_ev, "od": omega_od, "Xi": omega_Xi}[which](A)
        if A.d()(om) or not S.nondegenerate(A, om)[0]:
            continue
        log(f"    O({m}|{n};{u}), p = {p}, shape {which}")
        sigma_map(A, om, log)


if __name__ == "__main__":
    t0 = time.time()
    if len(sys.argv) > 1 and sys.argv[1] == "sigma":
        log = Log(os.path.join(HERE, "srsigma.log"))
        log("The obstruction map sigma of the Hamiltonian subalgebra")
        run_sigma(log)
        log(f"\ntotal {time.time() - t0:.1f} s")
        sys.exit(0)
    log = Log(os.path.join(HERE, "srestr.log"))
    log("Restricted structures of the algebras preserving the normal shapes")
    log(f"engine run of {time.strftime('%Y-%m-%d %H:%M')}")
    run_census(log)
    run_shear(log)
    run_bigp(log)
    log("")
    log(f"total {time.time() - t0:.1f} s")