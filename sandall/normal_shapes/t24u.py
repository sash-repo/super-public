#!/usr/bin/env python3
"""The standard non-alternating shape in an ODD number of indeterminates, and
its restricted structures for an ARBITRARY shearing vector.

The computation behind the Theorem on the restrictedness for an arbitrary
shearing vector, with the honest operator powers of srestr.py.  A square may
not be read off the values of D^2 on the GENERATORS: a derivation of O(m;u) is
not determined by those, and that shortcut would make the algebra look
"restricted for every u", whereas restrictedness holds for u = One alone.

Usage:  python3 t24u.py [u ...]      e.g.  python3 t24u.py 1,1,1 2,1,1

The shape here is the one of that Theorem,

    omega_I = (dx)^{(2)} + sum_{1<=i<=n} du_i du_{~i},   m = 2n+1,

x being the non-alternating indeterminate, of height 2^{k_0}.  Note that this
is the SAME shape as the omega_I of the present text, sum_a (dg_a)^{(2)}: over
a perfect field the two constant Gram matrices are congruent, being both
non-alternating and non-degenerate.  They are congruent by a LINEAR change,
which is an automorphism of O(m;u) only if the heights it mixes are equal; so
for u =/= One the two presentations are genuinely different data, and that is
why this run is not a repetition of run_shear.

Everything is exact over F_2.  Output: t24u.log.
"""
import sys, os, itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sform as S
import srestr as R

HERE = os.path.dirname(os.path.abspath(__file__))


def omega_std(A):
    """(dx_1)^{(2)} + sum_i dx_{i+1} dx_{n+i+1} over m = 2n+1 indeterminates."""
    assert A.n == 0 and A.m % 2 == 1
    n = (A.m - 1) // 2
    om = A.dx(0, 2)
    for i in range(n):
        om = A.add(om, A.mul(A.dx(1 + i), A.dx(1 + n + i)))
    return om


def predicted_obstructions(A, k):
    """eq. (obstru) of the text: the 2n functions x u^{(2^{k_u - 1})}
    and, if k_0 > 1, the function x^{(2^{k_0-1}+1)}."""
    n = (A.m - 1) // 2
    out = []
    for a in range(1, A.m):
        e = [0] * A.N
        e[0] = 1
        e[a] = 2 ** (k[a] - 1)
        out.append(tuple(e))
    if k[0] > 1:
        e = [0] * A.N
        e[0] = 2 ** (k[0] - 1) + 1
        out.append(tuple(e))
    return out


def run(log):
    # the shearing vectors are given on the command line, e.g. 1,1,1 2,1,1;
    # by default: every case the text quotes (m = 3 and m = 5)
    cases = [tuple(int(c) for c in a.split(",")) for a in sys.argv[1:]] or [
        (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 2),
        (3, 1, 1), (1, 3, 1), (1, 3, 3), (2, 3, 1),
        (1,) * 5, (2, 1, 1, 1, 1), (1, 2, 1, 2, 1)]

    for u in cases:
        m = len(u)
        A = S.Om(m, 0, 2, u, dh=2, dxh=2)
        om = omega_std(A)
        assert not A.d()(om), "the form is not closed"
        nd, _ = S.nondegenerate(A, om)
        log("")
        log("=" * 78)
        log(f"m = {m}, u = {u}, |u| = {sum(u)};  non-degenerate: {nd}")
        out = R.analyse(A, om, log, f"O({m};{u}), omega_std")

        # dimension check: dim h = 2^|u| - 1 + m,  dim Span{H_f} = 2^|u| - 1
        mons = A.fun_basis()
        idx = {mm: i for i, mm in enumerate(mons)}
        ev, od, bas = S.preserving(A, om)
        dh = len(R.Ech(A, idx, bas))
        H, hb = R.ham_basis(A, om)
        dham = len(R.Ech(A, idx, [D for _, D in hb]))
        pred = 2 ** sum(u)
        log(f"      dim h(omega) = {dh}, predicted 2^|u| - 1 + m = "
            f"{pred - 1 + m}: {dh == pred - 1 + m};  "
            f"dim Span(H_f) = {dham}, predicted 2^|u| - 1 = {pred - 1}: "
            f"{dham == pred - 1}")

        # the obstruction list against eq. (obstru)
        hf = [D for _, D in hb]
        rham = R.restricted(A, idx, hf, "Span{H_f}", log, verbose=False)
        byvec = {tuple(S.vf_vec(A, D, idx)): mm for mm, D in hb}
        got = set()
        for D in rham["bad_ev"] + rham["notfield_ev"] + rham["bad_od"] + \
                rham["notfield_od"]:
            got.add(byvec.get(tuple(S.vf_vec(A, D, idx)), ()))
        want = set(predicted_obstructions(A, u))
        log(f"      obstructions: {len(got)} found, {len(want)} predicted by "
            f"(obstru); the two sets agree: {got == want}")
        if got != want:
            log(f"        found only:     "
                f"{[R.fun_name(A, f) for f in sorted(got - want)]}")
            log(f"        predicted only: "
                f"{[R.fun_name(A, f) for f in sorted(want - got)]}")

        # the verdict of item 4) of the Theorem (M = 3, 5): a (2,4)-structure
        # iff every indeterminate of omega' has height 1 and k_0 <= 2
        tag = f"deg {R.fun_name(A, R.unit_mon(A, 0))}"
        got24 = out["grad"].get(tag, (False,))[0]
        pred = all(h == 1 for h in u[1:]) and u[0] <= 2
        log(f"      (2,4)-structure for the grading by deg x: {got24};  "
            f"N = {u}, so item 4) predicts {pred}: agree = {got24 == pred}")


if __name__ == "__main__":
    log = R.Log(os.path.join(HERE, "t24u.log"))
    run(log)
    log("")
    log("done.")
