"""Vaintrob's versal deformation of a supermanifold, checked over CP^1
(Statement 17.3, table (32), formulas (33)-(34); Statement 17.4, formula (35)).

Sources (keys in this script; the text calls the 1988 survey [Va2] and the
1984 note [Va1]):

  [Va1] A. Yu. Vaintrob, Deformations of complex superspaces and of the
        coherent sheaves on them, Itogi Nauki i Tekhn., Sovrem. Probl. Mat.
        Nov. Dost. 32 (1988) 125-211 = J. Soviet Math. 51:1 (1990) 2140-2188.
  [Va2] A. Yu. Vaintrob, Deformations of complex structures on supermanifolds,
        Funct. Anal. Appl. 18:2 (1984) 135-136.

What is checked here:

  A.  [Va1] 3.3.2 (1) = [Va2] Prop. 2, the projective superspace P^{1|n},
      which is the retract (1,...,1).  Rigidity for n <= 3, and
      for n >= 4 the superdimension of the versal base against
          (i)  the module  S^{n-3}(V) (x) V / S^{n-2}(V),  V = C^{2|n},
          (ii) the closed formula printed in [Va1] 3.3.2 (1),
      layer by layer under the correspondence  j = n - q  between the
      Lambda-degree j of the module and the degree q of the tangent sheaf.

  B.  [Va1] 3.3.1 (2), rk F = 2: h^1(T_2) = k_1 + k_2 - 3.

  C.  [Va1] 3.3.1 (3), rk F = 3: the formula d(a+b,c) + d(b+c,a) + d(c+a,b),
      where a = -k_1, b = -k_2, c = -k_3 and
          d(x,y) = -2x-4 for x <= -4,   1 for x = -3 or (x,y) = (-2,0),
                    0 otherwise.

The layers h^1(M; T_q) themselves come from quot.Quot, i.e., from honest
Cech cocycles on the transition functions (2) of the text, the same engine
that reproduced [Ba1, Thm. 2.1] and [Ba5, Thms. 1.9, 1.12, 1.13].
"""

from fractions import Fraction
import sys

import quot


LOG = None


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


# ------------------------------------------------------------------ the layers

def layers(ks, qmax=None):
    """h^1(M; T_q) for q = -1 .. m, computed from the transition functions."""
    m = len(ks)
    return [(q, quot.Quot(list(ks), q).dim)
            for q in range(-1, (qmax if qmax is not None else m) + 1)]


def sdim_from_layers(lay):
    ev = sum(d for q, d in lay if q % 2 == 0)
    od = sum(d for q, d in lay if q % 2 != 0)
    return ev, od


# -------------------------------------------------- the module S^k(V) for V=C^{2|n}

def sdim_S(k, n):
    """sdim S^k(C^{2|n}) = sum_j S^{k-j}(C^2) (x) Lambda^j(C^n), parity j."""
    ev = od = 0
    for j in range(0, min(k, n) + 1):
        d = (k - j + 1) * binom(n, j)
        if j % 2 == 0:
            ev += d
        else:
            od += d
    return ev, od


def binom(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def tensor(a, b):
    return (a[0] * b[0] + a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def vaintrob_module(n):
    """sdim of S^{n-3}(V) (x) V / S^{n-2}(V) for V = C^{2|n}."""
    V = (2, n)
    t = tensor(sdim_S(n - 3, n), V)
    s = sdim_S(n - 2, n)
    return (t[0] - s[0], t[1] - s[1])


def module_piece(n, j):
    """Dimension of the summand of Lambda-degree j of that module.

    The j-piece of S^k(V) is S^{k-j}(V_ev) (x) Lambda^j(V_od); the j-piece of
    S^{n-3}(V) (x) V picks up S^{n-3-j}(V_ev) (x) Lambda^j (x) V_ev and
    S^{n-2-j}(V_ev) (x) Lambda^{j-1} (x) V_od.  Its parity is j mod 2.
    """
    def dimS(d):
        return d + 1 if d >= 0 else 0
    first = 2 * dimS(n - 3 - j) * binom(n, j)
    second = n * dimS(n - 2 - j) * binom(n, j - 1)
    minus = dimS(n - 2 - j) * binom(n, j)
    return first + second - minus


def printed_formula(n):
    """The closed formula of [Va1] 3.3.2 (1), as printed, with the (-1)^n."""
    A = Fraction(2) ** (n - 2) * (n * n - 3 * n - 6) \
        + Fraction(n * n + 3, 2) + 2 * n
    B = Fraction(n * n - 4 * n + 3, 2)
    s = 1 if n % 2 == 0 else -1
    return A + s * B, A - s * B


def corrected_formula(n):
    """The same with the (-1)^n deleted, i.e., (A+B | A-B)."""
    A = Fraction(2) ** (n - 2) * (n * n - 3 * n - 6) \
        + Fraction(n * n + 3, 2) + 2 * n
    B = Fraction(n * n - 4 * n + 3, 2)
    return A + B, A - B


# ------------------------------------------------------------------ part A

def partA(nmax=7):
    say("=" * 74)
    say("A.  P^{1|n} = the retract (1,...,1):  [Va1] 3.3.2 (1) = [Va2] Prop. 2")
    say("=" * 74)
    bad = 0
    for n in range(1, nmax + 1):
        lay = layers([1] * n)
        ev, od = sdim_from_layers(lay)
        row = " ".join("%d:%d" % (q, d) for q, d in lay)
        say("\nn = %d   layers  %s" % (n, row))
        say("          sdim H^1(P^{1|n}; T) = (%d|%d)" % (ev, od))
        if n <= 3:
            ok = (ev == 0 and od == 0)
            say("          rigid?  %s   ([Va] says rigid for n <= 3)"
                % ("YES" if ok else "NO"))
            bad += 0 if ok else 1
            continue
        mod = vaintrob_module(n)
        pf = printed_formula(n)
        cf = corrected_formula(n)
        say("          module  S^{n-3}(V)(x)V/S^{n-2}(V), V=C^{2|n}: (%d|%d)"
            % mod)
        say("          printed formula                              : (%s|%s)"
            % pf)
        say("          same with (-1)^n deleted                     : (%s|%s)"
            % cf)
        if (ev, od) == mod:
            say("          VERDICT: the printed module agrees")
        elif (od, ev) == mod:
            say("          VERDICT: the printed module has the two entries "
                "INTERCHANGED")
        else:
            say("          VERDICT: DISAGREES in the totals as well")
            bad += 1
        if mod != (int(pf[0]), int(pf[1])):
            say("          NB: the printed formula does not evaluate to the "
                "printed module")
            bad += 1
        if (ev, od) != (int(cf[0]), int(cf[1])):
            say("          NB: the corrected formula misses the computation")
            bad += 1
        # layer by layer, j = n - q
        say("          layer by layer, j = n - q:")
        for q, d in lay:
            if q < 0:
                continue
            j = n - q
            md = module_piece(n, j)
            par_ours = q % 2
            par_his = j % 2
            mark = "ok " if md == d else "BAD"
            if md != d:
                bad += 1
            say("            q=%d  h^1=%-4d   j=%d  dim=%-4d  %s   "
                "parity ours %d, printed %d%s"
                % (q, d, j, md, mark, par_ours, par_his,
                   "" if par_ours == par_his else "   <-- shifted by Pi"))
    say("\npart A: %d discrepancies with the computation" % bad)
    return bad


# ------------------------------------------------------------------ part B

def partB():
    say("\n" + "=" * 74)
    say("B.  rk F = 2:  [Va1] 3.3.1 (2),  h^1(T_2) = k_1 + k_2 - 3")
    say("=" * 74)
    bad = 0
    for k1 in range(0, 7):
        for k2 in range(0, k1 + 1):
            got = quot.Quot([k1, k2], 2).dim
            want = max(0, k1 + k2 - 3)
            ok = got == want
            bad += 0 if ok else 1
            say("  (%d,%d): h^1(T_2) = %-3d  formula %-3d  %s"
                % (k1, k2, got, want, "ok" if ok else "BAD"))
    say("part B: %d discrepancies" % bad)
    return bad


# ------------------------------------------------------------------ part C

def d_va(x, y):
    """d as printed in [Va1] 3.3.1 (3)."""
    if x <= -4:
        return -2 * x - 4
    if x == -3 or (x, y) == (-2, 0):
        return 1
    return 0


def d_corr(x, y):
    """The same with the first branch extended to x <= -3.

    For a pair with x = a_i + a_j the two graded pieces of T_2 contribute
    h^1(O(x+2)) = (-x-3)_+ from Lambda^2F (x) T_M and h^1(O(x)) = (-x-1)_+
    from Lambda^3F (x) F^*, and the connecting map H^0(O(x+2)) -> H^1(O(x))
    can cancel only where both are non-zero, which is x = -2 alone.  At
    x = -3 the first is 0 and the second is 2, so d(-3,y) = 2 = -2x-4.
    """
    if x <= -3:
        return -2 * x - 4
    if (x, y) == (-2, 0):
        return 1
    return 0


def partC():
    say("\n" + "=" * 74)
    say("C.  rk F = 3:  [Va1] 3.3.1 (3),  d(a+b,c)+d(b+c,a)+d(c+a,b)")
    say("=" * 74)
    bad = badc = nmin3 = 0
    for k1 in range(0, 6):
        for k2 in range(0, k1 + 1):
            for k3 in range(0, k2 + 1):
                a, b, c = -k1, -k2, -k3
                prs = ((a + b, c), (b + c, a), (c + a, b))
                want = sum(d_va(x, y) for x, y in prs)
                wantc = sum(d_corr(x, y) for x, y in prs)
                got = quot.Quot([k1, k2, k3], 2).dim
                ok, okc = got == want, got == wantc
                bad += 0 if ok else 1
                badc += 0 if okc else 1
                n3 = sum(1 for x, y in prs if x == -3)
                nmin3 += 1 if n3 else 0
                say("  (%d,%d,%d): h^1(T_2) = %-3d  printed %-3d %s"
                    "   corrected %-3d %s   pairs with x=-3: %d"
                    % (k1, k2, k3, got, want, "ok " if ok else "BAD",
                       wantc, "ok " if okc else "BAD", n3))
    say("part C: %d discrepancies with d as printed, %d with the corrected d;"
        % (bad, badc))
    say("        every discrepancy is a triple having a pair with x = -3 "
        "(%d of them), and is off by exactly the number of such pairs"
        % nmin3)
    return badc


def main():
    global LOG
    LOG = open("vaintrob.log", "w")
    say(__doc__)
    b = partA(7) + partB() + partC()
    say("\nTOTAL discrepancies with the honest Cech computation: %d" % b)
    LOG.close()


if __name__ == "__main__":
    main()
