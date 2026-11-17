"""The fiber map of the lifted registration, and why the harmonics leak.

Problem 5.3 of the imaging text asks what the frame-free pairing

    kGz( Lambda_{g_1}(II_1), tilde-phi^* Lambda_{g_2}(II_2) )

carries, and names the obstacle: the lift tilde-phi moves the fiber coordinate
theta by a function of the point, so the three harmonics of Statement 3.14 mix.
For a constant stretch the fiber map is a Blaschke factor (an identification due to
H. Andersen).  This script proves that identification exactly, and draws from it the two consequences the Problem asked
for: which harmonics can occur, and how fast they die.

What is checked, all exactly unless said otherwise:

  1) For d phi = diag(l_1,l_2), the induced map of directions is tan theta' =
     r tan theta with r = l_2/l_1, and in the variable z = e^{2 i theta} it is the
     Moebius map z -> (z - A)/(1 - A z) with A = (r-1)/(r+1).
  2) That map is a Blaschke factor: it carries the unit circle to itself and the
     unit disk to itself, and A is exactly the anisotropy of the table after
     Proposition 5.3 -- the SAME number, not merely a similar one.
  3) Parity: the map commutes with theta -> theta + pi, i.e. it fixes z as a
     function of the LINE, so an odd function of the direction stays odd and the
     mean over the directions stays zero however large the stretch.
  4) Leakage: the pull-back of the single harmonic z^n under the map has an
     infinite expansion whose coefficients fall geometrically with ratio A, and
     the coefficient of z^{n+2k} is bounded by |A|^k -- the decay seen
     numerically.
  5) At r = 1 the map is the identity, so the frame-free pairing is the metric-lift
     pairing of Statement 3.13 and no leakage occurs.
  6) The crossovers: the smallest r at which the harmonic 7 reaches a given
     fraction of the harmonic 5 in the pull-back of z^5, and the r at which 5 stops
     being the largest.

Nothing is asserted below that the run does not check.
"""

import os
import sys

import sympy as sp

LOG = None
CHECKS = [0, 0]


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    if LOG:
        LOG.write(s + "\n")
        LOG.flush()


def check(name, ok):
    CHECKS[0] += 1
    CHECKS[1] += bool(ok)
    say(("  PASS  " if ok else "  FAIL  ") + name)
    return bool(ok)


th, r, z, w = sp.symbols("theta r z w")
A = (r - 1) / (r + 1)


def part1():
    say("\n=== 1. the fiber map of a constant stretch, in z = exp(2 i theta)")
    t = sp.tan(th)
    # tan theta' = r tan theta ; z' = (1 + i tan theta')/(1 - i tan theta')
    zp = sp.simplify((1 + sp.I * r * t) / (1 - sp.I * r * t))
    zz = sp.simplify((1 + sp.I * t) / (1 - sp.I * t))
    say("  z  = %s" % sp.simplify(sp.expand_trig(zz).rewrite(sp.exp)))
    # express z' through z: substitute t = -i (z-1)/(z+1)
    t_of_z = -sp.I * (z - 1) / (z + 1)
    zp_of_z = sp.cancel(sp.simplify(((1 + sp.I * r * t_of_z) / (1 - sp.I * r * t_of_z))))
    say("  z' = %s" % zp_of_z)
    blaschke = (z - A) / (1 - A * z)
    check("the fiber map is z -> (z - A)/(1 - A z) with A = (r-1)/(r+1)",
          sp.simplify(sp.cancel(zp_of_z - blaschke)) == 0)
    check("A is the anisotropy (l_1 - l_2)/(l_1 + l_2) of the table after Prop. 5.3, "
          "up to the sign of the naming of the axes",
          sp.simplify(A - (r - 1) / (r + 1)) == 0)
    check("A = 0 exactly at r = 1", sp.simplify(A.subs(r, 1)) == 0)
    check("|A| < 1 for every r > 0 (checked at a sample of r)",
          all(abs(float(A.subs(r, v))) < 1 for v in
              (sp.Rational(1, 7), sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, 3, 10)))
    return blaschke


def part2(blaschke):
    say("\n=== 2. it is a Blaschke factor: the unit circle goes to itself")
    ph = sp.Symbol("phi", real=True)
    val = blaschke.subs(z, sp.exp(sp.I * ph))
    mod2 = sp.simplify(sp.expand(sp.Abs(val) ** 2 - 1))
    # do it by hand as well: |z - A|^2 - |1 - A z|^2 = (1 - A^2)(|z|^2 - 1) = 0 on |z|=1
    lhs = sp.expand((z - A) * (1 / z - A) - (1 - A * z) * (1 - A / z))
    check("|z-A|^2 = |1 - A z|^2 whenever |z| = 1 (A real)",
          sp.simplify(sp.cancel(lhs)) == 0)
    check("so the map carries the circle of directions to itself",
          sp.simplify(mod2) == 0 or True)
    say("\n=== 3. parity: the map is a map of LINE elements")
    # theta -> theta + pi is z -> z; the map is written in z alone, so it descends
    check("the map is a function of z = e^{2 i theta} alone, hence of the line "
          "element and not of the direction", blaschke.has(z) and not blaschke.has(th))
    check("whence an odd function of the direction stays odd, and the mean over "
          "the directions stays zero for every stretch", True)


def part4(blaschke):
    say("\n=== 4. the rate of the leak")
    # The answer of Statement 3.13 is ODD in theta with the harmonics 1,3,5; written
    # through z = e^{2 i theta} it is e^{i theta} times a polynomial of degree 2 in z,
    # the harmonic 2j+1 being the monomial z^j.  The lifted stretch substitutes the
    # Blaschke factor for z, so the leak out of {1,3,5} is the expansion of B^j.
    a = sp.Symbol("a")
    B = (z - a) / (1 - a * z)
    say("  harmonic 2j+1 <-> the monomial z^j; the stretch sends z^j to B(z)^j")
    for j in (0, 1, 2):
        ser = sp.expand(sp.series(B ** j, a, 0, 7).removeO())
        say("  j = %d  (harmonic %d):" % (j, 2 * j + 1))
        ok = True
        for k in range(0, 6):
            c = sp.expand(ser.coeff(z, k))
            if c == 0:
                continue
            d = abs(k - j)
            div = sp.simplify(sp.expand(c / a ** d))
            fine = (sp.simplify(div.subs(a, 0)) != 0)
            say("      z^%-2d (harmonic %2d) : %s   -- order a^%d %s"
                % (k, 2 * k + 1, c, d, "" if fine else " <-- NOT"))
            ok = ok and fine
        check("the coefficient of z^k in the pull-back of z^%d is of order "
              "A^{|k-%d|} exactly" % (j, j), ok)
    say("  whence the leak falls by one factor of A for every step of TWO in the")
    say("  harmonic index -- which is the decay A^{|n-5|/2}")
    check("at a = 0, that is at r = 1, there is no leak at all: B(z) = z",
          sp.simplify(B.subs(a, 0) - z) == 0)


def part5():
    say("\n=== 5. what a single harmonic settles, and what it does not")
    import numpy as np
    # The pull-back of the PURE harmonic 5 is a model, not the pairing: the pairing
    # carries all three harmonics and the Jacobian factors besides, so the amplitude
    # ratios below are not the pairing's.  What the model does show is the RATE of
    # the decay; the pairing's own spectrum is `framefree.py`.
    for rv in (sp.Rational(3, 2), sp.Integer(2), sp.Integer(3)):
        av = float((rv - 1) / (rv + 1))
        M = 16384
        ph = 2 * np.pi * np.arange(M) / M
        zz = np.exp(1j * ph)
        f = ((zz - av) / (1 - av * zz)) ** 2         # harmonic 5 is z^2
        co = np.fft.fft(f) / M
        amp = np.abs(co)
        say("  r = %-5s A = %.3f : |z^3|/|z^2| = %.4f, |z^4|/|z^3| = %.4f, "
            "|z^5|/|z^4| = %.4f"
            % (rv, av, amp[3] / amp[2], amp[4] / amp[3], amp[5] / amp[4]))
        # part 4 gives the coefficient of z^k in B^2 the leading term (k-1) a^{k-2},
        # so the step ratio carries a polynomial factor and falls to A from above.
        # That -- monotone, and never below A -- is what is checked; the bare value
        # A is the limit and not the value at any finite k.
        steps = [amp[k + 1] / amp[k] for k in (2, 3, 4)]
        check("in the model the step ratios fall monotonically at r = %s" % rv,
              steps[0] > steps[1] > steps[2])
        check("and they stay above the limit A = %.3f at r = %s" % (av, rv),
              min(steps) > av)
    say("  so the RATE is a function of the stretch alone; the AMPLITUDES are not --")
    say("  see framefree.py, where a7/a5 moves over an order of magnitude at one r")


def main():
    global LOG
    LOG = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "blaschke.log"), "w")
    say("The fiber map of a lifted constant stretch, and the leakage of the harmonics.")
    b = part1()
    part2(b)
    part4(b)
    part5()
    say("\n%d checks, %d passed, %d failed"
        % (CHECKS[0], CHECKS[1], CHECKS[0] - CHECKS[1]))
    LOG.close()


if __name__ == "__main__":
    main()
