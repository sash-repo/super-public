#!/usr/bin/env python3
"""Superdimensions of the superspaces of the text, computed exactly:
q(n), sq(n), pq(n), psq(n) (cross-checked against the bases built by qsq.py),
po(0|m), h(0|m), h^(1)(0|m) (formula (22)), and h^(1)(0|4) vs psl(2|2).

Run:  python3 sdim.py
"""

from po import Po, rank, independent_subset

# --------------------------------------------------------------- queer series


def sdim_queer(n):
    """(q, sq, pq, psq) as (even|odd) pairs, from the realization (A,B)."""
    # q(n)  = {(A,B)}, A even, B odd, both in gl(n)
    q = (n * n, n * n)
    # sq(n) = queertraceless: tr B = 0
    sq = (n * n, n * n - 1)
    # pq(n) = q(n)/K.1_{2n},  psq(n) = sq(n)/K.1_{2n};  1_{2n} = (1_n,0) is even
    pq = (n * n - 1, n * n)
    psq = (n * n - 1, n * n - 1)
    return q, sq, pq, psq


def check_queer(nmax=7):
    """cross-check against qsq.py's own basis count."""
    import qsq
    ok = True
    for n in range(2, nmax + 1):
        for kind, expected in zip(('q', 'sq', 'pq', 'psq'), sdim_queer(n)):
            _, sub = qsq.build(n, kind)
            ev, od = sub.sdim()
            if (ev, od) != expected:
                print(f"  MISMATCH {kind}({n}): engine {ev}|{od} vs formula "
                      f"{expected[0]}|{expected[1]}")
                ok = False
    return ok


# ------------------------------------------------------------- Poisson series


def sdim_po(m):
    """po(0|m) = Lambda(m) with the Poisson bracket."""
    alg = Po(m)
    ev = sum(1 for msk in range(alg.dim) if alg.parity(msk) == 0)
    od = alg.dim - ev
    return ev, od


def sdim_h(m):
    """h(0|m) = po(0|m)/c, the centre c = K.1 being even."""
    ev, od = sdim_po(m)
    return ev - 1, od


def sdim_h1(m):
    """h^{(1)}(0|m) = [h,h], computed as [po,po] mod the centre."""
    alg = Po(m)
    n = alg.dim
    # span of all brackets of basis monomials
    vecs = []
    for a in range(n):
        for b in range(a, n):
            br = alg.bracket({a: 1}, {b: 1})
            if br:
                v = [0] * n
                for msk, c in br.items():
                    v[msk] = c
                vecs.append(v)
    # every bracket of two monomials is parity-homogeneous, so split by parity
    ev_vecs = [v for v in vecs
               if all(alg.parity(i) == 0 for i, c in enumerate(v) if c)]
    od_vecs = [v for v in vecs
               if all(alg.parity(i) == 1 for i, c in enumerate(v) if c)]
    assert len(ev_vecs) + len(od_vecs) == len(vecs), "inhomogeneous bracket"
    ev = rank([list(v) for v in ev_vecs]) if ev_vecs else 0
    od = rank([list(v) for v in od_vecs]) if od_vecs else 0
    # h^{(1)} = [po,po]/(K.1 \cap [po,po]);  {xi_1,eta_1} = 1, so 1 is there and
    # is even -- hence exactly one even dimension is lost
    return ev - 1, od, ev + od


def parity_of_top(m):
    return "even" if m % 2 == 0 else "odd"


if __name__ == "__main__":
    print("=== queer series: sdim (even|odd) ===")
    print(f"{'n':>3} {'q(n)':>12} {'sq(n)':>12} {'pq(n)':>12} {'psq(n)':>12}")
    for n in range(2, 8):
        q, sq, pq, psq = sdim_queer(n)
        f = lambda p: f"{p[0]}|{p[1]}"
        print(f"{n:>3} {f(q):>12} {f(sq):>12} {f(pq):>12} {f(psq):>12}")
    print("\ncross-check against qsq.py basis counts:",
          "OK" if check_queer() else "FAILED")

    print("\n=== Poisson series: sdim (even|odd) ===")
    print(f"{'m':>3} {'po(0|m)':>14} {'h(0|m)':>14} {'h^(1)(0|m)':>14} "
          f"{'dim [po,po]':>12} {'top monomial':>13}")
    for m in range(3, 9):
        po = sdim_po(m)
        h = sdim_h(m)
        e1, o1, dimbr = sdim_h1(m)
        f = lambda p: f"{p[0]}|{p[1]}"
        print(f"{m:>3} {f(po):>14} {f(h):>14} {f((e1, o1)):>14} "
              f"{dimbr:>12} {parity_of_top(m):>13}")

    print("\n=== gl(m|n) and its tautological module ===")
    for (a, b) in ((2, 2), (4, 4), (8, 8)):
        print(f"  V of sdim {a}|{b}:  gl({a}|{b}) of sdim "
              f"{a*a+b*b}|{2*a*b}, psl of sdim {a*a+b*b-2}|{2*a*b}")

    print("\n=== the two isomorphisms the text states ===")
    e1, o1, _ = sdim_h1(4)
    print(f"  h^(1)(0|4) of sdim {e1}|{o1}   vs psl(2|2) of sdim "
          f"{2*2+2*2-2}|{2*2*2}")
    print(f"  po(0|5) of sdim {sdim_po(5)[0]}|{sdim_po(5)[1]}   vs q(4) of sdim "
          f"{sdim_queer(4)[0][0]}|{sdim_queer(4)[0][1]}")
    print(f"  po(0|6) of sdim {sdim_po(6)[0]}|{sdim_po(6)[1]}   vs gl(4|4) of sdim "
          f"{4*4+4*4}|{2*4*4}")
