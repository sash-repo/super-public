"""Remark 2.4: which of vect(0|n), svect(0|n), the deformed svect(0|n) and h'(0|n) carry a
non-degenerate even 2-cocycle?

For each case named on the command line, the whole space Z^2_even of even 2-cocycles is
computed by `cocyc.analyse` (the cocycle identity on every triple of basis elements), and the
rank of a random element of Z^2_even is printed; the answer "NON-DEGENERATE EXISTS" means that
this rank equals dim g.  Arithmetic mod the prime P of cocyc.py.

This file only adds the case h'(0|n) to cocyc.py: h'(0|n) is spanned by the Hamiltonian fields
H_f, f a monomial of degree 1, ..., n-1 (frob.ham_fields), and its structure constants are
computed in that basis exactly as for the other algebras.

Run:  python3 census.py vect:2 vect:3 vect:4 vect:5 svect:3 svect:4 svect:5 svectdef:4 h:4 h:5 h:6
"""
import os
import sys
import time

sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"),
                os.path.dirname(os.path.abspath(__file__))]
import cocyc as C  # noqa: E402
import frob as F   # noqa: E402


def full(kind, n):
    vb, idx = C.build(n, 'vect')
    if kind == 'vect':
        C.analyse(f"vect(0|{n})", vb, n, idx)
    elif kind == 'h':
        C.analyse(f"h'(0|{n})", F.ham_fields(n), n, idx)
    else:
        _, _, div = C.build(n, 'svect')
        for sg in (0, 1):
            if kind == 'svect':
                cond = lambda X, sg=sg: div(X, sg)
            else:
                top = {tuple(range(n)): 1}
                cond = lambda X, sg=sg: C.padd(C.vf_apply(X, top),
                                               C.pmul(C.padd({(): 1}, top), div(X, sg)))
            try:
                g = C.subalg(n, cond, vb, idx)
                C.analyse(f"{kind}(0|{n}) sign{sg}", g, n, idx)
                break
            except AssertionError as e:
                print(f"  sign {sg}: not closed ({str(e)[:60]})", flush=True)


if __name__ == '__main__':
    for w in sys.argv[1:]:
        kind, n = w.split(':')
        t0 = time.time()
        full(kind, int(n))
        print(f"   [{time.time() - t0:.1f} s]", flush=True)
