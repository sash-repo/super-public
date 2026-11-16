r"""The odd Laplacian of Subsec. 1.9 (eqs. (dprime), (DelS), (Svanishes)).

In the special almost periplectic flat model C^{n|n} with q_i even, pi_i odd,

    d' := [H, d],   delta := [d, X^-],   S := -(1/2)[d', delta] = (1/2)[d, delta'],

and the odd Laplacian of eq. (DelS) is sum d^2/(d xi_i d q_i).  Checks that the two
coincide up to a factor, that [d, delta] = 0, and the parity of each operator.

Run (from this folder):  ../venv/bin/python laplace.py
"""
import os, sys; _HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(_HERE, "..", "lib"), _HERE]
from flatmodel import Model, d_operator


def periplectic(n):
    return Model([0] * n + [1] * n)


def report(n):
    Q = periplectic(n)
    q = lambda i: i                      # indices of q_1..q_n
    pi = lambda i: n + i                 # indices of pi_1..pi_n
    c = lambda a: Q.gen('c', a)          # left mult. by the differential
    b = lambda a: Q.gen('b', a)          # convolution with it
    t = lambda a: Q.gen('t', a)          # derivation along the coordinate

    d = d_operator(Q)
    Xp = {}
    Xm = {}
    S = {}
    for i in range(n):
        Xp = Q.add(Xp, Q.mul(c(q(i)), c(pi(i))))          # sum \hat q_i \hat pi_i
        Xm = Q.add(Xm, Q.mul(b(pi(i)), b(q(i))))          # sum d_{\hat pi_i} d_{\hat q_i}
        S = Q.add(S, Q.mul(t(q(i)), t(pi(i))))            # sum d_{q_i} d_{pi_i}

    H = Q.br(Xp, Xm)
    delta = Q.br(d, Xm)
    dpr = Q.br(H, d)
    deltapr = Q.br(Xm, dpr)
    Delta = Q.br(dpr, delta)
    Delta2 = Q.smul(-1, Q.br(d, deltapr))
    dd = Q.br(d, delta)

    par = Q.parity
    print('n = %d' % n)
    print('  parities: d %d, X^+ %d, X^- %d, H %d, delta %d, d\' %d' %
          (par(d), par(Xp), par(Xm), par(H), par(delta), par(dpr)))
    print('  [d, delta]            = 0 ?  %s   (parity %s)'
          % (not dd, 'odd' if dd else '-'))
    print('  [d\', delta] = -[d, delta\'] ?  %s'
          % (not Q.add(Delta, Q.smul(-1, Delta2))))
    print('  Delta := [d\', delta]  parity %d' % par(Delta))
    for lam in (1, -1, 2, -2):
        if not Q.add(Delta, Q.smul(-lam, S)):
            print('  Delta = %+d * S,  where S = sum d_{q_i} d_{pi_i}   <== ' % lam)
            break
    else:
        print('  Delta is NOT a multiple of S;  Delta =', Delta)
    print('  S parity %d' % par(S))


for n in (1, 2, 3, 4):
    report(n)
