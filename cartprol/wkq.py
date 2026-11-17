"""p = 2: the adjoint modules of wk(3;al), wk(4;al) (highest roots), and the Cartan prolongs of the
14-dimensional modules (1,0), (al,0), (0,1) over wk^(1)(3;al)/c and bgl^(1)(3;al)/c.
In the text: Statement 3.1, the row wk^(1)(3;alpha)/c of Table (1)."""
from ratf import F2alpha, AL, ONE, ZERO
from prol import Rep, Image, prolong
K = F2alpha(); a = AL
W3 = [[0, a, 0], [a, 0, 1], [0, 1, 0]]
W4 = [[0, a, 0, 0], [a, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]
for name, A, lam in (("wk3", W3, [1, 0, 0]), ("wk3", W3, [0, 1, 0]), ("wk4", W4, [0, 0, 0, 1])):
    r = Rep(A, [0]*len(A), 2, lam, field=K)
    img = Image(r)
    th = img.highest_root()
    lab = [sum((th[j] * A[i][j] for j in range(len(A))), ZERO) for i in range(len(A))]
    print(name, "via", lam, "image", img.dims(), "highest root", th, "labels", lab, flush=True)
    m = Rep(A, [0]*len(A), 2, lab, field=K)
    print("   L^theta dim", m.N, flush=True)
for lam in ([0, 1, 0], [0, a, 0], [a, 0, 1]):
    for par in ([0,0,0],[1,0,0]):
        r = Rep(W3, par, 2, lam, field=K)
        img = Image(r)
        print("wk3/bgl3", par, lam, "M", r.sdim(), "g0", img.dims(), flush=True)
        prolong(r, img, kmax=2)
