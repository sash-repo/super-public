#!/usr/bin/env python3
"""Driver: for each algebra of the tables, the image of g in gl(M), the highest
root (hence the highest weight of the adjoint module) and the Cartan prolong of
the given modules (printed to stdout).

In the text: Statement 3.1 and Table (1) (usage: runprol.py prol NAME LAMBDA 2)."""
import sys, signal, time
from prol import Rep, Image, prolong, labels_of, analyse
from hwmod import irrep, Fp

BIG = 1000003          # characteristic 0 is read mod this prime
MAXD = int(__import__("os").environ.get("MAXD", "400"))

def par_of(A, brown=()):
    out = []
    for i, row in enumerate(A):
        a = row[i]
        out.append(0 if (a == 2 or i in brown) else 1)
    return out

ALG = {
 "brj25": (5, [[0,-1],[-2,1]], ()),
 "brj23": (3, [[0,-1],[-2,1]], ()),
 "ag2":   (BIG, [[0,1,0],[-1,2,-3],[0,-1,2]], ()),
 "ab3":   (BIG, [[2,-1,0,0],[-3,0,1,0],[0,-1,2,-2],[0,0,-1,2]], ()),
 "ab3p3": (3, [[2,-1,0,0],[-3,0,1,0],[0,-1,2,-2],[0,0,-1,2]], ()),
 "el55":  (5, [[2,0,-1,0,0],[0,2,0,0,-1],[-1,0,0,-4,-4],[0,0,-4,0,-2],[0,-1,-4,-2,0]], ()),
 "el53":  (3, [[0,-1,0,0,0],[-1,0,0,-1,0],[0,0,2,-1,-1],[0,-1,-1,2,0],[0,0,-1,0,2]], ()),
 "el53at5": (5, [[0,-1,0,0,0],[-1,0,0,-1,0],[0,0,2,-1,-1],[0,-1,-1,2,0],[0,0,-1,0,2]], ()),
 "g16":   (3, [[2,-1,0],[-1,1,-1],[0,-1,0]], ()),
 "g23":   (3, [[2,-1,-1],[-1,2,-1],[-1,-1,0]], ()),
 "g33":   (3, [[2,-1,0,0],[-1,2,-1,0],[0,-2,2,-1],[0,0,-1,0]], ()),
 "g26":   (3, [[2,-1,0,0,0],[-1,2,-1,0,0],[0,-1,0,-1,-2],[0,0,-1,2,0],[0,0,-1,0,0]], ()),
 "g43":   (3, [[2,-1,0,0],[-1,2,-2,-1],[0,-1,2,0],[0,-1,0,0]], ()),
 "g36":   (3, [[0,-1,0,0],[-1,2,-1,0],[0,-1,1,-1],[0,0,-1,0]], ()),
 "g83":   (3, [[2,-1,0,0,0],[-1,2,-1,0,0],[0,-2,2,-1,0],[0,0,-1,2,-1],[0,0,0,1,0]], ()),
 "g46":   (3, [[2,-1,0,0,0,0],[-1,2,-1,0,0,0],[0,-1,2,-1,0,0],[0,0,-2,0,-2,-1],[0,0,0,-1,2,0],[0,0,0,-1,0,0]], ()),
 "g66":   (3, [[0,-1,0,0,0,0],[-1,0,-2,0,0,0],[0,-1,2,-1,0,0],[0,0,-2,0,-2,-1],[0,0,0,-1,2,0],[0,0,0,-1,0,0]], ()),
 "elM5": (5, [[2,0,-1,0,0],[0,0,0,-1,2],[-1,0,2,0,-1],[0,-1,0,0,0],[0,-1,-1,0,2]], ()),
 "elM3": (3, [[2,0,-1,0,0],[0,0,0,-1,2],[-1,0,2,0,-1],[0,-1,0,0,0],[0,-1,-1,0,2]], ()),
 "g2p5": (5, [[2,-1],[-3,2]], ()),
 "f4p3": (3, [[2,-1,0,0],[-1,2,-1,0],[0,-2,2,-1],[0,0,-1,2]], ()),
 "f4p0": (BIG, [[2,-1,0,0],[-1,2,-1,0],[0,-2,2,-1],[0,0,-1,2]], ()),
 "sp4p3": (3, [[2,-1],[-2,2]], ()),
 "sl3p3": (3, [[2,-1],[-1,2]], ()),
}

class TO(Exception):
    pass

def _alarm(s, f):
    raise TO()

def info(name, tl=600):
    p, A, brown = ALG[name]
    par = par_of(A, brown)
    n = len(A)
    # a faithful-enough module to read the root system: take the sum of the
    # fundamental ones is costly; the adjoint is found from any module on which
    # n_+ acts faithfully -- try each fundamental weight, keep the largest image
    best = None
    for i in range(n):
        lam = [1 if j == i else 0 for j in range(n)]
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(tl)
        try:
            r = Rep(A, par, p, lam, maxdim=MAXD)
            img = Image(r)
            d = img.dims()
            if best is None or sum(d) > sum(best[1]):
                best = (lam, d, img)
        except (TO, MemoryError, OverflowError):
            pass
        finally:
            signal.alarm(0)
    lam, d, img = best
    th = img.highest_root()
    K = Fp(p)
    lab = labels_of(A, K, th)
    lab = [x if x <= p // 2 else x - p for x in lab]
    print("%s: p=%s par=%s image of g in gl(L^%s) = %s; highest root %s; labels %s"
          % (name, p if p != BIG else 0, par, lam, d, th, lab), flush=True)
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(tl)
    try:
        m = irrep(A, par, p, lab, maxdim=3000)
        print("   L^%s = %s" % (lab, m.sdim()), flush=True)
    except Exception as e:
        print("   L^%s: %s" % (lab, type(e).__name__), flush=True)
    finally:
        signal.alarm(0)

def prol(name, lam, kmax=3, tl=3000):
    p, A, brown = ALG[name]
    par = par_of(A, brown)
    for ident, grad in ((False, False), (True, True)):
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(tl)
        try:
            r = Rep(A, par, p, lam, maxdim=2000)
            img = Image(r, extra_identity=ident, extra_grading=grad)
            print("%s lam=%s M=%s g0=%s%s" % (name, lam, r.sdim(), img.dims(),
                  " (+Id, +torus)" if ident else ""), flush=True)
            prolong(r, img, kmax=kmax)
        except TO:
            print("%s lam=%s: timeout" % (name, lam), flush=True)
        finally:
            signal.alarm(0)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "info":
        for nm in sys.argv[2:]:
            info(nm)
    elif cmd == "prol":
        nm = sys.argv[2]
        lam = [int(x) for x in sys.argv[3].split(",")]
        prol(nm, lam, kmax=int(sys.argv[4]) if len(sys.argv) > 4 else 3)
