"""Universal Virasoro vertex algebra V^c at a numeric central charge c (exact Fractions).
States: dict {descending tuple of parts>=2: Fraction} = L_{-n1}...L_{-nk}|0>.
General modes s_(n) v by the Borcherds identity (a=T, T_(p)=L_{p-1})."""
from fractions import Fraction as Fr
from functools import lru_cache
from math import comb

def binom(m, j):  # generalized binomial, m may be negative
    r = Fr(1)
    for i in range(j): r = r * (m - i) / (i + 1)
    return r

class VA:
    def __init__(self, c, conv=Fr):
        self.c = c if conv is not Fr else Fr(c)
        self.K = conv
        self.act = lru_cache(maxsize=None)(self._act)
        self.mode1 = lru_cache(maxsize=None)(self._mode1)
    def _act(self, m, mono):
        if not mono:
            return {} if m >= -1 else {(-m,): self.K(1)}
        n1, rest = mono[0], mono[1:]
        if m <= -2 and -m >= n1:
            return {(-m,) + mono: self.K(1)}
        out = {}
        for k, v in self.act(m, rest).items():
            for k2, v2 in self.act(-n1, k).items():
                out[k2] = out.get(k2, 0) + v * v2
        if m + n1 != 0:
            for k, v in self.act(m - n1, rest).items():
                out[k] = out.get(k, 0) + (m + n1) * v
        if m == n1:
            out[rest] = out.get(rest, 0) + self.c * (m**3 - m) / 12
        return {k: v for k, v in out.items() if v}
    def L(self, m, st):
        out = {}
        for k, v in st.items():
            for k2, v2 in self.act(m, k).items():
                out[k2] = out.get(k2, 0) + v * v2
        return {k: v for k, v in out.items() if v}
    def _mode1(self, s, n, v):
        """s_(n) applied to monomial v; s, v monomials (tuples). Returns dict."""
        wv = sum(v)
        if not s:
            return {v: self.K(1)} if n == -1 else {}
        k, sp = s[0], s[1:]
        m = 1 - k                      # s = T_(m) sp
        ws = sum(sp)
        out = {}
        def addd(d, f):
            for kk, vv in d.items():
                out[kk] = out.get(kk, 0) + f * vv
        j = 0
        # first sum: a_(m-j) b_(n+j) v ; b_(q) v nonzero needs result weight ws+wv-q-1>=0
        while n + j <= ws + wv - 1 or j <= wv + 1:
            bj = self.K(binom(m, j) * (-1) ** j)
            if n + j <= ws + wv - 1:
                w = self.mode1(sp, n + j, v)
                if w: addd(self.L(m - j - 1, w), bj)
            if j - 1 <= wv:            # a_(j) v = L_{j-1} v
                w = self.L(j - 1, {v: self.K(1)})
                if w:
                    w2 = self.mode({sp: self.K(1)}, m + n - j, w)
                    addd(w2, -bj * (1 if m % 2 == 0 else -1))
            j += 1
        return {kk: vv for kk, vv in out.items() if vv}
    def mode(self, st, n, vec):
        out = {}
        for s, a in st.items():
            for v, b in vec.items():
                for k, x in self.mode1(s, n, v).items():
                    out[k] = out.get(k, 0) + a * b * x
        return {k: v for k, v in out.items() if v}

def parts(n, mn=2):
    res = []
    def rec(rem, mx, acc):
        if rem == 0: res.append(tuple(acc)); return
        for p in range(min(rem, mx), mn - 1, -1):
            rec(rem - p, p, acc + [p])
    rec(n, n, [])
    return res

if __name__ == "__main__":
    V = VA(Fr(7, 3))
    T = {(2,): Fr(1)}
    # checks: T_(j) = L_{j-1}
    for j in range(0, 4):
        for v in parts(5):
            assert V.mode(T, j, {v: Fr(1)}) == V.L(j - 1, {v: Fr(1)})
    # vacuum axiom s_(-1)|0> = s
    for n in range(2, 8):
        for s in parts(n):
            assert V.mode({s: Fr(1)}, -1, {(): Fr(1)}) == {s: Fr(1)}, s
            for j in range(0, 4): assert V.mode({s: Fr(1)}, j, {(): Fr(1)}) == {}
    # skew-symmetry check a_(0)b vs b_(0)a for a=Lam
    Lam = {(2, 2): Fr(1), (4,): Fr(-3, 5)}
    print("ok")
