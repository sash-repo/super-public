"""GF(2^K) elements as Python objects with operator overloads (the interface hwmod.py needs)."""
MODS = {2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 8: 0b100011011,
        10: 0b10000001001, 12: 0b1000001010011}

class GFField:
    def __init__(self, K):
        self.K = K; self.mod = MODS[K]; self.q = 1 << K
        # exp/log tables with a primitive element found by search
        for g in range(2, self.q):
            e = [0] * (2 * self.q); x = 1; seen = set()
            for i in range(self.q - 1):
                e[i] = x; seen.add(x); x = self._mulraw(x, g)
            if len(seen) == self.q - 1:
                break
        for i in range(self.q - 1, 2 * self.q):
            e[i] = e[i - (self.q - 1)]
        self.exp = e; self.log = [0] * self.q
        for i in range(self.q - 1):
            self.log[e[i]] = i
        self.gen = g
    def _mulraw(self, a, b):
        r = 0
        while b:
            if b & 1: r ^= a
            b >>= 1; a <<= 1
            if a >> self.K: a ^= self.mod
        return r
    def mul(self, a, b):
        if a == 0 or b == 0: return 0
        return self.exp[self.log[a] + self.log[b]]
    def inv_(self, a):
        return self.exp[(self.q - 1 - self.log[a]) % (self.q - 1)]
    def el(self, v):
        return G(self, v)
    # hwmod interface
    def norm(self, x):
        if isinstance(x, G): return x
        return G(self, int(x) & 1)
    def is_zero(self, x):
        return self.norm(x).v == 0
    def inv(self, x):
        return G(self, self.inv_(self.norm(x).v))
    zero = property(lambda s: G(s, 0))
    one = property(lambda s: G(s, 1))

class G:
    __slots__ = ('F', 'v')
    def __init__(self, F, v): self.F = F; self.v = v
    def _o(self, o):
        return o.v if isinstance(o, G) else (int(o) & 1)
    def __add__(self, o): return G(self.F, self.v ^ self._o(o))
    __radd__ = __add__; __sub__ = __add__; __rsub__ = __add__
    def __neg__(self): return self
    def __mul__(self, o): return G(self.F, self.F.mul(self.v, self._o(o)))
    __rmul__ = __mul__
    def __eq__(self, o): return self.v == self._o(o)
    def __hash__(self): return hash(self.v)
    def __bool__(self): return self.v != 0
    def __repr__(self): return 'g%d' % self.v
