#!/usr/bin/env python3
r"""The rows of Table 25.5 of BGLLS (and Shen's algebra), as vector fields in
the coordinates of the paper.  Each row: coordinates (names, degrees,
parities), the negative part, and either the generators of g_0 or the flag
'complete prolong of the negative part' (BGLLS Sect. 2.11).  Coefficients are
over F_2, so every sign of the paper is dropped.  x^(k) is a divided power.

Transcribed from the typeset pages (1510.07255.pdf, pp. 49-88), not from the
text extraction, which loses the superscripts (2), (3).
"""
import itertools

ROWS = {}


def row(name, **kw):
    ROWS[name] = kw
    return kw


# ---------------------------------------------------------------- vle(9) Sect. 7
row('vle9', section='7', table='vle(3;N|6) -> vle(9;N)',
    coords=['x%d' % i for i in range(1, 10)],
    degs=[2, 2, 2, 1, 1, 1, 1, 1, 1],
    par=[0, 0, 0, 1, 1, 1, 1, 1, 1],
    neg=['dx1', 'dx2', 'dx3', 'dx4', 'dx5', 'dx6',
         'x5 dx3 + x6 dx2 + dx7', 'x4 dx3 + x6 dx1 + dx8', 'x4 dx2 + x5 dx1 + dx9'],
    g0='complete')

# ---------------------------------------------------------------- vle~(9) Sect. 8
# z1 (deg 2, even); z2,z3,z8,z9 even, z4..z7 odd, all deg 1
row('vle~9', section='8', table='vle(5;N|4) -> vle~(9;N)',
    coords=['z%d' % i for i in range(1, 10)],
    degs=[2, 1, 1, 1, 1, 1, 1, 1, 1],
    par=[0, 0, 0, 1, 1, 1, 1, 0, 0],
    neg=['dz1',
         'dz4', 'dz5',                        # x_i  <-> d_{2+i}, i = 2,3
         'dz6 + z4 dz1', 'dz7 + z5 dz1',      # xi1 xi_i <-> d_{4+i} + z_{2+i} d1
         'dz2', 'dz3',                        # xi_i <-> d_i
         'dz8 + z2 dz1', 'dz9 + z3 dz1'],     # xi1 x_i <-> d_{6+i} + z_i d1
    g0='complete')

# ---------------------------------------------------------------- kle(15) Sect. 9
_th = ['t12', 't13', 't14', 't15', 't23', 't24', 't25', 't34', 't35', 't45']
row('kle15', section='9', table='kle(5;N|10) -> kle(15;N)',
    coords=['x1', 'x2', 'x3', 'x4', 'x5'] + _th,
    degs=[2] * 5 + [1] * 10,
    par=[0] * 5 + [1] * 10,
    neg=['dx1', 'dx2', 'dx3', 'dx4', 'dx5',
         'dt12 + t34 dx5 + t45 dx3 + t35 dx4',
         'dt13 + t25 dx4 + t24 dx5 + t45 dx2',
         'dt14 + t23 dx5 + t35 dx2 + t25 dx3',
         'dt15 + t24 dx3 + t23 dx4 + t34 dx2',
         'dt23 + t45 dx1', 'dt24 + t35 dx1', 'dt25 + t34 dx1',
         'dt34', 'dt35', 'dt45'],
    g0='complete',
    g0check=[  # the Chevalley generators of sl(5) printed in Sect. 9
        'x1 dx2 + t23 t24 dx5 + t23 t25 dx4 + t24 t25 dx3 + t23 dt13 + t24 dt14 + t25 dt15',
        'x2 dx3 + t34 t35 dx1 + t13 dt12 + t34 dt24 + t35 dt25',
        'x3 dx4 + t14 dt13 + t24 dt23 + t45 dt35',
        'x4 dx5 + t15 dt14 + t25 dt24 + t35 dt34',
        'x2 dx1 + t13 t14 dx5 + t13 t15 dx4 + t14 t15 dx3 + t13 dt23 + t14 dt24 + t15 dt25',
        'x3 dx2 + t24 t25 dx1 + t12 dt13 + t24 dt34 + t25 dt35',
        'x4 dx3 + t13 dt14 + t23 dt24 + t35 dt45',
        'x5 dx4 + t14 dt15 + t24 dt25 + t34 dt35'])

# ---------------------------------------------------------------- kle~(15) Sect. 10
row('kle~15', section='10', table='kle(9;N|6) -> kle~(15;N)',
    coords=['x%d' % i for i in range(1, 16)],
    degs=[1] * 14 + [2],
    par=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    neg=['dx15',
         'dx1 + x11 dx15', 'dx2 + x12 dx15', 'dx3 + x13 dx15', 'dx4 + x14 dx15',
         'dx5 + x10 dx15', 'dx6 + x9 dx15', 'dx7 + x8 dx15',
         'dx8', 'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14'],
    g0=['x2 dx6 + x9 dx12 + x3 dx5 + x10 dx13 + x1 dx8 + x7 dx11 + x1 x7 dx15',
        'x2 dx7 + x8 dx12 + x4 dx5 + x10 dx14 + x1 dx9 + x6 dx11 + x1 x6 dx15',
        'x3 dx7 + x8 dx13 + x4 dx6 + x9 dx14 + x1 dx10 + x5 dx11 + x1 x5 dx15',
        'x2 dx10 + x5 dx12 + x2 x5 dx15 + x3 dx9 + x6 dx13 + x3 x6 dx15 + x4 dx8 + x7 dx14 + x4 x7 dx15',
        'x1 dx1 + x8 dx8 + x9 dx9 + x10 dx10 + x12 dx12 + x13 dx13 + x14 dx14 + x15 dx15',
        'x14 dx4 + x14^(2) dx15', 'x13 dx3 + x13^(2) dx15',
        'x12 dx2 + x12^(2) dx15', 'x11 dx1 + x11^(2) dx15'],
    srcH={'x11': 2, 'x12': 2, 'x13': 2, 'x14': 2})

# ---------------------------------------------------------------- kle_3(20) Sect. 11
row('kle3_20', section='11', table='kle(9;N|11) -> kle_3(20;N)',
    coords=['x%d' % i for i in range(1, 21)],
    degs=[1] * 12 + [2] * 6 + [3] * 2,
    # parity read off the weights: entries +-1 <=> a factor Pi
    par=[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    neg=['dx19', 'dx20',
         'dx13 + x7 dx19 + x8 dx20', 'dx14 + x9 dx19 + x10 dx20', 'dx15 + x11 dx19 + x12 dx20',
         'dx16', 'dx17', 'dx18',
         'dx1 + x5 dx15 + x6 dx14 + x9 dx16 + x11 dx17 + x18 dx20',
         'dx2 + x4 dx15 + x6 dx13 + x7 dx16 + x11 dx18 + x17 dx20',
         'dx3 + x4 dx14 + x5 dx13 + x7 dx17 + x9 dx18 + x16 dx20',
         'dx4 + x10 dx16 + x12 dx17 + x18 dx19',
         'dx5 + x8 dx16 + x12 dx18 + x17 dx19',
         'dx6 + x8 dx17 + x10 dx18 + x16 dx19',
         'dx7', 'dx8', 'dx9', 'dx10', 'dx11', 'dx12'],
    g0='complete')

# ---------------------------------------------------------------- kle_2(20) Sect. 12
row('kle2_20', section='12', table='kle(11;N|9) -> kle_2(20;N)',
    coords=['x%d' % i for i in range(1, 21)],
    degs=[1] * 16 + [2] * 4,
    par=[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    neg=['dx17', 'dx18', 'dx19', 'dx20',
         'dx1 + x9 dx17 + x10 dx18 + x11 dx19 + x15 dx20',
         'dx2 + x12 dx17 + x13 dx18 + x14 dx19 + x16 dx20',
         'dx3 + x8 dx18 + x6 dx19 + x12 dx20',
         'dx4 + x7 dx18 + x5 dx19 + x9 dx20',
         'dx5 + x8 dx17 + x13 dx20',
         'dx6 + x7 dx17 + x10 dx20',
         'dx7 + x14 dx20', 'dx8 + x11 dx20',
         'dx9', 'dx10', 'dx11', 'dx12', 'dx13', 'dx14', 'dx15', 'dx16'],
    g0='complete')

# ---------------------------------------------------------------- mb_3(11) Sect. 13
# u1..u3 (deg 2, even), e_i = eta_i, z_i = zeta_i (deg 1, odd), c1, c2 = chi (deg 3, odd)
_cyc = [(1, 2, 3), (2, 3, 1), (3, 1, 2)]
row('mb3_11', section='13', table='mb(3;N|8) -> mb_3(11;N)',
    coords=['u1', 'u2', 'u3', 'e1', 'e2', 'e3', 'z1', 'z2', 'z3', 'c1', 'c2'],
    degs=[2, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3],
    par=[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    neg=(['de%d + u%d dc1' % (i, i) for i in (1, 2, 3)]
         + ['dz%d + e%d du%d + e%d du%d + e%d e%d dc1 + u%d dc2' % (i, j, k, k, j, j, k, i)
            for (i, j, k) in _cyc]
         + ['du1', 'du2', 'du3', 'dc1', 'dc2']),
    g0='complete')

# ---------------------------------------------------------------- mb(9) Sect. 15
row('mb9', section='15', table='mb(4;N|5) -> mb(9;N)',
    coords=['x%d' % i for i in range(1, 10)],
    degs=[2] + [1] * 8,
    par=[1, 0, 0, 0, 1, 1, 1, 1, 0],
    neg=['dx1', 'dx2', 'dx3', 'dx4', 'dx9 + x8 dx1', 'dx8',
         'dx5 + x4 dx1', 'dx6 + x3 dx1', 'dx7 + x2 dx1'],
    g0=['x6 x7 dx1 + x4 dx8 + x6 dx2 + x7 dx3 + x9 dx5',
        'x3 dx2 + x7 dx6', 'x2 dx3 + x6 dx7', 'x4 dx3 + x6 dx5', 'x3 dx4 + x5 dx6',
        'x4^(2) dx1 + x4 dx5', 'x3^(2) dx1 + x3 dx6', 'x2^(2) dx1 + x2 dx7',
        'x8^(2) dx1 + x8 dx9',
        'x1 dx1 + x2 dx2 + x5 dx5 + x6 dx6 + x9 dx9'],
    srcH={'x2': 2, 'x3': 2, 'x4': 2, 'x8': 2})

# ---------------------------------------------------------------- mb_2(11) Sect. 16
row('mb2_11', section='16', table='mb(5;N|6) -> mb_2(11;N)',
    coords=['z1', 'z2', 'z3', 'z4', 'z5', 'z13', 'z14', 'z15', 'z23', 'z24', 'z25'],
    degs=[1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    par=[0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    neg=['dz3', 'dz4', 'dz5',
         'dz1', 'dz2', 'dz13 + z1 dz3', 'dz23 + z2 dz3',
         'dz14 + z1 dz4 + z25 dz3', 'dz24 + z2 dz4',
         'dz15 + z1 dz5 + z24 dz3', 'dz25 + z2 dz5'],
    g0='complete',
    g0check=[  # the four lowering operators printed in (16.2)
        'z13 dz14 + z23 dz24 + z15 dz2 + z25 dz1 + z3 dz4 + z15 z23 dz3 + z15 z24 dz4 + z15 z25 dz5',
        'z13 dz15 + z23 dz25 + z14 dz2 + z24 dz1 + z3 dz5 + z14 z23 dz3 + z14 z24 dz4 + z14 z25 dz5',
        'z2 dz1 + z13 dz23 + z14 dz24 + z15 dz25 + z14 z15 dz3',
        'z14 dz15 + z24 dz25 + z4 dz5'])

# ---------------------------------------------------------------- kas(10) Sect. 20
row('kas10', section='20', table='kas(5;N|5) -> kas(10;N)',
    coords=['x%d' % i for i in range(1, 11)],
    degs=[2, 2] + [1] * 8,
    par=[1, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    neg=['dx2', 'dx1', 'dx3', 'dx4', 'dx5', 'dx6',
         'x5 dx1 + dx7', 'x6 dx1 + dx8', 'x3 dx1 + x7 dx2 + dx9', 'x4 dx1 + x8 dx2 + dx10'],
    g0=['x7 dx3 + x9 dx5 + x8 dx4 + x10 dx6 + x7 x9 dx1 + x8 x10 dx1',     # xi1 Phi
        'x7 dx3 + x8 dx4 + x2 dx1',                                         # t xi1
        'x7 dx6 + x8 dx5 + x7 x8 dx1',                                      # xi1 eta2 eta3
        'x9 dx4 + x10 dx3 + x9 x10 dx1',                                    # xi1 xi2 xi3
        'x5 dx4 + x6 dx3 + x9 dx8 + x10 dx7 + x9 x10 dx2',                  # X+
        'x3 dx4 + x6 dx5 + x7 dx8 + x10 dx9',                               # X~+
        'x3 dx6 + x4 dx5 + x7 dx10 + x8 dx9 + x7 x8 dx2',                   # X-
        'x4 dx3 + x5 dx6 + x8 dx7 + x9 dx10',                               # X~-
        'x3 dx3 + x5 dx5 + x7 dx7 + x9 dx9',                                # D
        'x3 dx3 + x4 dx4 + x5 dx5 + x6 dx6 + x7 dx7 + x8 dx8 + x9 dx9 + x10 dx10',   # E
        'x4 dx4 + x6 dx6 + x8 dx8 + x10 dx10',                              # xi1 eta1
        'x2 dx2 + x3 dx3 + x4 dx4 + x9 dx9 + x10 dx10',                     # t + xi1 eta1
        'x3 x5 dx1 + x4 x6 dx1 + x5 x7 dx2 + x6 x8 dx2 + x1 dx2 + x3 dx7 + x4 dx8 + x5 dx9 + x6 dx10'])  # eta1

# ---------------------------------------------------------------- vas(8) Sect. 22
_vas0 = []
for i in range(1, 5):
    for j in range(1, 5):
        if i != j:
            _vas0.append('x%d dx%d + s%d ds%d' % (i, j, j, i))
for i in range(1, 5):
    for j in range(i + 1, 5):
        _vas0.append('x%d dx%d + x%d dx%d + s%d ds%d + s%d ds%d' % (i, i, j, j, i, i, j, j))
_vas0.append(' + '.join('s%d ds%d' % (i, i) for i in range(1, 5)))
for i in range(1, 5):
    _vas0.append('x%d ds%d' % (i, i))
for i in range(1, 5):
    for j in range(1, 5):
        if i != j:
            k, l = [a for a in range(1, 5) if a not in (i, j)]
            _vas0.append('x%d ds%d + s%d dx%d + s%d dx%d' % (i, j, k, l, l, k))
row('vas8', section='22', table='vas(4;N|4) -> vas(8;N)',
    coords=['x1', 'x2', 'x3', 'x4', 's1', 's2', 's3', 's4'],
    degs=[1] * 8, par=[0, 0, 0, 0, 1, 1, 1, 1],
    neg=['dx1', 'dx2', 'dx3', 'dx4', 'ds1', 'ds2', 'ds3', 'ds4'],
    g0=_vas0)

# ---------------------------------------------------------------- Shen's gs(2) Sect. 23
row('gs2', section='23', table="Shen's gs(2) = me^(1)(5;N)",
    coords=['x1', 'x2', 'x3', 'x4', 'x5'],
    degs=[3, 3, 2, 1, 1], par=[0] * 5,
    neg=['dx1', 'dx2', 'dx3', 'x3 dx2 + x4 x5 dx2 + dx4', 'x3 dx1 + x4 dx3 + dx5'],
    g0=['x1 dx1 + x3 dx3 + x4 dx4',
        'x5^(3) dx1 + x1 dx2 + x4 x5^(2) dx2 + x5^(2) dx3 + x5 dx4',
        'x2 dx1 + x4^(2) x5 dx1 + x4^(3) dx2 + x4^(2) dx3 + x4 dx5',
        'x2 dx2 + x3 dx3 + x5 dx5'],
    srcH={'x4': 2, 'x5': 2})


# ------------------------------------------------------------ generated rows
def _subsets(n):
    return [A for r in range(1, n + 1) for A in itertools.combinations(range(n), r)]


def gen_vle7():
    """vle(7): g_-1 = Pi(O(3;1)/K1), g_0 = c(vect(3;1)) (Table 25.5, Sect. 6)"""
    S = _subsets(3)
    names = ['y' + ''.join(str(a + 1) for a in A) for A in S]
    g0 = []
    for B in [()] + S:
        for j in range(3):
            terms = []
            for A in S:
                if j not in A:
                    continue
                R = set(A) - {j}
                if set(B) & R:
                    continue
                img = tuple(sorted(set(B) | R))
                if img:
                    terms.append('%s d%s' % (names[S.index(A)], names[S.index(img)]))
            if terms:
                g0.append(' + '.join(terms))
    g0.append(' + '.join('%s d%s' % (n, n) for n in names))
    return dict(section='6', table='vle(4;N|3) -> vle(7;N)', coords=names, degs=[1] * 7,
                par=[(len(A) + 1) % 2 for A in S], neg=['d' + n for n in names], g0=g0)


def gen_kas7tilde():
    """kas~(7): g_-1 = Vol_0(3;1), g_0 = c(vect(3;1)) acting by D + div D (Sect. 18)"""
    S = [()] + [A for A in _subsets(3) if len(A) < 3]          # functions without eta1eta2eta3
    names = ['z' + (''.join(str(a + 1) for a in A) or '0') for A in S]
    full = [()] + _subsets(3)
    g0 = []
    for B in full:
        for j in range(3):
            # D = eta^B d_j ;  div D = d_j(eta^B)
            divB = tuple(sorted(set(B) - {j})) if j in B else None
            terms = []
            for A in S:
                out = set()
                if j in A:
                    R = set(A) - {j}
                    if not (set(B) & R):
                        out ^= {tuple(sorted(set(B) | R))}
                if divB is not None and not (set(divB) & set(A)):
                    out ^= {tuple(sorted(set(divB) | set(A)))}
                for img in out:
                    assert len(img) < 3 or False, 'Vol_0 not invariant?'
                    terms.append('%s d%s' % (names[S.index(A)], names[S.index(img)]))
            if terms:
                g0.append(' + '.join(terms))
    g0.append(' + '.join('%s d%s' % (n, n) for n in names))
    return dict(section='18', table='kas(4;N|3) -> kas~(7;N)', coords=names, degs=[1] * 7,
                par=[len(A) % 2 for A in S], neg=['d' + n for n in names], g0=g0)


def gen_kas8():
    """kas(8): g_-1 = O(3;1), g_0 = O(3;1) (multiplications) x| d(svect^(1)(3;1))
    generated by (19.1): d_i, xi_i d_j, xi_i (xi_j d_j + xi_k d_k)  (Sect. 19)"""
    S = [()] + _subsets(3)
    names = ['w' + (''.join(str(a + 1) for a in A) or '0') for A in S]
    g0 = []
    for B in S:                                   # multiplication by xi^B
        terms = []
        for A in S:
            if not (set(A) & set(B)):
                img = tuple(sorted(set(A) | set(B)))
                terms.append('%s d%s' % (names[S.index(A)], names[S.index(img)]))
        g0.append(' + '.join(terms))
    ders = []
    for i in range(3):
        ders.append([((), i)])                                 # d_i
        for j in range(3):
            ders.append([((i,), j)])                           # xi_i d_j
        j, k = [a for a in range(3) if a != i]
        ders.append([((i, j), j), ((i, k), k)])                # xi_i (xi_j d_j + xi_k d_k)
    for D in ders:
        terms = []
        for A in S:
            out = set()
            for B, j in D:
                if j in A:
                    R = set(A) - {j}
                    if not (set(B) & R):
                        out ^= {tuple(sorted(set(B) | R))}
            for img in out:
                terms.append('%s d%s' % (names[S.index(A)], names[S.index(img)]))
        if terms:
            g0.append(' + '.join(terms))
    return dict(section='19', table='kas(4;N|4) -> kas(8;N)', coords=names, degs=[1] * 8,
                par=[len(A) % 2 for A in S], neg=['d' + n for n in names], g0=g0)


ROWS['vle7'] = gen_vle7()
ROWS['kas~7'] = gen_kas7tilde()
ROWS['kas8'] = gen_kas8()

# the paper's own fields, to be compared with the generated ones
KAS7T_PAPER = ['z1 dz0 + z12 dz2 + z13 dz3', 'z2 dz0 + z12 dz1 + z23 dz3',
               'z3 dz0 + z13 dz1 + z23 dz2',
               'z1 dz1 + z12 dz12 + z13 dz13', 'z2 dz2 + z12 dz12 + z23 dz23',
               'z3 dz3 + z13 dz13 + z23 dz23',
               'z1 dz2 + z13 dz23', 'z1 dz3 + z12 dz23', 'z2 dz1 + z23 dz13',
               'z2 dz3 + z12 dz13', 'z3 dz1 + z23 dz12', 'z3 dz2 + z13 dz12',
               'z1 dz23', 'z2 dz13', 'z3 dz12',
               'z2 dz12 + z3 dz13', 'z1 dz12 + z3 dz23', 'z1 dz13 + z2 dz23',
               'z0 dz1 + z3 dz13', 'z0 dz2 + z1 dz12', 'z0 dz3 + z2 dz23',
               'z0 dz12', 'z0 dz13', 'z0 dz23',
               'z0 dz0 + z1 dz1 + z2 dz2 + z3 dz3 + z12 dz12 + z13 dz13 + z23 dz23']
# (19.2) with x1..x8 <-> 1, xi1xi2, xi1xi3, xi2xi3, xi1, xi2, xi3, xi1xi2xi3
KAS8_MAP = {'x1': 'w0', 'x2': 'w12', 'x3': 'w13', 'x4': 'w23', 'x5': 'w1', 'x6': 'w2',
            'x7': 'w3', 'x8': 'w123'}
KAS8_PAPER = ['x1 dx1 + x2 dx2 + x3 dx3 + x4 dx4 + x5 dx5 + x6 dx6 + x7 dx7 + x8 dx8',
              'x1 dx5 + x4 dx8 + x6 dx2 + x7 dx3', 'x1 dx6 + x3 dx8 + x5 dx2 + x7 dx4',
              'x1 dx7 + x2 dx8 + x5 dx3 + x6 dx4', 'x1 dx2 + x7 dx8', 'x1 dx3 + x6 dx8',
              'x1 dx4 + x5 dx8', 'x1 dx8',
              'x2 dx6 + x3 dx7 + x5 dx1 + x8 dx4', 'x2 dx5 + x4 dx7 + x6 dx1 + x8 dx3',
              'x3 dx5 + x4 dx6 + x7 dx1 + x8 dx2',
              'x6 dx2 + x7 dx3', 'x5 dx2 + x7 dx4', 'x5 dx3 + x6 dx4',
              'x3 dx4 + x5 dx6', 'x2 dx4 + x5 dx7', 'x4 dx3 + x6 dx5', 'x2 dx3 + x6 dx7',
              'x4 dx2 + x7 dx5', 'x3 dx2 + x7 dx6',
              'x2 dx2 + x3 dx3 + x5 dx5 + x8 dx8', 'x2 dx2 + x4 dx4 + x6 dx6 + x8 dx8',
              'x3 dx3 + x4 dx4 + x7 dx7 + x8 dx8']

# ---------------------------------------------------------------- sb~(7) Sect. 21.2.1
# x1..x3 <-> xi_i vol, x4..x6 <-> xi_i xi_j vol, x7 <-> tau xi1xi2xi3 vol;
# g_-1 = Pi(Vol(0|3))/K(1 - tau xi1xi2xi3)vol, p(vol) = 1: x1..x3 odd, x4..x7 even
row('sb~7', section='21.2.1', table='sb~_mu(4|3) -> sb~(7;N)',
    coords=['x%d' % i for i in range(1, 8)], degs=[1] * 7,
    # the element (1 + tau xi1xi2xi3) mixes parities (tau is an ODD parameter),
    # so there is no superalgebra with these coordinates: dims only, parity off
    par=[0] * 7,
    neg=['dx%d' % i for i in range(1, 8)],
    g0=['x1 dx7 + x4 dx2 + x5 dx3 + x7 dx6', 'x2 dx7 + x4 dx1 + x6 dx3 + x7 dx5',
        'x3 dx7 + x5 dx1 + x6 dx2 + x7 dx4',
        'x2 dx1 + x6 dx5', 'x3 dx1 + x6 dx4', 'x1 dx2 + x5 dx6',
        'x1 dx1 + x2 dx2 + x5 dx5 + x6 dx6', 'x1 dx1 + x3 dx3 + x4 dx4 + x6 dx6',
        'x3 dx2 + x5 dx4', 'x1 dx3 + x4 dx6', 'x2 dx3 + x4 dx5',
        'x3 dx4', 'x2 dx5', 'x2 dx4 + x3 dx5', 'x1 dx6', 'x1 dx5 + x2 dx6',
        'x1 dx4 + x3 dx6'])

# lowest/highest-weight vectors of g_1 printed in the paper (for partial prolongs)
PARTIAL = {
    # Sect. 18.1, in the coordinates x1..x7 = z0, z12, z13, z23, z1, z2, z3
    'kas~7': dict(rename={'x1': 'z0', 'x2': 'z12', 'x3': 'z13', 'x4': 'z23',
                          'x5': 'z1', 'x6': 'z2', 'x7': 'z3'},
                  vecs={'v1': 'x1 x2 dx6 + x1 x3 dx7 + x1 x5 dx1 + x2 x5 dx2 + x3 x5 dx3 + x4 x5 dx4'
                              ' + x2 x7 dx4 + x3 x6 dx4 + x5 x6 dx6 + x5 x7 dx7',
                        'v2': 'x2 x3 dx3 + x2 x4 dx4 + x2 x5 dx5 + x2 x6 dx6 + x3 x6 dx7 + x4 x5 dx7 + x5 x6 dx1',
                        'v3': 'x2 x3 dx4 + x2 x5 dx6 + x3 x5 dx7 + x5^(2) dx1',
                        'w2': 'x1 x7 dx2', 'w3': 'x1^(2) dx2'}),
    # Sect. 19 (kas(8)), x1..x8 <-> 1, xi1xi2, xi1xi3, xi2xi3, xi1, xi2, xi3, xi1xi2xi3
    'kas8': dict(rename=KAS8_MAP,
                 vecs={'v1': 'x2 x3 dx3 + x2 x4 dx4 + x2 x5 dx5 + x2 x6 dx6 + x3 x6 dx7 + x4 x5 dx7'
                             ' + x5 x6 dx1 + x5 x8 dx3 + x6 x8 dx4',
                       'h1': 'x1 x6 dx2 + x1 x7 dx3 + x6 x7 dx8', 'h2': 'x1^(2) dx8'}),
    # Sect. 21.2.1
    'sb~7': dict(rename={},
                 vecs={'w1': 'x2 x3 dx7 + x2 x5 dx1 + x2 x6 dx2 + x2 x7 dx4 + x3 x4 dx1 + x3 x6 dx3'
                             ' + x3 x7 dx5 + x4 x6 dx4 + x5 x6 dx5',
                       'w2': 'x2 x3 dx1 + x2 x6 dx4 + x3 x6 dx5',
                       'w3': 'x3^(2) dx1 + x3 x6 dx4', 'v3': 'x1^(2) dx6'}),
    # Sect. 7 (vle(9))
    'vle9': dict(rename={},
                 vecs={'v1': 'x1 x4 dx3 + x1 x6 dx1 + x2 x5 dx3 + x2 x6 dx2 + x1 dx8 + x2 dx7'
                             ' + x4 x6 dx4 + x4 x9 dx7 + x5 x6 dx5 + x5 x9 dx8 + x6 x7 dx7 + x6 x8 dx8',
                       'v2': 'x1 x5 dx3 + x1 x6 dx2 + x1 dx7 + x5 x6 dx4 + x5 x9 dx7 + x6 x8 dx7'}),
}


def gen_sbt(n=4):
    """sb~(2^n - 1) for n even (Sect. 21.3): g_-1 = Pi(Vol(0|n))/K(1+Xi)vol,
    g_0 = (1+Xi) svect(0|n) acting by X(g vol) = ((1+Xi)D(g) + D(Xi) g) vol."""
    full = [()] + _subsets(n)
    top = tuple(range(n))
    S = [A for A in full if A]                         # vol == Xi vol in the quotient
    names = ['v' + ''.join(str(a + 1) for a in A) for A in S]

    def mul(A, B):
        if set(A) & set(B):
            return None
        return tuple(sorted(set(A) | set(B)))

    def der(j, A):
        return tuple(a for a in A if a != j) if j in A else None
    # vect(0|n) basis xi^B d_j ; svect = kernel of div(xi^B d_j) = d_j(xi^B)
    vb = [(B, j) for B in full for j in range(n)]
    from f2vf import kernel_bits
    imgs = []
    for B, j in vb:
        dB = der(j, B)
        imgs.append(0 if dB is None else 1 << full.index(dB))
    ker = kernel_bits(imgs)
    g0 = []
    for kv in ker:
        D = [vb[i] for i in range(len(vb)) if (kv >> i) & 1]
        DXi = set()
        for B, j in D:
            t = der(j, top)
            if t is not None:
                m = mul(B, t)
                if m is not None:
                    DXi ^= {m}
        terms = set()
        for A in S:
            out = set()
            for B, j in D:                              # (1+Xi) D(g)
                dA = der(j, A)
                if dA is None:
                    continue
                m = mul(B, dA)
                if m is None:
                    continue
                out ^= {m}
                m2 = mul(top, m)
                if m2 is not None:
                    out ^= {m2}
            for C in DXi:                                # D(Xi) g
                m = mul(C, A)
                if m is not None:
                    out ^= {m}
            if () in out:                                # vol == Xi vol
                out ^= {()}
                out ^= {top}
            for img in out:
                terms ^= {'%s d%s' % (names[S.index(A)], names[S.index(img)])}
        if terms:
            g0.append(' + '.join(sorted(terms)))
    # parity: Pi(xi^k vol), p(vol) = n mod 2
    par = [(len(A) + n + 1) % 2 for A in S]
    return dict(section='21.3', table='sb~(%d|%d) -> sb~(%d;N)' % (2 ** (n - 1), 2 ** (n - 1) - 1, 2 ** n - 1),
                coords=names, degs=[1] * len(S), par=par,
                neg=['d' + x for x in names], g0=g0)


ROWS['sb~15'] = gen_sbt(4)
