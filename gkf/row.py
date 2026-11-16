"""One whole cohomological degree of h^{(1)}(0|n): dim H^q_k for every internal degree k.

column.py and colbig.py walk a column (k fixed, q running); this walks a ROW (q fixed,
k running), which is what a table is read off by when only the top degree is missing.
Each k needs two ranks, d^{q-1} and d^q, and the second is the expensive one; the k's
are taken in increasing order of that cost, and any block whose dense FLINT matrix
would exceed --mem gigabytes is skipped and reported as skipped, so that a partial row
is never mistaken for a complete one.

Torus weight 0 throughout, as in colbig.py.

    ../venv/bin/python row.py <n> <q> <p> [--mem G] [--also p2]
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flint import nmod_mat                                        # noqa: E402
from po import Po                                                 # noqa: E402
from ce import CE                                                 # noqa: E402


def rank_block(ce, src, tgt, p, say):
    if not src or not tgt:
        return 0
    ti = {m: i for i, m in enumerate(tgt)}
    t = time.time()
    M = nmod_mat(len(src), len(tgt), p)
    nz = 0
    for i, mon in enumerate(src):
        for m2, v in ce.d_monomial(mon).items():
            j = ti.get(m2)
            if j is None:
                raise AssertionError("d left the block")
            M[i, j] = int(v) % p
            nz += 1
    tf = time.time() - t
    t = time.time()
    rk = M.rank()
    say(f"      {len(src)}x{len(tgt)}, nnz {nz}, rank {rk}"
        f"   [fill {tf:.1f} s, rank {time.time()-t:.1f} s]")
    del M
    return rk


def main():
    n, q, p = (int(x) for x in sys.argv[1:4])
    args = sys.argv[4:]
    mem = float(args[args.index("--mem") + 1]) if "--mem" in args else 8.0
    also = [int(args[i + 1]) for i, a in enumerate(args) if a == "--also"]
    only = ([int(x) for x in args[args.index("--ks") + 1].split(",")]
            if "--ks" in args else None)
    g = Po(n, "h1")
    ce = CE(g)
    assert not ce.check_d2()
    w = tuple([0] * len(ce.cwt[0]))
    os.makedirs("runs", exist_ok=True)
    out = open(f"runs/row-n{n}-q{q}.log", "a")

    def say(s):
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    say(f"\n==== n={n} q={q} p={p} also={also} mem={mem} GB ====")
    bas = {}
    for qq in (q - 1, q, q + 1):
        t = time.time()
        bas[qq] = {}
        for k in range(-3 * n, 3 * n + 1):
            b = ce.basis_block(qq, k, w)
            if b:
                bas[qq][k] = b
        say(f"  dim C^{qq}_k : "
            + ", ".join(f"{k}:{len(v)}" for k, v in sorted(bas[qq].items()))
            + f"   [{time.time()-t:.1f} s]")

    def cost(k):
        s, t = len(bas[q].get(k, [])), len(bas[q + 1].get(k, []))
        return s * t

    ks = sorted(bas[q], key=cost)
    if only is not None:
        ks = [k for k in ks if k in only]
    res = {}
    for k in ks:
        s, t = len(bas[q][k]), len(bas[q + 1].get(k, []))
        gb = s * t * 8 / 2 ** 30
        if gb > mem:
            say(f"    k={k}: SKIPPED, d^{q} is {s}x{t} = {gb:.1f} GB > {mem} GB")
            continue
        say(f"    k={k}  (d^{q} is {s}x{t} = {gb:.2f} GB)")
        rin = rank_block(ce, bas[q - 1].get(k, []), bas[q][k], p, say)
        rout = rank_block(ce, bas[q][k], bas[q + 1].get(k, []), p, say)
        h = s - rin - rout
        res[k] = (s, rin, rout, h)
        say(f"      => dim H^{q}_{{{k}}} = {h}   (mod {p})")
        for p2 in also:
            r2in = rank_block(ce, bas[q - 1].get(k, []), bas[q][k], p2, say)
            r2out = rank_block(ce, bas[q][k], bas[q + 1].get(k, []), p2, say)
            say(f"      => dim H^{q}_{{{k}}} = {s - r2in - r2out}   (mod {p2})"
                + ("" if (r2in, r2out) == (rin, rout) else "   *** DIFFERS ***"))

    say(f"\n  h^(1)(0|{n}), cohomological degree q = {q}, weight 0, mod {p}:")
    say("    k        " + "".join(f"{k:7d}" for k in sorted(res)))
    say("    dim C    " + "".join(f"{res[k][0]:7d}" for k in sorted(res)))
    say("    dim H    " + "".join(f"{res[k][3]:7d}" for k in sorted(res)))
    miss = [k for k in bas[q] if k not in res]
    if miss:
        say(f"    not computed: k = {sorted(miss)}")
    out.close()


if __name__ == "__main__":
    main()
