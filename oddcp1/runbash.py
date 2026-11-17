"""What "odd parameters added to Bashkin" needs: the ODD layers H^1(T_q), q odd,
of the retracts his three papers classify over Spec Cee.

Bashkin 1: CP^{1|4}, i.e. k = (1,1,1,1);  Bashkin 5: CP^{1|5}, k = (1,1,1,1,1);
Bashkin 2: the tuples (k,k,2,0).  Green's theorem makes the EVEN q the only
ones over Spec Cee, which is why the odd ones are absent from the papers.
"""
import bashkin as B
print(__doc__, flush=True)
for ks in ([1, 1, 1, 1], [1, 1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 2, 0],
           [4, 4, 2, 0], [2, 2, 2], [2, 2, 3]):
    m = len(ks)
    row = []
    for q in range(-1, m + 1):
        h0, h1 = B.h0_h1(ks, q)
        row.append((q, h0, h1))
    print(f"\nretract {tuple(ks)} (m = {m}):", flush=True)
    print("   q       " + "".join(f"{q:>6}" for q, _, _ in row), flush=True)
    print("   h^0     " + "".join(f"{h:>6}" for _, h, _ in row), flush=True)
    print("   h^1     " + "".join(f"{h:>6}" for _, _, h in row), flush=True)
    odd = [(q, h) for q, _, h in row if q % 2 == 1 and h]
    print(f"   the odd layers that carry obstructions: "
          f"{', '.join(f'H^1(T_{q}) = {h}' for q, h in odd) or 'none'}",
          flush=True)
