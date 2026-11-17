"""The sl(2)-module structure of H^1(T_1) against the reading of T_1.

m=1:  T_1 = Theta (x) E = O(2-k),                        H^1 = R_{k-4}.
m=2:  T_1 = T_1^{(1)} + T_1^{(2)}, where T_1^{(i)} is the extension
      0 -> O(-k_i) -> T_1^{(i)} -> O(2-k_i) -> 0  of class  k_j . (generator),
      hence  = V (x) O(1-k_i)  if k_j =/= 0  (B--O Prop. 18),
             = O(-k_i) + O(2-k_i)  if k_j = 0.
      So H^1(T_1^{(i)}) = V (x) R_{k_i-3} = R_{k_i-2} + R_{k_i-4}  (k_i >= 3),
      resp. R_{k_i-2} + R_{k_i-4} with the negative indices dropped.
"""
import oddpar as O


def predict(ks):
    m, out = len(ks), {}
    for i in range(m):
        ki, others = ks[i], [ks[j] for j in range(m) if j != i]
        if m == 1:
            degs = [ki - 4]
        elif all(k == 0 for k in others):           # the split, degenerate case
            degs = [ki - 2, ki - 4]
        else:                                       # V (x) O(1-k_i), k_i >= 3
            degs = [ki - 2, ki - 4] if ki >= 3 else []
        for d in degs:
            if d >= 0:
                out[d] = out.get(d, 0) + 1
    return out


print(f"{'retract':>10} {'dim':>4}  {'sl(2) decomposition':>26}  {'predicted':>26}")
print("-" * 74)
bad = 0
for ks in ([4], [5], [6], [7], [8], [9],
           [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [2, 1], [2, 2], [3, 1],
           [3, 2], [3, 3], [4, 1], [4, 2], [4, 3], [4, 4], [5, 2], [5, 3],
           [5, 4], [5, 5], [6, 6], [7, 5]):
    L = O.Layer(ks, 1)
    dec, pre = L.sl2_decomposition(), predict(ks)
    ok = dec == pre
    bad += not ok
    f = lambda d: " + ".join(f"R_{k}" + (f"^{v}" if v > 1 else "")
                             for k, v in sorted(d.items(), reverse=True)) or "0"
    print(f"{str(tuple(ks)):>10} {L.dim:>4}  {f(dec):>26}  {f(pre):>26}"
          f"  {'' if ok else '<-- MISMATCH'}")
print(f"\n{bad} mismatches out of 26")
