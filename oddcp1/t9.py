"""Section 16, first paragraph: the SL(2)-action on H^1(T_2) and H^1(T_4) of
the retract CP^{1|4} is trivial, as [Ba1, Thm. 1] claims; for comparison,
CP^{1|5}.
"""
import quot, sl2inv as S
ks=(1,1,1,1); m=4
print("Bashkin-1 Th1 claims: SL_2(C) acts TRIVIALLY on H^1(T_2) and H^1(T_4).")
for q in (2,4):
    Q=quot.Quot(ks,q)
    e,f=S.s_prime(1,m)
    inv=Q.invariants(e,f)
    print(f"   q={q}: dim H^1 = {Q.dim}, dim H^1^s = {len(inv)}  -> action trivial: {len(inv)==Q.dim}")
print()
print("For comparison, retract CP^{1|5} (Bashkin-5):")
for q in (2,4):
    Q=quot.Quot((1,1,1,1,1),q)
    e,f=S.s_prime(1,5)
    inv=Q.invariants(e,f)
    print(f"   q={q}: dim H^1 = {Q.dim}, dim H^1^s = {len(inv)}  -> trivial: {len(inv)==Q.dim}")
