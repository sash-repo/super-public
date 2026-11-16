from ab3dom import *
import sys
D = (sc(h,(-1,-1,-1,1)), (F(0),F(0),F(1),F(0)), (F(0),F(1),F(-1),F(0)), (F(1),F(-1),F(0),F(0)))
T = ((F(0),)*3+(F(1),), (h,-h,-h,-h), (F(0),F(0),F(1),F(0)), (F(0),F(1),F(-1),F(0)))
AD = cartan(D); AD[0] = [-2*x for x in AD[0]]; AT = [[2,-1,0,0],[-3,0,1,0],[0,-1,2,-2],[0,0,-1,2]]
print("even simple D", even_simple(D)); print("even simple T", even_simple(T))
which = sys.argv[1]
B, A = (D, AD) if which == "D" else (T, AT)
R = range(-4, 31) if which == "D" else range(-10, 16)
out = []
for lab in product(R, range(0, 7), range(0, 7), range(0, 7)) if which=="D" else product(range(0,9), R, range(0,7), range(0,7)):
    lam = lam_from_labels(B, A, lab)
    if dominant(B, lam): out.append(lab)
print(which, len(out))
import json; json.dump(out, open("dom_%s.json" % which, "w"))
