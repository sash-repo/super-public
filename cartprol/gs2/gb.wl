f = $ScriptCommandLine[[2]];
Get[f];
t = AbsoluteTiming[gb = GroebnerBasis[eqs, vars, Modulus -> 2]];
Print[f, "  time ", t[[1]], "  GB length ", Length[gb], "  GB = {1}? ", gb === {1}];
If[Length[gb] < 40, Print[gb]];
