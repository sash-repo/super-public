"""The expression Theta of formula (Gz10terms) of the text, term by term.

Each entry is (coefficient, chi_derivs, chi_slots, A_derivs, A_slots,
omegas_up, omegas_low) in the format of `iexpr.py`: the letters i, j, k are the
three output slots, every other letter a dummy bound by omega^{..}.
The integer coefficients are the printed ones; Gz_{1,0}(chi, A) = pi(Theta).
"""

from fractions import Fraction

TERMS = [
    (Fraction(3), (), ('a', 'c', 'j'), ('b', 'd'), ('i', 'k'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\chi_{acj}\,\partial_{b}\partial_{d}A_{ik}
    (Fraction(3), ('a',), ('c', 'j', 'k'), ('b',), ('i', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{cjk}\,\partial_{b}A_{id}
    (Fraction(1), ('a',), ('k', 'j', 'i'), ('c',), ('d', 'b'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{kji}\,\partial_{c}A_{db}
    (Fraction(-2), ('k',), ('i', 'a', 'c'), ('b',), ('d', 'j'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{k}\chi_{iac}\,\partial_{b}A_{dj}
    (Fraction(-3), ('a',), ('c', 'i', 'k'), ('j',), ('b', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{cik}\,\partial_{j}A_{bd}
    (Fraction(-3), ('a',), ('c', 'i', 'k'), ('d',), ('j', 'b'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{cik}\,\partial_{d}A_{jb}
    (Fraction(-2), (), ('a', 'j', 'c'), ('b', 'k'), ('i', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\chi_{ajc}\,\partial_{b}\partial_{k}A_{id}
    (Fraction(-3), (), ('a', 'j', 'k'), ('c', 'i'), ('d', 'b'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\chi_{ajk}\,\partial_{c}\partial_{i}A_{db}
    (Fraction(-1), ('a',), ('j', 'c', 'b'), ('i',), ('k', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{jcb}\,\partial_{i}A_{kd}
    (Fraction(-2), ('a', 'i'), ('k', 'c', 'b'), (), ('j', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\partial_{i}\chi_{kcb}\,A_{jd}
    (Fraction(-2), ('i',), ('a', 'c', 'k'), ('b',), ('d', 'j'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{i}\chi_{ack}\,\partial_{b}A_{dj}
    (Fraction(1), ('a', 'c'), ('k', 'i', 'j'), (), ('b', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\partial_{c}\chi_{kij}\,A_{bd}
    (Fraction(1), ('k',), ('i', 'a', 'c'), ('j',), ('d', 'b'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{k}\chi_{iac}\,\partial_{j}A_{db}
    (Fraction(-3), ('a',), ('b', 'c', 'i'), ('d',), ('j', 'k'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{bci}\,\partial_{d}A_{jk}
    (Fraction(-1), (), ('a', 'c', 'k'), ('b', 'i'), ('d', 'j'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\chi_{ack}\,\partial_{b}\partial_{i}A_{dj}
    (Fraction(2), ('a', 'k'), ('j', 'c', 'b'), (), ('d', 'i'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\partial_{k}\chi_{jcb}\,A_{di}
    (Fraction(1), ('a',), ('i', 'c', 'b'), ('k',), ('j', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{a}\chi_{icb}\,\partial_{k}A_{jd}
    (Fraction(-2), (), ('a', 'c', 'i'), ('d', 'j'), ('b', 'k'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\chi_{aci}\,\partial_{d}\partial_{j}A_{bk}
    (Fraction(1), ('j',), ('a', 'i', 'c'), ('d',), ('k', 'b'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{j}\chi_{aic}\,\partial_{d}A_{kb}
    (Fraction(1), ('j',), ('a', 'c', 'k'), ('i',), ('b', 'd'), [('a', 'b'), ('c', 'd')], []),  # \omega^{ab}\omega^{cd}\,\partial_{j}\chi_{ack}\,\partial_{i}A_{bd}
]
