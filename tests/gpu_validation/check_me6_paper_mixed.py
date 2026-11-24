"""Separate the Table 9/Eq. (41) and Eq. (43) reflection conventions."""
import re
from pathlib import Path

import mpmath as mp


def main():
    mp.mp.dps = 50
    source = (Path(__file__).resolve().parents[2]/'src/me6opti_operator.F90').read_text()

    def coefficients(name):
        section = source.split(name+'(9,3)=reshape', 1)[1].split('/),', 1)[0]
        ratios = re.findall(r'(-?\d+)\.e0_real_kind\s*/\s*(\d+)\.e0_real_kind', section)
        values = [mp.mpf(n)/mp.mpf(d) for n, d in ratios]
        if len(values) != 27:
            raise AssertionError(f'{name}: expected 27 stencil coefficients')
        return [values[i:i+9] for i in range(0, 27, 9)]

    interp = coefficients('me6opti_interp_table')
    penalty = coefficients('me6opti_penalty_table')
    outer = [mp.mpf(75)/64, -mp.mpf(25)/384, mp.mpf(3)/640]
    central = [mp.mpf(3)/4, -mp.mpf(3)/20, mp.mpf(1)/60]

    def polynomial(coef, z):
        return sum(c*mp.exp(p*z) for p, c in enumerate(coef, -4))

    def response(k, b, h, reflection):
        z = mp.j*k*h
        cross = sum(2*c*mp.sinh(i*z) for i, c in enumerate(central, 1))
        return sum(w*(polynomial(a, b*h)*(polynomial(a, z)*cross+polynomial(p, z))
                      -polynomial(a, -b*h)*(polynomial(a, -z)*cross+reflection*polynomial(p, -z)))
                   for w, a, p in zip(outer, interp, penalty))/h**2

    print('N Table9 Eq41_literal ASTR_odd')
    for n, reference in zip((20, 40, 80, 160, 320), ('0.0243', '0.000413', '6.65e-6', '1.05e-7', '1.65e-9')):
        h = 1/mp.mpf(n)
        k, b = 10, 2
        mean_errors = []
        for reflection in (1, -1):
            error = response(k, b, h, reflection) - mp.j*k*(b+mp.j*k)
            # On this fixture all (x,y) pairs with the same index sum share an error.
            mean_errors.append(sum(min(i, 2*n-i)*abs(mp.im(error*mp.exp((b+mp.j*k)*i*h)))
                                   for i in range(1, 2*n))/(10*n*n))
        if format(float(mean_errors[0]), '.3g') != format(float(reference), '.3g'):
            raise AssertionError(f'Table 9 differs at N={n}')
        print(n, reference, *(mp.nstr(e, 16) for e in mean_errors))
    literal, astr = [mp.re(response(mp.pi, 0, 1, sign)) for sign in (1, -1)]
    paper_cutoff = -mp.mpf(29192)/3750
    if abs(literal) > mp.mpf('1e-46') or abs(astr-paper_cutoff) > mp.mpf('1e-46'):
        raise AssertionError('cut-off responses do not match the Eq. (41)/(43) split')
    print('cutoff', mp.nstr(literal, 16), mp.nstr(astr, 16))


if __name__ == '__main__':
    main()
