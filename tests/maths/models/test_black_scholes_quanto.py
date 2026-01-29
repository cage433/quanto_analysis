from unittest import TestCase

from modelling.maths.models.black_scholes import BlackScholes
from modelling.maths.models.black_scholes_quanto import BlackScholesQuanto
from modelling.maths.models.option_right import CALL, PUT
from modelling.maths.random.random_number_generator import RandomNumberGenerator
from quantity.quantity_test_utils import QtyTestUtils
from random_test.random_test_case import RandomisedTest


class BlackScholesTestCase(TestCase, QtyTestUtils):


    @RandomisedTest(number_of_runs=30)
    def test_delta(self, rng: RandomNumberGenerator):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        fx_rate = rng.uniform(0.8, 1.2)
        fx_rate = 1.0
        F_vol = rng.uniform(0.1, 0.5)
        fx_vol = rng.uniform(0.1, 0.5)
        rho = rng.uniform(-1, 1)
        T = rng.uniform(0.1, 1.0)
        dF = 0.01
        right = rng.choice(CALL, PUT)
        def quanto(f):
            return BlackScholesQuanto(right, S=f, fixed_fx_rate=fx_rate, X=K, S_vol=F_vol, fx_vol=fx_vol, rho=rho, T=T)
        c_up = quanto(F + dF)
        c = quanto(F)
        c_dn = quanto(F - dF)
        numeric_delta = (c_up.value - c_dn.value) / (dF * 2.0)
        self.assertAlmostEqual(c.delta, numeric_delta, delta = 1e-4)
