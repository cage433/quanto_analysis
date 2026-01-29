from unittest import TestCase

from modelling.maths.models.black_scholes import BlackScholes
from modelling.maths.models.option_right import CALL, PUT
from modelling.maths.random.random_number_generator import RandomNumberGenerator
from quantity.quantity_test_utils import QtyTestUtils
from random_test.random_test_case import RandomisedTest


class BlackScholesTestCase(TestCase, QtyTestUtils):

    @RandomisedTest(number_of_runs=100)
    def test_intrinsic(self, rng):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol, T = rng.choice((0, 1), (1, 0), (0, 0))   # Any of these will lead to the intrinsic value being returned
        bs = BlackScholes(rng.choice(CALL, PUT), S=F, X=K, vol=vol, T=T)
        expected = bs.right.intrinsic(F, K)
        self.assertAlmostEqual(expected, bs.value, delta=1e-6)

    def test_known_values(self):
        self.assertAlmostEqual(
            BlackScholes(right=CALL, S=100, X=100, vol=0.2, T=1).value,
            7.965567,
            delta=1e-6
        )
        self.assertAlmostEqual(
            BlackScholes(right=PUT, S=150, X=100, vol=0.2, T=1).value,
            0.192475,
            delta=1e-6
        )

    @RandomisedTest(number_of_runs=30)
    def test_put_call_parity(self, rng):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol = rng.uniform()
        T = rng.uniform()
        call_value = BlackScholes(CALL, S=F, X=K, vol=vol, T=T).value
        put_value = BlackScholes(PUT, S=F, X=K, vol=vol, T=T).value
        self.assertAlmostEqual(
            call_value - put_value,
            F - K,
            delta=1e-5
        )

    @RandomisedTest(number_of_runs=30)
    def test_delta(self, rng: RandomNumberGenerator):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol = rng.uniform(0.1, 0.5)
        T = rng.uniform(0.1, 1.0)
        dF = 0.01
        right = rng.choice(CALL, PUT)
        c_up = BlackScholes(right, S=F + dF, X=K, vol=vol, T=T)
        c = BlackScholes(right, S=F, X=K, vol=vol, T=T)
        c_dn = BlackScholes(right, S=F - dF, X=K, vol=vol, T=T)
        numeric_delta = (c_up.value - c_dn.value) / (dF * 2.0)
        self.assertAlmostEqual(c.delta, numeric_delta, delta = 1e-4)

    @RandomisedTest(number_of_runs=30)
    def test_gamma(self, rng):
        F, K = [rng.uniform(90, 110) for _ in range(2)]
        vol = 0.1 + rng.uniform() * 0.5
        T = 0.1 + rng.uniform()
        call = BlackScholes(CALL, S=F, X=K, vol=vol, T=T)
        put = BlackScholes(PUT, S=F, X=K, vol=vol, T=T)
        self.assertAlmostEqual(call.gamma, put.gamma, delta=1e-6)
        dF = 0.01
        c_up = BlackScholes(CALL, S=F + dF, X=K, vol=vol, T=T)
        c_dn = BlackScholes(CALL, S=F - dF, X=K, vol=vol, T=T)
        numeric_gamma = (c_up.value - 2 * call.value + c_dn.value) / (dF * dF)
        self.assertAlmostEqual(call.gamma, numeric_gamma, delta = 1e-4)
