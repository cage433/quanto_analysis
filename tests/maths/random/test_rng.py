import unittest

from modelling.maths.random.random_number_generator import RandomNumberGenerator
from random_test.random_test_case import RandomisedTest


class RNGTests(unittest.TestCase):
    def test_reproducibility(self):
        terms = list(range(100))
        rng = RandomNumberGenerator(seed=1234)
        rng2 = RandomNumberGenerator(seed=1234)

        self.assertEqual(
            rng.shuffle(terms),
            rng2.shuffle(terms),
        )

        self.assertNotEqual(
            rng.shuffle(terms),
            rng.shuffle(terms),
        )

    @RandomisedTest(number_of_runs=10)
    def test_single_element_random_times(self, rng):
        def check_single_value(times, t):
            self.assertEqual(len(times), 1)
            self.assertAlmostEqual(times[0], t, delta=1e-6)

        times = rng.random_times(n_times=1)
        check_single_value(times, times[0])

        t0 = rng.uniform()
        check_single_value(
            rng.random_times(n_times=1, t0=t0),
            t0
        )

        T = rng.uniform()
        check_single_value(
            rng.random_times(n_times=1, T=T),
            T
        )

        T = t0
        check_single_value(
            rng.random_times(n_times=1, t0=t0, T=T),
            T
        )

        with self.assertRaises(AssertionError):
            rng.random_times(n_times=1, t0=t0, T=t0 + rng.uniform())

    @RandomisedTest(number_of_runs=10)
    def test_random_times_with_multiple_elements(self, rng):
        n_times = rng.randint(2, 20)

        def check_increasing(ts):
            self.assertEqual(len(ts), n_times)
            for i_t in range(len(ts) - 1):
                self.assertGreaterEqual(ts[i_t + 1], ts[i_t])

        times = rng.random_times(n_times)
        check_increasing(times)
        self.assertEqual(len(times), n_times)

        t0 = rng.uniform()
        times = rng.random_times(n_times, t0=t0)
        check_increasing(times)
        self.assertEqual(times[0], t0)

        T = rng.uniform()
        times = rng.random_times(n_times, T=T)
        check_increasing(times)
        self.assertAlmostEqual(times[-1], T, delta=1e-6)

        T = t0 + rng.uniform()
        times = rng.random_times(n_times, t0=t0, T=T)
        check_increasing(times)
        self.assertAlmostEqual(times[0], t0, delta=1e-6)
        self.assertAlmostEqual(times[-1], T, delta=1e-6)

    @RandomisedTest(number_of_runs=10)
    def test_invalid_random_times(self, rng):
        with self.assertRaises(AssertionError):
            rng.random_times(n_times=0)

        with self.assertRaises(AssertionError):
            rng.random_times(n_times=1, t0=1.0, T=2.0)
