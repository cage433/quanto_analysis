import unittest
from contextlib import redirect_stdout
from io import StringIO

from random_test.random_test_case import RandomisedTest


class RandomTestTestSuite(unittest.TestCase):

    def test_distinct_seeds_are_used(self):

        @RandomisedTest(number_of_runs=100, num_allowed_failures=80)
        def test_some_heads(rng):
            self.assertTrue(rng.is_heads())

        @RandomisedTest(number_of_runs=100)
        def test_all_heads(rng):
            self.assertTrue(rng.is_heads())

        @RandomisedTest(number_of_runs=100, num_allowed_failures=80)
        def test_some_tails(rng):
            self.assertFalse(rng.is_heads())

        @RandomisedTest(number_of_runs=100)
        def test_all_tails(rng):
            self.assertFalse(rng.is_heads())

        test_some_heads()
        test_some_tails()

        def check_test_fails(test_func):
            stream = StringIO()
            with redirect_stdout(stream):
                with self.assertRaises(AssertionError):
                    test_func()
                self.assertTrue(stream.getvalue())

        check_test_fails(test_all_heads)
        check_test_fails(test_all_tails)

    def test_fails_after_breaching_allowed_failure_limit(self):
        @RandomisedTest(number_of_runs=10, num_allowed_failures=5)
        def test_should_fail(rng):
            self.assertEqual(1, 2)

        stream = StringIO()
        with redirect_stdout(stream):
            with self.assertRaises(AssertionError):
                test_should_fail()
            self.assertTrue(stream.getvalue())