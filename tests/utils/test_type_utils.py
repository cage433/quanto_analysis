import unittest

from modelling.utils.type_utils import checked_type, checked_list_type


class TypeUtilsTest(unittest.TestCase):
    def test_checked_type(self):
        checked_type("foo", str)

        with self.assertRaises(AssertionError):
            checked_type(1, str)

    def test_checked_list_type(self):
        checked_list_type(["foo"], str)
        checked_list_type([], str)

        with self.assertRaises(AssertionError):
            checked_list_type("foo", str)

        with self.assertRaises(AssertionError):
            checked_list_type([1], str)
