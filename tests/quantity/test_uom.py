import unittest

from modelling.quantity.uom import USD, MWH, SCALAR, GBP, EUR


class UOMTestCase(unittest.TestCase):
    def test_numerator_and_denominator(self):
        self.assertEqual(USD, (USD / MWH).numerator)
        self.assertEqual(MWH, (USD / MWH).denominator)
        self.assertEqual(SCALAR, USD.denominator)
        self.assertEqual(SCALAR, USD.inverse.numerator)
        self.assertEqual(USD, USD.inverse.denominator)

    def test_sorting(self):
        self.assertEqual(sorted([GBP, USD]), [GBP, USD])
        self.assertEqual(sorted([USD, GBP]), [GBP, USD])
        self.assertEqual(sorted([USD, GBP, EUR / MWH]), [EUR / MWH, GBP, USD])
