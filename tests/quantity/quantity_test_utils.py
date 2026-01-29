from typing import Optional

from modelling.quantity.quantity import Qty


class QtyTestUtils:
    def assertVeryClose(self, q1: Qty, q2: Qty, delta: Optional[Qty] = None):
        delta = delta or Qty(1e-6, q1.uom)
        if abs(q1 - q2) > delta:
            raise AssertionError(f"{q1} not withing {delta} of {q2}")
