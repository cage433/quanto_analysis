from functools import total_ordering
from numbers import Number
from typing import Union

from modelling.quantity.uom import UOM, SCALAR
from modelling.utils.type_utils import checked_type


@total_ordering
class Qty:
    def __init__(self, value: Number, uom: UOM):
        self.value: float = checked_type(value, Number)
        self.uom: UOM = checked_type(uom, UOM)

    def __str__(self):
        return self.fmt(n_dp=3)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, Qty) and self.value == other.value and self.uom == other.uom

    def fmt(self, n_dp):
        uom_text = "" if self.uom == SCALAR else f" {self.uom}"
        return f"{round(self.value, n_dp)}{uom_text}"

    def __lt__(self, other: 'Qty'):
        if self.uom != other.uom:
            raise ValueError(f"Cannot compare Qty with different uom, {self} vs {other}")
        return self.value < other.value

    def __mul__(self, other: Union['Qty', Number]) -> 'Qty':
        if isinstance(other, Number):
            return self * Qty.to_qty(other)
        return Qty(self.value * other.value, self.uom * other.uom)

    def __add__(self, other: 'Qty') -> 'Qty':
        if self.uom != other.uom:
            raise ValueError(f"Cannot add Qty with different uom, {self} vs {other}")
        return Qty(self.value + other.value, self.uom)

    def __neg__(self) -> 'Qty':
        return Qty(-self.value, self.uom)

    def negate(self) -> 'Qty':
        return Qty(-self.value, self.uom)

    def __sub__(self, other: 'Qty') -> 'Qty':
        return self + (-other)

    def __truediv__(self, other: Union['Qty', Number]) -> 'Qty':
        if isinstance(other, Number):
            return self / Qty.to_qty(other)
        return Qty(self.value / other.value, self.uom / other.uom)

    @property
    def inverse(self) -> 'Qty':
        return Qty(1 / self.value, self.uom.inverse)

    def checked_value(self, uom: UOM) -> float:
        assert self.uom == uom, f"Expected uom {uom}, this is {self}"
        return self.value

    @property
    def checked_scalar_value(self) -> float:
        return self.checked_value(SCALAR)

    def max(self, other) -> 'Qty':
        return max(self, other)

    @property
    def abs(self) -> 'Qty':
        return Qty(abs(self.value), self.uom)

    def __abs__(self) -> 'Qty':
        return self.abs

    @property
    def is_zero(self):
        return self.value == 0

    def is_almost_zero(self, eps=1e-6):
        return abs(self.value) < eps

    @staticmethod
    def to_qty(qty_or_number) -> 'Qty':
        if isinstance(qty_or_number, Number):
            return Qty(qty_or_number, SCALAR)
        return checked_type(qty_or_number, Qty)

    @staticmethod
    def sum(qtys: list['Qty']) -> 'Qty':
        assert len(qtys) > 0, "Can't sum empty quantities"
        s = qtys[0]
        for q in qtys[1:]:
            s += q
        return s