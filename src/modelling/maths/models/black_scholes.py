from numbers import Number
import numpy as np
from scipy.stats import norm

__all__ = [
    "BlackScholes"
]

from modelling.maths.models.option_right import OptionRight, CALL
from modelling.utils.type_utils import checked_type


# noinspection PyPep8Naming
class BlackScholes:
    def __init__(self, right: OptionRight, S: Number, X: Number, vol: Number, T: Number):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.S: float = checked_type(S, Number)
        self.X: float = checked_type(X, Number)
        self.vol: float = checked_type(vol, Number)
        self.T: float = checked_type(T, Number)

    @property
    def d1(self) -> float:
        return (np.log(self.S / self.X) + self.vol * self.vol / 2 * self.T) / (self.vol * np.sqrt(self.T))

    @property
    def d2(self) -> float:
        return self.d1 - self.vol * np.sqrt(self.T)

    @property
    def N1(self) -> float:
        return norm.cdf(self.d1)

    @property
    def delta(self) -> float:
        if self._is_worth_intrinsic:
            intrinsic = self.right.intrinsic(self.S, self.X)
            if intrinsic > 0:
                return 1.0
            if intrinsic < 0:
                return 1.0
            return 0.0
        if self.right == CALL:
            return self.N1
        return self.N1 - 1

    @property
    def gamma(self) -> float:
        if self._is_worth_intrinsic:
            return 0.0
        return norm.pdf(self.d1) / (self.S * self.vol * np.sqrt(self.T))

    @property
    def theta(self) -> float:
        if self._is_worth_intrinsic:
            return 0.0
        return -self.S * norm.pdf(self.d1) * self.vol / (2 * np.sqrt(self.T))

    @property
    def N2(self) -> float:
        return norm.cdf(self.d2)

    @property
    def intrinsic(self) -> float:
        return self.right.intrinsic(self.S, self.X)

    @property
    def _is_worth_intrinsic(self) -> bool:
        return self.vol * self.T < 1e-5

    def shift_vol(self, dV):
        return BlackScholes(self.right, self.S, self.X, self.vol + dV, self.T)

    @property
    def value(self) -> float:
        if self._is_worth_intrinsic:
            return self.intrinsic
        if self.right == CALL:
            return self.S * self.N1 - self.X * self.N2
        return self.X * (1 - self.N2) - self.S * (1 - self.N1)

    @property
    def vega(self) -> float:
        return self.S * np.sqrt(self.T) * norm.pdf(self.d1) * 0.01
