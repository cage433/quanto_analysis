from numbers import Number

import numpy as np

from modelling.maths.models.black_scholes import BlackScholes
from modelling.maths.models.option_right import OptionRight
from modelling.utils.type_utils import checked_number


# noinspection PyPep8Naming
class BlackScholesQuanto:
    def __init__(
            self,
            right: OptionRight,
            S: Number,
            fixed_fx_rate: Number,
            X: Number,
            S_vol: Number,
            fx_vol: Number,
            rho: Number,
            T: Number,
    ):
        self.fixed_fx_rate: float = checked_number(fixed_fx_rate)
        self.conv_adj: float = rho * S_vol * fx_vol * T
        F_adj = S * np.exp(-self.conv_adj)
        self.bs = BlackScholes(
            right,
            F_adj,
            X,
            S_vol,
            T
        )


    @property
    def delta(self) -> float:
        bs_delta = self.bs.delta
        return bs_delta * np.exp(- self.conv_adj) * self.fixed_fx_rate

    @property
    def value(self) -> float:
        return self.bs.value * self.fixed_fx_rate

