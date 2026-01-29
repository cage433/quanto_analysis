from numbers import Number

import numpy as np
from numpy import ndarray
from tabulate import tabulate

from modelling.maths.models.black_scholes import BlackScholes
from modelling.maths.models.option_right import OptionRight, CALL
from modelling.maths.random.random_number_generator import RandomNumberGenerator
from modelling.quantity.quantity import Qty
from modelling.quantity.uom import UOM, MT, USD
from modelling.utils.type_utils import checked_type, checked_number

# Done as a trial run for the more interesting quanto option cases
class VanillaOption:
    def __init__(
            self,
            right: OptionRight,
            X: Qty,
            T: Number
    ):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.X: Qty = checked_type(X, Qty)
        self.T: float = checked_number(T)
        self.ccy: UOM = self.X.uom.numerator
        self.qty_uom: UOM = self.X.uom.denominator

    def black_scholes(self, market_data: 'MarketData', t: float) -> BlackScholes:
        S = market_data.S.checked_value(self.X.uom)
        K = self.X.checked_value(self.X.uom)
        return BlackScholes(self.right, S, K, market_data.vol, self.T - t)

    def delta(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.delta, self.qty_uom)

    def value(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.value, self.ccy)


class MarketData:
    def __init__(
            self,
            S: Qty,
            vol: Number,
    ):
        self.S: Qty = checked_type(S, Qty)
        self.vol: float = checked_number(vol)
        self.price_uom: UOM = self.S.uom
        self.ccy: UOM = self.price_uom.numerator
        self.qty_uom: UOM = self.price_uom.denominator


class Portfolio:
    def __init__(self, option: VanillaOption, underlying: Qty, cash: Qty):
        self.option: VanillaOption = checked_type(option, VanillaOption)
        self.underlying: Qty = checked_type(underlying, Qty)
        self.cash: Qty = checked_type(cash, Qty)

    def value(self, market_data: MarketData, t: float) -> Qty:
        return self.option.value(market_data, t) + self.underlying * market_data.S + self.cash

    def delta(self, market_data: MarketData, t: float) -> Qty:
        option_delta = self.option.delta(market_data, t)
        return option_delta + self.underlying

    def rehedge(self, market_data: MarketData, t: float) -> 'Portfolio':
        option_delta = self.option.delta(market_data, t)
        underlying_hedge = - option_delta
        d_underlying = (underlying_hedge - self.underlying)
        cost_of_hedge = - d_underlying * market_data.S
        rehedged_portfolio = Portfolio(
            self.option,
            -option_delta,
            self.cash + cost_of_hedge
        )
        return rehedged_portfolio


def shift_prices(rng: RandomNumberGenerator, prices_sample: ndarray, mu: float, vol: Number, dt: float):
    shift = np.exp(rng.normal(size=prices_sample.shape) * vol * np.sqrt(dt) + mu * dt)
    return np.multiply(prices_sample, shift)


# Dynamically hedge an option through its lifecycle. If we do this over a number of paths then the
# average value should change little. Of course that isn't true for a single path, as there is no
# guarantee that observed vol matches implied
def dynamically_hedge_option(rng: RandomNumberGenerator, option: VanillaOption, market_data: MarketData,
                             n_time_steps: int, n_paths: int):
    dt = option.T / (n_time_steps - 1)
    times = np.arange(dt, option.T + dt, dt)
    Fs = np.ones(n_paths) * market_data.S.value
    pfs = [Portfolio(option, Qty(0, market_data.qty_uom), Qty(0, market_data.ccy)) for _ in range(n_paths)]
    pfs = [pf.rehedge(market_data, 0) for pf in pfs]
    for t in times:
        mu = rng.uniform(-0.05, 0.05)
        Fs = shift_prices(rng, Fs, mu, market_data.vol, dt)
        mds = [MarketData(Qty(F, market_data.price_uom), market_data.vol) for F in Fs]
        hedged_pfs = [pf.rehedge(md, t) for pf, md in zip(pfs, mds)]
        pfs = hedged_pfs
    mds = [MarketData(Qty(F, market_data.price_uom), market_data.vol) for F in Fs]
    values = [pf.value(md, option.T).checked_value(option.ccy) for pf, md in zip(pfs, mds)]
    return values


if __name__ == '__main__':
    def random_market_data(rng: RandomNumberGenerator):
        S = Qty(rng.uniform(90, 110), USD / MT)
        vol = rng.uniform(0.1, 0.5)
        return MarketData(S=S, vol=vol)

    def random_option(rng: RandomNumberGenerator):
        T = rng.uniform(0.2, 1.5)
        X = Qty(rng.uniform(90, 110), USD / MT)
        return VanillaOption(CALL, X=X, T=T)

    for i_run in range(10):
        print()
        rng = RandomNumberGenerator.random()
        option = random_option(rng)
        market_data = random_market_data(rng)

        rows = [["Analytic", option.value(market_data, t=0).checked_value(USD)]]

        n_time_steps = 100
        n_paths = 1000
        dynamically_hedged_values = dynamically_hedge_option(rng, option, market_data, n_time_steps, n_paths)
        mean = np.mean(dynamically_hedged_values)
        se = np.std(dynamically_hedged_values) / np.sqrt(n_paths)
        rows.append(["Dynamically Hedged", mean, se])

        print(
            tabulate(rows, floatfmt="1.3f", headers=["", "Value", "Std Err"])
        )
