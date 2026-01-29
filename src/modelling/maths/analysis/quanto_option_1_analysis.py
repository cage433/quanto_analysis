from functools import cached_property
from numbers import Number
from typing import Tuple

import numpy as np
from numpy import ndarray
from tabulate import tabulate

from modelling.maths.models.black_scholes import BlackScholes
from modelling.maths.models.option_right import OptionRight, CALL
from modelling.maths.random.random_number_generator import RandomNumberGenerator
from modelling.quantity.quantity import Qty
from modelling.quantity.uom import UOM, MT, USD, EUR
from modelling.utils.type_utils import checked_type, checked_number, checked_qty

# Analyses the first of Haug's currency translation options.
#
#   Call payoff is max(S * E - X, 0)
# where
#   S - underlying (in foreign ccy)
#   X - strike (in domestic ccy)
#   E - fx rate
#

class QuantoOption1:
    def __init__(
            self,
            foreign_ccy: UOM,
            right: OptionRight,
            X: Qty,
            T: Number
    ):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.X: Qty = checked_qty(X)
        self.foreign_ccy: UOM = checked_type(foreign_ccy, UOM)
        self.domestic_price_uom: UOM = self.X.uom

        self.domestic_ccy: UOM = self.domestic_price_uom.numerator
        self.fx_uom: UOM = self.domestic_ccy / self.foreign_ccy
        assert self.fx_uom.is_fx_uom, "Not an FX uom"
        self.T: float = checked_number(T)
        self.qty_uom: UOM = self.X.uom.denominator
        self.foreign_price_uom: UOM = self.foreign_ccy / self.qty_uom

    def black_scholes(self, market_data: 'MarketData', t: float) -> BlackScholes:
        F_domestic = (market_data.price * market_data.fx_rate).checked_value(self.domestic_price_uom)
        K = self.X.checked_value(self.domestic_price_uom)
        return BlackScholes(
            self.right,
            F_domestic,
            K,
            market_data.domestic_price_vol,
            self.T - t
        )

    def delta(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.delta, self.qty_uom)

    def value(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.value, self.domestic_ccy)

    def mc_value(self, rng: RandomNumberGenerator, n_paths: int, market_data: 'MarketData', t: float) -> Tuple[
        Qty, Qty]:

        normals = rng.normal(size=(n_paths, 2))
        rho = market_data.rho
        z_0 = normals[:, 0]
        z_1 = normals[:, 1]
        z_prices = z_0
        z_fx = z_0 * rho + np.sqrt(1 - rho * rho) * z_1
        v_p = market_data.price_vol
        v_fx = market_data.fx_vol
        dt = self.T - t
        prices = market_data.price.checked_value(self.foreign_price_uom) * np.exp(
            z_prices * v_p * np.sqrt(dt)
            - rho * v_p * v_fx * dt
            - 0.5 * v_p * v_p * dt
        )
        fx_rates = market_data.fx_rate.checked_value(self.fx_uom) * np.exp(
            z_fx * v_fx * np.sqrt(dt)
            - 0.5 * v_fx * v_fx * dt
        )
        K = self.X.checked_value(self.domestic_price_uom)
        if self.right == CALL:
            payoffs = np.maximum(np.multiply(prices, fx_rates) - K, 0)
        else:
            payoffs = np.maximum(K - np.multiply(prices, fx_rates), 0)

        mean = np.mean(payoffs)
        se = np.std(payoffs) / np.sqrt(n_paths)
        return (
            Qty(mean, self.domestic_ccy),
            Qty(se, self.domestic_ccy),
        )

class MarketData:
    def __init__(
            self,
            price: Qty,
            fx_rate: Qty,
            price_vol: float,
            fx_vol: float,
            rho: float
    ):
        self.price: Qty = checked_qty(price)
        self.fx_rate: Qty = checked_qty(fx_rate)
        assert self.fx_rate.uom.is_fx_uom
        self.price_vol: float = checked_number(price_vol)
        self.fx_vol: float = checked_number(fx_vol)
        self.rho: float = checked_number(rho)

    def with_price_and_fx_rate(self, new_price: Qty, new_fx_rate: Qty) -> 'MarketData':
        return MarketData(
            new_price,
            new_fx_rate,
            self.price_vol,
            self.fx_vol,
            self.rho
        )

    @cached_property
    def domestic_price_vol(self):
        v1 = self.price_vol
        v2 = self.fx_vol
        return np.sqrt(
            v1 * v1
            + 2 * self.rho * v1 * v2
            + v2 * v2
        )



class Portfolio:
    def __init__(self, option: QuantoOption1, underlying: Qty, cash: Qty):
        self.option: QuantoOption1 = checked_type(option, QuantoOption1)
        self.underlying: Qty = checked_type(underlying, Qty)
        self.cash: Qty = checked_type(cash, Qty)
        assert self.underlying.uom == option.qty_uom, f"Expected {option.qty_uom}, got {self.underlying}"
        assert self.cash.uom == option.domestic_ccy, "Mismatching ccy"

    def value(self, market_data: MarketData, t: float) -> Qty:
        return self.option.value(market_data, t) + self.underlying * market_data.price * market_data.fx_rate + self.cash

    def delta(self, market_data: MarketData, t: float) -> Qty:
        option_delta = self.option.delta(market_data, t)
        return option_delta + self.underlying

    def rehedge(self, market_data: MarketData, t: float) -> 'Portfolio':
        option_delta = self.option.delta(market_data, t)
        underlying_hedge = - option_delta
        d_underlying = (underlying_hedge - self.underlying)
        cost_of_hedge = - d_underlying * market_data.price * market_data.fx_rate
        rehedged_portfolio = Portfolio(
            self.option,
            -option_delta,
            self.cash + cost_of_hedge
        )
        return rehedged_portfolio


def normals(rng: RandomNumberGenerator, n_paths: int) -> ndarray:
    assert n_paths % 2 == 0, "require even number of paths"
    n1 = rng.normal(size=int(n_paths / 2))
    n2 = n1 * -1
    return np.concatenate([n1, n2])


def shift_prices_and_fx(
        rng: RandomNumberGenerator,
        prices_sample: ndarray,
        fx_sample: ndarray,
        price_mu: float,
        fx_mu: float,
        price_vol: float,
        fx_vol: float,
        rho: float,
        dt: float
) -> Tuple[ndarray, ndarray]:
    normals = rng.normal(size=(len(prices_sample), 2))
    z0 = normals[:, 0]
    z1 = normals[:, 1]
    z_prices = z0
    z_fx = z0 * rho + z1 * np.sqrt(1 - rho * rho)

    def apply_shift(sample, z, vol, mu) -> ndarray:
        shift = np.exp(z * vol * np.sqrt(dt) + mu * dt)
        return np.multiply(sample, shift)

    shifted_prices = apply_shift(prices_sample, z_prices, price_vol, price_mu)
    shifted_fx = apply_shift(fx_sample, z_fx, fx_vol, fx_mu)

    return shifted_prices, shifted_fx


# Dynamically hedge an option through its lifecycle. If we do this over a number of paths then the
# average value should change little. Of course that isn't true for a single path, as there is no
# guarantee that observed vol matches implied
def dynamically_hedged_option_values(
        rng: RandomNumberGenerator,
        option: QuantoOption1,
        market_data: MarketData,
        n_time_steps: int,
        n_paths: int
):
    dt = option.T / (n_time_steps - 1)
    times = np.arange(dt, option.T + dt, dt)
    F_sample = np.ones(n_paths) * market_data.price.value
    FX_sample: ndarray = np.ones(n_paths) * market_data.fx_rate.value
    pfs = [Portfolio(option, Qty(0, option.qty_uom), Qty(0, option.domestic_ccy)) for _ in range(n_paths)]
    pfs = [pf.rehedge(market_data, 0) for pf in pfs]
    mds = [
        market_data.with_price_and_fx_rate(
            Qty(F, option.foreign_price_uom),
            Qty(FX, option.fx_uom)
        ) for F, FX in zip(F_sample, FX_sample)
    ]
    for t in times:
        price_mu, fx_mu = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
        F_sample, FX_sample = shift_prices_and_fx(
            rng,
            F_sample,
            FX_sample,
            price_mu,
            fx_mu,
            market_data.price_vol,
            market_data.fx_vol,
            market_data.rho,
            dt)
        mds = [
            market_data.with_price_and_fx_rate(
                Qty(F, option.foreign_price_uom),
                Qty(FX, option.fx_uom)
            ) for F, FX in zip(F_sample, FX_sample)
        ]
        hedged_pfs = [pf.rehedge(md, t) for pf, md in zip(pfs, mds)]
        pfs = hedged_pfs
    values = [pf.value(md, option.T).checked_value(option.domestic_ccy) for pf, md in zip(pfs, mds)]
    return values




if __name__ == '__main__':
    def random_market_data(rng: RandomNumberGenerator) -> MarketData:
        S = Qty(rng.uniform(90, 110), EUR / MT)
        price_vol = rng.uniform(0.1, 0.5)
        fx_vol = rng.uniform(0.1, 0.5)
        fx_rate = Qty(rng.uniform(0.9, 1.1), USD / EUR)
        rho = rng.uniform(-1, 1)
        return MarketData(price=S, fx_rate=fx_rate, price_vol=price_vol, fx_vol=fx_vol, rho=rho)


    def random_option(rng: RandomNumberGenerator) -> QuantoOption1:
        T = rng.uniform(0.2, 1.5)
        X = Qty(rng.uniform(90, 110), USD / MT)
        return QuantoOption1(EUR, CALL, X=X, T=T)

    for i_run in range(10):
        print(f"\nRun {i_run}")
        seed = RandomNumberGenerator.random().random_seed()
        rows = []
        rng = RandomNumberGenerator(seed)
        option = random_option(rng)
        market_data = random_market_data(rng)

        rows.append(["Analytic", option.value(market_data, t = 0).checked_value(USD)])

        # Monte carlo
        mc_value, mc_se = option.mc_value(rng, n_paths=1_000_000, market_data=market_data, t=0)
        rows.append(["Monte Carlo", mc_value.checked_value(USD), mc_se.checked_value(USD)])


        # Dynamically hedged
        n_hedging_paths = 1000
        dynamically_hedged_values = dynamically_hedged_option_values(rng, option, market_data, n_time_steps=50,
                                                                     n_paths=n_hedging_paths)
        hedging_mean = np.mean(dynamically_hedged_values)
        hedging_se = np.std(dynamically_hedged_values) / np.sqrt(n_hedging_paths)
        rows.append(["Dynamically Hedged", hedging_mean, hedging_se])
        print(tabulate(rows, floatfmt="1.3f", headers=[f"Seed = {seed}", "Value", "Std Err"]))
