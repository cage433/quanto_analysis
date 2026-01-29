from functools import cached_property
from numbers import Number
from typing import Tuple

import numpy as np
from numpy import ndarray
from tabulate import tabulate

from modelling.maths.models.black_scholes_quanto import BlackScholesQuanto
from modelling.maths.models.option_right import OptionRight, CALL, PUT
from modelling.maths.random.random_number_generator import RandomNumberGenerator
from modelling.quantity.quantity import Qty
from modelling.quantity.uom import UOM, MT, USD, EUR
from modelling.utils.type_utils import checked_type, checked_number, checked_qty


#
# The second of Haug's currency translation options
#
# Call payoff is EP * max(S - X, 0)
# where
#
#   S - underlying (in foreign ccy)
#   X - strike (in foreign ccy)
#   EP - _fixed_, i.e. pre-determined fx rate
#
class QuantoOption2:
    def __init__(
            self,
            right: OptionRight,
            EP: Qty,
            X: Qty,
            T: Number
    ):
        self.right: OptionRight = checked_type(right, OptionRight)
        self.EP: Qty = checked_qty(EP)
        self.X: Qty = checked_qty(X)
        self.T: float = checked_number(T)

        self.fx_uom: UOM = self.EP.uom
        self.foreign_price_uom: UOM = self.X.uom
        self.foreign_ccy: UOM = self.EP.uom.denominator
        self.domestic_ccy: UOM = self.EP.uom.numerator
        assert self.domestic_ccy == EP.uom.numerator, "Mismatching UOM"
        self.qty_uom: UOM = self.X.uom.denominator

    def black_scholes(self, market_data: 'MarketData', t: float) -> BlackScholesQuanto:
        S = market_data.price.checked_value(self.X.uom)
        X = self.X.value
        return BlackScholesQuanto(
            self.right,
            S,
            self.EP.value,
            X,
            market_data.price_vol,
            market_data.fx_vol,
            market_data.rho,
            self.T - t
        )

    def delta(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.delta, self.domestic_ccy / self.foreign_price_uom)

    def value(self, market_data: 'MarketData', t: float) -> Qty:
        bs = self.black_scholes(market_data, t)
        return Qty(bs.value, self.domestic_ccy)

    def mc_value(
            self,
            rng: RandomNumberGenerator,
            n_paths: int,
            market_data: 'MarketData',
            t: float
    ) -> Tuple[Qty, Qty]:
        normals = rng.normal(size=(n_paths))
        rho = market_data.rho
        v_p = market_data.price_vol
        v_fx = market_data.fx_vol
        z_prices = normals
        dt = self.T - t
        prices = market_data.price.checked_value(self.foreign_price_uom) * np.exp(
            z_prices * v_p * np.sqrt(dt)
            - rho * v_p * v_fx * dt
            - 0.5 * v_p * v_p * dt
        )
        K = self.X.checked_value(self.foreign_price_uom)
        if self.right == CALL:
            payoffs = np.maximum(prices - K, 0)
        else:
            payoffs = np.maximum(-prices + K, 0)

        mean = np.mean(payoffs)
        se = np.std(payoffs) / np.sqrt(n_paths)
        fx = self.EP.checked_value(self.fx_uom)
        return (
            Qty(mean * fx, self.domestic_ccy),
            Qty(se * fx, self.domestic_ccy),
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

    def with_price(self, new_price: Qty) -> 'MarketData':
        return self.with_price_and_fx_rate(new_price, self.fx_rate)

    def with_fx(self, new_fx_rate: Qty) -> 'MarketData':
        return self.with_price_and_fx_rate(self.price, new_fx_rate)

    @cached_property
    def domestic_price_vol(self):
        v1 = self.price_vol
        v2 = self.fx_vol
        return np.sqrt(
            v1 * v1
            + 2 * self.rho * v1
            * v2 + v2 * v2
        )


class Portfolio:
    def __init__(self, option: QuantoOption2, underlying: Qty, foreign_cash, domestic_cash: Qty):
        self.option: QuantoOption2 = checked_type(option, QuantoOption2)
        self.underlying: Qty = checked_type(underlying, Qty)
        self.foreign_cash = checked_type(foreign_cash, Qty)
        self.domestic_cash: Qty = checked_type(domestic_cash, Qty)
        assert self.underlying.uom == option.qty_uom, f"Expected {option.qty_uom}, got {self.underlying}"
        assert self.domestic_cash.uom == option.domestic_ccy, "Mismatching ccy"
        assert self.foreign_cash.uom == option.foreign_ccy, "Mismatching ccy"

    def value(self, market_data: MarketData, t: float) -> Qty:
        return self.option.value(market_data,
                                 t) + self.underlying * market_data.price * market_data.fx_rate + self.foreign_cash * market_data.fx_rate + self.domestic_cash

    def numeric_delta(self, market_data: MarketData, t: float) -> Qty:
        dF = Qty(0.001, EUR / MT)
        vup = self.value(market_data.with_price(market_data.price + dF), t)
        vdn = self.value(market_data.with_price(market_data.price - dF), t)
        return (vup - vdn) / (dF * 2.0)

    def numeric_fx_delta(self, market_data: MarketData, t: float) -> Qty:
        dFX = Qty(0.01, USD / EUR)
        v0 = self.value(market_data, t)
        v1 = self.value(market_data.with_fx(market_data.fx_rate + dFX), t)
        return (v1 - v0) / dFX

    def delta(self, market_data: MarketData, t: float) -> Qty:
        option_delta = self.option.delta(market_data, t)
        return option_delta + self.underlying

    def rehedge_price_risk(self, market_data: MarketData, t: float) -> 'Portfolio':
        option_derivative = self.option.delta(market_data, t)
        option_delta = option_derivative / market_data.fx_rate
        underlying_hedge = - option_delta
        d_underlying = (underlying_hedge - self.underlying)
        cost_of_hedge = - d_underlying * market_data.price * market_data.fx_rate
        rehedged_portfolio = Portfolio(
            self.option,
            -option_delta,
            self.foreign_cash,
            self.domestic_cash + cost_of_hedge
        )
        return rehedged_portfolio

    def rehedge_fx_risk(self, market_data: MarketData, t: float) -> 'Portfolio':
        current_fx_risk = self.underlying * market_data.price + self.foreign_cash
        d_foreign_cash = -current_fx_risk
        cost_of_hedge = -d_foreign_cash * market_data.fx_rate
        rehedged_portfolio = Portfolio(
            self.option,
            self.underlying,
            self.foreign_cash + d_foreign_cash,
            self.domestic_cash + cost_of_hedge
        )
        return rehedged_portfolio

    def rehedge(self, market_data: MarketData, t: float) -> 'Portfolio':
        return self.rehedge_price_risk(market_data, t).rehedge_fx_risk(market_data, t)


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


def dynamically_hedge_option(rng: RandomNumberGenerator, option: QuantoOption2, market_data: MarketData,
                             n_time_steps: int, n_paths: int):
    dt = option.T / (n_time_steps - 1)
    times = np.arange(dt, option.T + dt, dt)
    F_sample = np.ones(n_paths) * market_data.price.value
    FX_sample: ndarray = np.ones(n_paths) * market_data.fx_rate.value
    pfs = [Portfolio(option, Qty(0, option.qty_uom), Qty(0, option.foreign_ccy), Qty(0, option.domestic_ccy)) for _ in
           range(n_paths)]
    pfs = [pf.rehedge(market_data, 0) for pf in pfs]
    for t in times:
        price_mu, fx_mu = rng.uniform(-0.05, 0.01), rng.uniform(-0.05, 0.01)
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


def random_option(rng: RandomNumberGenerator) -> 'QuantoOption2':
    T = rng.uniform(0.2, 0.5)
    K = Qty(rng.uniform(90, 110), EUR / MT)
    fixed_fx_rate = Qty(rng.uniform(0.9, 1.1), USD / EUR)
    right = rng.choice(CALL, PUT)
    return QuantoOption2(right, fixed_fx_rate, X=K, T=T)


def random_market_data(rng: RandomNumberGenerator) -> MarketData:
    F = Qty(rng.uniform(90, 110), EUR / MT)
    price_vol = rng.uniform(0.1, 0.5)
    fx_vol = rng.uniform(0.1, 0.5)
    fx_rate = Qty(rng.uniform(0.9, 1.1), USD / EUR)
    rho = rng.uniform(-1, 1)
    return MarketData(price=F, fx_rate=fx_rate, price_vol=price_vol, fx_vol=fx_vol, rho=rho)


def run_random_option_scenario(rng: RandomNumberGenerator):
    n_time_steps = 100
    n_paths = 500
    option = random_option(rng)
    market_data = random_market_data(rng)
    dynamically_hedge_option(rng, option, market_data, n_time_steps, n_paths)


def compare_mc_and_analytic_values(rng: RandomNumberGenerator):
    n_paths = 2_000_000
    option = random_option(rng)
    market_data = random_market_data(rng)
    analytic = option.value(market_data, t=0)
    mc_value, mc_se = option.mc_value(rng, n_paths, market_data, t=0)
    print(f"Value {analytic}, MC {mc_value} ({mc_se})")


if __name__ == '__main__':
    def random_market_data(rng: RandomNumberGenerator) -> MarketData:
        F = Qty(rng.uniform(90, 110), EUR / MT)
        price_vol = rng.uniform(0.1, 0.5)
        fx_vol = rng.uniform(0.1, 0.5)
        fx_rate = Qty(rng.uniform(0.9, 1.1), USD / EUR)
        rho = rng.uniform(-1, 1)
        return MarketData(price=F, fx_rate=fx_rate, price_vol=price_vol, fx_vol=fx_vol, rho=rho)


    def run_random_option_scenario(rng: RandomNumberGenerator):
        n_time_steps = 100
        n_paths = 500
        option = random_option(rng)
        market_data = random_market_data(rng)
        dynamically_hedge_option(rng, option, market_data, n_time_steps, n_paths)


    for i_run in range(10):
        print(f"\nRun {i_run}")
        seed = RandomNumberGenerator.random().random_seed()

        rng = RandomNumberGenerator(seed)
        option = random_option(rng)
        market_data = random_market_data(rng)

        table = [["Analytic", option.value(market_data, t=0).checked_value(USD)]]

        n_paths = 2_000_000
        mc_value, mc_se = option.mc_value(rng, n_paths, market_data, t=0)
        table.append(
            ["Monte Carlo", mc_value.checked_value(USD), mc_se.checked_value(USD)]
        )

        n_hedge_paths = 500
        hedged_values = dynamically_hedge_option(rng, option, market_data, n_time_steps=100, n_paths=n_hedge_paths)
        table.append(
            ["Dynamically Hedged", np.mean(hedged_values), np.std(hedged_values) / np.sqrt(n_hedge_paths)]
        )
        print(tabulate(table, floatfmt="1.3f", headers=[f"Seed = {seed}", "Value", "Std Err"]))

