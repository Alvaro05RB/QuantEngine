from datetime import date

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from quant_engine.options import Option


def time_to_maturity(expiry_date: date, pricing_date: date | None = None):
    if pricing_date is None:
        pricing_date = date.today()
    return max((expiry_date - pricing_date).days / 365, 0.0)


def calculate_d1(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    maturity: float,
):
    return (
        np.log(spot_price / option.strike)
        + (risk_free + 0.5 * volatility**2) * maturity
    ) / (volatility * np.sqrt(maturity))


def black_scholes(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
    is_call: bool = True,
):
    phi = 1.0 if is_call else -1.0
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        return np.maximum(
            phi * (spot_price - option.strike), 0
        )  # Returns if the option was executed
    d1 = calculate_d1(option, spot_price, volatility, risk_free, maturity)
    d2 = d1 - volatility * np.sqrt(maturity)
    return phi * spot_price * norm.cdf(phi * d1) - phi * option.strike * np.exp(
        -risk_free * maturity
    ) * norm.cdf(phi * d2)


def delta(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
    is_call: bool = True,
):
    phi = 0.0 if is_call else -1.0
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        if is_call:
            if spot_price > option.strike:
                return 1.0
            elif spot_price == option.strike:
                return 0.5
            return 0.0
        else:
            if spot_price < option.strike:
                return -1.0
            elif spot_price == option.strike:
                return -0.5
            return 0.0
    return (
        norm.cdf(calculate_d1(option, spot_price, volatility, risk_free, maturity))
        + phi
    )


def gamma(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
):
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        return 0.0
    d1 = calculate_d1(option, spot_price, volatility, risk_free, maturity)
    return norm.pdf(d1) / (spot_price * volatility * np.sqrt(maturity))


def vega(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
):
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        return 0.0
    d1 = calculate_d1(option, spot_price, volatility, risk_free, maturity)

    return spot_price * np.sqrt(maturity) * norm.pdf(d1)


def theta(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
    is_call: bool = True,
):
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        return 0.0
    d1 = calculate_d1(option, spot_price, volatility, risk_free, maturity)
    phi = 1.0 if is_call else -1.0

    return -(spot_price * norm.pdf(d1) * volatility) / (
        2 * np.sqrt(maturity)
    ) - phi * risk_free * option.strike * np.exp(-risk_free * maturity) * norm.cdf(
        phi * (d1 - volatility * np.sqrt(maturity))
    )


def rho(
    option: Option,
    spot_price: float,
    volatility: float,
    risk_free: float,
    pricing_date: date | None = None,
    is_call: bool = True,
):
    maturity = time_to_maturity(option.expiry, pricing_date)
    if maturity <= 0:
        return 0.0
    d1 = calculate_d1(option, spot_price, volatility, risk_free, maturity)
    phi = 1.0 if is_call else -1.0

    return (
        phi
        * option.strike
        * maturity
        * np.exp(-risk_free * maturity)
        * norm.cdf(phi * (d1 - volatility * np.sqrt(maturity)))
    )


def get_implied_volatility(
    option: Option,
    spot_price: float,
    market_price: float,
    risk_free: float,
    pricing_date: date | None = None,
    is_call: bool = True,
):
    def objective(vol):
        return (
            black_scholes(option, spot_price, vol, risk_free, pricing_date, is_call)
            - market_price
        )

    return brentq(objective, 0.0001, 5.0)
