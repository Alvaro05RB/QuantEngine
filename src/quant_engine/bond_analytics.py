from datetime import date

import numpy as np
from bonds import Bond, PaymentFrequency
from scipy.optimize import brentq


def dirty_price_from_yield(bond: Bond, ytm: float, settlement_date: date) -> float:
    cash_flow_list = bond.cash_flows(settlement_date)
    if cash_flow_list.empty:
        return 0.0
    cf = cash_flow_list["cash_flow"].to_numpy()
    time_fractions = cash_flow_list["year_fraction"].to_numpy()
    if bond.frequency == PaymentFrequency.ZERO_COUPON:
        df = 1 / (1 + ytm) ** time_fractions
    else:
        compounding_factor = 1 + ytm / bond.frequency
        exponents = bond.frequency * time_fractions
        df = 1 / compounding_factor**exponents
    return float(np.sum(cf * df))


def clean_price_from_yield(bond: Bond, ytm: float, settlement_date: date) -> float:
    return dirty_price_from_yield(bond, ytm, settlement_date) - bond.accrued_interest(
        settlement_date
    )


def yield_to_maturity(bond: Bond, settlement_date: date, market_price: float):
    def objective(ytm):
        return clean_price_from_yield(bond, ytm, settlement_date) - market_price

    return brentq(objective, -0.1, 2)


def macaulay_duration(bond: Bond, ytm: float, settlement_date: date) -> float:
    cash_flow_list = bond.cash_flows(settlement_date)

    if cash_flow_list.empty:
        return 0.0

    cf = cash_flow_list["cash_flow"].to_numpy()
    tau = cash_flow_list["year_fraction"].to_numpy()

    if bond.frequency == PaymentFrequency.ZERO_COUPON:
        df = 1 / (1 + ytm) ** tau
    else:
        compounding_factor = 1 + ytm / bond.frequency
        df = 1 / compounding_factor ** (bond.frequency * tau)

    pv_cash_flows = cf * df
    dirty_price = np.sum(pv_cash_flows)

    if dirty_price == 0.0:
        return 0.0

    return float(np.sum(tau * pv_cash_flows) / dirty_price)


def modified_duration(bond: Bond, ytm: float, settlement_date: date):
    d_mac = macaulay_duration(bond, ytm, settlement_date)

    if bond.frequency == PaymentFrequency.ZERO_COUPON:
        return d_mac / (1 + ytm)
    return d_mac / (1 + ytm / bond.frequency)


def dv01(bond: Bond, ytm: float, settlement_date: date):
    return (
        modified_duration(bond, ytm, settlement_date)
        * dirty_price_from_yield(bond, ytm, settlement_date)
        * 0.0001
    )


def convexity(bond: Bond, ytm: float, settlement_date: date):
    cash_flow_list = bond.cash_flows(settlement_date)
    if cash_flow_list.empty:
        return 0.0

    cf = cash_flow_list["cash_flow"].to_numpy()
    time_fractions = cash_flow_list["year_fraction"].to_numpy()

    m = 1.0 if bond.frequency == PaymentFrequency.ZERO_COUPON else float(bond.frequency)

    compounding_factor = 1.0 + ytm / m
    df = 1.0 / compounding_factor ** (m * time_fractions)

    pv = cf * df
    dirty_price = np.sum(pv)
    if dirty_price == 0.0:
        return 0.0

    numerator = np.sum(time_fractions * (time_fractions + 1.0 / m) * pv)
    denominator = dirty_price * (compounding_factor**2)

    return float(numerator / denominator)
