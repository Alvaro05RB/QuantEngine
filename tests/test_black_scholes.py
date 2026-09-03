from datetime import date

import numpy as np
import pytest

from quant_engine import black_scholes
from quant_engine.options import Option


def test_time_to_maturity_standard():
    expiry_date = date(2027, 1, 1)
    option = Option(strike=100, expiry=expiry_date)
    maturity = black_scholes.time_to_maturity(option.expiry, date(2026, 1, 1))
    assert maturity == pytest.approx(1.0)


def test_time_to_maturity_past_expiry():
    expiry_date = date(2026, 1, 1)
    option = Option(strike=100, expiry=expiry_date)
    maturity = black_scholes.time_to_maturity(option.expiry, date(2026, 6, 1))
    assert maturity == 0.0


def test_black_scholes_benchmark():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05

    call_price = black_scholes.black_scholes(
        option, spot, vol, r, pricing_date, is_call=True
    )
    put_price = black_scholes.black_scholes(
        option, spot, vol, r, pricing_date, is_call=False
    )

    assert call_price == pytest.approx(10.4506, abs=1e-4)
    assert put_price == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    option = Option(strike=120.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 7, 15)
    spot = 150.0
    vol = 0.60
    r = 0.10
    maturity = black_scholes.time_to_maturity(option.expiry, pricing_date)
    call_price = black_scholes.black_scholes(
        option, spot, vol, r, pricing_date, is_call=True
    )
    put_price = black_scholes.black_scholes(
        option, spot, vol, r, pricing_date, is_call=False
    )
    assert (call_price - put_price) == pytest.approx(
        spot - option.strike * np.exp(-r * maturity), abs=1e-4
    )


def test_expiration_payoff():
    option = Option(strike=120.0, expiry=date(2026, 7, 15))
    expiry = option.expiry
    r, vol = 0.05, 0.20

    # Spot = 150: Call is ITM, Put is OTM
    assert (
        black_scholes.black_scholes(option, 150.0, vol, r, expiry, is_call=True) == 30.0
    )
    assert (
        black_scholes.black_scholes(option, 150.0, vol, r, expiry, is_call=False) == 0.0
    )

    # Spot = 90: Call is OTM, Put is ITM
    assert (
        black_scholes.black_scholes(option, 90.0, vol, r, expiry, is_call=True) == 0.0
    )
    assert (
        black_scholes.black_scholes(option, 90.0, vol, r, expiry, is_call=False) == 30.0
    )


def test_black_scholes_asymptotes():
    option = Option(strike=100.0, expiry=date(2026, 9, 15))
    expiry = option.expiry
    pricing_date = date(2026, 7, 15)
    r, vol = 0.05, 0.20
    T = black_scholes.time_to_maturity(expiry, pricing_date)

    # Spot = 5000: Call is deep ITM, Put is deep OTM
    assert black_scholes.black_scholes(
        option, 5000.0, vol, r, pricing_date, is_call=True
    ) == pytest.approx(5000 - option.strike * np.exp(-r * T), abs=1e-2)
    assert black_scholes.black_scholes(
        option, 5000.0, vol, r, pricing_date, is_call=False
    ) == pytest.approx(0.0, abs=1e-4)

    # Spot = 1: Call is deep OTM, Put is deep ITM
    assert black_scholes.black_scholes(
        option, 0.1, vol, r, pricing_date, is_call=True
    ) == pytest.approx(0.0, abs=1e-4)
    assert black_scholes.black_scholes(
        option, 0.1, vol, r, pricing_date, is_call=False
    ) == pytest.approx(option.strike * np.exp(-r * T) - 0.1, abs=1e-2)


def test_call_delta():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    delta = black_scholes.delta(option, spot, vol, r, pricing_date, True)
    assert delta == pytest.approx(0.6368, abs=1e-4)


def test_put_delta():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    delta = black_scholes.delta(option, spot, vol, r, pricing_date, False)
    assert delta == pytest.approx(-0.3632, abs=1e-4)


def test_delta_expiration():
    option = Option(strike=100.0, expiry=date(2026, 1, 1))
    pricing_date = date(2026, 1, 1)
    vol = 0.20
    r = 0.05

    assert black_scholes.delta(option, 110, vol, r, pricing_date, True) == 1.0
    assert black_scholes.delta(option, 110, vol, r, pricing_date, False) == 0.0

    assert black_scholes.delta(option, 90, vol, r, pricing_date, True) == 0.0
    assert black_scholes.delta(option, 90, vol, r, pricing_date, False) == -1.0

    assert black_scholes.delta(option, 100, vol, r, pricing_date, True) == 0.5
    assert black_scholes.delta(option, 100, vol, r, pricing_date, False) == -0.5


def test_gamma():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    gamma = black_scholes.gamma(option, spot, vol, r, pricing_date)
    assert gamma == pytest.approx(0.0188, abs=1e-4)


def test_gamma_expiration():
    option = Option(strike=100.0, expiry=date(2026, 1, 1))
    pricing_date = date(2026, 1, 1)
    vol = 0.20
    r = 0.05
    spot = 100
    assert black_scholes.gamma(option, spot, vol, r, pricing_date) == 0.0


def test_vega():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    vega = black_scholes.vega(option, spot, vol, r, pricing_date)
    assert vega == pytest.approx(37.524, abs=1e-4)


def test_vega_expiration():
    option = Option(strike=100.0, expiry=date(2026, 1, 1))
    pricing_date = date(2026, 1, 1)
    vol = 0.20
    r = 0.05
    spot = 100
    assert black_scholes.vega(option, spot, vol, r, pricing_date) == 0.0


def test_call_theta():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    theta = black_scholes.theta(option, spot, vol, r, pricing_date, True)
    assert theta == pytest.approx(-6.414, abs=1e-4)


def test_put_theta():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    theta = black_scholes.theta(option, spot, vol, r, pricing_date, False)
    assert theta == pytest.approx(-1.6579, abs=1e-4)


def test_theta_expiration():
    option = Option(strike=100.0, expiry=date(2026, 1, 1))
    pricing_date = date(2026, 1, 1)
    vol = 0.20
    r = 0.05
    spot = 100
    assert black_scholes.theta(option, spot, vol, r, pricing_date, True) == 0.0
    assert black_scholes.theta(option, spot, vol, r, pricing_date, False) == 0.0


def test_call_rho():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    rho = black_scholes.rho(option, spot, vol, r, pricing_date, True)
    assert rho == pytest.approx(53.2325, abs=1e-4)


def test_put_rho():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    vol = 0.20
    r = 0.05
    rho = black_scholes.rho(option, spot, vol, r, pricing_date, False)
    assert rho == pytest.approx(-41.8905, abs=1e-4)


def test_rho_expiration():
    option = Option(strike=100.0, expiry=date(2026, 1, 1))
    pricing_date = date(2026, 1, 1)
    vol = 0.20
    r = 0.05
    spot = 100
    assert black_scholes.rho(option, spot, vol, r, pricing_date, True) == 0.0
    assert black_scholes.rho(option, spot, vol, r, pricing_date, False) == 0.0


def test_get_implied_volatility_round_trip():
    option = Option(strike=100.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    r = 0.05
    true_vol = 0.25

    # 1. Round-trip for Call
    call_mkt_price = black_scholes.black_scholes(
        option, spot, true_vol, r, pricing_date, is_call=True
    )
    recovered_call_iv = black_scholes.get_implied_volatility(
        option, spot, call_mkt_price, r, pricing_date, is_call=True
    )
    assert recovered_call_iv == pytest.approx(true_vol, abs=1e-4)

    # 2. Round-trip for Put
    put_mkt_price = black_scholes.black_scholes(
        option, spot, true_vol, r, pricing_date, is_call=False
    )
    recovered_put_iv = black_scholes.get_implied_volatility(
        option, spot, put_mkt_price, r, pricing_date, is_call=False
    )
    assert recovered_put_iv == pytest.approx(true_vol, abs=1e-4)


def test_arbitrage_violation():
    option = Option(strike=80.0, expiry=date(2027, 1, 1))
    pricing_date = date(2026, 1, 1)
    spot = 100.0
    r = 0.05
    impossible_market_price = 5.0  # Intrinsic value is >= 20.0

    with pytest.raises(ValueError):
        black_scholes.get_implied_volatility(
            option, spot, impossible_market_price, r, pricing_date, is_call=True
        )
