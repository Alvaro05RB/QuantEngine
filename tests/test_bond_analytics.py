from datetime import date

import bond_analytics as ba
import pytest
from bonds import Bond, PaymentFrequency


@pytest.fixture
def par_semi_annual_bond():
    return Bond(
        issue_date=date(2024, 1, 1),
        maturity=date(2026, 1, 1),
        coupon_rate=0.05,
        face_value=100.0,
        frequency=PaymentFrequency.SEMI_ANNUAL,
    )


@pytest.fixture
def zero_coupon_bond():
    return Bond(
        issue_date=date(2024, 1, 1),
        maturity=date(2025, 1, 1),
        coupon_rate=0.0,
        face_value=100.0,
        frequency=PaymentFrequency.ZERO_COUPON,
    )


def test_par_bond_pricing(par_semi_annual_bond):
    settlement = date(2024, 1, 1)
    ytm = 0.05
    clean_px = ba.clean_price_from_yield(par_semi_annual_bond, ytm, settlement)
    assert pytest.approx(clean_px, abs=1e-6) == 100.0


def test_ytm_round_trip(par_semi_annual_bond):
    settlement = date(2024, 6, 1)  # Mid-coupon date
    target_ytm = 0.0625

    clean_px = ba.clean_price_from_yield(par_semi_annual_bond, target_ytm, settlement)
    recovered_ytm = ba.yield_to_maturity(par_semi_annual_bond, settlement, clean_px)

    assert pytest.approx(recovered_ytm, abs=1e-6) == target_ytm


def test_zero_coupon_duration_equals_maturity(zero_coupon_bond):
    settlement = date(2024, 1, 1)
    ytm = 0.04
    mac_dur = ba.macaulay_duration(zero_coupon_bond, ytm, settlement)

    # 2024 is a leap year (366 days)
    expected_tau = 366 / 365.0
    assert pytest.approx(mac_dur, abs=1e-4) == expected_tau


def test_dv01_matches_bump_and_reprice(par_semi_annual_bond):
    settlement = date(2024, 6, 1)
    ytm = 0.05
    shift = 0.0001  # 1 bp

    analytic_dv01 = ba.dv01(par_semi_annual_bond, ytm, settlement)

    # Numerical bump: down 1 bp vs base
    p_base = ba.dirty_price_from_yield(par_semi_annual_bond, ytm, settlement)
    p_down = ba.dirty_price_from_yield(par_semi_annual_bond, ytm - shift, settlement)
    numerical_dv01 = p_down - p_base

    assert pytest.approx(analytic_dv01, rel=1e-3) == numerical_dv01


def test_convexity_taylor_expansion(par_semi_annual_bond):
    settlement = date(2024, 1, 1)
    ytm = 0.05
    dy = 0.0050  # 50 bps jump

    p0 = ba.dirty_price_from_yield(par_semi_annual_bond, ytm, settlement)
    p_bumped = ba.dirty_price_from_yield(par_semi_annual_bond, ytm + dy, settlement)
    actual_change = p_bumped - p0

    d_mod = ba.modified_duration(par_semi_annual_bond, ytm, settlement)
    c = ba.convexity(par_semi_annual_bond, ytm, settlement)

    # Taylor series: ΔP ≈ P * (-D_mod * dy + 0.5 * Convexity * dy^2)
    taylor_change = p0 * (-d_mod * dy + 0.5 * c * (dy**2))

    assert pytest.approx(actual_change, rel=1e-3) == taylor_change
