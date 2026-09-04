from datetime import date

import numpy as np
import pytest

from quant_engine.bonds import Bond, DayCountConvention, PaymentFrequency
from quant_engine.curve_bootstrapper import bootstrap_curve


def test_bootstrapper_repricing():
    settlement = date(2024, 1, 1)

    # 1. Benchmark 1-year zero coupon bond
    b1 = Bond(
        maturity=date(2025, 1, 1),
        issue_date=settlement,
        coupon_rate=0.0,
        face_value=100.0,
        frequency=PaymentFrequency.ZERO_COUPON,
        day_count=DayCountConvention.ACT_365,
    )
    p1 = 96.0  # market clean price

    # 2. Benchmark 2-year annual coupon bond
    b2 = Bond(
        maturity=date(2026, 1, 1),
        issue_date=settlement,
        coupon_rate=0.05,
        face_value=100.0,
        frequency=PaymentFrequency.ANNUAL,
        day_count=DayCountConvention.ACT_365,
    )
    p2 = 101.0  # market clean price

    curve = bootstrap_curve([(b1, p1), (b2, p2)], settlement)

    # Verify Bond 1 re-prices exactly to p1
    cf1 = b1.cash_flows(settlement)
    pv1 = sum(
        row["cash_flow"] * curve.discount_factor(row["year_fraction"])
        for _, row in cf1.iterrows()
    )
    assert pytest.approx(pv1, abs=1e-6) == p1

    # Verify Bond 2 re-prices exactly to p2
    cf2 = b2.cash_flows(settlement)
    pv2 = sum(
        row["cash_flow"] * curve.discount_factor(row["year_fraction"])
        for _, row in cf2.iterrows()
    )
    assert pytest.approx(pv2, abs=1e-6) == p2
