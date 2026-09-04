from datetime import date

import numpy as np
from bonds import Bond
from curves import ZeroCurve


def bootstrap_curve(
    instruments: list[tuple[Bond, float]],
    settlement_date: date,
) -> ZeroCurve:
    """Bootstraps a ZeroCurve from a list of (Bond, clean_price) tuples.

    Instruments must be sorted in ascending order of maturity.
    """
    # 1. Sort instruments by maturity
    sorted_instruments = sorted(instruments, key=lambda item: item[0].maturity)

    times = [0.0]
    discount_factors = [1.0]

    for bond, clean_px in sorted_instruments:
        dirty_px = clean_px + bond.accrued_interest(settlement_date)
        cf_table = bond.cash_flows(settlement_date)

        if cf_table.empty:
            continue

        # Terminal cash flow (face + final coupon)
        terminal_flow = cf_table.iloc[-1]["cash_flow"]
        terminal_tau = cf_table.iloc[-1]["year_fraction"]

        # Discount intermediate cash flows with current partial curve
        pv_intermediate = 0.0
        if len(cf_table) > 1:
            temp_curve = ZeroCurve(
                reference_date=settlement_date,
                times=np.array(times),
                discount_factors=np.array(discount_factors),
            )
            for _, row in cf_table.iloc[:-1].iterrows():
                df = temp_curve.discount_factor(row["year_fraction"])
                pv_intermediate += row["cash_flow"] * df

        # Solve for terminal discount factor
        d_terminal = (dirty_px - pv_intermediate) / terminal_flow

        times.append(terminal_tau)
        discount_factors.append(d_terminal)

    return ZeroCurve(
        reference_date=settlement_date,
        times=np.array(times),
        discount_factors=np.array(discount_factors),
    )
