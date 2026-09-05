from dataclasses import dataclass
from datetime import date
from enum import Enum, IntEnum

import pandas as pd
from dateutil.relativedelta import relativedelta


class PaymentFrequency(IntEnum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    ZERO_COUPON = 0


class DayCountConvention(Enum):
    ACT_365 = "ACT_365"
    ACT_ACT_ICMA = "ACT_ACT_ICMA"
    THIRTY_360 = "THIRTY_360"


@dataclass(frozen=True)
class Bond:
    maturity: date
    issue_date: date
    coupon_rate: float
    face_value: float = 100
    frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL
    day_count: DayCountConvention = DayCountConvention.ACT_ACT_ICMA

    def _all_payment_dates(self) -> list[date]:
        if self.frequency == PaymentFrequency.ZERO_COUPON:
            dates = [self.maturity]
        else:
            step_months = 12 // self.frequency
            current = self.maturity
            dates = []
            while current > self.issue_date:
                dates.append(current)
                current -= relativedelta(months=step_months)
        dates.reverse()
        return dates

    def _day_count_calculations(self, start_date, end_date) -> float:
        if self.day_count == DayCountConvention.ACT_365:
            return (end_date - start_date).days / 365.0
        if self.day_count == DayCountConvention.ACT_ACT_ICMA:
            if self.frequency == PaymentFrequency.ZERO_COUPON:
                return (end_date - start_date).days / 365.0

            dates = [self.issue_date] + self._all_payment_dates()
            tprev = max([d for d in dates if d <= start_date])
            tnext = min([d for d in dates if d > start_date])
            period_len = (tnext - tprev).days

            # Case 1: end_date falls within the current coupon period
            if end_date <= tnext:
                return ((end_date - start_date).days / period_len) / self.frequency

            # Case 2: end_date is a future payment date
            fractional_first_period = (tnext - start_date).days / period_len
            subsequent_periods = (
                dates.index(end_date) - dates.index(tnext) if end_date in dates else 0
            )
            return (fractional_first_period + subsequent_periods) / self.frequency
        if self.day_count == DayCountConvention.THIRTY_360:
            y1, m1, d1 = start_date.year, start_date.month, start_date.day
            y2, m2, d2 = end_date.year, end_date.month, end_date.day
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 >= 30:
                d2 = 30
            day_diff = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
            return day_diff / 360.0
        raise ValueError(f"Unsupported day count convention: {self.day_count}")

    def cash_flows(self, settlement_date: date) -> pd.DataFrame:
        empty_df = pd.DataFrame(columns=["date", "cash_flow", "year_fraction"])
        if settlement_date >= self.maturity:
            return empty_df
        dates = self._all_payment_dates()
        dates = [d for d in dates if d > settlement_date]
        records = []
        for cf_date in dates:
            record = {
                "date": cf_date,
                "cash_flow": (
                    self.face_value * self.coupon_rate / self.frequency
                    if self.frequency != PaymentFrequency.ZERO_COUPON
                    else 0.0
                ),
                "year_fraction": self._day_count_calculations(settlement_date, cf_date),
            }
            records.append(record)
        if records:
            records[-1]["cash_flow"] += self.face_value
            return pd.DataFrame(records)
        else:
            return empty_df

    def accrued_interest(self, settlement_date: date) -> float:
        if (
            self.frequency == PaymentFrequency.ZERO_COUPON
            or settlement_date < self.issue_date
            or settlement_date >= self.maturity
        ):
            return 0.0
        else:
            dates = self._all_payment_dates()
            dates.insert(0, self.issue_date)
            if settlement_date in dates:
                return 0.0
            tprev = max([d for d in dates if d <= settlement_date])
            tnext = min([d for d in dates if d > settlement_date])
            accrued_days = (settlement_date - tprev).days
            period_days = (tnext - tprev).days
            coupon = self.coupon_rate * self.face_value / self.frequency
            return coupon * (accrued_days / period_days)
