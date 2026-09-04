from dataclasses import dataclass
from datetime import date
from enum import IntEnum
from dateutil.relativedelta import relativedelta
import pandas as pd

class PaymentFrequency(IntEnum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    ZERO_COUPON = 0

@dataclass(frozen=True)
class Bond:
    maturity: date
    issue_date: date
    coupon_rate: float
    face_value: float = 100
    frequency: PaymentFrequency = PaymentFrequency.SEMI_ANNUAL

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
                "year_fraction": (cf_date - settlement_date).days / 365.0,
            }
            records.append(record)
        if records:
            records[-1]["cash_flow"] += self.face_value
            return pd.DataFrame(records)
        else:
            return empty_df

    def accrued_interest(self,settlement_date : date) ->float:
        if self.frequency == PaymentFrequency.ZERO_COUPON or settlement_date < self.issue_date or settlement_date >= self.maturity:
            return 0.0
        else:
            dates = self._all_payment_dates()
            dates.insert(0,self.issue_date)
            if settlement_date in dates:
                return 0.0
            tprev = max([d for d in dates if d <=settlement_date])
            tnext = min([d for d in dates if d >settlement_date])
            accrued_days = (settlement_date - tprev).days
            period_days = (tnext - tprev).days
            coupon = self.coupon_rate * self.face_value / self.frequency
            return coupon * (accrued_days / period_days)
