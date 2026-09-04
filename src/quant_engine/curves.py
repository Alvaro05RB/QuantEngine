from dataclasses import dataclass
from datetime import date
from enum import Enum

import numpy as np


class InterpolationMethod(Enum):
    LOG_LINEAR = "LOG_LINEAR"


@dataclass
class ZeroCurve:
    reference_date: date
    times: np.ndarray
    discount_factors: np.ndarray
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR

    def discount_factor(self, t: float | date) -> float:
        """Returns D(t) using log-linear interpolation."""
        if isinstance(t, date):
            # Convert date to year fraction relative to reference_date
            tau = (t - self.reference_date).days / 365.0
        else:
            tau = float(t)

        if tau <= 0.0:
            return 1.0

        if tau > self.times[-1]:
            raise ValueError(
                f"Cannot extrapolate discount factor beyond curve maturity "
                f"({tau:.2f}y > {self.times[-1]:.2f}y)"
            )

        log_d = np.interp(tau, self.times, np.log(self.discount_factors))
        return float(np.exp(log_d))

    def zero_rate(self, t: float | date, continuous: bool = True) -> float:
        if isinstance(t, date):
            tau = (t - self.reference_date).days / 365.0
        else:
            tau = float(t)
        if tau <= 0.0:
            first_tau = self.times[1] if len(self.times) > 1 else 1e-4
            return self.zero_rate(first_tau, continuous=continuous)
        d_tau = self.discount_factor(tau)
        if continuous:
            return float(-np.log(d_tau) / tau)
        else:
            return float(d_tau ** (-1.0 / tau) - 1.0)

    def forward_rate(self, t1: float, t2: float) -> float:
        return (
            1 / (t2 - t1) * ((self.discount_factor(t1) / self.discount_factor(t2)) - 1)
        )
