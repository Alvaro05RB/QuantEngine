from datetime import date
from dataclasses import dataclass

@dataclass(frozen=True)
class Option:
    strike :float
    expiry: date

