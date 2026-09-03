from datetime import date


class Option:
    def __init__(self, strike: float, expiry: date):
        self.strike = strike
        self.expiry = expiry
