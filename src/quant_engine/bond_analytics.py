import numpy as np
from bonds import Bond
from datetime import date

def dirty_price_from_yield(bond:Bond, ytm:float, settlement_date :date) ->float:
    return 0.0

def clean_price_from_yield(bond:Bond, ytm:float, settlement_date: date)-> float:
    return dirty_price_from_yield(bond,ytm,settlement_date) - bond.accrued_interest(settlement_date)
