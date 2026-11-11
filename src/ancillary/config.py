"""Shared constants for the simulator, model and optimiser.

Everything here is a SIMULATION ASSUMPTION. Partner names label scenarios built on
the public profile of each kind of partner. Traffic mixes, price sensitivity,
revenue share, markup limits and booking-rate floors are invented for the exercise
and are not real contract terms or real data. Supplier names are fictional.
"""
from __future__ import annotations

import numpy as np

# ---- markets ---------------------------------------------------------------
MARKETS = ["IE", "GB", "DE", "FR", "ES", "IT", "US", "AU"]
MARKET_PRICE_MULT = np.array([1.15, 1.05, 1.00, 0.95, 0.85, 0.90, 1.10, 1.12])
MARKET_SENS = np.array([0.90, 1.00, 1.10, 1.00, 1.25, 1.15, 0.85, 0.95])

# ---- partners (the B2B customers) -------------------------------------------
PARTNERS = ["Ryanair", "easyJet", "Southwest Airlines", "Qantas", "Airbnb", "eDreams ODIGEO"]
PARTNER_SLUG = ["ryanair", "easyjet", "southwest", "qantas", "airbnb", "edreams"]
PARTNER_KIND = ["Low-fare airline", "Low-fare airline", "Loyalty-led airline",
                "Flag carrier", "Travel marketplace", "Online travel agency"]
# Share of all sessions, and where each partner's traffic comes from.
PARTNER_TRAFFIC = np.array([0.28, 0.20, 0.20, 0.08, 0.10, 0.14])
#                                 IE    GB    DE    FR    ES    IT    US    AU
PARTNER_MARKET_PROBS = np.array([
    [0.14, 0.24, 0.16, 0.10, 0.20, 0.16, 0.00, 0.00],   # Ryanair
    [0.04, 0.40, 0.12, 0.18, 0.14, 0.12, 0.00, 0.00],   # easyJet
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],   # Southwest Airlines
    [0.00, 0.12, 0.00, 0.00, 0.00, 0.00, 0.08, 0.80],   # Qantas
    [0.00, 0.00, 0.00, 0.17, 0.17, 0.15, 0.38, 0.13],   # Airbnb
    [0.05, 0.15, 0.15, 0.15, 0.25, 0.20, 0.05, 0.00],   # eDreams ODIGEO
])
PARTNER_MOBILE = np.array([0.72, 0.64, 0.52, 0.50, 0.66, 0.35])
PARTNER_RETURNING = np.array([0.35, 0.30, 0.50, 0.40, 0.30, 0.20])
PARTNER_LEAD_SCALE = np.array([16.0, 18.0, 22.0, 30.0, 25.0, 28.0])  # gamma(2, scale) days
PARTNER_SENS = np.array([1.15, 1.10, 0.90, 0.90, 1.00, 1.25])
# Revenue share paid to the partner (assumed equal for all, edit to experiment).
PARTNER_SHARE = np.array([0.30, 0.30, 0.30, 0.30, 0.30, 0.30])
# Commercial guardrails per partner (assumptions).
PARTNER_MARKUP_MIN = np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08])
PARTNER_MARKUP_MAX = np.array([0.32, 0.32, 0.40, 0.44, 0.36, 0.28])
# Largest tolerated relative drop in booking rate versus the current-style rule.
PARTNER_CONV_FLOOR = np.array([0.05, 0.05, 0.03, 0.05, 0.05, 0.10])

# ---- cars and suppliers -----------------------------------------------------
CAR_CLASSES = ["mini", "economy", "compact", "suv", "van"]
CLASS_CAPACITY = np.array([2, 4, 4, 5, 7])
CLASS_BASE_COST = np.array([22.0, 27.0, 33.0, 52.0, 60.0])

SUPPLIERS = ["Alpha", "Bravo", "Cobalt", "Delta", "Ember", "Fjord"]
SUPPLIER_RATING = np.array([4.5, 4.2, 3.8, 4.0, 3.5, 4.6])
SUPPLIER_COST_MULT = np.array([1.10, 1.00, 0.92, 0.97, 0.88, 1.15])

N_OFFERS = 6
BASELINE_MARKUP = 0.20
# Markups the optimiser may choose from: 8% .. 44% in 4 point steps.
MARKUP_GRID = np.round(np.arange(0.08, 0.441, 0.04), 2)

MODEL_VERSION = "lr-v2-partners"
