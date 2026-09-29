"""Exogenous energy signals: day-ahead price, on-site PV and grid carbon intensity.

Real data: one day (2026-09-28, DE-LU bidding zone, 15-min resolution) retrieved
from the Fraunhofer ISE Energy-Charts API (price data: Bundesnetzagentur | SMARD.de,
CC BY 4.0).  It is used *only for testing*.  Training uses synthetic days drawn from
the parametric generator :func:`synthetic_day`, whose parameter ranges were fixed a
priori from typical German autumn/spring days and were not fitted to the test day.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass

import numpy as np

SLOTS_PER_DAY = 96
SLOT_MIN = 15


@dataclass
class DayProfile:
    name: str
    price: np.ndarray    # (96,) EUR/MWh
    solar: np.ndarray    # (96,) normalised solar shape in [0, 1]
    co2: np.ndarray      # (96,) gCO2eq/kWh

    def minute_arrays(self, start_min: int, horizon: int, pv_kwp: float):
        """Per-minute signals for t = 0..horizon-1 when the shift starts at
        ``start_min`` minutes after local midnight.  Beyond 24 h the day is
        repeated cyclically (stated as a modelling assumption in the paper)."""
        t = (start_min + np.arange(horizon)) // SLOT_MIN % SLOTS_PER_DAY
        return (self.price[t].astype(np.float64), pv_kwp * self.solar[t].astype(np.float64),
                self.co2[t].astype(np.float64))


def load_energy_charts_day(data_dir: str) -> DayProfile:
    """Build the real test day from the raw Energy-Charts JSON responses."""
    p = json.load(open(os.path.join(data_dir, "smard_de_lu_prices.json")))
    pw = json.load(open(os.path.join(data_dir, "smard_de_lu_power.json")))
    c = json.load(open(os.path.join(data_dir, "germany_co2eq.json")))
    price = np.asarray(p["price"], dtype=np.float64)
    assert price.shape[0] == SLOTS_PER_DAY
    solar = [x for x in pw["production_types"] if x["name"] == "Solar"][0]["data"]
    solar = np.asarray([0.0 if v is None else v for v in solar], dtype=np.float64)
    # the power series ends at 22:00 local time; solar output is zero at night
    solar = np.concatenate([solar, np.zeros(SLOTS_PER_DAY - solar.shape[0])])
    co2m, co2f = c["co2eq"], c["co2eq_forecast"]
    # measured values where available, otherwise the Energy-Charts forecast
    co2 = np.asarray([co2m[i] if co2m[i] is not None else co2f[i] for i in range(SLOTS_PER_DAY)],
                     dtype=np.float64)
    tz = _dt.timezone(_dt.timedelta(hours=2))
    date = _dt.datetime.fromtimestamp(p["unix_seconds"][0], tz).strftime("%Y-%m-%d")
    return DayProfile(f"DE-LU {date}", price, solar / solar.max(), co2)


def synthetic_day(rng: np.random.Generator, name: str = "synthetic") -> DayProfile:
    """Random day with morning/evening price peaks, a PV-induced midday price dip
    and a carbon intensity that decreases with the solar share."""
    h = (np.arange(SLOTS_PER_DAY) + 0.5) * SLOT_MIN / 60.0
    # --- solar shape (clear-sky bell modulated by clouds)
    noon = rng.uniform(12.8, 13.8)
    width = rng.uniform(2.0, 3.3)
    clear = np.exp(-0.5 * ((h - noon) / width) ** 2)
    clear[np.abs(h - noon) > 2.6 * width] = 0.0
    cloud = np.clip(1.0 - rng.uniform(0.0, 0.6) * _smooth_noise(rng, SLOTS_PER_DAY, 12), 0.05, 1.0)
    amp = rng.uniform(0.25, 1.0)
    solar = amp * clear * cloud
    # --- price (EUR/MWh)
    base = rng.uniform(60.0, 160.0)
    morning = rng.uniform(20.0, 120.0) * np.exp(-0.5 * ((h - rng.uniform(7.0, 8.5)) / 1.2) ** 2)
    evening = rng.uniform(40.0, 220.0) * np.exp(-0.5 * ((h - rng.uniform(18.5, 20.5)) / 1.6) ** 2)
    dip = rng.uniform(40.0, 160.0) * solar
    price = base + morning + evening - dip + rng.uniform(5.0, 20.0) * _smooth_noise(rng, SLOTS_PER_DAY, 4)
    price = np.maximum(price, -50.0)
    # --- carbon intensity (gCO2eq/kWh)
    co2 = (rng.uniform(300.0, 560.0) - rng.uniform(100.0, 260.0) * solar
           + rng.uniform(20.0, 90.0) * np.exp(-0.5 * ((h - 19.5) / 2.0) ** 2)
           + rng.uniform(10.0, 30.0) * _smooth_noise(rng, SLOTS_PER_DAY, 8))
    co2 = np.clip(co2, 80.0, 750.0)
    s = solar / solar.max() if solar.max() > 0 else solar
    return DayProfile(name, price, s, co2)


def _smooth_noise(rng, n, k):
    x = rng.standard_normal(n + 2 * k)
    ker = np.ones(k) / k
    y = np.convolve(x, ker, mode="same")[k:k + n]
    return y / (np.abs(y).max() + 1e-9)
