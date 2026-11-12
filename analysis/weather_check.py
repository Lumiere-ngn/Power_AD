#!/usr/bin/env python
"""Check the city weather against the signs' own sensors (8 March to 31 July 2026 only).

Three checks per sign, none of which changes any data:
  1. City match. The sign's daily mean temperature, with the season removed (minus a centred 15-day
     mean), is correlated with the outside temperature of each of the six weather areas. If the
     agency -> city reading is right, the sign's own area should fit best. Signs without an enclosure
     temperature sensor use the board temperature.
  2. Hourly agreement of enclosure temperature and humidity with the outside values.
  3. Light timing. The light sensor is correlated with solar radiation shifted by -3 to +3 hours;
     the best shift should be 0 if the city (its longitude) and the hour alignment are right.
For the signs without a city, the best-fitting area is printed as a hint only.

Reads analysis/out/features.parquet and weather.parquet. Prints a summary (kept in
analysis/out/weather_check.log) and writes analysis/out/weather_check.csv (one row per sign).
Usage: python -I analysis/weather_check.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "analysis/out"
END = "2026-08-01"  # the test months stay unused
AREA = {"EXO": "Montreal", "RTL": "Montreal", "TTC": "Toronto", "Barcelona": "Barcelona", "NJT": "Newark",
        "BCT": "South Florida", "BalHarbour": "South Florida", "LA": "Los Angeles"}
AREA_WEATHER = {"Montreal": "EXO", "Toronto": "TTC", "Barcelona": "Barcelona", "Newark": "NJT",
                "South Florida": "BCT", "Los Angeles": "LA"}
MIN_HOURS, MIN_DAYS = 24 * 30, 30


def detrend(s: pd.Series) -> pd.Series:
    """Daily values minus a centred 15-day mean: keeps the day-to-day weather, drops the season."""
    return s - s.rolling(15, center=True, min_periods=8).mean()


def main() -> None:
    cols = ["device", "agency", "time", "temperature", "board_temp_celsius", "lightsensor", "humidity",
            "wx_temp_c", "wx_solar_wm2", "wx_humidity_pct"]
    f = pd.read_parquet(OUT / "features.parquet", columns=cols)
    f = f[f["time"] < END]
    wx = pd.read_parquet(OUT / "weather.parquet")
    wx = wx[wx["time"] < END]
    area_day = {a: detrend(wx[wx["agency"] == ag].set_index("time")["wx_temp_c"].resample("D").mean())
                for a, ag in AREA_WEATHER.items()}

    rows = []
    for dev, g in f.groupby("device", observed=True):
        g = g.set_index("time")
        agency = g["agency"].iloc[0]
        own = AREA.get(agency)
        sensor = "temperature" if g["temperature"].notna().sum() >= MIN_HOURS else "board_temp_celsius"
        daily = g[sensor].resample("D")
        d = daily.mean()[daily.count() >= 18]
        if len(d) < MIN_DAYS:
            continue
        d = detrend(d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D", tz="UTC", unit="ms")))
        r = {a: d.corr(s) for a, s in area_day.items()}
        best = max(r, key=lambda a: -9 if np.isnan(r[a]) else r[a])
        row = {"device": dev, "agency": agency, "area": own, "sensor": sensor, "days": int(d.notna().sum()),
               "best_area": best, "r_best": r[best], "r_own": r.get(own, np.nan)}
        if own:
            if sensor == "temperature":
                row["r_temp_hourly"] = g["temperature"].corr(g["wx_temp_c"])
                row["temp_minus_outside"] = (g["temperature"] - g["wx_temp_c"]).median()
            if g["humidity"].notna().sum() >= MIN_HOURS:
                row["r_humidity_hourly"] = g["humidity"].corr(g["wx_humidity_pct"])
            if g["lightsensor"].notna().sum() >= MIN_HOURS:
                lag = {k: g["lightsensor"].corr(g["wx_solar_wm2"].shift(k)) for k in range(-3, 4)}
                row["light_best_shift_h"] = max(lag, key=lag.get)
                row["r_light_hourly"] = lag[0]
        rows.append(row)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "weather_check.csv", index=False)

    K = R[R["area"].notna()].copy()
    K["match"] = K["best_area"] == K["area"]
    print(f"signs with a city and at least {MIN_DAYS} days before August: {len(K)}")
    print(K.groupby("agency", observed=True).agg(
        signs=("device", "size"), own_area_fits_best=("match", "sum"), r_own=("r_own", "median"),
        r_temp_hourly=("r_temp_hourly", "median"), temp_minus_outside=("temp_minus_outside", "median"),
        r_humidity_hourly=("r_humidity_hourly", "median"), r_light_hourly=("r_light_hourly", "median"),
    ).round(2).to_string())
    print(f"own area fits best for {int(K['match'].sum())} of {len(K)} signs")
    if (~K["match"]).any():
        print("signs whose own area does not fit best:")
        print(K[~K["match"]][["device", "agency", "sensor", "days", "best_area", "r_best", "r_own"]].round(2).to_string(index=False))
    L = K.dropna(subset=["light_best_shift_h"])
    print(f"light sensor, best shift in hours (signs): {L['light_best_shift_h'].astype(int).value_counts().sort_index().to_dict()}")
    print(L.groupby("agency", observed=True)["light_best_shift_h"].agg(
        lambda s: s.astype(int).value_counts().sort_index().to_dict()).to_string())
    U = R[R["area"].isna()]
    print("signs without a city: best-fitting area (a hint only, not used)")
    print(U.groupby(["agency", "best_area"], observed=True).agg(signs=("device", "size"), r_best=("r_best", "median")).round(2).to_string())


if __name__ == "__main__":
    main()
