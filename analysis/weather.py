#!/usr/bin/env python
"""Download hourly city weather for each agency, for the feature table.

The signs carry no coordinates, only an `agency` label. Each agency with a nameable city gets one
point (the city centre), and the hourly weather at that point is used for all of its signs. This is
city-level weather: it follows the regional weather (fronts, heat waves, cloudy days), not the shade
or wind at one stop. The agency -> city table below is our reading of the labels and is not confirmed
by BusPas. The agencies `buspas`, `demo`, `K` and `show` have no city and get no weather.

Source: Open-Meteo historical weather API (ERA5 reanalysis and its higher-resolution variants, grid of
about 10 to 25 km), https://open-meteo.com, data under CC BY 4.0. Only the city coordinates and the
dates are sent.

Each row describes the same hour as a row of the feature table, [time, time + 1 h) in UTC:
  * wx_solar_wm2, wx_precip_mm: the API labels these with the END of the hour they cover, so they are
    moved back one hour;
  * the other columns are instantaneous at the full hour, so the row holds the mean of the values at
    the start and the end of the hour.

Output: analysis/out/weather.parquet (agency, time, wx_*) and analysis/out/weather_sites.csv.
features.py joins the weather on (agency, time) when the file exists.
Usage: python -I analysis/weather.py [--start 2026-03-07] [--end 2026-10-07]
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/out"
API = "https://archive-api.open-meteo.com/v1/archive"

# agency label -> (city used, latitude, longitude). Our reading of the labels, to be confirmed by BusPas.
SITES = {
    "EXO": ("Montreal, Quebec", 45.50, -73.57),
    "RTL": ("Longueuil, Quebec", 45.53, -73.52),
    "TTC": ("Toronto, Ontario", 43.65, -79.38),
    "Barcelona": ("Barcelona, Spain", 41.39, 2.17),
    "NJT": ("Newark, New Jersey", 40.74, -74.17),
    "BCT": ("Fort Lauderdale, Florida", 26.12, -80.14),
    "BalHarbour": ("Bal Harbour, Florida", 25.89, -80.13),
    "LA": ("Los Angeles, California", 34.05, -118.24),
}
# API variable -> (column, True if the value covers the hour that ENDS at the label)
VARS = {
    "temperature_2m": ("wx_temp_c", False),
    "relative_humidity_2m": ("wx_humidity_pct", False),
    "dew_point_2m": ("wx_dewpoint_c", False),
    "cloud_cover": ("wx_cloud_pct", False),
    "wind_speed_10m": ("wx_wind_ms", False),
    "shortwave_radiation": ("wx_solar_wm2", True),
    "precipitation": ("wx_precip_mm", True),
}


def fetch(lat: float, lon: float, start: str, end: str) -> dict:
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": ",".join(VARS), "timezone": "GMT", "wind_speed_unit": "ms",
    })
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "buspas-analysis/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # network hiccup or rate limit: wait and retry
            if attempt == 3:
                raise
            print(f"  retry after error: {e}", flush=True)
            time.sleep(5 * (attempt + 1))


def to_buckets(hourly: dict) -> pd.DataFrame:
    """API hourly block -> one row per hour bucket [time, time + 1 h)."""
    raw = pd.DataFrame(hourly)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).astype("datetime64[ms, UTC]")  # as in the feature table
    raw = raw.set_index("time").astype(float)
    out = pd.DataFrame(index=raw.index[:-1])
    for var, (col, covers_previous_hour) in VARS.items():
        nxt = raw[var].shift(-1)
        out[col] = (nxt if covers_previous_hour else (raw[var] + nxt) / 2).iloc[:-1].astype("float32")
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-03-07", help="first UTC day (one day before the data starts)")
    ap.add_argument("--end", default="2026-10-07", help="last UTC day (one day after the data ends)")
    args = ap.parse_args()

    frames, sites = [], []
    for agency, (city, lat, lon) in SITES.items():
        j = fetch(lat, lon, args.start, args.end)
        w = to_buckets(j["hourly"])
        w.insert(0, "agency", agency)
        frames.append(w)
        sites.append({"agency": agency, "city": city, "latitude": lat, "longitude": lon,
                      "grid_latitude": j["latitude"], "grid_longitude": j["longitude"],
                      "grid_elevation_m": j["elevation"], "hours": len(w),
                      "hours_missing": int(w.filter(like="wx_").isna().any(axis=1).sum())})
        print(f"{agency:<11} {city:<26} {len(w):>5} hours, {sites[-1]['hours_missing']} with a missing value", flush=True)
        time.sleep(1)

    wx = pd.concat(frames, ignore_index=True)
    wx.to_parquet(OUT / "weather.parquet", index=False)
    pd.DataFrame(sites).to_csv(OUT / "weather_sites.csv", index=False)
    print(f"weather: {len(wx):,} rows, {wx['time'].min()} to {wx['time'].max()} -> analysis/out/weather.parquet")


if __name__ == "__main__":
    main()
