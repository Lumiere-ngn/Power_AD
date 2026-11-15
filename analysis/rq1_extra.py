"""Extra breakdowns of the RQ1 test predictions. UNPLANNED: not in the protocol, added on 15 November 2026
after the third run had been read, in answer to a reviewer.

Nothing is fitted here. It only re-aggregates out/rq1_test_predictions.parquet (the locked third run) joined
with the hourly table. Error = mean |measured - predicted| over hours; bias = mean (measured - predicted),
so a positive bias means the prediction is too low.

  .venv/bin/python -I analysis/rq1_extra.py      -> analysis/out/rq1_extra.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # -I drops the script directory
import rq_common as C  # noqa: E402

TEMP_BANDS = [-np.inf, 30, 40, 50, 60, np.inf]      # measured or predicted CPU/GPU/board temperature, °C
AIR_BANDS = [-np.inf, 10, 15, 20, 25, 30, np.inf]   # city air temperature, °C
MEM_BANDS = [-np.inf, 4, 6, np.inf]                 # memory in use, GB


def stats(g: pd.DataFrame) -> dict:
    a, b = (g.y - g.pred).abs(), (g.y - g.baseline).abs()
    return {"hours": int(len(g)), "devices": int(g.device.nunique()), "mean_measured": float(g.y.mean()),
            "mae": float(a.mean()), "bias": float((g.y - g.pred).mean()),
            "skill": float(1 - a.sum() / b.sum()) if b.sum() > 0 else None}


def by(d: pd.DataFrame, key) -> dict:
    return {str(k): stats(g) for k, g in d.groupby(key, observed=True, dropna=False) if len(g)}


def band(v: pd.Series, edges: list, unit: str) -> pd.Series:
    labels = [f"below {edges[1]} {unit}"] + [f"{a} to {b} {unit}" for a, b in zip(edges[1:-2], edges[2:-1])] + [f"{edges[-2]} {unit} or more"]
    return pd.cut(v, edges, labels=labels, right=False)


def main():
    p = pd.read_parquet(C.OUT / "rq1_test_predictions.parquet")
    f = C.load()
    f["device"] = f["device"].astype(str)
    f["agency"] = f["agency"].astype(str)
    fit = f[f["time"] < C.CAL]                       # the hours the test models were fitted on (8 March - 30 June)
    out = {"note": "UNPLANNED re-aggregation of the third-run test predictions (15 November 2026). Nothing refitted."}

    for y in C.TARGETS:
        d = p[p.target == y].merge(f.drop(columns=[y]), on=["device", "time"], how="left", validate="one_to_one")
        r = {"all": stats(d)}

        # --- hardware configuration of the scored devices
        cfg = (np.where(d.v56 == 1, "56 V", "12 V") + pd.Series(np.where(d.has_mppt == 1, ", solar", ", no solar"), index=d.index)
               + np.where(d.has_env == 1, ", enclosure sensor", ", no enclosure sensor")
               + np.where(d.has_ds == 1, ", video analytics", ", no video analytics"))
        r["by_configuration"] = by(d, cfg)
        r["by_enclosure_sensor"] = by(d, np.where(d.has_env == 1, "enclosure sensor", "no enclosure sensor"))
        r["by_video_analytics_device"] = by(d, np.where(d.has_ds == 1, "has video analytics", "no video analytics"))

        # --- workload level in that hour
        app = np.where(d.deepstream_app_running.isna(), "not reported", np.where(d.deepstream_app_running >= 0.5, "running", "not running"))
        r["by_app_running"] = by(d, app)
        r["by_memory_in_use"] = by(d, band(d.mem_used_gb, MEM_BANDS, "GB"))
        pir = d[d.pir_count_rate.notna()]
        r["by_motion_count_third"] = by(pir, pd.qcut(pir.pir_count_rate, 3, labels=["lowest third", "middle third", "highest third"]))

        if y in C.THERMAL:
            # --- each device's own coolest / hottest 10 % of measured hours (same rule as rq1.py for the hottest)
            q10 = d.groupby("device").y.transform(lambda v: v.quantile(0.1))
            q90 = d.groupby("device").y.transform(lambda v: v.quantile(0.9))
            own = np.where(d.y > q90, "hottest 10 %", np.where(d.y < q10, "coolest 10 %", "middle 80 %"))
            r["by_own_level"] = by(d, own)
            inside, above = (d.y >= d.lo) & (d.y <= d.hi), d.y > d.hi
            r["band_by_own_level"] = {k: {"inside": float(inside[own == k].mean()), "above": float(above[own == k].mean())}
                                      for k in ("coolest 10 %", "middle 80 %", "hottest 10 %")} | {
                                      "all": {"inside": float(inside.mean()), "above": float(above.mean())}}
            hb = (d.y - d.pred)[d.y > q90].groupby(d.device).mean()
            r["hottest_10pct_devices"] = {"devices": int(len(hb)), "predicted_too_low": int((hb > 0).sum()),
                                          "too_low_by_more_than_1": int((hb > 1).sum()), "median_bias": float(hb.median())}
            worst = d.device == "bp3235300141"   # the device with the largest CPU/GPU error (section 1)
            r["by_own_level_without_bp3235300141"] = by(d[~worst], own[~worst])
            r["by_measured_band_without_bp3235300141"] = by(d[~worst], band(d.y[~worst], TEMP_BANDS, "°C"))
            r["by_configuration_without_bp3235300141"] = by(d[~worst], cfg[~worst])
            # --- fixed thresholds: by measured value, by predicted value, by city air temperature
            r["by_measured_band"] = by(d, band(d.y, TEMP_BANDS, "°C"))
            r["by_predicted_band"] = by(d, band(d.pred, TEMP_BANDS, "°C"))
            r["by_air_temperature"] = by(d[d.wx_temp_c.notna()], band(d.wx_temp_c[d.wx_temp_c.notna()], AIR_BANDS, "°C"))
            # --- was the hour outside what the device showed in the fit months?
            lim = fit.groupby("device").agg(y_max=(y, "max"), y_p99=(y, lambda v: v.quantile(0.99)),
                                            wx_max=("wx_temp_c", "max"), wx_p99=("wx_temp_c", lambda v: v.quantile(0.99)))
            d = d.join(lim, on="device")
            hot = d[d.y > q90]
            rng = {}
            for name, col, ref in [("reading above the device's fit-months maximum", "y", "y_max"),
                                   ("reading above the device's fit-months 99th percentile", "y", "y_p99"),
                                   ("air temperature above the city's fit-months maximum", "wx_temp_c", "wx_max"),
                                   ("air temperature above the city's fit-months 99th percentile", "wx_temp_c", "wx_p99")]:
                rng[name] = {"share_of_all_hours": float((d[col] > d[ref]).mean()),
                             "share_of_hottest_10pct": float((hot[col] > hot[ref]).mean()),
                             "hottest_10pct_outside": stats(hot[hot[col] > hot[ref]]) if (hot[col] > hot[ref]).any() else None,
                             "hottest_10pct_inside": stats(hot[~(hot[col] > hot[ref])])}
            r["training_range"] = rng
            mon = f[f.device.isin(d.device.unique())].assign(month=lambda x: x.time.dt.strftime("%Y-%m")).groupby("month")
            r["by_month_context"] = {m: {"mean_reading": float(g[y].mean()), "p90_reading": float(g[y].quantile(0.9)),
                                         "mean_air": float(g.wx_temp_c.mean()), "p90_air": float(g.wx_temp_c.quantile(0.9))} for m, g in mon}
        out[y] = r

    (C.OUT / "rq1_extra.json").write_text(json.dumps(C.clean(out), indent=1))

    def show(title, t, keys=("hours", "devices", "mean_measured", "mae", "bias", "skill")):
        print(f"\n{title}")
        print(pd.DataFrame(t).T[list(keys)].round(2).to_string())

    for y in ["cpu_temp_celsius", "gpu_temp_celsius", "board_temp_celsius", "input_power_watts"]:
        print(f"\n================ {y}   all: {out[y]['all']}")
        for k, v in out[y].items():
            if k.startswith("by_") and k != "by_month_context":
                show(k, v)
        if "training_range" in out[y]:
            for k, v in out[y]["training_range"].items():
                o, i = v["hottest_10pct_outside"], v["hottest_10pct_inside"]
                print(f"\n{k}: {v['share_of_all_hours']:.1%} of all hours, {v['share_of_hottest_10pct']:.1%} of hottest-10% hours")
                print("   outside:", o and {a: round(b, 2) for a, b in o.items() if b is not None})
                print("   inside: ", {a: round(b, 2) for a, b in i.items() if b is not None})
            print(pd.DataFrame(out[y]["by_month_context"]).T.round(1).to_string())


if __name__ == "__main__":
    main()
