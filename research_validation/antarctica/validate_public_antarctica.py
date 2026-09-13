#!/usr/bin/env python3
"""Reproduce the JPL Antarctica mass time series from the RL06.3Mv04 CRI grid.

The public L3 teaching copy is byte-identical to the cited JPL RL06.3Mv04
NetCDF snapshot through 2026-01. The validation target is the WMO dashboard's
archived NASA/JPL GRACE series, whose history records that it came from the
official PO.DAAC Level-4 Antarctica product and was zeroed on June 2005.

Primary reconstruction:
  * use CRI-filtered lwe_thickness as supplied (no extra leakage correction);
  * use the supplied land_mask;
  * select Antarctic land south of 60 S;
  * integrate equivalent-water thickness over spherical grid-cell area;
  * do not apply optional scale_factor, because this is a continent-scale
    integral rather than a sub-mascon downscaling application;
  * rebase both series to June 2005 before comparison.

Additional candidates are reported only as diagnostics, not silently selected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

R_EARTH_M = 6_371_008.8
CM_WATER_TO_GT_PER_M2 = 1e-11  # cm /100 * 1000 kg/m^3 / 1e12 kg/Gt


def read_badc_csv(path: Path) -> dict[tuple[int, int], float]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    try:
        marker = lines.index("data")
    except ValueError as exc:
        raise ValueError(f"BADC CSV missing data marker: {path}") from exc
    rows = csv.DictReader(lines[marker + 1 :])
    out: dict[tuple[int, int], float] = {}
    for row in rows:
        if not row.get("year") or not row.get("month") or not row.get("data"):
            continue
        out[(int(row["year"]), int(row["month"]))] = float(row["data"])
    return out


def cell_areas(lat_bounds: np.ndarray, lon_bounds: np.ndarray) -> np.ndarray:
    lat_s = np.deg2rad(np.minimum(lat_bounds[:, 0], lat_bounds[:, 1]))
    lat_n = np.deg2rad(np.maximum(lat_bounds[:, 0], lat_bounds[:, 1]))
    d_sin = np.sin(lat_n) - np.sin(lat_s)
    dlon = np.deg2rad(np.abs(lon_bounds[:, 1] - lon_bounds[:, 0]))
    # Guard the 0/360 wrap convention if it ever appears in a future file.
    dlon = np.where(dlon > np.pi, 2 * np.pi - dlon, dlon)
    return (R_EARTH_M**2) * d_sin[:, None] * dlon[None, :]


def metrics(calc: np.ndarray, ref: np.ndarray, years: np.ndarray) -> dict[str, float | int]:
    err = calc - ref
    x = years - years.mean()
    calc_slope = float(np.polyfit(x, calc, 1)[0])
    ref_slope = float(np.polyfit(x, ref, 1)[0])
    return {
        "n": int(calc.size),
        "bias_gt": float(np.mean(err)),
        "mae_gt": float(np.mean(np.abs(err))),
        "rmse_gt": float(np.sqrt(np.mean(err**2))),
        "max_abs_gt": float(np.max(np.abs(err))),
        "correlation": float(np.corrcoef(calc, ref)[0, 1]),
        "reconstructed_trend_gt_per_year": calc_slope,
        "reference_trend_gt_per_year": ref_slope,
        "trend_difference_gt_per_year": calc_slope - ref_slope,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--l3", required=True)
    p.add_argument("--l4", required=True)
    p.add_argument("--out-dir", default="results/antarctica")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = read_badc_csv(Path(args.l4))

    with Dataset(args.l3) as ds:
        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lat_bounds = np.asarray(ds.variables["lat_bounds"][:], dtype=np.float64)
        lon_bounds = np.asarray(ds.variables["lon_bounds"][:], dtype=np.float64)
        land = np.asarray(ds.variables["land_mask"][:], dtype=np.float64) > 0.5
        scale = np.asarray(ds.variables["scale_factor"][:], dtype=np.float64)
        time_var = ds.variables["time"]
        dates = num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, "calendar", "standard"))
        keys = [(int(d.year), int(d.month)) for d in dates]
        areas = cell_areas(lat_bounds, lon_bounds)
        south60 = lat[:, None] <= -60.0

        masks = {
            "antarctic_land_unscaled": land & south60,
            "south60_all_cells_unscaled": np.broadcast_to(south60, land.shape),
            "antarctic_land_scaled": land & south60,
        }
        series = {name: [] for name in masks}
        lwe_var = ds.variables["lwe_thickness"]
        for ti in range(len(keys)):
            lwe = np.ma.filled(lwe_var[ti, :, :], np.nan).astype(np.float64)
            for name, mask in masks.items():
                vals = lwe * areas * CM_WATER_TO_GT_PER_M2
                if name == "antarctic_land_scaled":
                    vals = vals * scale
                series[name].append(float(np.nansum(np.where(mask, vals, np.nan))))

    baseline = (2005, 6)
    if baseline not in keys or baseline not in ref:
        raise ValueError("June 2005 baseline missing from L3 or reference")
    base_i = keys.index(baseline)
    common_i = [i for i, key in enumerate(keys) if key in ref]
    common_keys = [keys[i] for i in common_i]
    ref_arr = np.asarray([ref[k] for k in common_keys], dtype=np.float64)
    ref_arr = ref_arr - float(ref[baseline])
    years = np.asarray([y + (m - 0.5) / 12.0 for y, m in common_keys], dtype=np.float64)

    results: dict[str, dict] = {}
    comparison_rows = []
    rebased_series = {}
    for name, raw in series.items():
        arr = np.asarray(raw, dtype=np.float64)
        arr = arr - arr[base_i]
        calc = arr[common_i]
        rebased_series[name] = calc
        results[name] = metrics(calc, ref_arr, years)

    primary = "antarctic_land_unscaled"
    primary_metrics = results[primary]
    status = "PASS_VALIDATED" if primary_metrics["correlation"] >= 0.99 and abs(primary_metrics["trend_difference_gt_per_year"]) <= 10.0 else "VALIDATION_MISMATCH"

    for j, (y, m) in enumerate(common_keys):
        row = {"year": y, "month": m, "reference_gt": float(ref_arr[j])}
        for name in rebased_series:
            row[name + "_gt"] = float(rebased_series[name][j])
        comparison_rows.append(row)

    with (out / "antarctica_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)

    report = {
        "status": status,
        "primary_method": primary,
        "baseline": "2005-06",
        "region": "supplied land_mask at grid centers south of or equal to 60 S",
        "area_model": f"spherical Earth radius {R_EARTH_M} m using supplied lat/lon bounds",
        "scale_factor_applied_primary": False,
        "extra_GIA_correction_applied": False,
        "extra_CRI_or_leakage_correction_applied": False,
        "overlap_start": f"{common_keys[0][0]:04d}-{common_keys[0][1]:02d}",
        "overlap_end": f"{common_keys[-1][0]:04d}-{common_keys[-1][1]:02d}",
        "metrics": results,
    }
    (out / "antarctica_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ANTARCTICA_VALIDATION " + json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
