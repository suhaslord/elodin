# Antarctica RL06.3Mv04 validation result

Status: **PASS_VALIDATED**

Validation run: GitHub Actions `34781870941` (`Antarctica public validation`).

## Inputs

- JPL GRACE/GRACE-FO Mascon RL06.3Mv04 CRI Level-3 public teaching-copy snapshot through 2026-01, checksum verified (`MD5 d7aa7765f8e84a1b29c686aeae476523`).
- WMO Climate Dashboard archive of the NASA/JPL Antarctica Level-4 series, checksum verified (`MD5 c2bebfc718c79e60d24c661b06eb47f8`). Its provenance records the official PO.DAAC Antarctica mass product as its source and the June-2005 zeroing operation.

## Reproduction method

- Integrate `lwe_thickness` over supplied `land_mask` grid cells at latitudes <= -60 degrees.
- Use supplied latitude/longitude bounds for spherical cell area with Earth radius 6,371,008.8 m.
- Convert cm liquid-water-equivalent thickness to Gt with water density 1000 kg/m^3.
- Rebase both reconstructed and reference series to 2005-06.
- Do **not** re-apply CRI/leakage correction or GIA correction; those are already represented in the Level-3 product.
- Do **not** apply the optional sub-mascon `scale_factor` for the continent-scale integral.

## Result

Overlap: 2002-04 through 2025-10, 248 common months.

- Pearson correlation: **0.9999728410**
- RMSE: **9.5124 Gt**
- MAE: **5.3214 Gt**
- Mean bias: **-5.1656 Gt**
- Maximum absolute monthly difference: **83.8833 Gt**
- Reconstructed linear trend: **-130.5293 Gt/yr**
- Reference linear trend: **-129.9967 Gt/yr**
- Trend difference: **-0.5326 Gt/yr**

This validates the Level-3-to-Antarctica-total-mass reproduction method for the publicly mirrored RL06.3Mv04 snapshot. The current official PO.DAAC files extend beyond this public snapshot; this result does not claim validation of months not present in the public inputs.
