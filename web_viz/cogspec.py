#!/usr/bin/env python
"""
The wire format of the COGs the browser streams. Single source of truth for Python.

The JS mirror lives in index.html (`SCALE` / `NOD` / the band constants). If you change
anything here, change it there -- there is no way to make one file the source of truth for
both, so the two are kept adjacent in the docs instead.

WHY int16 WITH NO GDAL scale/offset TAGS
----------------------------------------
maplibre-cog-protocol has two read paths and they disagree about scale/offset:

    setColorFunction(url, fn)   -> fn gets the RAW tile buffer. Tags NOT applied.
    locationValues(url, ...)    -> tags ARE applied.

So if we wrote int16 *with* `scale=0.01`, the heatmap would colour raw integers (529) while
the popup for the same pixel printed 5.29 deg. Silent, and it would look like a model bug.

The fix is to write no tags at all, ship raw integers, and divide by SCALE in ONE JS helper
that both paths call. That keeps the two paths honest by construction.

float32 -> int16 costs nothing real here: 0.01 deg on a horizon angle is 1/27th of the solar
disc, and it takes the file every client streams from 1029 MB to ~190 MB.
"""
from __future__ import annotations

import numpy as np

DATA_VERSION = 2          # bump on every re-export; filenames carry it, so caches can be
                          # immutable and a new run never serves stale bytes

EVI_TIF = f"evi.v{DATA_VERSION}.tif"      # streamed by every client, on every repaint
DEM_TIF = f"dem.v{DATA_VERSION}.tif"      # hillshade, 3D terrain, and the popup's elevation
IEEC_TIF = f"ieec.v{DATA_VERSION}.tif"    # Generalitat/IEEC binary map, for comparison

I16_NODATA = -32768

# band order, name, multiplier, and the physical range that has to survive int16
#   value_int16 = round(value_physical * scale)
EVI_BANDS = [
    ("horizon_deg",  100.0),   # 0..21.4 deg   -> 0..2142     (0.01 deg quantum)
    ("sun_alt_deg",  100.0),   # 3.2..6.7 deg  -> 319..673
    ("alt_drop_deg", 1000.0),  # 0..0.31 deg   -> 0..306      (needs the extra digit: it is
                               #                               the C2->C3 fall, and f_vis
                               #                               divides by it)
    ("duration_s",    10.0),   # 0..103 s      -> 0..1028
    ("occluder_km",  100.0),   # 0..29.8 km    -> 0..2980     (10 m quantum)
]
EVI_SCALE = [s for _, s in EVI_BANDS]
EVI_NAMES = [n for n, _ in EVI_BANDS]

# Elevation is NOT in evi.tif: it is byte-for-byte the DEM we already ship for the
# hillshade, and duplicating it cost ~56 MB of egress for nothing. The popup reads it
# from dem.tif with a second locationValues() call, which is one cached tile.
DEM_SCALE = 1.0            # plain metres. int16 is the DEM's natural dtype anyway.

# COG driver options. predictor 3 is the FLOATING-POINT predictor and is invalid for
# integers; 2 is horizontal differencing, which is what int16 wants.
COG_F32 = dict(driver="COG", compress="DEFLATE", predictor=3, blocksize=512,
               overview_resampling="average", BIGTIFF="IF_SAFER")
COG_I16 = dict(driver="COG", compress="DEFLATE", predictor=2, blocksize=512,
               overview_resampling="average", BIGTIFF="IF_SAFER")
# Categorical: averaging a 0/1 mask would invent values that are neither.
COG_U8_CAT = dict(driver="COG", compress="DEFLATE", blocksize=512,
                  overview_resampling="nearest", BIGTIFF="IF_SAFER")


def quantize(arr: np.ndarray, scale: float) -> np.ndarray:
    """Physical float -> raw int16, with non-finite mapped to I16_NODATA.

    Saturates rather than wrapping. A silent int16 wraparound would put a 21 deg ridge at
    -21 deg and paint the ugliest pixel on the map bright green.
    """
    out = np.full(arr.shape, I16_NODATA, dtype=np.int16)
    m = np.isfinite(arr)
    v = np.rint(arr[m] * scale)
    if v.size:
        lo, hi = float(v.min()), float(v.max())
        if lo <= I16_NODATA or hi > 32767:
            raise ValueError(
                f"int16 overflow: scaled range {lo:.0f}..{hi:.0f} does not fit "
                f"({I16_NODATA + 1}..32767). Lower the scale in cogspec.EVI_BANDS.")
    out[m] = v.astype(np.int16)
    return out


def dequantize(raw: np.ndarray, scale: float) -> np.ndarray:
    """Raw int16 -> physical float32, nodata -> NaN. The Python mirror of JS `dec()`."""
    out = raw.astype(np.float32) / scale
    out[raw == I16_NODATA] = np.nan
    return out
