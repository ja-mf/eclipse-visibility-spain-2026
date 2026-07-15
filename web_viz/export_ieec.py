#!/usr/bin/env python
"""
Reproject the Generalitat/IEEC visibility map into a COG the browser can overlay.

Theirs is the map everyone in Catalonia is actually looking at
(https://eclipsi2026.cat), and it is binary: a place either "sees" the eclipse or it does
not. Ours is continuous. Putting one on top of the other is the fastest way to see where
the two models disagree -- and disagreement is the interesting signal, because it is almost
always a clearance-threshold argument (their map behaves like it demands ~1 deg of
clearance; the solar disc only needs 0.27 deg).

Input is what we scraped during validation:
    generalitat_eclipse_map_data/visibility.tif   EPSG:25831, uint8, 1 = visible, 0 = not

Output:
    data/ieec.v1.tif   EPSG:3857 uint8 COG, 255 = outside their map

NEAREST resampling everywhere, including the overviews: this is a categorical mask, and
averaging 0 and 1 invents a 0.5 that means nothing.

Run:  conda run -n geography python eclipse/web_viz/export_ieec.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import rioxarray  # noqa: F401  (registers .rio)
from rasterio.enums import Resampling

HERE = os.path.dirname(os.path.abspath(__file__))
ECLIPSE = os.path.dirname(HERE)
REPO = os.path.dirname(ECLIPSE)
sys.path.insert(0, HERE)

from cogspec import COG_U8_CAT, IEEC_TIF  # noqa: E402

SRC = os.path.join(REPO, "generalitat_eclipse_map_data", "visibility.tif")
OUT = os.path.join(HERE, "data")
NODATA = 255


def main():
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} -- run the Generalitat scrape first")

    da = rioxarray.open_rasterio(SRC).squeeze(drop=True)
    a = da.values
    print(f"source {da.rio.crs} {a.shape} values {sorted(np.unique(a).tolist())}")

    # Their nodata is unset in visibility.tif but 255 is used as "outside the map".
    da = da.rio.write_nodata(NODATA)
    web = da.rio.reproject("EPSG:3857", resampling=Resampling.nearest, nodata=NODATA)
    web = web.astype(np.uint8)

    v = web.values
    n_vis = int((v == 1).sum())
    n_not = int((v == 0).sum())
    print(f"reprojected {web.shape}: visible {n_vis/1e3:.0f}k px, "
          f"not visible {n_not/1e3:.0f}k px, outside {int((v == NODATA).sum())/1e3:.0f}k px")

    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, IEEC_TIF)
    web.rio.to_raster(dst, **COG_U8_CAT)
    print(f"wrote {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
