#!/usr/bin/env python
"""
Export the path of totality (limits, central line, duration contours) as GeoJSON,
plus the Generalitat's official observation points for contrast.

Run:  conda run -n geography python eclipse/web_viz/export_vectors.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ECLIPSE = os.path.dirname(HERE)
ROOT = os.path.dirname(ECLIPSE)
sys.path.insert(0, ECLIPSE)

import besselian as B  # noqa: E402
import path as P  # noqa: E402

OUT = os.path.join(HERE, "data")

# Iberia + a margin. 0.02 deg (~2 km) resolves the limits well enough to draw;
# the limits are a smooth curve, not a fractal.
BBOX = (-11.0, 35.0, 6.0, 46.0)
STEP = 0.02


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    E = B.Elements()
    fc = P.to_geojson(E, bbox=BBOX, step=STEP)

    kinds = {}
    for f in fc["features"]:
        kinds[f["properties"]["kind"]] = kinds.get(f["properties"]["kind"], 0) + 1
    print(f"features: {kinds}")

    dst = os.path.join(OUT, "totality.geojson")
    with open(dst, "w") as f:
        json.dump(fc, f)
    print(f"wrote {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")

    # central line sanity: where is it, and what is the max duration on it?
    cl = P.central_line(E)
    if cl:
        best = max(cl, key=lambda r: r[3])
        print(f"central line: {len(cl)} samples, "
              f"max duration {best[3]:.0f}s at ({best[0]:.3f}, {best[1]:.3f}) {B.hms(best[2])} UT")
        near = [r for r in cl if -2.5 < r[1] < 2.5 and 39 < r[0] < 43]
        if near:
            print("  over our AOI:")
            for la, lo, ut, d in near[::3]:
                print(f"    {la:7.3f}, {lo:7.3f}   {B.hms(ut)} UT   {d:5.0f}s")

    # the official observation points, for on-map contrast
    src = os.path.join(ROOT, "generalitat_eclipse_map_data",
                       "observation_points_labeled.geojson")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(OUT, "observation_points.geojson"))
        print("copied observation_points.geojson")

    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
