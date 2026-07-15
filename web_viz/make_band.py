#!/usr/bin/env python
"""
The totality-band polygon, buffered -- the mask every Spain parcel is clipped to.

Output outside the path carries no eclipse information, and (worse) the non-band *land*
inside each parcel rectangle holds real horizon values that compress nothing like sea. So
we nodata everything outside mag>=1 plus a small margin that catches coastal totality and
keeps the edge from feathering. Vectorised from the coarse Besselian grid; the 0.01 deg
jaggies vanish when rasterised onto a 50 m output.
"""
import json
import os
import sys

import numpy as np
import shapely.geometry as sg
import shapely.ops as ops
from affine import Affine
from rasterio import features

HERE = os.path.dirname(os.path.abspath(__file__))
ECLIPSE = os.path.dirname(HERE)
sys.path.insert(0, ECLIPSE)

import besselian as B  # noqa: E402
import path as P  # noqa: E402

BBOX = (-10.0, 36.5, 5.6, 44.6)
STEP = 0.01
BUFFER = 0.12                       # deg (~13 km): coastal totality + clean edges
MIN_AREA = 0.05                     # deg^2; drop specks from the coarse rasterisation
OUT = os.path.join(HERE, "data", "band.geojson")


def main():
    e = B.Elements()
    lons, lats, mag, _ = P.magnitude_grid(e, BBOX, step=STEP)
    inside = mag >= 1.0
    aff = Affine.translation(lons[0], lats[0]) * Affine.scale(STEP, STEP)
    polys = [sg.shape(g) for g, v in features.shapes(inside.astype("uint8"),
                                                      mask=inside, transform=aff)]
    band = ops.unary_union([p for p in polys if p.area > MIN_AREA]).buffer(BUFFER)
    feat = {"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"kind": "totality_band", "buffer_deg": BUFFER},
        "geometry": sg.mapping(band)}]}
    with open(OUT, "w") as f:
        json.dump(feat, f)
    npoly = 1 if band.geom_type == "Polygon" else len(band.geoms)
    print(f"band: {band.geom_type}({npoly}) area {band.area:.0f} deg^2 -> {OUT}")


if __name__ == "__main__":
    main()
