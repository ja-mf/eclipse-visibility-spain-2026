#!/usr/bin/env python
"""
Drive the Spain-wide parcelated EVI export on upf.

Splits the totality band into longitude strips (each ~1.5 deg wide, lat window hugged to
the band), then for each strip shells out to export_cogs.py --mode evi (resumable: skips
parcels whose tif already exists). The shared DEM is exported once over the union bbox.
Finally aggregates every parcel's subsample into fixed global colour domains + an evi_norm
reference and writes stats.json with the parcel manifest the browser routes clicks through.

The ray-march is memory-bandwidth-bound and flat past ~4 workers, so parcels run one at a
time on 8 workers (more is wasted). Expect ~3-4 min/strip, ~40 min total.

Run:
    conda run -n geography python eclipse/web_viz/run_parcels.py [--workers 8]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import rasterio

HERE = os.path.dirname(os.path.abspath(__file__))
ECLIPSE = os.path.dirname(HERE)
sys.path.insert(0, ECLIPSE)

import besselian as B  # noqa: E402
import path as P  # noqa: E402
import score as S  # noqa: E402

from cogspec import DATA_VERSION, DEM_TIF, EVI_NAMES, EVI_SCALE, I16_NODATA  # noqa: E402

OUT = os.path.join(HERE, "data")
BBOX = (-10.0, 36.5, 5.6, 44.6)      # coarse grid the parcel strips are derived from
STEP = 0.02
STRIDE_LON = 1.5                      # parcel width in degrees
LAT_PAD = 0.30                        # how far past the band edge each strip reaches
HALO = 0.35                           # fetch halo (deg) so edge rays complete
from export_cogs import AZIMUTHS, BANDS, WEB_IDX  # noqa: E402


def parcels():
    """Longitude strips hugged to the band's lat extent."""
    e = B.Elements()
    lons, lats, mag, _ = P.magnitude_grid(e, BBOX, step=STEP)
    inband = mag >= 1.0
    out = []
    a = BBOX[0]
    while a < BBOX[2] - 1e-9:
        b = min(a + STRIDE_LON, BBOX[2])
        col = (lons >= a) & (lons < b)
        rows = np.where(inband[:, col].any(axis=1))[0]
        if rows.size:
            s = max(BBOX[1], float(lats[rows.min()]) - LAT_PAD)
            n = min(BBOX[3], float(lats[rows.max()]) + LAT_PAD)
            out.append((round(a, 2), round(s, 2), round(b, 2), round(n, 2)))
        a = b
    return out


def sh(cmd):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    os.makedirs(OUT, exist_ok=True)

    # 0. the totality-band mask polygon
    sh([sys.executable, os.path.join(HERE, "make_band.py")])
    mask = os.path.join(OUT, "band.geojson")

    pcs = parcels()
    print(f"{len(pcs)} parcels")
    for i, p in enumerate(pcs):
        print(f"  p{i:02d} {p}")

    # 1. shared DEM over the union bbox, masked to the band
    w = min(p[0] for p in pcs); s = min(p[1] for p in pcs)
    e = max(p[2] for p in pcs); n = max(p[3] for p in pcs)
    dem_bbox = (w - LAT_PAD, s - LAT_PAD, e + LAT_PAD, n + LAT_PAD)
    sh([sys.executable, os.path.join(HERE, "export_cogs.py"), "--mode", "dem",
        "--bbox", *map(str, dem_bbox), "--out-res", "100", "--name", "spain",
        "--mask", mask])

    # 2. one EVI parcel at a time (resumable)
    for i, p in enumerate(pcs):
        name = f"p{i:02d}"
        if os.path.exists(os.path.join(OUT, f"evi.{name}.v{DATA_VERSION}.tif")):
            print(f"skip {name} (exists)")
            continue
        sh([sys.executable, os.path.join(HERE, "export_cogs.py"), "--mode", "evi",
            "--bbox", *map(str, p), "--name", name, "--mask", mask,
            "--halo", str(HALO), "--workers", str(workers)])

    # 3. aggregate subsamples -> global domains + evi_ref
    stacks, evis = [], []
    for i in range(len(pcs)):
        z = np.load(os.path.join(OUT, f"subsample.p{i:02d}.npz"))
        stacks.append(z["stack"]); evis.append(z["evi"])
    big = np.concatenate([s.reshape(s.shape[0], -1) for s in stacks], axis=1)
    evibig = np.concatenate([e.ravel() for e in evis])
    dom = {}
    for i, (name, _) in enumerate(BANDS):     # 6-band stack order; p1/p99 over all Spain
        v = big[i][np.isfinite(big[i])]
        if v.size:
            dom[name] = [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
    dom["horizon_deg"][0] = 0.0
    dom["duration_s"] = [0.0, float(np.nanmax(big[WEB_IDX[3]]))]
    evi_ref = float(np.nanmax(evibig[np.isfinite(evibig)])) or 1.0

    # 4. manifest: bounds/shape straight from each written tif
    plist = []
    for i, p in enumerate(pcs):
        f = os.path.join(OUT, f"evi.p{i:02d}.v{DATA_VERSION}.tif")
        with rasterio.open(f) as d:
            plist.append({"name": f"p{i:02d}", "file": os.path.basename(f),
                          "bbox_4326": list(p),
                          "bbox_3857": [float(v) for v in d.bounds],
                          "shape": [d.height, d.width]})
    with rasterio.open(os.path.join(OUT, DEM_TIF)) as d:
        dem_3857 = [float(v) for v in d.bounds]

    E = B.Elements()
    stats = {"domains": dom, "out_res_m": 50.0, "azimuths": list(AZIMUTHS),
             "bbox_4326": list(dem_bbox), "bbox_3857": dem_3857,
             "sun": S.sun_fit(tuple(dem_bbox), elements=E), "cartoon": S.CARTOON,
             "files": {"dem": DEM_TIF}, "parcels": plist,
             "evi_bands": EVI_NAMES, "evi_scale": EVI_SCALE,
             "evi_ref": evi_ref, "int16_nodata": I16_NODATA}
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nwrote stats.json: {len(plist)} parcels, evi_ref {evi_ref:.3f}")
    print("domains:")
    for k, (a, b) in dom.items():
        print(f"  {k:14s} {a:8.2f} .. {b:8.2f}")


if __name__ == "__main__":
    main()
