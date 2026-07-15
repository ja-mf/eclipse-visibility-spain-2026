#!/usr/bin/env python
"""
Export the model's *components* as web-ready COGs.

We ship the components, not a baked EVI heatmap: the browser recombines them per-pixel
(maplibre-cog-protocol's setColorFunction), so the EVI weights are sliders rather than
constants frozen into a raster. See ../PIPELINE.md.

Two hard requirements of maplibre-cog-protocol:
  * output must be EPSG:3857 -- it does not warp
  * COG with internal overviews, so the browser range-fetches only the zoom it needs

The horizon is ray-marched at the DEM's native resolution (accuracy) and only then
downsampled to `--out-res` for the web (file size). Do not downsample first.

Three modes:
  full  -- one bbox: evi + dem + stats.json (the laptop/AOI path)
  evi   -- one parcel of the Spain grid: writes evi.<name>.v<VER>.tif + a subsample for
           global domain aggregation. No dem, no stats.
  dem   -- the single shared DEM over a bbox, no ray-march.

In evi/dem modes `--mask` clips output to a geojson polygon (the totality band): outside
the path there is no eclipse and the non-band land would bloat the file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr
from rasterio import features as rfeatures
from rasterio.warp import transform_geom

HERE = os.path.dirname(os.path.abspath(__file__))
ECLIPSE = os.path.dirname(HERE)
sys.path.insert(0, ECLIPSE)

import besselian as B  # noqa: E402
import horizon as H  # noqa: E402
import score as S  # noqa: E402

from cogspec import (COG_I16, DATA_VERSION, DEM_TIF, EVI_NAMES, EVI_SCALE,  # noqa: E402
                     I16_NODATA, quantize)

OUT = os.path.join(HERE, "data")
DEMDIR = os.path.join(ECLIPSE, "dem")

# Ebre delta -> Tarragona -> Teruel -> Valencia, plus a 30 km buffer W and N for the
# horizon search (occluders are WNW; 30 km is provably enough -- REFERENCE.md 3).
DEFAULT_BBOX = (-2.40, 39.30, 1.95, 42.05)

# Sun azimuth spans ~284-287 across the AOI. Ray-march three azimuths and interpolate to
# each pixel's own azimuth; a single-azimuth run leaves a ~1 deg systematic.
AZIMUTHS = (283.0, 285.0, 287.0)

# The internal stack. elev_m is computed and used for the colour domains and by
# export_candidates.py, but it is NOT written into evi.tif -- it is the DEM, which we
# already ship for the hillshade, and shipping it twice cost ~56 MB of egress for nothing.
BANDS = [
    ("horizon_deg", "terrain horizon angle at the sun's azimuth"),
    ("sun_alt_deg", "sun altitude at mid-totality"),
    ("alt_drop_deg", "how far the sun falls between C2 and C3"),
    ("duration_s", "duration of totality (0 = outside the path)"),
    ("elev_m", "ground elevation"),
    ("occluder_km", "distance to the limiting occluder"),
]
_NAMES = [n for n, _ in BANDS]
# which of them, in which order, go into the file the browser streams (see cogspec.py)
WEB_IDX = [_NAMES.index(n) for n in EVI_NAMES]


def utm_crs_for(lon_c):
    """ETRS89 UTM zone for a central longitude. Iberia spans zones 29-32; using one zone
    for all of it adds ~1.3% horizontal scale error in Galicia, so pick per parcel."""
    zone = int(np.floor((lon_c + 180.0) / 6.0) + 1)
    return f"EPSG:258{zone}"


def get_dem(bbox, name, crs):
    """Fetch + reproject the DEM once; reuse it on later runs."""
    utm = os.path.join(DEMDIR, f"{name}_utm.tif")
    if os.path.exists(utm):
        print(f"reusing {utm}")
        return rioxarray.open_rasterio(utm).squeeze(drop=True)

    from dem_stitcher import stitch_dem
    os.makedirs(DEMDIR, exist_ok=True)
    print(f"fetching glo_30 for {bbox} into {crs} ...")
    arr, meta = stitch_dem(bounds=list(bbox), dem_name="glo_30",
                           dst_tile_dir=Path(DEMDIR) / "tiles",
                           overwrite_existing_tiles=False)
    wgs = os.path.join(DEMDIR, f"{name}_wgs84.tif")
    with rasterio.open(wgs, "w", driver="GTiff", height=meta["height"], width=meta["width"],
                       count=1, dtype=arr.dtype, crs=meta["crs"],
                       transform=meta["transform"], nodata=np.nan) as dst:
        dst.write(arr, 1)
    dem = rioxarray.open_rasterio(wgs).squeeze(drop=True).rio.reproject(crs)
    dem.rio.to_raster(utm)
    return dem


def load_mask_geom(path):
    if not path:
        return None
    with open(path) as f:
        fc = json.load(f)
    return fc["features"][0]["geometry"]


def mask_array(geom, shape, transform, crs):
    """Rasterise a 4326 geojson geometry onto an output grid (bool, True = keep)."""
    g = transform_geom("EPSG:4326", crs, geom)
    return rfeatures.rasterize([g], out_shape=shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)


def compute_stack(bbox, name, workers, crs):
    """DEM -> circumstances -> horizon. Returns the physical float stack + bookkeeping."""
    t0 = time.time()
    dem = get_dem(bbox, name, crs)
    Z = dem.values.astype(np.float32)
    nodata = ~np.isfinite(Z)
    Zf = np.where(nodata, np.nanmedian(Z), Z).astype(np.float32)
    px = float(dem.x[1] - dem.x[0])
    py = float(dem.y[1] - dem.y[0])
    print(f"DEM {Z.shape} = {Z.size/1e6:.1f} Mpx @ {px:.1f} m  ({dem.rio.crs})  "
          f"elev {np.nanmin(Z):.0f}-{np.nanmax(Z):.0f} m")

    inv = pyproj.Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
    XX, YY = np.meshgrid(dem.x.values, dem.y.values)
    lon, lat = inv.transform(XX, YY)
    del XX, YY

    E = B.Elements()
    circ = S.coarse_circumstances(lat, lon, Z, E, stride=64, keys=S.SCORE_KEYS)
    del lat, lon
    az = circ["az_mid"]
    inpath = circ["is_total"]
    print(f"circumstances ({time.time()-t0:.0f}s): sun alt "
          f"{np.nanmin(circ['alt_mid']):.2f}-{np.nanmax(circ['alt_mid']):.2f}deg, "
          f"az {az[az > 0].min():.2f}-{az.max():.2f}deg, "
          f"in-path {100*inpath.mean():.0f}% of tile, "
          f"max totality {np.nanmax(circ['dur']):.0f}s")

    # Interpolate the three ray-marched azimuths to each pixel's own sun azimuth (tent
    # basis). One azimuth's rasters exist at a time -- holding all three plus their
    # distance rasters is 6 x the array and this box is not infinite.
    azc = np.clip(az, AZIMUTHS[0], AZIMUTHS[-1])
    hor = np.zeros_like(Z)
    occ = np.zeros_like(Z)
    ray_ok = ~nodata
    for k, a in enumerate(AZIMUTHS):
        t = time.time()
        h, d, rok = H.horizon_raster_mp(Zf, px, py, a, d_min=150.0, d_max=30000.0, eye_h=1.7,
                                        valid=~nodata, workers=workers, progress=True)
        ray_ok &= rok
        del rok
        lo_ = AZIMUTHS[k - 1] if k > 0 else None
        hi_ = AZIMUTHS[k + 1] if k < len(AZIMUTHS) - 1 else None
        w = np.zeros_like(Z)
        if lo_ is not None:
            m = (azc > lo_) & (azc <= a)
            w[m] = ((azc[m] - lo_) / (a - lo_))
        if hi_ is not None:
            m = (azc > a) & (azc < hi_)
            w[m] = ((hi_ - azc[m]) / (hi_ - a))
        if lo_ is None:
            w[azc <= a] = 1.0
        if hi_ is None:
            w[azc >= a] = 1.0
        hor += w * h
        occ += w * d
        del h, d, w
        print(f"  horizon az={a}: {time.time()-t:.0f}s")
    del Zf, azc

    # EVI on the native grid, for the global domain + the browser's evi_norm reference.
    evi = S.evi(circ, hor)["evi"]

    stack = np.stack([
        hor,
        circ["alt_mid"],
        circ["alt_c2"] - circ["alt_c3"],
        np.where(inpath, circ["dur"], 0.0).astype(np.float32),
        Z,
        occ,
    ]).astype(np.float32)
    del hor, occ, circ
    stack[:, nodata] = np.nan
    stack[:, ~ray_ok] = np.nan      # geometric margin: the ray left the data
    evi[~ray_ok] = np.nan
    evi[nodata] = np.nan
    dropped = int((~ray_ok).sum() - nodata.sum())
    print(f"incomplete-ray pixels dropped: {dropped/1e6:.1f} Mpx "
          f"({100*dropped/max(1, (~nodata).sum()):.1f}% of valid)")
    return dict(stack=stack, evi=evi, dem=dem, t0=t0)


def _evi_path(name):
    tag = name if name and name != "aoi" else None
    stem = f"evi.{tag}." if tag else "evi."
    return os.path.join(OUT, f"{stem}v{DATA_VERSION}.tif")


def run_evi(args, crs):
    """One parcel: ray-march a halo-padded DEM, write the masked evi COG cropped tight to
    --bbox (so parcels are non-overlapping), plus a subsample for global domains."""
    os.makedirs(OUT, exist_ok=True)
    geom = load_mask_geom(args.mask)
    w, s, e, n = args.bbox
    fb = (w - args.halo, s - args.halo, e + args.halo, n + args.halo)  # fetch with halo
    out = compute_stack(fb, args.name, args.workers, crs)
    stack, evi, dem = out["stack"], out["evi"], out["dem"]

    da = xr.DataArray(stack, dims=("band", "y", "x"),
                      coords={"band": np.arange(1, len(BANDS) + 1), "y": dem.y, "x": dem.x})
    da = da.rio.write_crs(dem.rio.crs).rio.write_nodata(np.nan)
    web = da.rio.reproject("EPSG:3857", resolution=args.out_res)
    eviw = xr.DataArray(evi[None], dims=("band", "y", "x"),
                        coords={"band": [1], "y": dem.y, "x": dem.x}).rio.write_crs(
        dem.rio.crs).rio.write_nodata(np.nan).rio.reproject("EPSG:3857", resolution=args.out_res)
    # crop both to the true parcel bbox (drop the halo from the output)
    web = web.rio.clip_box(w, s, e, n, crs="EPSG:4326")
    eviw = eviw.rio.clip_box(w, s, e, n, crs="EPSG:4326")

    if geom is not None:
        keep = mask_array(geom, (web.rio.height, web.rio.width), web.rio.transform(),
                          web.rio.crs)
        vals = web.values.copy()
        vals[:, ~keep] = np.nan
        web = xr.DataArray(vals, dims=web.dims, coords=web.coords).rio.write_crs(
            web.rio.crs).rio.write_nodata(np.nan)

    q = np.stack([quantize(web.values[i], s) for i, s in zip(WEB_IDX, EVI_SCALE)])
    qa = xr.DataArray(q, dims=("band", "y", "x"),
                      coords={"band": np.arange(1, len(EVI_NAMES) + 1),
                              "y": web.y, "x": web.x})
    qa = qa.rio.write_crs(web.rio.crs).rio.write_nodata(I16_NODATA)
    qa.attrs["long_name"] = tuple(EVI_NAMES)
    comp = _evi_path(args.name)
    qa.rio.to_raster(comp, **COG_I16)
    print(f"wrote {comp}  {qa.shape}  ({os.path.getsize(comp)/1e6:.0f} MB)")

    # subsample the cropped web grid for global domains + evi_norm ref.
    np.savez_compressed(os.path.join(OUT, f"subsample.{args.name}.npz"),
                        stack=web.values[:, ::8, ::8], evi=eviw.values[0, ::8, ::8])
    print(f"done ({time.time()-out['t0']:.0f}s)")


def run_dem(args):
    """The single shared DEM over a (large) bbox, masked to the band cutline.

    Streamed through gdalwarp from a VRT of the cached GLO-30 tiles. Materialising the
    union DEM (a 1.7 Gpx array) in rioxarray and reprojecting it OOMs; gdalwarp warps
    block-wise with bounded RAM. GLO-30 already ships int16 metres with ocean as nodata,
    so we keep int16 and normalise the nodata to I16_NODATA.
    """
    import glob
    import subprocess

    os.makedirs(OUT, exist_ok=True)
    tile_dir = Path(DEMDIR) / "tiles"
    tiles = sorted(glob.glob(str(tile_dir / "*.tif")))
    if len(tiles) < 20:                       # cache miss: fetch tiles for this bbox once
        from dem_stitcher import stitch_dem
        os.makedirs(DEMDIR, exist_ok=True)
        print(f"fetching glo_30 tiles for {tuple(args.bbox)} ...")
        stitch_dem(bounds=list(args.bbox), dem_name="glo_30",
                   dst_tile_dir=tile_dir, overwrite_existing_tiles=False)
        tiles = sorted(glob.glob(str(tile_dir / "*.tif")))

    vrt = os.path.join(DEMDIR, f"{args.name}.vrt")
    subprocess.run(["gdalbuildvrt", vrt, *tiles], check=True)
    demc = os.path.join(OUT, DEM_TIF)
    cut = ["-cutline", args.mask, "-crop_to_cutline"] if args.mask else []
    subprocess.run(["gdalwarp", "-t_srs", "EPSG:3857",
                    "-tr", str(args.out_res), str(args.out_res), "-r", "bilinear",
                    "-of", "COG", "-co", "COMPRESS=DEFLATE", "-co", "PREDICTOR=2",
                    "-co", "BIGTIFF=IF_SAFER", "-co", "BLOCKSIZE=512",
                    "-dstnodata", str(I16_NODATA), *cut, vrt, demc], check=True)
    print(f"wrote {demc}  ({os.path.getsize(demc)/1e6:.0f} MB)  int16 metres")


def run_full(args, crs):
    """The original single-bbox export: evi + dem + stats.json."""
    os.makedirs(OUT, exist_ok=True)
    out = compute_stack(tuple(args.bbox), args.name, args.workers, crs)
    stack, evi, dem = out["stack"], out["evi"], out["dem"]

    da = xr.DataArray(stack, dims=("band", "y", "x"),
                      coords={"band": np.arange(1, len(BANDS) + 1), "y": dem.y, "x": dem.x})
    da = da.rio.write_crs(dem.rio.crs).rio.write_nodata(np.nan)
    web = da.rio.reproject("EPSG:3857", resolution=args.out_res)

    q = np.stack([quantize(web.values[i], s) for i, s in zip(WEB_IDX, EVI_SCALE)])
    qa = xr.DataArray(q, dims=("band", "y", "x"),
                      coords={"band": np.arange(1, len(EVI_NAMES) + 1),
                              "y": web.y, "x": web.x})
    qa = qa.rio.write_crs(web.rio.crs).rio.write_nodata(I16_NODATA)
    qa.attrs["long_name"] = tuple(EVI_NAMES)
    qa.rio.to_raster(os.path.join(OUT, f"evi.v{DATA_VERSION}.tif"), **COG_I16)

    dweb = dem.rio.write_nodata(np.nan).rio.reproject("EPSG:3857", resolution=args.out_res)
    di = xr.DataArray(quantize(dweb.values, 1.0), dims=("y", "x"),
                      coords={"y": dweb.y, "x": dweb.x})
    di = di.rio.write_crs(dweb.rio.crs).rio.write_nodata(I16_NODATA)
    di.rio.to_raster(os.path.join(OUT, DEM_TIF), **COG_I16)

    sub = stack[:, ::8, ::8]
    dom = {}
    for i, (name, _) in enumerate(BANDS):
        v = sub[i][np.isfinite(sub[i])]
        if v.size:
            dom[name] = [float(np.percentile(v, 1)), float(np.percentile(v, 99))]
    dom["duration_s"] = [0.0, float(np.nanmax(sub[3]))]
    dom["horizon_deg"] = [0.0, float(np.percentile(sub[0][np.isfinite(sub[0])], 99))]
    E = B.Elements()
    stats = {"domains": dom,
             "bbox_3857": [float(v) for v in web.rio.bounds()],
             "bbox_4326": [float(v) for v in args.bbox],
             "out_res_m": args.out_res,
             "azimuths": list(AZIMUTHS),
             "sun": S.sun_fit(tuple(args.bbox), elements=E),
             "cartoon": S.CARTOON,
             "files": {"evi": f"evi.v{DATA_VERSION}.tif", "dem": DEM_TIF},
             "evi_bands": EVI_NAMES,
             "evi_scale": EVI_SCALE,
             "evi_ref": float(np.nanmax(evi)),
             "int16_nodata": I16_NODATA}
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote evi+dem+stats  ({time.time()-out['t0']:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="full", choices=["full", "evi", "dem"])
    ap.add_argument("--bbox", nargs=4, type=float, default=list(DEFAULT_BBOX),
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--out-res", type=float, default=50.0)
    ap.add_argument("--name", default="aoi")
    ap.add_argument("--workers", type=int, default=None,
                    help="ray-march processes (default: all cores). 1 = serial.")
    ap.add_argument("--mask", default=None, help="geojson polygon to clip evi/dem output to")
    ap.add_argument("--halo", type=float, default=0.0,
                    help="evi mode: pad the fetch bbox by this many deg so edge rays complete")
    args = ap.parse_args()

    lon_c = 0.5 * (args.bbox[0] + args.bbox[2])
    crs = utm_crs_for(lon_c)
    if args.mode == "evi":
        run_evi(args, crs)
    elif args.mode == "dem":
        run_dem(args)
    else:
        run_full(args, crs)


if __name__ == "__main__":
    main()
