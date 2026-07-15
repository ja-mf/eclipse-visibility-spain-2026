"""
Path of totality: central line, north/south limits, duration contours.

Derived from our own Besselian elements, so the path is guaranteed consistent with the
heatmap. (The IEEC band sits ~1 km north of ours -- see REFERENCE.md 6b. Keeping both on
the map makes that difference visible instead of hidden.)

Central line: at time t the shadow axis pierces the ellipsoid where the observer's
fundamental-plane coords (xi, eta) equal the axis coords (x, y). We seed with the closed-form
spherical solution and then Newton-refine against the exact `observer_xyz`, which is the same
validated code the rest of the model uses -- no second implementation to keep in sync.

Limits: contour magnitude == 1 on a lat/lon grid. Same model, same answer, no extra algebra.
"""
from __future__ import annotations

import numpy as np

import besselian as B

D2R = np.pi / 180.0
R2D = 180.0 / np.pi
FLAT = B.FLAT


def central_point(t, elements: B.Elements, iters=12):
    """(lat, lon) where the shadow axis hits the ellipsoid at element-time t, or (nan, nan)."""
    el = elements.at(t)
    x, y, d, mu = float(el["x"]), float(el["y"]), float(el["d"]), float(el["mu"])

    r2 = x * x + y * y
    if r2 >= 1.0:
        return np.nan, np.nan                       # axis misses the Earth

    # --- spherical seed ---
    zeta = np.sqrt(1.0 - r2)
    sin_phi1 = y * np.cos(d) + zeta * np.sin(d)
    cos_phi1_cos_th = -y * np.sin(d) + zeta * np.cos(d)
    theta = np.arctan2(x, cos_phi1_cos_th)
    phi1 = np.arctan2(sin_phi1, np.hypot(x, cos_phi1_cos_th))
    lat = np.arctan(np.tan(phi1) / (FLAT * FLAT)) * R2D     # geocentric -> geodetic
    lon = (theta * R2D - mu * R2D)

    # --- Newton refine on the exact (flattened) observer position ---
    for _ in range(iters):
        f0 = _resid(lat, lon, t, elements, x, y)
        if not np.all(np.isfinite(f0)):
            return np.nan, np.nan
        h = 1e-4
        J = np.column_stack([
            (_resid(lat + h, lon, t, elements, x, y) - f0) / h,
            (_resid(lat, lon + h, t, elements, x, y) - f0) / h,
        ])
        try:
            step = np.linalg.solve(J, -f0)
        except np.linalg.LinAlgError:
            break
        lat += step[0]
        lon += step[1]
        if np.max(np.abs(step)) < 1e-10:
            break

    # undo the ephemeris-meridian shift that observer_xyz applied internally
    lon = ((lon + 180.0) % 360.0) - 180.0
    return float(lat), float(lon)


def _resid(lat, lon, t, elements, x, y):
    el = elements.at(t)
    xi, eta, _, _, _ = B.observer_xyz(lat, B.eph_lon(lon, elements), 0.0, el)
    return np.array([float(xi) - x, float(eta) - y])


def central_line(elements: B.Elements, t0=-2.0, t1=2.0, step=1.0 / 60.0):
    """Central line as (lat, lon, ut_hours, duration_s) samples."""
    out = []
    for t in np.arange(t0, t1 + 1e-9, step):
        la, lo = central_point(float(t), elements)
        if not np.isfinite(la):
            continue
        c = B.contacts(la, lo, 0.0, elements, t_guess=float(t))
        if not bool(c["is_total"]):
            continue
        out.append((la, lo, float(elements.ut_hours(t)), float(c["duration_s"])))
    return out


def magnitude_grid(elements: B.Elements, bbox, step=0.05):
    """Max eclipse magnitude and totality duration on a lat/lon grid."""
    w, s, e, n = bbox
    lons = np.arange(w, e + step, step)
    lats = np.arange(s, n + step, step)
    LO, LA = np.meshgrid(lons, lats)
    c = B.contacts(LA, LO, 0.0, elements)
    mag = np.asarray(c["mag_max"], dtype=float)
    dur = np.where(c["is_total"], c["duration_s"], 0.0)
    return lons, lats, mag, np.asarray(dur, dtype=float)


def contours(lons, lats, field, levels):
    """Contour a lat/lon field -> {level: [ [[lon,lat],...], ... ]}."""
    from contourpy import contour_generator

    cg = contour_generator(x=lons, y=lats, z=field)
    out = {}
    for lv in levels:
        lines = [np.asarray(ln) for ln in cg.lines(lv) if len(ln) >= 2]
        out[lv] = [ln.tolist() for ln in lines]
    return out


def to_geojson(elements: B.Elements, bbox=(-11.0, 35.0, 6.0, 46.0), step=0.05):
    """FeatureCollection: north/south limits, central line, duration contours."""
    lons, lats, mag, dur = magnitude_grid(elements, bbox, step)

    feats = []

    # limits of totality = the magnitude==1 contour
    for line in contours(lons, lats, mag, [1.0])[1.0]:
        feats.append({
            "type": "Feature",
            "properties": {"kind": "limit", "name": "limit of totality"},
            "geometry": {"type": "LineString", "coordinates": line},
        })

    # duration contours inside the path
    for lv in (30.0, 60.0, 90.0, 120.0):
        for line in contours(lons, lats, dur, [lv])[lv]:
            feats.append({
                "type": "Feature",
                "properties": {"kind": "duration", "seconds": lv},
                "geometry": {"type": "LineString", "coordinates": line},
            })

    # central line
    cl = central_line(elements)
    if cl:
        feats.append({
            "type": "Feature",
            "properties": {"kind": "centreline", "name": "central line"},
            "geometry": {"type": "LineString",
                         "coordinates": [[lo, la] for la, lo, _, _ in cl]},
        })
        for la, lo, ut, d in cl:
            if abs((ut * 60) % 5) < 0.35:          # a tick every ~5 min of UT
                feats.append({
                    "type": "Feature",
                    "properties": {"kind": "centreline_tick",
                                   "ut": B.hms(ut), "duration_s": round(d)},
                    "geometry": {"type": "Point", "coordinates": [lo, la]},
                })

    return {"type": "FeatureCollection", "features": feats}
