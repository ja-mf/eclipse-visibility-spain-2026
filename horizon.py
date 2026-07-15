"""
Terrain horizon angle along a fixed azimuth, for a whole DEM at once.

For the 2026-08-12 eclipse the sun sits at ~2-6 deg altitude and its azimuth is
~285-287 deg everywhere in the AOI. So we never need a full 360 deg viewshed --
just the horizon elevation angle along one (or a few) azimuths. That is a single
running maximum along a ray, which we can do as a sequence of whole-array shifts.

Two physical corrections matter at a 5 deg sun:

  * curvature + refraction. Folded into an effective earth radius,
    R_eff = R / (1 - k) with k = 0.13, so an occluder at distance d is lowered by
    d^2 / (2 R_eff). That is 6.8 m at 10 km, 61 m at 30 km.

  * the near field. GLO-30 is a *surface* model: it contains trees, buildings and
    DEM noise. Without a minimum ray distance the horizon is set by whatever pixel
    you happen to stand next to -- a 5 m bump 30 m away subtends 9 deg and would
    veto the site. A real observer takes two steps to the side. `d_min` (default
    150 m) skips that, and `eye_h` puts the observer's eyes above the ground.

Occluders beyond ~20 km cannot reach a 5 deg sun unless they rise ~2 km above the
observer, so `d_max` of 30 km is generous. See REFERENCE.md.
"""
from __future__ import annotations

import os
from multiprocessing import get_context

import numpy as np

R_EARTH = 6371000.0
K_REFRACT = 0.13
R_EFF = R_EARTH / (1.0 - K_REFRACT)


def step_schedule(d_min=150.0, d_max=30000.0, px=30.0):
    """
    Ray sample distances: dense near the observer, sparse far away.

    Angular resolution is what matters, and it degrades as 1/d, so we can afford
    coarser sampling far out. This keeps the step count ~400 instead of ~1200.
    """
    segs = [(d_min, 3000.0, px), (3000.0, 10000.0, 2 * px), (10000.0, d_max, 4 * px)]
    d = np.concatenate([np.arange(a, b, s) for a, b, s in segs if a < b])
    return d[(d >= d_min) & (d <= d_max)]


def _shift(z, di, dj):
    """Shift array by (di rows, dj cols) with edge clamping. Returns a view-like copy."""
    out = np.empty_like(z)
    ni, nj = z.shape
    si = slice(max(0, di), min(ni, ni + di))
    sj = slice(max(0, dj), min(nj, nj + dj))
    ti = slice(max(0, -di), min(ni, ni - di))
    tj = slice(max(0, -dj), min(nj, nj - dj))
    out[ti, tj] = z[si, sj]
    # clamp edges by replicating the border of the valid region
    if di > 0:
        out[ni - di:, :] = out[ni - di - 1:ni - di, :]
    elif di < 0:
        out[:-di, :] = out[-di:-di + 1, :]
    if dj > 0:
        out[:, nj - dj:] = out[:, nj - dj - 1:nj - dj]
    elif dj < 0:
        out[:, :-dj] = out[:, -dj:-dj + 1]
    return out


def _shift_valid(m, di, dj):
    """Shift a bool mask; anything shifted in from outside the array becomes False."""
    out = np.zeros_like(m)
    ni, nj = m.shape
    si = slice(max(0, di), min(ni, ni + di))
    sj = slice(max(0, dj), min(nj, nj + dj))
    ti = slice(max(0, -di), min(ni, ni - di))
    tj = slice(max(0, -dj), min(nj, nj - dj))
    out[ti, tj] = m[si, sj]
    return out


def horizon_raster(z, px, py, azimuth_deg, d_min=150.0, d_max=30000.0, eye_h=1.7,
                   valid=None, progress=False):
    """
    Horizon elevation angle (degrees) along `azimuth_deg` for every pixel of `z`.

    z     : 2-D elevation array (metres), row 0 = north, in a *metric* CRS.
    px    : pixel size in x (metres, positive, east-increasing).
    py    : pixel size in y (metres, negative if row index increases southward).
    valid : optional bool mask of real (non-nodata) elevation.

    Returns (horizon_deg, dist_km, ray_ok).

    `ray_ok` is False wherever the ray left the data -- either off the array or into
    nodata. Those pixels had their samples edge-clamped, so their horizon is
    UNDER-estimated and their clearance is fictitious; they otherwise win any ranking
    outright. Computing it here is exact by construction (we already visit every offset)
    and costs one bool array, versus guessing with a geometric margin, which leaks.
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    z0 = z + np.float32(eye_h)
    if valid is None:
        valid = np.ones(z.shape, dtype=bool)

    ax = np.sin(np.radians(azimuth_deg))    # east component
    ay = np.cos(np.radians(azimuth_deg))    # north component

    # We race ~400 ray samples to find the LARGEST elevation angle. tan is strictly
    # increasing on (-90, 90), so the sample that maximises the angle is exactly the sample
    # that maximises its tangent, dz/dist -- and a divide is an order of magnitude cheaper
    # than an arctan2 over 181 Mpx. Race the tangents, convert ONCE at the end. Same answer,
    # ~400 fewer transcendental passes over the whole DEM; this was over half the runtime.
    best_t = np.full(z.shape, -1e30, dtype=np.float32)      # tan of the horizon angle so far
    bestd = np.zeros(z.shape, dtype=np.float32)
    ray_ok = valid.copy()

    seen = set()
    for d in step_schedule(d_min, d_max, abs(px)):
        dj = int(round(d * ax / px))          # column offset (east)
        di = int(round(d * ay / py))          # row offset (py is negative -> north is -row)
        if (di, dj) in seen or (di == 0 and dj == 0):
            continue
        seen.add((di, dj))

        dist = float(np.hypot(dj * px, di * py))
        if dist < d_min:
            continue
        drop = np.float32(dist * dist / (2.0 * R_EFF))

        t = _shift(z, di, dj)                 # fresh array -- safe to mutate in place
        t -= z0
        t -= drop
        t /= np.float32(dist)                 # tangent of the elevation angle

        upd = t > best_t
        np.copyto(best_t, t, where=upd)       # `where=` beats boolean fancy-indexing: it
        np.copyto(bestd, np.float32(dist), where=upd)   # builds no index array
        ray_ok &= _shift_valid(valid, di, dj)

    best = np.degrees(np.arctan(best_t)).astype(np.float32)   # the only transcendental pass

    if progress:
        print(f"horizon: {len(seen)} ray steps, az={azimuth_deg}, "
              f"d={d_min:.0f}-{d_max:.0f} m, "
              f"{100*ray_ok.mean():.1f}% of pixels have a complete ray")
    return best, bestd / 1000.0, ray_ok


# --------------------------------------------------------------------------------------
# Parallel: split into ROW stripes.
#
# The rays only ever go WNW (az 283-287 deg), i.e. ~29 km west and ~8 km north over the
# 30 km search. That asymmetry decides the decomposition:
#
#   * split by COLUMNS and every stripe needs a 29 km (1120 px) halo to its west -- on a
#     1200 px-wide stripe that is ~90% redundant work. Pointless.
#   * split by ROWS and every stripe needs only an 8 km (~340 px) halo to its north, and no
#     halo at all east/west because we never cut in that direction. On a 1030-row stripe
#     that is ~30% redundant work, and it parallelises 12 ways.
#
# With a halo >= the largest northward row offset, every output pixel's entire ray lies
# inside its padded stripe, so the result is EXACT -- not an approximation, and not merely
# close: the assertion in the __main__ block below checks it is bit-identical to the serial
# version.
# --------------------------------------------------------------------------------------
_SHARED = {}          # populated pre-fork; children inherit it copy-on-write

# The whole design depends on FORK. macOS defaults to 'spawn', which would (a) re-import the
# parent module in every child and (b) pickle the entire 725 MB DEM down a pipe, once per
# worker -- which costs more than the ray-march it is trying to parallelise. Ask for fork
# explicitly; if the platform will not give it, stay serial rather than quietly do that.
try:
    _CTX = get_context("fork")
except ValueError:                                        # pragma: no cover
    _CTX = None


def _halo_rows(azimuth_deg, d_max, py):
    """Rows of padding a stripe needs, and on which side. Positive = north (smaller row)."""
    ay = float(np.cos(np.radians(azimuth_deg)))
    n = int(np.ceil(d_max * abs(ay) / abs(py))) + 2       # +2 for the int() rounding of di
    return (n, 0) if ay > 0 else (0, n)                   # (north, south)


def _stripe(job):
    r0, r1, p0, p1, kw = job
    z = _SHARED["z"][p0:p1]
    v = _SHARED["valid"][p0:p1]
    h, d, ok = horizon_raster(z, valid=v, **kw)
    o = r0 - p0                                            # drop the halo before returning
    n = r1 - r0
    return r0, h[o:o + n], d[o:o + n], ok[o:o + n]


def horizon_raster_mp(z, px, py, azimuth_deg, d_min=150.0, d_max=30000.0, eye_h=1.7,
                      valid=None, workers=None, progress=False):
    """horizon_raster(), fanned out over row stripes. Same arguments, same answer."""
    workers = workers or os.cpu_count() or 1
    nrows = z.shape[0]
    hn, hs = _halo_rows(azimuth_deg, d_max, py)
    halo = hn + hs
    # Each stripe must be comfortably taller than its own halo, or we burn more on redundant
    # rows than we save on cores. Cap the worker count rather than silently doing that.
    workers = max(1, min(workers, nrows // max(1, 2 * halo)))
    if workers < 2 or _CTX is None:
        return horizon_raster(z, px, py, azimuth_deg, d_min, d_max, eye_h, valid, progress)

    if valid is None:
        valid = np.ones(z.shape, dtype=bool)
    _SHARED["z"] = np.ascontiguousarray(z, dtype=np.float32)
    _SHARED["valid"] = valid

    kw = dict(px=px, py=py, azimuth_deg=azimuth_deg, d_min=d_min, d_max=d_max, eye_h=eye_h)
    edges = np.linspace(0, nrows, workers + 1).round().astype(int)
    jobs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            jobs.append((int(a), int(b),
                         int(max(0, a - hn)), int(min(nrows, b + hs)), kw))

    best = np.empty(z.shape, np.float32)
    dist = np.empty(z.shape, np.float32)
    rok = np.empty(z.shape, bool)
    # fork: the children inherit _SHARED without copying it (COW). See _CTX above.
    with _CTX.Pool(processes=len(jobs)) as pool:
        for r0, h, d, ok in pool.imap_unordered(_stripe, jobs):
            n = h.shape[0]
            best[r0:r0 + n] = h
            dist[r0:r0 + n] = d
            rok[r0:r0 + n] = ok
    _SHARED.clear()

    if progress:
        print(f"horizon(mp): az={azimuth_deg}, {len(jobs)} row stripes, "
              f"halo {hn + hs} rows ({100 * (hn + hs) * len(jobs) / nrows:.0f}% redundant), "
              f"{100 * rok.mean():.1f}% of pixels have a complete ray")
    return best, dist, rok
