"""
Eclipse visibility scoring: combine eclipse geometry with the terrain horizon.

The model, per pixel:

    f_vis   fraction of the C2..C3 totality window during which the sun sits clear
            of the terrain horizon. This is the HARD GATE -- it is 0 if a ridge
            eats the sun, and it is the entire point of the project.
    dur     seconds of totality (0 outside the path).
    alt     sun altitude at mid-totality (deg). Drives airmass, hence how much
            atmosphere the corona has to shine through.
    clear   alt - horizon, the margin above the ridge (deg).

    EVI = f_vis * (dur/DUR_REF)^0.5 * Q_alt(alt) * Q_margin(clear)

f_vis is computed exactly rather than by sampling: over the ~60-90 s of totality
the sun's altitude falls monotonically and very nearly linearly, so the visible
fraction is just where the descending line alt(t) crosses the (constant) horizon.
"""
from __future__ import annotations

import numpy as np

import besselian as B
import horizon as H

DUR_REF = 120.0        # s, roughly the best totality anywhere on this eclipse's track
DISK_R = 0.27          # deg, solar/lunar angular radius -- clearing the *whole* disk


#: everything coarse_circumstances *can* upsample. Ask for only what you need --
#: each full-res band is 0.6 GB at 155 Mpx, so the default set is not free.
ALL_KEYS = ("dur", "mag", "alt_c2", "alt_c3", "alt_mid", "az_mid", "t_c2", "t_c3")

#: what the scoring model actually consumes
SCORE_KEYS = ("dur", "alt_c2", "alt_c3", "alt_mid", "az_mid")


def coarse_circumstances(lat2d, lon2d, z2d, elements, stride=64, keys=ALL_KEYS):
    """
    Eclipse circumstances on a strided subgrid, bilinearly upsampled to full res.

    C2/C3/alt/az vary over hundreds of km; a 64-px (~1.6 km) grid resolves them far
    better than we need, and it turns a 155 M-point contact solve into a 40 k one.
    Elevation *does* enter the contact solve, so we pass the real z at each node.

    Output is **float32**. `contacts` works in float64, and upsampling all 8 bands at
    f64 costs ~10 GB at 155 Mpx -- enough to swap-kill a 16 GB machine. Pass `keys` to
    upsample only the bands you will actually read.
    """
    from scipy.ndimage import zoom

    sl = (slice(None, None, stride), slice(None, None, stride))
    la, lo, zz = lat2d[sl], lon2d[sl], z2d[sl]
    zz = np.nan_to_num(zz, nan=float(np.nanmedian(z2d)))

    c = B.contacts(la, lo, zz, elements)
    tot = c["is_total"]

    # Outside the path C2/C3 are undefined. Fall back to t_max there so alt/az stay
    # finite and the bilinear upsample is not poisoned by holes -- `dur` and
    # `is_total` carry the "no totality here" information instead.
    t2 = np.where(tot, c["C2"], c["t_max"])
    t3 = np.where(tot, c["C3"], c["t_max"])
    alt_c2, _ = B.sun_altaz(la, lo, t2, elements)
    alt_c3, az_m = B.sun_altaz(la, lo, t3, elements)
    alt_m, _ = B.sun_altaz(la, lo, c["t_max"], elements)
    dur = np.where(tot, c["duration_s"], 0.0)

    src = dict(dur=dur, mag=c["mag_max"], alt_c2=alt_c2, alt_c3=alt_c3,
               alt_mid=alt_m, az_mid=az_m,
               t_c2=np.where(tot, c["C2_ut"], np.nan),
               t_c3=np.where(tot, c["C3_ut"], np.nan))

    out = {}
    fac = [lat2d.shape[i] / la.shape[i] for i in (0, 1)]
    crop = (slice(0, lat2d.shape[0]), slice(0, lat2d.shape[1]))
    for k in keys:
        v = np.nan_to_num(src[k], nan=0.0).astype(np.float32)
        out[k] = zoom(v, fac, order=1)[crop]
    out["is_total"] = zoom(tot.astype(np.uint8), fac, order=0)[crop].astype(bool)
    return out


def visible_fraction(alt_c2, alt_c3, hor, thresh=0.0):
    """
    Fraction of totality with the sun clear of the horizon by `thresh` degrees.

    alt(t) falls monotonically from alt_c2 to alt_c3, so this is a line crossing.
    thresh=0      -> sun's centre clear of the ridge
    thresh=DISK_R -> the whole disk clear
    """
    need = hor + thresh
    span = alt_c2 - alt_c3                       # > 0, the sun is setting
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (alt_c2 - need) / np.where(span > 1e-9, span, np.nan)
    f = np.where(span > 1e-9, f, (alt_c2 > need).astype(float))
    return np.clip(f, 0.0, 1.0)


def airmass(alt_deg):
    """Kasten-Young relative airmass. ~11 at 5 deg, ~26 at 2 deg, ~1 at zenith."""
    a = np.clip(alt_deg, 0.05, 90.0)
    return 1.0 / (np.sin(np.radians(a)) + 0.50572 * (a + 6.07995) ** -1.6364)


def q_alt(alt_deg, tau=0.20):
    """
    Corona brightness relative to a zenith sun, from atmospheric extinction:
    I / I0 = exp(-tau * X). At 2 deg (X~26) you keep ~1 % of the light; at 6 deg
    (X~9.3) ~16 %. Normalised so that the best altitude in the AOI scores ~1.
    """
    return np.exp(-tau * airmass(alt_deg))


def q_margin(clear_deg, soft=1.5):
    """
    Reward clear sky *above* the ridge. A sun scraping the skyline is technically
    visible but sits in haze, dust and whatever the DEM does not know about (trees,
    buildings, a parked lorry). Saturates once you have `soft` degrees of margin.
    """
    return np.clip(clear_deg / soft, 0.0, 1.0)


def evi(circ, hor, thresh=DISK_R):
    """Eclipse Visibility Index in [0, 1], plus the component rasters."""
    clear = circ["alt_mid"] - hor
    f = visible_fraction(circ["alt_c2"], circ["alt_c3"], hor, thresh)
    qa = q_alt(circ["alt_mid"])
    qm = q_margin(clear)
    dur = np.clip(circ["dur"], 0.0, None)

    score = f * np.sqrt(dur / DUR_REF) * qa * qm
    score = np.where(circ["is_total"], score, 0.0)

    # q_alt is an absolute transmission (~0.19 at a 6 deg sun), so raw EVI never
    # approaches 1. Keep the physical value and expose a rank-friendly normalised
    # twin for the heatmap.
    hi = np.nanmax(score) if np.isfinite(score).any() else 1.0
    return dict(evi=score, evi_norm=score / hi if hi > 0 else score,
                f_vis=f, clear=clear, q_alt=qa, q_margin=qm,
                airmass=airmass(circ["alt_mid"]), dur=dur, horizon=hor)


def sun_fit(bbox, elements=None, nlat=31, nlon=41):
    """
    Quadratic surface fits for mid-totality UT and the sun's azimuth over the AOI.

    The browser needs two things at an arbitrary click that are not in the COG: *when*
    totality happens there (to aim PeakFinder's sun) and *which way* to look (to aim
    Street View). Neither deserves a raster band -- both are smooth to the point of being
    boring across the AOI, so a degree-2 fit in (lon, lat) reproduces mid-totality to
    ~0.01 s and the azimuth to ~0.01 deg using six numbers each.

    Do NOT replace this with a constant. Mid-totality moves 5.5 minutes across the AOI,
    which is 1.4 deg of solar azimuth -- nearly three solar diameters, i.e. exactly the
    error that would make "is that ridge in the way?" unanswerable.

    Height is ignored: 2000 m of elevation shifts mid-totality by 0.7 s.
    """
    E = elements or B.Elements()
    w, s, e, n = bbox
    lon0, lat0 = 0.5 * (w + e), 0.5 * (s + n)
    LO, LA = np.meshgrid(np.linspace(w, e, nlon), np.linspace(s, n, nlat))
    c = B.contacts(LA, LO, np.zeros_like(LA), E)
    _, az = B.sun_altaz(LA, LO, c["t_max"], E)

    dx, dy = (LO - lon0).ravel(), (LA - lat0).ravel()
    A = np.stack([np.ones_like(dx), dx, dy, dx * dx, dx * dy, dy * dy], axis=1)

    out = {"lon0": lon0, "lat0": lat0,
           "terms": ["1", "dx", "dy", "dx*dx", "dx*dy", "dy*dy"],
           "date_utc": "2026-08-12", "rms": {}}
    for key, field in (("t_max_ut_s", c["t_max_ut"] * 3600.0), ("az_mid_deg", az)):
        coef, *_ = np.linalg.lstsq(A, np.asarray(field).ravel(), rcond=None)
        out[key] = [float(v) for v in coef]
        out["rms"][key] = float(np.sqrt((((A @ coef) - np.asarray(field).ravel()) ** 2).mean()))
    return out


# Solar/lunar angular radii and the moon's apparent velocity relative to the sun, in the
# local horizontal frame (deg, deg/s). Global for this eclipse; the browser draws the
# sky cartoon from these plus each pixel's alt/drop/dur/horizon. See web_viz/index.html.
CARTOON = {"r_sun": 0.2630, "r_moon": 0.2717, "vx": -1.5627e-4, "vy": 6.0408e-5,
           "az_rate": 2.664e-3}
