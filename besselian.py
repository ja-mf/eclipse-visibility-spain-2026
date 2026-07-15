"""
Besselian local circumstances for the 2026-08-12 total solar eclipse.

Vectorised over (lat, lon, height) so it can be evaluated on a whole grid at once.
Elements from the NASA/Espenak Five Millennium Canon (see besselian_2026-08-12.json).

Reference: Explanatory Supplement to the Astronomical Almanac, ch. 8;
Meeus, "Elements of Solar Eclipses". Sign conventions follow Meeus.
"""
from __future__ import annotations

import json
import os

import numpy as np

D2R = np.pi / 180.0
R2D = 180.0 / np.pi
FLAT = 0.99664719          # b/a for the reference ellipsoid
EARTH_A = 6378.137         # km, equatorial radius

_HERE = os.path.dirname(os.path.abspath(__file__))


# The canon tabulates delta_T = 75.4 s, but that was an old extrapolation. Earth rotation
# sped up since; the current IERS-based estimate for 2026-08 is ~69.1 s (skyfield's
# `Time.delta_t`). Using the canon value puts every contact time 6.3 s off and biases the
# ephemeris-meridian longitude correction. The elements themselves are functions of TDT and
# are delta_T-independent -- only the TDT->UT conversion and the meridian offset use it.
DELTA_T_2026 = 69.1


class Elements:
    """Besselian element polynomials for one eclipse."""

    def __init__(self, path: str | None = None, delta_t: float | None = None):
        path = path or os.path.join(_HERE, "besselian_2026-08-12.json")
        with open(path) as f:
            self.raw = json.load(f)
        e = self.raw["elements"]
        self.t0 = e["t0_tdt_hour"]
        self.dt_canon = self.raw["greatest_eclipse"]["delta_t_sec"]
        self.dt = DELTA_T_2026 if delta_t is None else delta_t
        self.cx = np.array(e["x"])
        self.cy = np.array(e["y"])
        self.cd = np.array(e["d_deg"])
        self.cmu = np.array(e["mu_deg"])
        self.cl1 = np.array(e["l1"])
        self.cl2 = np.array(e["l2"])
        self.tanf1 = e["tan_f1"]
        self.tanf2 = e["tan_f2"]

    # --- polynomials and their time derivatives (per hour) ---
    @staticmethod
    def _poly(c, t):
        t = np.asarray(t, dtype=float)
        return sum(ci * t**i for i, ci in enumerate(c))

    @staticmethod
    def _dpoly(c, t):
        t = np.asarray(t, dtype=float)
        return sum(i * ci * t ** (i - 1) for i, ci in enumerate(c) if i >= 1)

    def at(self, t):
        """All elements at t (hours from t0, TDT). Angles in radians, rates per hour."""
        return dict(
            x=self._poly(self.cx, t),
            y=self._poly(self.cy, t),
            dx=self._dpoly(self.cx, t),
            dy=self._dpoly(self.cy, t),
            d=self._poly(self.cd, t) * D2R,
            dd=self._dpoly(self.cd, t) * D2R,
            mu=self._poly(self.cmu, t) * D2R,
            dmu=self._dpoly(self.cmu, t) * D2R,
            l1=self._poly(self.cl1, t),
            l2=self._poly(self.cl2, t),
        )

    def ut_hours(self, t):
        """Convert element time t -> UT decimal hours on 2026-08-12."""
        return self.t0 + np.asarray(t, dtype=float) - self.dt / 3600.0


def eph_lon(lon, elements: "Elements"):
    """
    Longitude referred to the *ephemeris meridian*.

    mu is tabulated against a meridian 1.002738*deltaT east of Greenwich, so the
    observer's east longitude must be shifted before forming the local hour angle.
    Omitting this costs ~0.32 deg of longitude here -> ~0.2 deg of sun altitude,
    which at a 5 deg sun is not a rounding error.
    """
    return np.asarray(lon, dtype=float) - 1.002738 * elements.dt * 15.0 / 3600.0


def observer_xyz(lat, lon, height_m, el):
    """Observer position (xi, eta, zeta) in the Besselian fundamental frame.

    `lon` must already be ephemeris-meridian corrected (see `eph_lon`).
    """
    phi = np.asarray(lat, dtype=float) * D2R
    lam = np.asarray(lon, dtype=float) * D2R
    h = np.asarray(height_m, dtype=float) / (EARTH_A * 1000.0)

    u = np.arctan(FLAT * np.tan(phi))
    rsp = FLAT * np.sin(u) + h * np.sin(phi)      # rho * sin(phi')
    rcp = np.cos(u) + h * np.cos(phi)             # rho * cos(phi')

    theta = el["mu"] + lam                        # local hour angle of the shadow axis
    xi = rcp * np.sin(theta)
    eta = rsp * np.cos(el["d"]) - rcp * np.cos(theta) * np.sin(el["d"])
    zeta = rsp * np.sin(el["d"]) + rcp * np.cos(theta) * np.cos(el["d"])

    # time derivatives (per hour)
    dxi = el["dmu"] * rcp * np.cos(theta)
    deta = el["dmu"] * xi * np.sin(el["d"]) - zeta * el["dd"]
    return xi, eta, zeta, dxi, deta


def state(lat, lon, height_m, t, elements: Elements):
    """Core geometry at element-time t. All arrays broadcast together."""
    el = elements.at(t)
    xi, eta, zeta, dxi, deta = observer_xyz(lat, eph_lon(lon, elements), height_m, el)

    u = el["x"] - xi
    v = el["y"] - eta
    a = el["dx"] - dxi
    b = el["dy"] - deta
    n2 = a * a + b * b
    n = np.sqrt(n2)

    m = np.hypot(u, v)
    L1p = el["l1"] - zeta * elements.tanf1     # penumbral radius at observer
    L2p = el["l2"] - zeta * elements.tanf2     # umbral radius (negative => total)

    mag = (L1p - m) / (L1p + L2p)
    return dict(el=el, xi=xi, eta=eta, zeta=zeta, u=u, v=v, a=a, b=b, n=n, n2=n2,
                m=m, L1p=L1p, L2p=L2p, mag=mag)


def sun_altaz(lat, lon, t, elements: Elements, refract=True):
    """Apparent sun altitude/azimuth (deg). dec = d, hour angle = mu + lon."""
    el = elements.at(t)
    phi = np.asarray(lat, dtype=float) * D2R
    H = el["mu"] + eph_lon(lon, elements) * D2R
    d = el["d"]
    sinh = np.sin(phi) * np.sin(d) + np.cos(phi) * np.cos(d) * np.cos(H)
    alt = np.arcsin(np.clip(sinh, -1, 1)) * R2D
    az = np.arctan2(-np.cos(d) * np.sin(H),
                    np.sin(d) * np.cos(phi) - np.cos(d) * np.sin(phi) * np.cos(H)) * R2D % 360.0
    if refract:
        alt = alt + refraction(alt)
    return alt, az


def refraction(alt_deg):
    """Bennett refraction, degrees. Valid down to and slightly below the horizon."""
    a = np.asarray(alt_deg, dtype=float)
    r = 1.02 / np.tan((a + 10.3 / (a + 5.11)) * D2R) / 60.0   # arcmin -> deg
    return np.where(a > -1.0, r, 0.0)


def t_max(lat, lon, height_m, elements: Elements, t_guess=0.5, iters=6):
    """Element-time of maximum eclipse (closest approach to the shadow axis)."""
    t = np.full(np.broadcast(np.asarray(lat, float), np.asarray(lon, float)).shape,
                float(t_guess)) if np.ndim(lat) or np.ndim(lon) else float(t_guess)
    t = np.asarray(t, dtype=float)
    for _ in range(iters):
        s = state(lat, lon, height_m, t, elements)
        tau = -(s["u"] * s["a"] + s["v"] * s["b"]) / s["n2"]
        t = t + tau
    return t


def contacts(lat, lon, height_m, elements: Elements, t_guess=0.5, iters=8):
    """
    Return dict with t_max, magnitude, and the four contact times (element-hours).
    C1/C4 use the penumbral radius L1', C2/C3 the umbral radius |L2'|.
    NaN where the contact does not occur (e.g. C2/C3 outside the path of totality).
    """
    tm = t_max(lat, lon, height_m, elements, t_guess)
    sm = state(lat, lon, height_m, tm, elements)

    out = dict(t_max=tm, mag_max=sm["mag"], m_max=sm["m"],
               L1p=sm["L1p"], L2p=sm["L2p"],
               is_total=sm["m"] < np.abs(sm["L2p"]),
               is_partial=sm["m"] < sm["L1p"])

    def solve(which, umbral):
        # sign: -1 = immersion (C1/C2), +1 = emersion (C3/C4)
        sign = -1.0 if which in ("C1", "C2") else 1.0
        t = np.array(tm, dtype=float, copy=True)
        for _ in range(iters):
            s = state(lat, lon, height_m, t, elements)
            L = np.abs(s["L2p"]) if umbral else s["L1p"]
            # sin(psi) = (u*b - v*a) / (n * L)
            sinpsi = np.clip((s["u"] * s["b"] - s["v"] * s["a"]) / (s["n"] * L), -1, 1)
            cospsi = np.sqrt(np.maximum(0.0, 1 - sinpsi**2))
            tau = -(s["u"] * s["a"] + s["v"] * s["b"]) / s["n2"] + sign * (L / s["n"]) * cospsi
            t = t + tau
        return t

    out["C1"] = np.where(out["is_partial"], solve("C1", False), np.nan)
    out["C4"] = np.where(out["is_partial"], solve("C4", False), np.nan)
    out["C2"] = np.where(out["is_total"], solve("C2", True), np.nan)
    out["C3"] = np.where(out["is_total"], solve("C3", True), np.nan)
    out["duration_s"] = (out["C3"] - out["C2"]) * 3600.0
    out["partial_duration_s"] = (out["C4"] - out["C1"]) * 3600.0

    for k in ("C1", "C2", "C3", "C4", "t_max"):
        out[k + "_ut"] = elements.ut_hours(out[k])
    return out


def hms(ut_hours):
    """Decimal UT hours -> 'HH:MM:SS' string (scalar)."""
    h = float(ut_hours)
    if not np.isfinite(h):
        return "--:--:--"
    s = round(h * 3600.0)
    return f"{int(s // 3600) % 24:02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"
