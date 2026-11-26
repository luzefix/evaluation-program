"""Synthetic data files for tests and demos."""

from pathlib import Path

import numpy as np

from evalprog.core.rsm.analysis import angles_to_q, expected_q, pv2d, q_to_angles
from evalprog.core.xrd.peaks import bragg_two_theta, ka2_position, pseudo_voigt
from evalprog.materials import get_film, get_substrate


def xrd_scan(substrate, z=0.05, height_um=50.0, film_c=3.95, seed=1):
    rng = np.random.default_rng(seed)
    x = np.arange(15, 110, 0.005)
    y = np.full_like(x, 20.0)
    D = np.degrees(-2 * height_um * 1e-3 / 240.0)
    shift = lambda tt: z + D * np.cos(np.radians(tt) / 2)
    for n, d, _ in substrate.reflections():
        tt = bragg_two_theta(d)
        if np.isfinite(tt) and tt < 108:
            c1 = tt + shift(tt)
            y += pseudo_voigt(x, c1, 2e5 / n, 0.02, 0.6) + pseudo_voigt(x, ka2_position(c1), 1e5 / n, 0.02, 0.6)
    if film_c:
        for l in (1, 2, 3, 4):
            tt = bragg_two_theta(film_c / l)
            if tt < 108:
                c1 = tt + shift(tt)
                y += pseudo_voigt(x, c1, 3000 / l, 0.15, 0.3) + pseudo_voigt(x, ka2_position(c1), 1500 / l, 0.15, 0.3)
    return x, rng.poisson(y).astype(float)


def write_xy(path, x, y):
    np.savetxt(path, np.column_stack([x, y]), fmt="%.4f")


def rsm_grid(d_tt=0.03, d_om=0.12, seed=2):
    sub = get_substrate("SrTiO3", "(001)")
    film = get_film("SrRuO3")
    hkl = (1, 0, 3)
    a_s = sub.inplane_spacing()
    nu = film.poisson
    c_f = film.c - 2 * nu / (1 - nu) * (a_s - film.a)
    qs = expected_q(hkl, a_s, sub.d_oop, sign=-1)
    qf = expected_q(hkl, a_s, c_f, sign=-1)
    tt0, om0 = q_to_angles(*qs)
    tt = np.arange(tt0 - 2.5, tt0 + 1.0, 0.01)
    offs = np.arange(-0.8, 0.8, 0.01)
    TT, OF = np.meshgrid(tt, offs)
    OM = TT / 2 + (om0 - tt0 / 2) + OF
    qx, qz = angles_to_q(TT, OM)
    I = pv2d(qx, qz, 1e5, *qs, 0.0015, 0.0015, 0.0, 0.5) + pv2d(qx, qz, 3e3, *qf, 0.004, 0.012, 0.0, 0.4) + 5
    I = np.random.default_rng(seed).poisson(I).astype(float)
    return TT + d_tt, OM + d_om, I, qf


def write_xrdml_map(path, TT, OM, I):
    scans = []
    for i in range(TT.shape[0]):
        scans.append(
            '<scan scanAxis="2Theta-Omega"><dataPoints>'
            f'<positions axis="2Theta" unit="deg"><startPosition>{TT[i, 0]:.5f}</startPosition>'
            f'<endPosition>{TT[i, -1]:.5f}</endPosition></positions>'
            f'<positions axis="Omega" unit="deg"><startPosition>{OM[i, 0]:.5f}</startPosition>'
            f'<endPosition>{OM[i, -1]:.5f}</endPosition></positions>'
            '<commonCountingTime unit="seconds">1</commonCountingTime>'
            f'<intensities unit="counts">{" ".join(str(int(v)) for v in I[i])}</intensities>'
            '</dataPoints></scan>')
    path.write_text('<?xml version="1.0"?><xrdMeasurements xmlns="http://www.xrdml.com/XRDMeasurement/1.5">'
                    '<xrdMeasurement><usedWavelength><kAlpha1 unit="Angstrom">1.5405980</kAlpha1>'
                    '<kAlpha2 unit="Angstrom">1.5444260</kAlpha2><ratioKAlpha2KAlpha1>0.5</ratioKAlpha2KAlpha1>'
                    '</usedWavelength>' + "".join(scans) + '</xrdMeasurement></xrdMeasurements>')


def write_qd_sc(path, tc=30.0):
    T = np.r_[np.linspace(60, 2, 600), np.linspace(2, 60, 600) + 0.2]
    t = np.arange(T.size) * 2.0
    R1 = (100 + 0.5 * T) / (1 + np.exp(-(T - tc) / 0.25))
    R2 = 2 * R1
    lines = ["[Header]", "; ETO", "[Data]",
             "Comment,Time Stamp (sec),Temperature (K),Magnetic Field (Oe),Resistance Ch1 (Ohms),Resistance Ch2 (Ohms)"]
    lines += [f",{a},{b},0,{c},{d}" for a, b, c, d in zip(t, T, R1, R2)]
    path.write_text("\n".join(lines))


LOCKIN_HEADER = ("1.000,_Harm_,317.000,_Hz_,0.30000000,_V_,j= ,4102844.639,_Orders=_,,0.000,°,A=,8.0,"
                 "I=,{I},R=,9140.000")


def write_lockin_sweep(path, T=10.0, I=1e-5, bmax=90000.0, step=50.0, r0=100.0, hall=2e-6, mr=1e-9,
                       v2=3e-11, v3=1e-12, seed=0):
    """Field loop +Bmax -> -Bmax -> +Bmax (Oe) as written by the LabVIEW program."""
    rng = np.random.default_rng(seed)
    down = np.arange(bmax, -bmax - step / 2, -step)
    B = np.r_[down, down[::-1][1:]]
    # anomalous-Hall-like loop with coercive field 1000 Oe (down: switches at -Bc, up: at +Bc)
    bc = np.r_[np.full(down.size, -1000.0), np.full(down.size - 1, 1000.0)]
    n = B.size
    t = np.arange(n) * 0.5
    V1 = I * (r0 + mr * B ** 2 + hall * B + 0.05 * np.tanh((B - bc) / 500))
    V2 = v2 * B
    V3 = v3 * (1 + 1e-10 * B ** 2)
    cols = [t, np.full(n, T) + rng.normal(0, 0.005, n), B, np.zeros(n), np.full(n, 317.0),
            V1 + rng.normal(0, 1e-9, n), np.zeros(n), np.zeros(n), V2 + rng.normal(0, 1e-12, n),
            V3, 0.2 * V3, np.full(n, I), np.zeros(n)]
    lines = [LOCKIN_HEADER.format(I=I)]
    lines += ["\t".join(f"{c[i]:.10g}" for c in cols) for i in range(n)]
    path.write_text("\n".join(lines))


def write_lockin_rt(path, I=1e-5, r0=50.0, slope=5.0, T0=10.0):
    """R(T) at zero field: R = r0 + slope*T, centred so that R(T0) = r0 + slope*T0."""
    T = np.linspace(2, 30, 400)
    n = T.size
    cols = [np.arange(n) * 1.0, T, np.zeros(n), np.zeros(n), np.full(n, 317.0),
            I * (r0 + slope * T), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
            np.full(n, I), np.zeros(n)]
    lines = [LOCKIN_HEADER.format(I=I)]
    lines += ["\t".join(f"{c[i]:.10g}" for c in cols) for i in range(n)]
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- AFM


def afm_terraces(n=256, size_nm=2000.0, step=0.3905, terrace_nm=250.0, tilt=(2e-3, -1e-3),
                 row_sigma=0.3, scars=((60, 0.8), (150, -0.6)), pits=8, seed=0):
    """Vicinal surface: terraces of width terrace_nm (steps along x), plus an
    instrument tilt, random line offsets, two scars and some pits (2 nm deep).
    Returns (z_measured, z_true_levelled) in nm."""
    rng = np.random.default_rng(seed)
    dx = size_nm / n
    y, x = np.mgrid[:n, :n] * dx
    edge = x + 40 * np.sin(2 * np.pi * y / size_nm)        # meandering step edges
    true = -step * np.floor(edge / terrace_nm)
    for _ in range(pits):
        cx, cy = rng.uniform(0, size_nm, 2)
        true = np.where((x - cx) ** 2 + (y - cy) ** 2 < 30.0 ** 2, true - 2.0, true)
    z = true + tilt[0] * x + tilt[1] * y + rng.normal(0, 0.02, true.shape)
    z += rng.normal(0, row_sigma, n)[:, None]
    for row, h in scars:
        z[row, n // 4: n // 4 + n // 3] += h
    return z, true


def write_nanoscope(path, channels, size_nm=2000.0, zsens=7.471396):
    """Minimal NanoScope v9 file. channels: list of (name, z_nm) with 32-bit data
    and 'Bytes/pixel: 2' in the header, as written by NanoScope 9."""
    ny, nx = channels[0][1].shape
    zscale_v = 10.0
    lsb_nm = zscale_v / 65536 * zsens
    header_len = 8192
    blocks = []
    off = header_len
    for name, _ in channels:
        blocks.append(
            "\\*Ciao image list\r\n"
            f"\\Data offset: {off}\r\n\\Data length: {nx * ny * 4}\r\n\\Bytes/pixel: 2\r\n"
            f"\\Samps/line: {nx}\r\n\\Number of lines: {ny}\r\n"
            f"\\Scan Size: {size_nm / 1e3:g} {size_nm / 1e3:g} ~m\r\n\\Line Direction: Retrace\r\n"
            f'\\@2:Image Data: S [{name}] "{name}"\r\n'
            f"\\@2:Z scale: V [Sens. Zsens] (0.006713867 V/LSB) {zscale_v} V\r\n")
        off += nx * ny * 4
    head = ("\\*File list\r\n\\Version: 0x09700106\r\n\\Data length: 8192\r\n"
            "\\*Ciao scan list\r\n"
            f"\\Scan Size: {size_nm:g} nm\r\n\\Samps/line: {nx}\r\n\\Lines: {ny}\r\n"
            f"\\@Sens. Zsens: V {zsens} nm/V\r\n" + "".join(blocks) + "\\*File list end\r\n")
    hb = head.encode("latin-1")
    hb += b"\x00" * (header_len - len(hb))
    data = b"".join(np.flipud(np.round(z / lsb_nm)).astype("<i4").tobytes() for _, z in channels)
    Path(path).write_bytes(hb + data)
    return lsb_nm
