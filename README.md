# Evaluation Program

Desktop application (Qt / PySide6) for evaluating thin-film measurements:
growth and structural characterisation (XRD, RSM) and low-temperature transport.
All analysis code lives in `evalprog.core` and can be used from scripts or
notebooks without the GUI.

| Tab | Status |
| --- | --- |
| XRD (symmetric 2θ-ω) | substrate database, automatic substrate peak fit, zero-shift and sample-height correction, film peaks, out-of-plane strain |
| RSM | 2D pseudo-Voigt fits, substrate-based Δ2θ/Δω correction, in-/out-of-plane lattice, strain, relaxation |
| R(T) | QD ETO / Resistivity option / Zurich lock-in (LabVIEW) files, T_c analysis, magnetic kinks, MIT, RRR, ρ and R_sheet |
| R(H) / Harmonics | Zurich lock-in (LabVIEW) field sweeps with 1f/2f/3f: symmetrisation, R(T), heating from 3f, normal Hall subtraction, R_H and n, series comparisons with power-law fits |
| AFM | Bruker NanoScope `.spm` / ASCII matrices: scar removal, plane / polynomial / terrace levelling, line alignment, roughness, terrace levels and step heights, profiles, PSD, pits / islands |
| MPMS, Ellipsometry, TEM | planned |

## Installation

Python ≥ 3.10.

```bash
git clone <repo-url> evaluation-program
cd evaluation-program
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .
evalprog            # or: python -m evalprog   or: python run.py
```

In Spyder run `run.py` with "Execute in an external system terminal" (or set the
graphics backend to Qt).

### Stand-alone app

```bash
pip install pyinstaller
pyinstaller evaluation_program.spec
```

The app is created in `dist/EvaluationProgram/` (Windows `EvaluationProgram.exe`,
macOS `EvaluationProgram.app`). Build on the operating system you want to run it on.

## XRD

1. **Load scan**: `.xy` (two columns, as converted from `.xrdml`) or `.xrdml` directly.
2. **Substrate**: choose substrate and orientation, *Find & fit substrate peaks*.
   All allowed substrate reflections inside the scan range are located (± search
   window around the calculated position) and the strongest ones (up to *Max. peaks*)
   are fitted with a pseudo-Voigt Kα1/Kα2 doublet (Kα2 position from Bragg's law,
   intensity ratio 0.5, optionally asymmetric) on a linear background. Poisson weights
   (1/√I) are used. An extra peak (film peak, shoulder) inside the window is added
   automatically when it improves the fit (BIC).
3. **Correction**: Δ2θ = Z + D·cos θ.
   * 1 peak → zero shift Z only;
   * ≥ 2 peaks → Z and the sample-displacement term D. The sample height is
     s = −D·R/2 (R = goniometer radius, 240 mm for the Empyrean).
   The corrected axis is used for everything afterwards and can be saved as `.xy`.
4. **Film peaks**: drag a range in the upper plot, set phase and (h k l), *Fit selected
   range*. Substrate peaks inside the range are fitted simultaneously. The table gives
   the corrected position, d and the out-of-plane lattice parameter; several reflections
   of one phase are combined (weighted mean) and compared with the bulk value.

## RSM

1. **Load** `.xrdml` area scans or the PANalytical CSV export (`[Scan points]` block).
2. **Substrate reference**: substrate, orientation, in-plane direction and (h k l)
   of the measured substrate reflection (pseudocubic for perovskites, e.g. STO (1 0 3);
   NdGaO3 (110)o uses pseudocubic indices with the [1-10]o or [001]o in-plane spacing).
   *Find substrate peak & correct offset* fits the substrate peak and shifts 2θ and ω
   so that it sits at the calculated position. A rectangle drawn around the
   substrate can be used instead (*Use current ROI*).
3. **Film peak**: drag a rectangle, set phase and (h k l) in the film cell, *Fit ROI*.
   The fit is a rotated 2D pseudo-Voigt with a plane background:
   * weighting *poisson* (default) or *log* (emphasises tails). *none* reproduces the
     old unweighted fit, which is why its RMS was dominated by the peak top;
   * the fitted substrate peak can be subtracted first, so a large ROI does not pull
     the film peak towards the substrate;
   * *robust loss* reduces the influence of streaks / detector artefacts.
4. **Results**: a_ip = 2π√(h²+k²)/|Qx|, c = 2πl/Qz, strains against the bulk
   values, relaxation R = (a_ip − a_sub)/(a_bulk − a_sub), the unstrained lattice
   parameter from the Poisson ratio a_r = (c·a0/c0 + k·a_ip)/(1 + k), k = 2ν/(1−ν),
   and the effective Poisson ratio ν_eff = ε_zz/(ε_zz − 2ε_xx). Selecting a row shows
   Qx and Qz line cuts through the peak (data and fit).

## R(T)

* **Files**: QD MultiVu `.dat` (ETO channels, Resistivity option bridges), LabVIEW
  lock-in files (instrument header with `I=`; each demodulator voltage → R = V/I) and
  any delimited table with a header line. Several files/channels can be overlaid.
* **Branch**: cooling / warming are separated automatically (time order).
* **Quantity**: R, ρ (bar: ρ = R·w·t/L; van der Pauw: ρ = π t/ln2 · R · f), R_sheet,
  R/R(300 K).
* **Superconducting**: normal-state line fitted above the transition;
  T_c,onset (90 %), T_c,mid (50 %), T_c,zero (1 %), onset from the tangent at T_mid,
  ΔT_c (90–10 %), all thresholds adjustable.
* **Magnetic / kinks**: most prominent extrema of dR/dT and d²R/dT² (Savitzky–Golay,
  window adjustable), resistance minima/maxima.
* **Metal–insulator**: T_MI from the extremum of d ln R/dT for both branches and the
  hysteresis width; resistance minimum.
* Click into the plot to add a manual marker. Results are collected in a table and
  can be exported; curves are exported with Origin-style long-name/unit header lines.

## R(H) / Harmonics

Port of the stand-alone higher-harmonics program (same algorithms, English UI).
Files are classified automatically as field sweeps or R(T) curves.

* **Evaluation**: loop split into down/up sweep, resampled on a symmetric field grid,
  sym(B) = [S_down(B) + S_up(−B)]/2, antisym(B) = [S_down(B) − S_up(−B)]/2; output in V or ρ.
  A measurement restarted within one file is detected (last section / each / all).
* **R(T)**: R = ReZ1f/I from R(T) files or imported tables, smoothed dR/dT.
* **Heating (3f)**: ΔT₂ω = 2|V₃f|/(I|dR/dT|) with an automatically matched R(T) curve.
* **Hall**: fit R₀·B (+ A·sgn B) for |B| ≥ B_min, subtract R₀·B, R_H and n.
* **Comparison**: any set of sweeps or an automatically selected series (only T, I, f or
  angle varies), normalisation (÷Iⁿ, MR, ÷S(0)), value at B₀ or maximum vs. parameter,
  log-log power-law fit.

All results go to `<measurement folder>/output/` (CSV with Origin header lines) and
`output/figures/`.

## AFM

* **Files**: Bruker / Veeco NanoScope (`.spm`, `.000` …, all channels; height in nm
  from Z scale × Zsens) and Gwyddion-style ASCII matrices.
* **Corrections** (in this order, each can be switched off):
  1. background + line alignment, then **scar removal**: streaks 1–3 lines wide that
     differ from the lines above *and* below by > n·σ (σ = robust line-to-line noise),
     at least *min. length* pixels long, are replaced by interpolation along the slow axis;
  2. **background**: plane, polynomial (2nd/3rd order, for piezo bow) or
     *terraces (facet level)*: the plane whose slope is the mean local gradient after
     rejecting step edges, so terraces of a vicinal surface become horizontal (a
     least-squares plane tilts them by the miscut);
  3. **line alignment**: median per line (rough films, particles) or *median of
     differences* (terraces, large features: aligns each line to the previous one);
     mean / linear / quadratic per line;
  4. pits and particles (> n·σ) are masked and the fits of 2 and 3 repeated without them;
  5. optional spike removal (median filter), optional smoothing (Gaussian σ or median
     window in nm; reduces noise for step heights, lowers Sq), zero level.
  With *terraces* + *median of differences* the final line offsets and the terrace
  slope are fitted jointly (z = c_terrace + r_line + a·x, edges excluded), so the line
  offsets do not drift like a pure line-to-line alignment does on noisy images.
  *Terrace bow order* (2 or 3, default 2; 0 = off) adds a 2D polynomial to that fit
  (piezo bow / sample bending). Its curvature is measured inside the terraces, with
  one free offset per terrace, so the staircase itself is kept. If the step edges are
  not resolved (one terrace region covers more than half of the image) the bow is
  skipped and the log says so.
* **Step heights of substrates** (e.g. BOE-terminated SrTiO3):
  * *profile line* mode: drag across the steps; *profile width* averages along the step
    edges (dashed band). **Profile ⟂ steps (auto)** sets the line through the image
    centre perpendicular to the step edges (structure tensor of the step-edge
    gradients) and the width to 0.4 × terrace width (terrace width = reference
    step / mean slope), narrow enough that meandering edges are not smeared.
    The profile is fitted with a staircase h = a + b·s + Σ H_k·Θ(s − e_k) (edge zones
    excluded, residual tilt b free); fitted steps below min. step/2 and pits/islands
    (two opposite edges returning to the same level) are dropped. Result: every step
    height ± error, mean |step| ± std/√N, terrace width and miscut angle;
  * *region* mode: drag a rectangle over a few clean terraces; the height histogram of
    the region is fitted with one Gaussian per level, the level spacing gives the step
    heights and a linear fit L = L0 + n·d over all levels gives one step height ± error
    (n from the reference step, e.g. 0.3905 nm). Works also for meandering steps.
* **Analysis**: Sq, Sa, Sz, Ssk, Sku (whole image and without masked features),
  correlation length (ACF = 1/e), height histogram with automatic level detection
  and step heights in units of a reference step (e.g. 0.3905 nm for SrTiO3),
  line profiles (drag a line, adjustable averaging width), 1D PSD along both axes,
  pits / islands relative to a local reference level (count, density, area fraction,
  size, depth, volume).
* **Export**: corrected image as ASCII matrix (opens in Gwyddion), profile, results
  and feature tables as CSV, figures as PNG, into `<data folder>/output/`.

## Materials database

Built-in tables: `src/evalprog/materials/data/substrates.json` and `films.json`.

Substrates: SrTiO3 (001)/(110)/(111), LaAlO3 (001)pc, LSAT (001), MgO (001),
NdGaO3 (110)o/(001)o, DyScO3 (110)o, GdScO3 (110)o, SrLaAlO4 (001).
Films: SrIrO3, SrRuO3, CaIrO3, NdNiO3, La2CuO4, La1.85Sr0.15CuO4.

The values are room-temperature literature values; Poisson ratios are typical
numbers and should be checked for the respective system. To add or change entries
without touching the code, use *Materials → Open user materials folder* and put
entries with the same structure into `~/.evaluation_program/substrates.json` or
`films.json`, then *Materials → Reload materials*.

## Using the analysis without the GUI

```python
from evalprog import materials
from evalprog.core.xrd.io import load_scan
from evalprog.core.xrd.substrate import find_and_fit

scan = load_scan("sample.xy")
res = find_and_fit(scan.two_theta, scan.intensity, materials.get_substrate("SrTiO3", "(001)"))
print(res.correction.describe())
two_theta_corrected = res.correction.apply(scan.two_theta)
```

## Development

```bash
pip install -e .[dev]
pytest
```

Tests use synthetic data (`tests/synth.py`). Real example files for each instrument
should be added to `tests/data/` as they become available.

## Roadmap

* MPMS M(T), M(H)
* Ellipsometry
* GenX integration for XRR (run GenX fits / read `.hgx` results)
* eLabFTW: upload results and figures to experiments via the REST API
* Python control of the Zurich Instruments lock-ins (zhinst)
