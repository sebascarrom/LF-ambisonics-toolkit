"""
spatial_metrics.py
==================

Spatial analysis tools for first-order Ambisonic Room Impulse Responses.

This module provides directional analysis of B-format room impulse
responses by estimating the active acoustic intensity vector and the
Direction of Arrival (DOA) of incoming sound energy.

The implementation follows the classical first-order Ambisonics model
and is intended as a complementary analysis to the ISO 3382-1 Lateral
Fraction (LF), allowing visualization of the spatial evolution of the
sound field.

Implemented analyses
--------------------
- Instantaneous acoustic intensity (proxy)
- Direction of Arrival (DOA)
- Directional reflectograms
- Polar energy maps
- Intensity vector visualization

Planned extensions
------------------
- Diffuseness estimation
- Lateral Energy Fraction maps
- Lateral Energy Fraction Curve (LFC)
- Interaural Cross-Correlation (IACC)
- Spatial statistics

References
----------
ISO 3382-1:2009

Merimaa, J., & Pulkki, V. (2005).
Spatial Impulse Response Rendering (SIRR).

Cacavelos, J., Bonelli, A., & Bidondo, A. (2016).
Development of a 3D Impulse Response Interpretation Algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# import sys

# sys.path.append(r"C:\Users\Pc-usuario\LF-ambisonics-toolkit")

from src.acoustic_core import (
    BFormatSignals,
    detect_rir_end,
    detect_onset_noise_floor,
)


# ============================================================================
# DATA CLASS
# ============================================================================

"""
Container holding the results of a directional analysis.

Parameters
----------
time_ms
    Time axis corresponding to each analysis window.

Ix, Iy, Iz
    Integrated active intensity components.

magnitude
    Intensity vector magnitude.

azimuth_deg
    Horizontal Direction of Arrival.

elevation_deg
    Vertical Direction of Arrival.

window_ms
    Integration window used for the analysis.
"""

@dataclass(slots=True)
class DirectionalAnalysis:

    sample_index: np.ndarray

    # Time axis relative to the detected RIR onset (0 = transient).
    # Pre-roll content (captured before the onset) appears at negative time.
    time_ms: np.ndarray

    Ix: np.ndarray
    Iy: np.ndarray
    Iz: np.ndarray

    magnitude: np.ndarray

    azimuth_deg: np.ndarray
    elevation_deg: np.ndarray

    window_ms: float

    # Onset / end reference points, kept for synchronized plotting.
    onset_sample: int = 0
    onset_ms: float = 0.0
    end_sample: int = 0
    
# @dataclass(slots=True)
# class SpatialStatistics:
#     """
#     Statistical descriptors of the directional sound field.

#     Reserved for future implementations such as diffuseness,
#     directional spread and energy-weighted statistics.
#     """

#     mean_azimuth: float
#     std_azimuth: float

#     mean_elevation: float
#     std_elevation: float

#     peak_intensity: float


def summary(self) -> str:

    lines = []

    lines.append("Directional Analysis")
    lines.append("--------------------")
    lines.append(f"Samples        : {len(self.time_ms)}")
    lines.append(f"Window         : {self.window_ms:.2f} ms")
    lines.append(f"Max intensity  : {np.max(self.magnitude):.4e}")
    lines.append(f"Mean azimuth   : {np.mean(self.azimuth_deg):.2f}°")
    lines.append(f"Mean elevation : {np.mean(self.elevation_deg):.2f}°")

    return "\n".join(lines)

# ============================================================================
# INTENSITY
# ============================================================================

def compute_intensity(
    bformat: BFormatSignals,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a proxy of the instantaneous active acoustic intensity vector.

    In first-order Ambisonics, the omnidirectional channel W is
    proportional to the acoustic pressure while the dipole channels
    X, Y and Z are proportional to the particle velocity components.

    Therefore,

        Ix ∝ W · X
        Iy ∝ W · Y
        Iz ∝ W · Z

    are proportional to the active acoustic intensity components.

    The proportionality constant (ρc)^−1 is omitted because only the
    direction of arrival is required for visualization and spatial
    analysis.

    Parameters
    ----------
    bformat : BFormatSignals
        First-order B-format signals.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays containing the X, Y and Z components of the
        instantaneous intensity proxy.
    """

    Ix = bformat.W * bformat.X
    Iy = bformat.W * bformat.Y
    Iz = bformat.W * bformat.Z

    return Ix, Iy, Iz


# ============================================================================
# DIRECTION OF ARRIVAL
# ============================================================================

def compute_doa(
    Ix: np.ndarray,
    Iy: np.ndarray,
    Iz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the Direction of Arrival (DOA) from acoustic intensity vectors.

    The active intensity vector is converted from Cartesian coordinates
    (Ix, Iy, Iz) to spherical coordinates.

    Coordinate system
    -----------------
        +X : Front
        -X : Back

        +Y : Left
        -Y : Right

        +Z : Up
        -Z : Down

    Therefore

        Azimuth:
            0°   = Front
            +90° = Left
            ±180° = Back
            -90° = Right

        Elevation:
            +90° = Up
            0°   = Horizontal plane
            -90° = Down

    Parameters
    ----------
    Ix, Iy, Iz : ndarray
        Components of the active intensity vector.

    Returns
    -------
    azimuth_deg : ndarray
        Horizontal direction of arrival.

    elevation_deg : ndarray
        Vertical direction of arrival.
    """

    horizontal = np.sqrt(Ix**2 + Iy**2)

    horizontal = np.maximum(
        horizontal,
        np.finfo(float).eps
    )

    azimuth = np.degrees(
        np.arctan2(Iy, Ix)
    )

    elevation = np.degrees(
        np.arctan2(Iz, horizontal)
    )

    return azimuth, elevation


# ============================================================================
# WINDOWED AVERAGING
# ============================================================================
# ============================================================================
# WINDOWED INTENSITY INTEGRATION
# ============================================================================

def window_intensity(
    Ix,
    Iy,
    Iz,
    fs,
    window_ms=1.0,
    t_start_ms=0.0,
    t_end_ms=None,
):
    """
    Integrate the acoustic intensity vector over consecutive
    time windows within a selected time interval.

    Parameters
    ----------
    Ix, Iy, Iz : ndarray
        Instantaneous intensity components.

    fs : int
        Sampling frequency [Hz].

    window_ms : float
        Integration window length.

    t_start_ms : float
        Beginning of the analysis interval.

    t_end_ms : float or None
        End of the analysis interval.
        If None, the signal is analyzed until its end.

    Returns
    -------
    Ix_win, Iy_win, Iz_win
        Integrated intensity components.

    time_ms
        Time corresponding to each window center.

    sample_index
        Sample index corresponding to each window center.
    """

    # ----------------------------------------------------------
    # Convert analysis interval to samples
    # ----------------------------------------------------------

    start_sample = max(
        int(round(t_start_ms * fs / 1000)),
        0,
    )

    if t_end_ms is None:
        end_sample = len(Ix)
    else:
        end_sample = min(
            int(round(t_end_ms * fs / 1000)),
            len(Ix),
        )

    if end_sample <= start_sample:
        raise ValueError(
            "Invalid analysis interval."
        )

    # ----------------------------------------------------------
    # Crop signals
    # ----------------------------------------------------------

    Ix = Ix[start_sample:end_sample]
    Iy = Iy[start_sample:end_sample]
    Iz = Iz[start_sample:end_sample]

    # ----------------------------------------------------------
    # Window size
    # ----------------------------------------------------------

    samples_per_window = max(
        int(round(window_ms * fs / 1000)),
        1,
    )

    n_windows = len(Ix) // samples_per_window

    if n_windows == 0:
        raise ValueError(
            "Signal is shorter than the selected averaging window."
        )

    # ----------------------------------------------------------
    # Output arrays
    # ----------------------------------------------------------

    Ix_win = np.zeros(n_windows)
    Iy_win = np.zeros(n_windows)
    Iz_win = np.zeros(n_windows)

    time_ms = np.zeros(n_windows)
    sample_index = np.zeros(n_windows, dtype=int)

    dt = 1.0 / fs

    # ----------------------------------------------------------
    # Integrate intensity over consecutive windows
    # ----------------------------------------------------------

    for i in range(n_windows):

        start = i * samples_per_window
        end = start + samples_per_window

        Ix_win[i] = np.trapezoid(Ix[start:end], dx=dt)
        Iy_win[i] = np.trapezoid(Iy[start:end], dx=dt)
        Iz_win[i] = np.trapezoid(Iz[start:end], dx=dt)

        center = start + samples_per_window // 2

        # Absolute sample index in the original RIR
        sample_index[i] = start_sample + center

        # Absolute time in the original RIR
        time_ms[i] = sample_index[i] / fs * 1000.0

    return (
        Ix_win,
        Iy_win,
        Iz_win,
        time_ms,
        sample_index,
    )

# ============================================================================
# COMPLETE DIRECTIONAL ANALYSIS
# ============================================================================

def analyze_directionality(
    bformat: BFormatSignals,
    window_ms: float = 1.0,
    onset_sample: int | None = None,
    end_sample: int | None = None,
    preroll_ms: float = 20.0,
    normalize_vectors: bool = False,
) -> DirectionalAnalysis:
    """
    Complete directional analysis pipeline.

    Processing steps
    ----------------
    1. Compute instantaneous acoustic intensity.
    2. Detect the RIR onset (transient) and the useful end of the RIR.
    3. Integrate intensity over temporal windows, from
       (onset - preroll_ms) up to end_sample.
    4. Compute intensity magnitude.
    5. Estimate Direction of Arrival (DOA).

    The returned time axis (``result.time_ms``) is referenced to the
    detected onset: t=0 corresponds to the RIR transient, content
    captured before it (pre-roll / noise floor) is reported at
    negative time.

    Parameters
    ----------
    onset_sample : int or None
        Sample index of the RIR transient. If None, it is detected
        automatically via detect_onset_noise_floor() on the W channel.
    end_sample : int or None
        Sample index marking the useful end of the RIR. If None, it is
        detected automatically via detect_rir_end().
    preroll_ms : float
        Amount of pre-onset content (in ms) to include in the analysis
        / plots, shown at negative time. Use 0.0 to start exactly at
        the onset.
    """

    # --------------------------------------------------------------
    # Step 1 — Instantaneous intensity
    # --------------------------------------------------------------

    Ix, Iy, Iz = compute_intensity(bformat)

    # --------------------------------------------------------------
    # Step 2 — Detect onset and end of useful RIR
    # --------------------------------------------------------------

    if onset_sample is None:
        onset_sample = detect_onset_noise_floor(
            bformat.W,
            bformat.fs,
        )

    onset_ms = onset_sample / bformat.fs * 1000.0

    if end_sample is None:
        end_sample = detect_rir_end(
            bformat.W,
            bformat.fs,
        )

    t_end_ms = end_sample / bformat.fs * 1000.0

    t_start_ms = max(onset_ms - preroll_ms, 0.0)

    # --------------------------------------------------------------
    # Step 3 — Window integration
    # --------------------------------------------------------------

    (
        Ix,
        Iy,
        Iz,
        time_ms,
        sample_index,
    ) = window_intensity(
        Ix,
        Iy,
        Iz,
        fs=bformat.fs,
        window_ms=window_ms,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
    )

    # Reference the time axis to the detected onset (t=0 = transient).
    time_ms = time_ms - onset_ms

    # --------------------------------------------------------------
    # Step 4 — Intensity magnitude
    # --------------------------------------------------------------

    magnitude = np.sqrt(
        Ix**2 +
        Iy**2 +
        Iz**2
    )

    # --------------------------------------------------------------
    # Step 5 — Optional normalization
    # --------------------------------------------------------------

    if normalize_vectors:

        eps = np.finfo(float).eps

        Ix = Ix / (magnitude + eps)
        Iy = Iy / (magnitude + eps)
        Iz = Iz / (magnitude + eps)

    # --------------------------------------------------------------
    # Step 6 — Direction of Arrival
    # --------------------------------------------------------------

    azimuth_deg, elevation_deg = compute_doa(
        Ix,
        Iy,
        Iz,
    )

    # --------------------------------------------------------------
    # Step 7 — Build result object
    # --------------------------------------------------------------

    return DirectionalAnalysis(

        sample_index=sample_index,

        time_ms=time_ms,

        Ix=Ix,
        Iy=Iy,
        Iz=Iz,

        magnitude=magnitude,

        azimuth_deg=azimuth_deg,

        elevation_deg=elevation_deg,

        window_ms=window_ms,

        onset_sample=onset_sample,
        onset_ms=onset_ms,
        end_sample=end_sample,
    )


# ============================================================================
# PLOTS
# ============================================================================
def plot_waveform(
    bformat: BFormatSignals,
    ax=None,
    onset_sample: int = 0,
    end_sample: int | None = None,
):
    """
    Plot the omnidirectional (W) channel of the B-format impulse response.

    If onset_sample / end_sample are provided, only that portion of the
    signal is plotted and the time axis is referenced to the onset
    (t=0 = RIR transient; pre-roll content shown at negative time).
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))

    if end_sample is None:
        end_sample = len(bformat.W)

    samples = np.arange(end_sample)

    onset_ms = onset_sample / bformat.fs * 1000.0

    time_ms = (
        samples
        / bformat.fs
        * 1000.0
        - onset_ms
    )

    ax.plot(
        time_ms,
        bformat.W[:end_sample],
        color="black",
        linewidth=0.8,
    )

    ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_ylabel("Amplitude")
    ax.set_title("Omnidirectional Channel (W)")
    ax.grid(True)

    return ax
# ============================================================================
# COMPLETE DIRECTIONAL PANEL
# ============================================================================

def plot_directional_panel(
    bformat: BFormatSignals,
    result: DirectionalAnalysis,
):
    """
    Complete synchronized visualization of the Room Impulse Response and its
    directional characteristics.

    Layout
    ------
    1. Waveform (W channel)
    2. Time–Azimuth Directional Reflectogram with intensity vectors
    3. Elevation vs Time

    This visualization combines the ideas of:

        • Merimaa & Pulkki (2005) — SIRR
        • Cacavelos, Bonelli & Bidondo (2016)

    allowing simultaneous interpretation of

        - arrival time
        - direction of arrival
        - intensity magnitude
        - direction of acoustic energy flow
    """

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(15, 9),
        sharex=True,
        constrained_layout=True,
    )

    # ------------------------------------------------------------------
    # 0) Relative-intensity magnitude expressed in decibels (dB).
    #
    # Ix = W·X etc. behave as an intensity-like (power) quantity, so a
    # 10·log10 mapping is used. Values are referenced to the maximum
    # magnitude found within the plotted window, and floored at
    # db_floor (default -60 dB) to avoid -inf for near-zero samples.
    # ------------------------------------------------------------------

    db_floor = -60.0
    eps = np.finfo(float).eps

    ref = np.max(result.magnitude)
    ref = ref if ref > 0 else eps

    magnitude_db = 10.0 * np.log10(
        np.maximum(result.magnitude, eps) / ref
    )
    magnitude_db = np.maximum(magnitude_db, db_floor)

    # ------------------------------------------------------------------
    # 1) Waveform
    # ------------------------------------------------------------------

    plot_waveform(
        bformat,
        ax=axs[0],
        onset_sample=result.onset_sample,
        end_sample=result.end_sample,
    )

    # ------------------------------------------------------------------
    # 2) Directional Reflectogram
    # ------------------------------------------------------------------

    ax = axs[1]

    sc = ax.scatter(
        result.time_ms,
        result.azimuth_deg,
        c=magnitude_db,
        cmap="viridis",
        vmin=db_floor,
        vmax=0.0,
        s=12,
        alpha=0.35,
        zorder=1,
    )

    # ----------------------------------------------------------
    # Normalize vectors only for display
    # ----------------------------------------------------------

    Ix = result.Ix / (result.magnitude + eps)
    Iy = result.Iy / (result.magnitude + eps)

    # Vertical component of the arrows
    #
    # Iy controls the position (left/right),
    # while Ix controls the arrow orientation along time.
    #

    scale_y = 10.0       # degrees
    scale_x = 0.7        # ms

    ax.quiver(

        result.time_ms,

        result.azimuth_deg,

        scale_x * Ix,

        scale_y * Iy,

        magnitude_db,

        cmap="viridis",

        clim=(db_floor, 0.0),

        angles="xy",

        scale_units="xy",

        scale=1,

        width=0.0022,

        headwidth=3,

        headlength=4,

        headaxislength=3.5,

        pivot="middle",

        zorder=2,

    )

    ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_ylabel("Azimuth (°)")

    ax.set_ylim(-180, 180)

    ax.set_title(
        "Directional Reflectogram (Time–Azimuth–Intensity)"
    )

    ax.grid(True)

    plt.colorbar(
        sc,
        ax=ax,
        label="Relative Intensity (dB)",
    )

    # ------------------------------------------------------------------
    # 3) Elevation
    # ------------------------------------------------------------------

    ax = axs[2]

    sc2 = ax.scatter(

        result.time_ms,

        result.elevation_deg,

        c=magnitude_db,

        cmap="viridis",

        vmin=db_floor,

        vmax=0.0,

        s=12,

        alpha=0.7,

    )

    ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.set_ylabel("Elevation (°)")

    ax.set_xlabel("Time relative to onset (ms)")

    ax.set_ylim(-90, 90)

    ax.grid(True)

    plt.colorbar(
        sc2,
        ax=ax,
        label="Relative Intensity (dB)",
    )

    # ------------------------------------------------------------------
    # Shared time axis: limited to the analyzed window
    # (onset - preroll  →  end_sample), t=0 at the RIR transient.
    # ------------------------------------------------------------------

    axs[0].set_xlim(
        result.time_ms.min(),
        result.time_ms.max(),
    )

    fig.suptitle(
        "3D Directional Analysis of the Room Impulse Response",
        fontsize=15,
        fontweight="bold",
    )

    return fig, axs