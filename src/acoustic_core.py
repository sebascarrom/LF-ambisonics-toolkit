"""
acoustic_core.py
================
Core processing module for Room Impulse Response (RIR) analysis.
Implements A-format to B-format conversion and Early Lateral Energy
Fraction (LF) calculation per octave band according to ISO 3382-1.

Methodological notes
--------------------
- A→B conversion uses the standard SoundField tetrahedral matrix
  (frequency-independent approximation, valid for first-order arrays
  with standard geometry: azimuth ±45°/±135°, elevation ±35.26°).
- Octave-band filtering uses zero-phase Butterworth bandpass filters
  (sosfiltfilt) to avoid phase distortion in the integration windows.
- Onset detection is performed on the broadband W channel before
  filtering, so that the ISO time windows [0–80 ms] and [5–80 ms]
  are referenced to the actual arrival of the direct sound.
- LF is computed independently per octave band; the reported mean
  follows the arithmetic average convention over valid bands.

References
----------
ISO 3382-1:2009 — Acoustics — Measurement of room acoustic parameters —
    Part 1: Performance spaces.
Barron, M. & Marshall, A.H. (1981). Spatial impression due to early
    lateral reflections in concert halls. JSV, 77(2), 211–232.
Bradley, J.S. (2011). Review of objective room acoustics measures and
    future needs. Applied Acoustics, 72, 713–720.
"""

import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# ─── Constants ────────────────────────────────────────────────────────────────

# ISO 3382-1 octave band center frequencies (Hz)
OCTAVE_BANDS_HZ: list[int] = [125, 250, 500, 1000, 2000, 4000]

# ISO 3382-1 LF integration windows (ms)
LF_T1_MS: float = 5.0    # start of lateral energy window
LF_T2_MS: float = 80.0   # end of both windows (common upper limit)

# Butterworth filter order (effective order doubles with sosfiltfilt)
FILTER_ORDER: int = 3

# Onset detection: threshold below peak energy (dB)
ONSET_THRESHOLD_DB: float = -20.0


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class BFormatSignals:
    """
    First-order B-format (Ambisonic) signals decoded from A-format.

    Channels
    --------
    W : omnidirectional pressure component (zeroth order)
    X : front-back figure-8 velocity component
    Y : left-right figure-8 velocity component  ← used for LF
    Z : up-down figure-8 velocity component
    fs : sample rate in Hz
    """
    W:  np.ndarray
    X:  np.ndarray
    Y:  np.ndarray
    Z:  np.ndarray
    fs: int


@dataclass
class LFResult:
    """
    Result container for a single LF measurement.

    Attributes
    ----------
    LF_per_band : dict {center_freq_hz (int): LF_value (float)}
                  LF value per octave band; NaN if band was skipped.
    LF_mean     : arithmetic mean of LF over all valid bands.
    onset_sample: detected RIR onset (sample index, pre-filtering).
    onset_ms    : onset time offset from sample 0 in milliseconds.
    fs          : sample rate in Hz.
    bands_hz    : list of center frequencies used in analysis.
    """
    LF_per_band:  dict
    LF_mean:      float
    onset_sample: int
    onset_ms:     float
    fs:           int
    bands_hz:     list = field(default_factory=lambda: OCTAVE_BANDS_HZ.copy())

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = ["─" * 42,
                 f"  Onset detected at : {self.onset_ms:.2f} ms",
                 f"  {'Band (Hz)':<12} {'LF':>8}",
                 "  " + "─" * 22]
        for fc in self.bands_hz:
            val = self.LF_per_band.get(fc, float('nan'))
            flag = "  ← mid" if fc in (500, 1000) else ""
            lines.append(f"  {fc:<12} {val:>8.4f}{flag}")
        lines += ["  " + "─" * 22,
                  f"  {'Mean LF':<12} {self.LF_mean:>8.4f}",
                  "─" * 42]
        return "\n".join(lines)


# ─── Module 1: A-format → B-format conversion ─────────────────────────────────

# Channel name aliases so the same conversion function accepts both
# naming conventions (Usina SP200 vs Catedral Soyuz 013).
CHANNEL_ALIASES: dict[str, str] = {
    # SoundField SP200 (Usina del Arte)
    "LeftFront":  "LF",
    "LeftBack":   "LB",
    "RightFront": "RF",
    "RightBack":  "RB",
    # Soyuz 013 Ambisonic (Catedral Metropolitana)
    "FRONT L UP":   "LF",
    "FRONT R DOWN": "RF",
    "BACK L DOWN":  "LB",
    "BACK R UP":    "RB",
    # Already canonical
    "LF": "LF", "RF": "RF", "LB": "LB", "RB": "RB",
}


def normalize_aformat_keys(signals: dict) -> dict:
    """
    Normalize A-format channel names to canonical keys: LF, RF, LB, RB.

    Accepts naming conventions from both SP200 (Usina) and Soyuz 013
    (Catedral). Raises KeyError if an unrecognized key is found.
    """
    normalized = {}
    for key, data in signals.items():
        canonical = CHANNEL_ALIASES.get(key)
        if canonical is None:
            raise KeyError(
                f"Unrecognized A-format channel name: '{key}'. "
                f"Known names: {list(CHANNEL_ALIASES.keys())}"
            )
        normalized[canonical] = data
    for expected in ("LF", "RF", "LB", "RB"):
        if expected not in normalized:
            raise KeyError(f"Missing A-format channel '{expected}' after normalization.")
    return normalized


def aformat_to_bformat(signals_aformat: dict, fs: int) -> BFormatSignals:
    """
    Convert A-format tetrahedral signals to first-order B-format.

    Standard SoundField tetrahedral geometry:
        LF (LeftFront / FRONT L UP) :  azimuth +45°,  elevation +35.26°
        RF (RightFront/ FRONT R DOWN):  azimuth -45°,  elevation -35.26°
        LB (LeftBack  / BACK L DOWN) :  azimuth +135°, elevation -35.26°
        RB (RightBack / BACK R UP)   :  azimuth -135°, elevation +35.26°

    Conversion matrix (frequency-independent approximation):
        W = 0.5 · (LF + RF + LB + RB)   [omnidirectional]
        X = 0.5 · (LF + RF − LB − RB)   [front-back dipole]
        Y = 0.5 · (−LF + RF − LB + RB)  [lateral dipole, used in LF]
        Z = 0.5 · (LF − RF − LB + RB)   [vertical dipole]

    Parameters
    ----------
    signals_aformat : dict with keys 'LF', 'RF', 'LB', 'RB'
                      (or aliases — see CHANNEL_ALIASES).
    fs              : sample rate in Hz.

    Returns
    -------
    BFormatSignals
    """
    signals = normalize_aformat_keys(signals_aformat)

    LF = signals["LF"].astype(np.float64)
    RF = signals["RF"].astype(np.float64)
    LB = signals["LB"].astype(np.float64)
    RB = signals["RB"].astype(np.float64)

    lengths = {len(LF), len(RF), len(LB), len(RB)}
    if len(lengths) != 1:
        raise ValueError(
            f"Channel length mismatch: LF={len(LF)}, RF={len(RF)}, "
            f"LB={len(LB)}, RB={len(RB)}. All channels must be the same length."
        )

    W = 0.5 * (LF + RF + LB + RB)
    X = 0.5 * (LF + RF - LB - RB)
    Y = 0.5 * (-LF + RF - LB + RB)
    Z = 0.5 * (LF - RF - LB + RB)

    return BFormatSignals(W=W, X=X, Y=Y, Z=Z, fs=fs)


# ─── Module 2: RIR onset detection ────────────────────────────────────────────

def detect_onset(signal_w: np.ndarray,
                 threshold_db: float = ONSET_THRESHOLD_DB) -> int:
    """
    Detect the onset of the RIR from the W (omnidirectional) channel.

    Strategy: find the first sample whose squared amplitude exceeds
    a threshold relative to the signal's peak squared amplitude.
    This is applied to the broadband (unfiltered) W channel before
    octave-band processing, ensuring a single consistent time reference
    for all bands.

    Parameters
    ----------
    signal_w     : W channel array (broadband, float64).
    threshold_db : threshold in dB below peak energy (default -20 dB).
                   Lower values → earlier (more sensitive) detection.
                   Higher values → later (more conservative) detection.

    Returns
    -------
    onset_sample : integer sample index of the detected onset.

    Raises
    ------
    ValueError if no sample exceeds the threshold (signal may be silent
    or entirely noise).

    KNOWN LIMITATION — read before trusting this on real recordings
    -------------------------------------------------------------------
    This detector is relative to the GLOBAL PEAK of the signal. It
    silently assumes the loudest moment in the file IS the direct
    sound. If a later, non-acoustic transient (an edit/punch-in
    artifact from a DAW-extracted segment, a handling noise, a louder
    specular reflection) is louder than the genuine direct sound, this
    function will lock onto that later event instead — every threshold
    value will agree with each other (all relative to the same wrong
    peak), which can look like "stability" while being wrong. This was
    observed empirically: see detect_onset_noise_floor() below for a
    more robust alternative, and ALWAYS visually inspect the waveform
    (see debug_LF.py) before trusting either detector on new data.
    """
    energy = signal_w ** 2
    peak_energy = np.max(energy)

    if peak_energy == 0:
        raise ValueError("W channel is silent (all zeros). Cannot detect onset.")

    threshold_linear = peak_energy * (10.0 ** (threshold_db / 10.0))
    candidates = np.where(energy >= threshold_linear)[0]

    if len(candidates) == 0:
        raise ValueError(
            f"No sample exceeded the onset threshold ({threshold_db} dB). "
            "Try lowering |threshold_db|."
        )

    return int(candidates[0])


def detect_onset_noise_floor(signal_w: np.ndarray,
                              fs: int,
                              noise_window_ms: float = 50.0,
                              threshold_db: float = 15.0,
                              frame_ms: float = 1.0,
                              min_consecutive_frames: int = 3) -> int:
    """
    Robust onset detection based on rise above the estimated NOISE FLOOR,
    instead of relative to the global peak (see detect_onset() limitation).

    Rationale
    ---------
    detect_onset() fails silently when a later, non-acoustic event in the
    recording is louder than the genuine direct sound — common in RIR
    segments manually cut from a longer multitrack DAW session, where
    edit boundaries or stray transients can be the loudest broadband
    event in the file.

    This alternative instead:
      1. Estimates the noise floor from the first `noise_window_ms` of
         the signal (assumed to be silence/pre-roll — verify this
         assumption visually before trusting the result).
      2. Computes short-time energy in non-overlapping frames of
         `frame_ms`.
      3. Finds the FIRST frame whose energy exceeds
         (noise_floor + threshold_db) AND stays above that level for at
         least `min_consecutive_frames` — this sustained-rise requirement
         filters out brief single-frame clicks/glitches that aren't
         genuine sustained acoustic events (a real direct sound is
         followed by reflections/decay, not silence).

    Parameters
    ----------
    signal_w                : W channel (broadband).
    fs                      : sample rate in Hz.
    noise_window_ms         : duration used to estimate the noise floor,
                              taken from the very start of the signal.
    threshold_db            : how far above the noise floor (dB) a frame
                              must rise to be considered the onset.
    frame_ms                : analysis frame size for short-time energy.
    min_consecutive_frames  : consecutive frames that must stay above
                              threshold to accept the detection (filters
                              isolated clicks/artifacts).

    Returns
    -------
    onset_sample : sample index of the start of the first qualifying frame.

    Raises
    ------
    ValueError if no sustained rise above the noise floor is found, or if
    noise_window_ms exceeds the signal length.
    """
    frame_len = max(1, int(frame_ms / 1000 * fs))
    noise_len = int(noise_window_ms / 1000 * fs)

    if noise_len >= len(signal_w):
        raise ValueError(
            f"noise_window_ms ({noise_window_ms} ms) is longer than the "
            "signal itself."
        )

    noise_floor_energy = np.mean(signal_w[:noise_len] ** 2)
    if noise_floor_energy == 0:
        noise_floor_energy = 1e-12  # avoid div-by-zero / -inf dB

    threshold_linear = noise_floor_energy * (10.0 ** (threshold_db / 10.0))

    n_frames = len(signal_w) // frame_len
    if n_frames < min_consecutive_frames:
        raise ValueError("Signal too short for the requested frame settings.")

    frame_energies = np.array([
        np.mean(signal_w[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    above = frame_energies >= threshold_linear

    for i in range(n_frames - min_consecutive_frames + 1):
        if np.all(above[i:i + min_consecutive_frames]):
            return i * frame_len

    raise ValueError(
        f"No se encontró una subida sostenida de al menos "
        f"{min_consecutive_frames} frame(s) de {frame_ms} ms por encima "
        f"de {threshold_db} dB sobre el piso de ruido estimado "
        f"(de los primeros {noise_window_ms} ms). Probá ajustar "
        "threshold_db, noise_window_ms, o verificar visualmente si los "
        "primeros ms del archivo son realmente silencio."
    )


# ─── Module 3: Octave-band Butterworth filtering ──────────────────────────────

def octave_band_filter(signal_in: np.ndarray,
                       center_freq_hz: float,
                       fs: int,
                       order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply a zero-phase octave-band Butterworth bandpass filter.

    Cutoff frequencies follow the standard octave-band definition:
        f_low  = f_center / sqrt(2)
        f_high = f_center * sqrt(2)

    Zero-phase filtering (sosfiltfilt) is used to preserve the temporal
    structure of the RIR and avoid shifting energy across the integration
    windows. This is critical for the LF calculation.

    Parameters
    ----------
    signal_in      : input signal array (float64).
    center_freq_hz : octave band center frequency in Hz.
    fs             : sample rate in Hz.
    order          : Butterworth filter order (effective order = 2*order
                     after forward-backward filtering).

    Returns
    -------
    Bandpass-filtered signal (same length as input).

    Raises
    ------
    ValueError if f_high >= Nyquist (band not representable at this fs).
    """
    nyquist = fs / 2.0
    f_low  = center_freq_hz / np.sqrt(2)
    f_high = center_freq_hz * np.sqrt(2)

    if f_high >= nyquist:
        raise ValueError(
            f"Octave band {center_freq_hz} Hz: upper cutoff {f_high:.1f} Hz "
            f"exceeds or equals Nyquist ({nyquist:.1f} Hz) for fs={fs} Hz. "
            "Increase sample rate or exclude this band."
        )

    if f_low <= 0:
        raise ValueError(
            f"Octave band {center_freq_hz} Hz: lower cutoff {f_low:.2f} Hz is invalid."
        )

    sos = sig.butter(
        order,
        [f_low / nyquist, f_high / nyquist],
        btype="bandpass",
        output="sos"
    )
    return sig.sosfiltfilt(sos, signal_in)


# ─── Module 4: LF calculation (single band) ───────────────────────────────────

def compute_LF_band(W_band: np.ndarray,
                    Y_band: np.ndarray,
                    fs: int,
                    onset: int,
                    t1_ms: float = LF_T1_MS,
                    t2_ms: float = LF_T2_MS) -> float:
    """
    Calculate the Early Lateral Energy Fraction (LF) for one octave band.

    Definition (ISO 3382-1, Annex A):
        LF = ∫[t1, t2] p_Y²(t) dt  /  ∫[0, t2] p_W²(t) dt

    where:
        - p_Y : band-filtered Y channel (lateral figure-8 response)
        - p_W : band-filtered W channel (omnidirectional response)
        - t=0 : onset of the direct sound (corrected via onset parameter)
        - t1  : 5 ms  (excludes direct sound contribution)
        - t2  : 80 ms (upper limit of early reflections)

    Both p_Y and p_W are squared (energy), so the sign of the dipole
    response does not affect the result — negative values contribute
    equally to the integral after squaring.

    Parameters
    ----------
    W_band : octave-band filtered W channel (omnidirectional).
    Y_band : octave-band filtered Y channel (lateral figure-8).
    fs     : sample rate in Hz.
    onset  : sample index of RIR onset (from detect_onset, broadband).
    t1_ms  : start of lateral energy window in ms (default 5 ms).
    t2_ms  : common upper integration limit in ms (default 80 ms).

    Returns
    -------
    LF value (float, dimensionless). Typical range: 0.0 – 0.35 in
    concert halls. Returns np.nan if W energy is zero.
    """
    i_onset = onset
    i1 = i_onset + int(round((t1_ms / 1000.0) * fs))
    i2 = i_onset + int(round((t2_ms / 1000.0) * fs))

    sig_len = min(len(W_band), len(Y_band))

    if i2 > sig_len:
        raise ValueError(
            f"Upper integration limit (onset + {t2_ms} ms = sample {i2}) "
            f"exceeds signal length ({sig_len} samples = "
            f"{sig_len / fs * 1000:.1f} ms). RIR may be too short."
        )

    energy_Y_num = np.sum(Y_band[i1:i2] ** 2)   # numerator:   Y, [5–80 ms]
    energy_W_den = np.sum(W_band[i_onset:i2] ** 2)  # denominator: W, [0–80 ms]

    if energy_W_den == 0.0:
        return float("nan")

    return float(energy_Y_num / energy_W_den)


# ─── Module 5: Full per-band LF pipeline ──────────────────────────────────────

def compute_LF_fullband(bformat: BFormatSignals,
                         bands_hz: list = None,
                         t1_ms:    float = LF_T1_MS,
                         t2_ms:    float = LF_T2_MS,
                         onset_threshold_db: float = ONSET_THRESHOLD_DB
                         ) -> LFResult:
    """
    Full LF analysis pipeline: onset detection → octave filtering → LF per band.

    Parameters
    ----------
    bformat             : BFormatSignals object (W, X, Y, Z + fs).
    bands_hz            : octave band center frequencies to analyze.
                          Default: [125, 250, 500, 1000, 2000, 4000] Hz.
    t1_ms / t2_ms       : ISO 3382-1 integration window limits in ms.
    onset_threshold_db  : energy threshold for onset detection (dB below peak).

    Returns
    -------
    LFResult with per-band values, arithmetic mean, and onset metadata.
    """
    if bands_hz is None:
        bands_hz = OCTAVE_BANDS_HZ

    # Step 1 — Detect onset on broadband W (single reference for all bands)
    onset = detect_onset(bformat.W, threshold_db=onset_threshold_db)
    onset_ms = (onset / bformat.fs) * 1000.0

    # Step 2 — Per-band filtering and LF computation
    LF_per_band: dict[int, float] = {}

    for fc in bands_hz:
        try:
            W_band = octave_band_filter(bformat.W, fc, bformat.fs)
            Y_band = octave_band_filter(bformat.Y, fc, bformat.fs)
            lf_val = compute_LF_band(
                W_band, Y_band, bformat.fs, onset, t1_ms, t2_ms
            )
            LF_per_band[fc] = lf_val
        except ValueError as e:
            print(f"  [WARNING] Band {fc} Hz skipped — {e}")
            LF_per_band[fc] = float("nan")

    # Step 3 — Arithmetic mean over valid bands
    valid = [v for v in LF_per_band.values() if not np.isnan(v)]
    LF_mean = float(np.mean(valid)) if valid else float("nan")

    return LFResult(
        LF_per_band=LF_per_band,
        LF_mean=LF_mean,
        onset_sample=onset,
        onset_ms=onset_ms,
        fs=bformat.fs,
        bands_hz=list(bands_hz),
    )


# ─── Module 6: I/O — load A-format signals ────────────────────────────────────

def _normalize_wav(data: np.ndarray) -> np.ndarray:
    """
    Normalize a WAV array to float64 in the range [-1, 1].

    scipy.io.wavfile returns:
      - int16  → divide by 2**15  (32768)
      - int32  → divide by 2**31  (2147483648)
      - float32/float64 → already in [-1, 1], just cast
    """
    if data.dtype == np.int16:
        return data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        return data.astype(np.float64) / 2147483648.0
    else:
        return data.astype(np.float64)  # float32 / float64: passthrough

def load_aformat_mono_files(paths: dict) -> tuple[dict, int]:
    """
    Load A-format from 4 separate mono WAV files.

    Parameters
    ----------
    paths : dict mapping channel names to file paths.
            Keys can use any naming convention supported by CHANNEL_ALIASES.
            Example (Usina):
                {"LF": "P1_LF.wav", "RF": "P1_RF.wav",
                 "LB": "P1_LB.wav", "RB": "P1_RB.wav"}
            Example (Catedral):
                {"FRONT L UP":   "P1_FLU.wav",
                 "FRONT R DOWN": "P1_FRD.wav",
                 "BACK L DOWN":  "P1_BLD.wav",
                 "BACK R UP":    "P1_BRU.wav"}

    Returns
    -------
    (signals_dict, fs) — dict with canonical keys LF/RF/LB/RB, sample rate.
    """
    signals: dict[str, np.ndarray] = {}
    fs_set:  set[int] = set()

    for key, path in paths.items():
        fs_i, data = wav.read(str(path))
        fs_set.add(fs_i)
        if data.ndim > 1:
            raise ValueError(
                f"File '{path}' has {data.shape[1]} channels. "
                "Expected mono (single-channel) WAV."
            )
        signals[key] = _normalize_wav(data)

    if len(fs_set) != 1:
        raise ValueError(
            f"Sample rate mismatch across files: {fs_set}. "
            "All channels must share the same sample rate."
        )

    return signals, fs_set.pop()


def load_aformat_multichannel(path: Union[str, Path],
                               channel_order: list = None) -> tuple[dict, int]:
    """
    Load A-format from a single 4-channel WAV file.

    Parameters
    ----------
    path          : path to 4-channel WAV file.
    channel_order : list of 4 channel names in track order.
                    Default: ['LF', 'RF', 'LB', 'RB'].
                    Can use any names supported by CHANNEL_ALIASES.

    Returns
    -------
    (signals_dict, fs) — dict with canonical keys LF/RF/LB/RB, sample rate.
    """
    if channel_order is None:
        channel_order = ["LF", "RF", "LB", "RB"]

    if len(channel_order) != 4:
        raise ValueError(
            f"channel_order must have exactly 4 entries, got {len(channel_order)}."
        )

    fs, data = wav.read(str(path))

    if data.ndim == 1:
        raise ValueError("WAV file is mono. Expected 4-channel interleaved file.")
    if data.shape[1] != 4:
        raise ValueError(
            f"WAV file has {data.shape[1]} channels. Expected 4 (A-format)."
        )

    signals = {
        name: _normalize_wav(data[:, i])
        for i, name in enumerate(channel_order)
    }
    return signals, fs


# ─── Convenience entry point ──────────────────────────────────────────────────

def analyze_rir(source: Union[str, Path, dict],
                channel_order: list = None,
                bands_hz:      list = None) -> LFResult:
    """
    Full pipeline entry point: load → convert → detect onset → filter → LF.

    Parameters
    ----------
    source        : str/Path → 4-channel WAV file.
                    dict     → {channel_name: file_path} for 4 mono WAV files.
    channel_order : channel ordering (only used for multichannel WAV).
    bands_hz      : octave band center frequencies (default: ISO standard set).

    Returns
    -------
    LFResult

    Examples
    --------
    # From a 4-channel WAV (Usina, standard order):
    result = analyze_rir("P1_4ch.wav")
    print(result.summary())

    # From 4 mono files (Catedral naming):
    result = analyze_rir({
        "FRONT L UP":   "meas/P2_FLU.wav",
        "FRONT R DOWN": "meas/P2_FRD.wav",
        "BACK L DOWN":  "meas/P2_BLD.wav",
        "BACK R UP":    "meas/P2_BRU.wav",
    })
    print(result.summary())
    """
    if isinstance(source, dict):
        signals, fs = load_aformat_mono_files(source)
    else:
        signals, fs = load_aformat_multichannel(source, channel_order)

    bformat = aformat_to_bformat(signals, fs)
    return compute_LF_fullband(bformat, bands_hz=bands_hz)


# ─── Module 7: B-format export (for cross-validation in EASERA, etc.) ─────────

def export_bformat_wav(bformat: BFormatSignals,
                       output_path: Union[str, Path],
                       layout: str = "interleaved",
                       order: str = "WYZX",
                       fuma_normalize_w: bool = False) -> Path:
    """
    Export decoded B-format signals to WAV file(s) for cross-validation
    against third-party software (e.g. EASERA, dEQ, Soundfield plugins).

    Two export layouts are supported:

    - 'interleaved' : a single 4-channel WAV file, channel order set by
                      `order`. This is the layout most B-format-aware
                      software (including EASERA's Ambisonic/Soundfield
                      analysis module) expects for direct import.
    - 'separate'    : four mono WAV files (suffixed _W, _X, _Y, _Z),
                      useful if the target software requires per-channel
                      import or if you want to inspect each channel
                      independently (e.g. in an audio editor).

    Bit depth: files are written as 32-bit float PCM. This avoids
    clipping and quantization loss — the A→B conversion can produce
    sample values outside the original A-format's amplitude range
    (sums of 4 channels scaled by 0.5), and float WAV has no fixed
    full-scale ceiling. Most modern DAWs and analysis tools (including
    EASERA) read 32-bit float WAV natively.

    IMPORTANT — channel order / normalization caveat:
    This export writes the *unweighted* FuMa-style W,X,Y,Z signals as
    produced by aformat_to_bformat() (no -3 dB pad on W, no SN3D/N3D
    renormalization). If your comparison software assumes a specific
    Ambisonic convention (FuMa vs AmbiX/ACN-SN3D), verify its channel
    order and gain convention before comparing levels directly — LF
    itself is a ratio (energy of Y over energy of W) and is therefore
    invariant to a uniform overall gain, but it is NOT invariant to a
    relative gain mismatch between W and Y, so this matters if you're
    cross-checking absolute LF values rather than just waveform shapes.

    Parameters
    ----------
    bformat     : BFormatSignals to export.
    output_path : for layout='interleaved', the full path of the output
                  .wav file (e.g. "P4_Bformat.wav").
                  for layout='separate', the path is used as a prefix
                  (e.g. "P4_Bformat.wav" → "P4_Bformat_W.wav", ..._X.wav, ...).
    layout      : 'interleaved' (default) or 'separate'.
    order       : channel order string for the interleaved file, using
                  letters W/X/Y/Z. Default 'WXYZ' (SoundField/FuMa-style
                  raw order). Use 'WYZX' if your target software expects
                  AmbiX/ACN channel ordering (note: AmbiX additionally
                  expects SN3D-normalized X/Y/Z gains, NOT applied here).

    Returns
    -------
    Path (interleaved) or list[Path] (separate) of the file(s) written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Aplicar normalización FuMa en W si se solicita:
    # W_FuMa = W * (1/√2) — reduce W 3 dB para igualar la convención de la
    # mayoría de los plugins y software de análisis (EASERA, Reaper, etc.).
    # Esto es necesario para que el LF calculado externamente coincida con el
    # nuestro, ya que LF = Y²/W² y la escala relativa W/Y sí importa.
    W = bformat.W * (1.0 / np.sqrt(2)) if fuma_normalize_w else bformat.W
    channels = {"W": W, "X": bformat.X, "Y": bformat.Y, "Z": bformat.Z}

    if layout == "separate":
        written = []
        for ch_letter, data in channels.items():
            ch_path = output_path.with_name(
                f"{output_path.stem}_{ch_letter}{output_path.suffix or '.wav'}"
            )
            wav.write(str(ch_path), bformat.fs, data.astype(np.float32))
            written.append(ch_path)
        return written

    elif layout == "interleaved":
        if sorted(order) != sorted("WXYZ"):
            raise ValueError(
                f"order must be a permutation of 'WXYZ', got '{order}'."
            )
        stacked = np.stack([channels[letter] for letter in order], axis=1)
        wav.write(str(output_path), bformat.fs, stacked.astype(np.float32))
        return output_path

    else:
        raise ValueError(
            f"layout must be 'interleaved' or 'separate', got '{layout}'."
        )


# ─── Module 8: Azimuth rotation diagnostic (orientation misalignment) ─────────

def scan_azimuth_rotation(bformat: BFormatSignals,
                          onset: int,
                          bands_hz: list = None,
                          angles_deg: np.ndarray = None) -> tuple:
    """
    Diagnostic: scan a horizontal-plane rotation angle φ and recompute the
    mean LF at each angle, to test whether the microphone's assumed
    orientation (front axis pointing at the source) matches reality.

    Rationale
    ---------
    aformat_to_bformat() assumes the SP200/Soyuz tetrahedral geometry is
    aligned with the source: X = front-back (relative to source), Y =
    left-right (relative to source/listener). If the physical microphone
    was rotated by some angle θ relative to that assumed alignment during
    the measurement, the computed Y channel mixes true lateral and true
    frontal energy, and LF will be systematically wrong — NOT due to any
    bug, but due to a geometric calibration mismatch between the mic's
    physical orientation and the source-receiver axis.

    Because X and Y form a true 2D vector pair under first-order
    Ambisonics, a rotation by angle φ of the horizontal plane is exactly:
        Y_rot(φ) = X·sin(φ) + Y·cos(φ)
        X_rot(φ) = X·cos(φ) − Y·sin(φ)
    (This is the same transform applied to filtered X/Y per band — linear
    time-invariant filtering commutes with this fixed-coefficient mix, so
    filtering once and rotating is equivalent to rotating then filtering,
    and much cheaper.)

    If LF(φ) varies substantially across φ and peaks/stabilizes at some
    consistent non-zero angle across most octave bands, that is evidence
    of a real orientation mismatch — and the angle that recovers the
    expected/literature LF value is an estimate of the actual mic
    rotation offset for that measurement. If LF(φ) is roughly flat across
    all φ, an orientation mismatch is NOT a good explanation, and the
    issue likely lies elsewhere (channel mislabeling, mic defect, etc.)

    Parameters
    ----------
    bformat    : BFormatSignals (uses W, X, Y; Z unused for LF).
    onset      : sample index of the RIR onset (use the validated onset,
                 e.g. from detect_onset_noise_floor).
    bands_hz   : octave bands to average over (default: ISO standard set).
    angles_deg : array of angles in degrees to scan (default: 0-350 in
                 10° steps).

    Returns
    -------
    (angles_deg, mean_lf_per_angle, lf_per_band_per_angle)
        angles_deg            : the angles scanned, in degrees.
        mean_lf_per_angle     : mean LF across bands_hz at each angle.
        lf_per_band_per_angle : dict {band_hz: array of LF per angle},
                                for inspecting per-band stability.
    """
    if bands_hz is None:
        bands_hz = OCTAVE_BANDS_HZ
    if angles_deg is None:
        angles_deg = np.arange(0, 360, 10)

    # Filter W, X, Y once per band (rotation is applied AFTER filtering —
    # mathematically equivalent, much cheaper than re-filtering per angle).
    Wf_bands = {fc: octave_band_filter(bformat.W, fc, bformat.fs) for fc in bands_hz}
    Xf_bands = {fc: octave_band_filter(bformat.X, fc, bformat.fs) for fc in bands_hz}
    Yf_bands = {fc: octave_band_filter(bformat.Y, fc, bformat.fs) for fc in bands_hz}

    lf_per_band_per_angle = {fc: [] for fc in bands_hz}
    mean_lf_per_angle = []

    for theta_deg in angles_deg:
        theta = np.deg2rad(theta_deg)
        lf_vals = []
        for fc in bands_hz:
            Y_rot = Xf_bands[fc] * np.sin(theta) + Yf_bands[fc] * np.cos(theta)
            lf = compute_LF_band(Wf_bands[fc], Y_rot, bformat.fs, onset)
            lf_per_band_per_angle[fc].append(lf)
            lf_vals.append(lf)
        mean_lf_per_angle.append(np.nanmean(lf_vals))

    lf_per_band_per_angle = {fc: np.array(v) for fc, v in lf_per_band_per_angle.items()}
    return angles_deg, np.array(mean_lf_per_angle), lf_per_band_per_angle


# ─── Module 9: Automatic azimuth alignment (no physical aiming required) ──────

def estimate_direct_sound_azimuth(bformat: BFormatSignals,
                                  onset: int,
                                  window_ms: float = 5.0) -> float:
    """
    Estimate the azimuth of the direct sound's arrival direction from the
    B-format active sound intensity vector (Merimaa & Pulkki method),
    computed over a short window right after onset.

    Why this matters
    -----------------
    This eliminates the need for the microphone to be physically aimed at
    the source. Per Protheroe (2015), "Lateral Fraction Measurements with
    a 3D Microphone Array" (in this project's references): professional
    3-D LF measurement systems (IRIS/TetraMic) do NOT require accurate
    physical aiming — "the software identifies the horizontal direction
    of the direct sound using a sound intensity technique and then
    synthesises a horizontal figure-of-8 microphone with the null in this
    direction." This function implements that same principle. Dick &
    Vigeant (2016, also in this project's references) describe an
    equivalent beamforming-based approach for spherical arrays.

    Method
    ------
    Active intensity (horizontal components), proportional to the
    time-averaged product of the omnidirectional and each dipole signal
    over the direct sound window:
        Ix = mean(W · X)
        Iy = mean(W · Y)
        azimuth = atan2(Iy, Ix)

    Parameters
    ----------
    bformat   : BFormatSignals.
    onset     : sample index of the RIR onset (use a validated onset,
                e.g. from detect_onset_noise_floor).
    window_ms : duration after onset used to estimate the direct sound
                direction. Default 5 ms — the ISO 3382-1 t1, capturing
                only the direct sound before early reflections arrive.

    Returns
    -------
    azimuth_rad : estimated azimuth of the direct sound, in radians,
                  relative to the X axis (the mic's nominal "front").
                  0 means the direct sound already arrives exactly along
                  X — no correction needed.

    Raises
    ------
    ValueError if the window exceeds the signal length, or if intensity
    is exactly zero (silent direct-sound window).
    """
    i0 = onset
    i1 = onset + int(window_ms / 1000.0 * bformat.fs)

    if i1 > len(bformat.W):
        raise ValueError(
            f"Window [onset, onset+{window_ms}ms] (sample {i0}:{i1}) "
            f"exceeds signal length ({len(bformat.W)} samples)."
        )

    W_seg = bformat.W[i0:i1]
    X_seg = bformat.X[i0:i1]
    Y_seg = bformat.Y[i0:i1]

    Ix = float(np.mean(W_seg * X_seg))
    Iy = float(np.mean(W_seg * Y_seg))

    if Ix == 0.0 and Iy == 0.0:
        raise ValueError(
            "Active intensity is zero in the direct sound window — "
            "cannot estimate azimuth (check onset / window placement)."
        )

    return float(np.arctan2(Iy, Ix))


def rotate_horizontal_bformat(bformat: BFormatSignals,
                              azimuth_rad: float) -> BFormatSignals:
    """
    Rotate the horizontal plane (X, Y) of a B-format signal so the new X
    axis points exactly at the direct sound direction and the new Y axis
    is the correctly-aligned lateral dipole. W and Z are unchanged.

    Parameters
    ----------
    bformat     : BFormatSignals to rotate.
    azimuth_rad : azimuth to correct for, typically the output of
                  estimate_direct_sound_azimuth(). The signal is rotated
                  by -azimuth_rad to bring the direct sound onto the X axis.

    Returns
    -------
    New BFormatSignals with corrected X, Y (W, Z, fs unchanged).
    """
    theta = -azimuth_rad
    X_rot = bformat.X * np.cos(theta) - bformat.Y * np.sin(theta)
    Y_rot = bformat.X * np.sin(theta) + bformat.Y * np.cos(theta)
    return BFormatSignals(W=bformat.W, X=X_rot, Y=Y_rot, Z=bformat.Z, fs=bformat.fs)


def auto_align_bformat(bformat: BFormatSignals,
                       onset: int,
                       window_ms: float = 5.0) -> tuple:
    """
    Convenience wrapper: estimate the direct sound azimuth and apply the
    rotation in one step.

    Returns
    -------
    (aligned_bformat, azimuth_rad) — the rotated BFormatSignals and the
    azimuth (radians) that was corrected for, so callers can log/report
    the correction angle actually applied for each measurement.
    """
    azimuth_rad = estimate_direct_sound_azimuth(bformat, onset, window_ms)
    aligned = rotate_horizontal_bformat(bformat, azimuth_rad)
    return aligned, azimuth_rad


# ─── Module 9: A-format channel balance diagnostic ────────────────────────────

def check_aformat_channel_balance(signals_aformat: dict,
                                   fs: int,
                                   onset: int,
                                   window_ms: float = 80.0) -> dict:
    """
    Diagnostic: compute relative RMS levels of the 4 A-format capsules
    within the RIR window, to check for a hardware gain mismatch
    (damaged cable, loose connector, miscalibrated preamp channel)
    BEFORE attributing an apparent azimuth misalignment (see
    scan_azimuth_rotation) to genuine microphone-source geometry.

    Rationale
    ---------
    A single capsule with anomalous gain propagates into BOTH the X and
    Y dipole channels (each A-format capsule contributes to every
    B-format channel with a ±0.5 coefficient), and can produce a
    band-consistent apparent "rotation" in scan_azimuth_rotation that is
    actually a hardware artifact, not a geometric one. If the 4 capsules
    are reasonably balanced (within ~1-2 dB of each other, the typical
    matching tolerance for this kind of tetrahedral array) a hardware
    gain issue is unlikely, and a genuine orientation misalignment
    becomes the more probable explanation.

    Parameters
    ----------
    signals_aformat : dict with the 4 A-format channels (canonical or
                      aliased names — see CHANNEL_ALIASES).
    fs              : sample rate in Hz.
    onset           : sample index of the RIR onset.
    window_ms       : duration of the analysis window from onset
                      (default 80 ms, matching the LF integration window).

    Returns
    -------
    dict {channel: rms_relative_db}, where rms_relative_db is each
    capsule's RMS level in dB relative to the mean RMS of all 4 capsules
    (0 dB = exactly average; positive = louder than average).
    """
    signals = normalize_aformat_keys(signals_aformat)
    i2 = onset + int(window_ms / 1000.0 * fs)

    rms = {}
    for ch in ("LF", "RF", "LB", "RB"):
        seg = signals[ch][onset:i2].astype(np.float64)
        rms[ch] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) > 0 else 0.0

    mean_rms = np.mean(list(rms.values()))
    if mean_rms == 0:
        return {ch: float("nan") for ch in rms}

    rel_db = {ch: 20.0 * np.log10(v / mean_rms) if v > 0 else float("-inf")
              for ch, v in rms.items()}
    return rel_db


# ─── Module 10: A-format channel alignment (fixes desynchronized exports) ─────

def detect_onsets_per_channel(signals_aformat: dict,
                               fs: int,
                               noise_window_ms: float = 50.0,
                               threshold_db: float = 15.0,
                               frame_ms: float = 1.0,
                               min_consecutive_frames: int = 3) -> dict:
    """
    Run detect_onset_noise_floor() independently on each of the 4 raw
    A-format channels (BEFORE B-format conversion).

    Rationale
    ---------
    When the 4 capsule signals are exported/extracted as separate mono
    files (e.g. manually cut from a longer multitrack DAW session), each
    file can end up with a different amount of leading silence if the
    cut/export point wasn't pixel-identical across all 4 tracks. The
    matrix conversion (aformat_to_bformat) assumes all 4 channels are
    sample-aligned — if they aren't, W/X/Y/Z become corrupted combinations
    of misaligned signals, and any onset detection run on the resulting W
    will anchor near whichever channel(s) happen to start earliest,
    silently missing real content in the channels that start later.

    Parameters mirror detect_onset_noise_floor().

    Returns
    -------
    dict {channel: onset_sample}, canonical keys LF/RF/LB/RB.
    """
    signals = normalize_aformat_keys(signals_aformat)
    onsets = {}
    for ch in ("LF", "RF", "LB", "RB"):
        onsets[ch] = detect_onset_noise_floor(
            signals[ch].astype(np.float64), fs,
            noise_window_ms=noise_window_ms, threshold_db=threshold_db,
            frame_ms=frame_ms, min_consecutive_frames=min_consecutive_frames
        )
    return onsets


def align_aformat_channels(signals_aformat: dict,
                            fs: int,
                            onsets: dict = None,
                            **onset_kwargs) -> tuple:
    """
    Align the 4 raw A-format channels to a common time reference by
    trimming each channel's leading samples so all 4 onsets coincide.

    This MUST be run before aformat_to_bformat() if the 4 channels were
    exported/extracted separately and might not be sample-aligned (see
    detect_onsets_per_channel() for the rationale).

    Parameters
    ----------
    signals_aformat : dict with the 4 A-format channels (any supported
                      naming convention).
    fs              : sample rate in Hz.
    onsets          : optional pre-computed {channel: onset_sample} dict
                      (from detect_onsets_per_channel). If None, it is
                      computed internally using onset_kwargs.
    **onset_kwargs  : passed to detect_onsets_per_channel() if onsets is
                      not provided (noise_window_ms, threshold_db, etc.)

    Returns
    -------
    (aligned_signals, onsets, common_onset)
        aligned_signals : dict {LF,RF,LB,RB: np.ndarray}, all trimmed to
                          the same length, with onsets coinciding at
                          sample index `common_onset`.
        onsets          : the per-channel onsets that were detected/used.
        common_onset    : the sample index (== min(onsets.values())) at
                          which all 4 channels' onsets now align — pass
                          this directly as the `onset` argument to
                          compute_LF_band()/compute_LF_fullband() instead
                          of re-running detect_onset() on the aligned W.
    """
    signals = normalize_aformat_keys(signals_aformat)

    if onsets is None:
        onsets = detect_onsets_per_channel(signals_aformat, fs, **onset_kwargs)

    common_onset = min(onsets.values())
    shifts = {ch: onsets[ch] - common_onset for ch in onsets}

    trimmed = {ch: signals[ch][shifts[ch]:] for ch in ("LF", "RF", "LB", "RB")}
    min_len = min(len(v) for v in trimmed.values())
    aligned = {ch: trimmed[ch][:min_len] for ch in trimmed}

    return aligned, onsets, common_onset


def fine_align_channels(aligned_signals: dict,
                         fs: int,
                         common_onset: int,
                         reference_channel: str = "LF",
                         search_radius_ms: float = 5.0,
                         correlation_window_ms: float = 30.0) -> tuple:
    """
    Refine a COARSE per-channel alignment (from align_aformat_channels)
    to sample-level precision using cross-correlation against a
    reference channel.

    Why this is needed in addition to align_aformat_channels()
    -------------------------------------------------------------------
    detect_onset_noise_floor() (used internally by align_aformat_channels)
    has a temporal resolution limited by `frame_ms` (default 1 ms = 48
    samples at 48 kHz). That is enough to fix GROSS misalignment between
    channels caused by inconsistent manual file export (tens to hundreds
    of ms), but 1 ms of residual error is much larger than a full audio
    cycle at mid/high frequencies (1 ms = exactly one cycle at 1000 Hz;
    0.25 ms = one cycle at 4000 Hz). Combining channels with that much
    residual misalignment scrambles the PHASE relationship between
    capsules at those frequencies, producing unstable or physically
    implausible LF values even after coarse alignment fixes the low
    bands.

    This function searches, within ±search_radius_ms of the
    coarse-aligned position, for the INTEGER SAMPLE lag that maximizes
    cross-correlation between each channel and a reference channel
    (default LF) over a short window right after the common onset. This
    achieves single-sample precision (down to ~1/fs resolution, e.g.
    ~21 microseconds at 48 kHz) — far finer than frame-based onset
    detection, and adequate for phase-coherent combination up to several
    kHz.

    Parameters
    ----------
    aligned_signals  : dict {LF,RF,LB,RB: np.ndarray}, already coarsely
                       aligned (output of align_aformat_channels).
    fs               : sample rate in Hz.
    common_onset     : sample index of the shared onset (from
                       align_aformat_channels).
    reference_channel: which channel to align the others against
                       (default 'LF'; any reasonably loud channel works).
    search_radius_ms : how far (in ms) to search for the best lag around
                       the coarse-aligned position (default ±5 ms — should
                       comfortably exceed any residual frame-quantization
                       error from the coarse step).
    correlation_window_ms : length of the window used to compute
                       cross-correlation (default 30 ms, capturing the
                       direct sound and first reflections — enough
                       energy for a robust correlation estimate).

    Returns
    -------
    (refined_signals, lags)
        refined_signals : dict with the same channels, fine-shifted.
        lags            : dict {channel: lag_samples} — the integer
                          sample shift applied to each channel (positive
                          = channel content was delayed, shifted earlier
                          to align; reference channel always has lag 0).
    """
    radius = int(round(search_radius_ms / 1000.0 * fs))
    win_len = int(round(correlation_window_ms / 1000.0 * fs))

    ref_sig = aligned_signals[reference_channel]
    if common_onset + win_len > len(ref_sig):
        raise ValueError(
            "correlation_window_ms extends beyond signal length at the "
            "given common_onset. Use a shorter window or check the onset."
        )
    ref_window = ref_sig[common_onset: common_onset + win_len]

    refined = {}
    lags = {}

    for ch, sig in aligned_signals.items():
        if ch == reference_channel:
            refined[ch] = sig
            lags[ch] = 0
            continue

        best_lag = 0
        best_corr = -np.inf
        for lag in range(-radius, radius + 1):
            start = common_onset + lag
            if start < 0 or start + win_len > len(sig):
                continue
            seg = sig[start: start + win_len]
            corr = float(np.dot(seg, ref_window))
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        lags[ch] = best_lag
        if best_lag > 0:
            refined[ch] = np.concatenate([sig[best_lag:], np.zeros(best_lag)])
        elif best_lag < 0:
            refined[ch] = np.concatenate([np.zeros(-best_lag), sig[:best_lag]])
        else:
            refined[ch] = sig

    return refined, lags


# ─── Quick test (run as script) ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python acoustic_core.py <4ch_rir.wav> [LF RF LB RB]")
        print("       python acoustic_core.py <4ch_rir.wav>")
        sys.exit(0)

    wav_path = sys.argv[1]
    ch_order = sys.argv[2:6] if len(sys.argv) >= 6 else None

    print(f"\nAnalyzing: {wav_path}")
    result = analyze_rir(wav_path, channel_order=ch_order)
    print(result.summary())
