"""
Automatic video<->demo synchronisation (replaces hand-picking anchors).

Two stages:

  1. Candidate events from the broadcast. extract_audio() pulls the mono
     waveform via ffmpeg; detect_onsets() finds sharp energy onsets (the
     round-start "go" beep and bomb-plant beeps are loud, abrupt, and recur
     once per round, which is exactly what we exploit). This stage is a
     scaffold: real broadcasts have crowd noise and casters, so you will tune
     the threshold, or swap in a template match against a clip of the actual
     round-start sound.

  2. Robust alignment. align_events() takes the broadcast candidate times and
     the demo's round_start ticks and recovers tick = a*frame + b WITHOUT you
     matching them up by hand. It is a Hough-style vote: every (broadcast time,
     demo tick) pairing implies an offset b given the known frame/tick rate
     ratio a; the true offset wins the most votes, and a final least-squares
     refit on the inliers gives the SyncModel + residual drift. This tolerates
     missed and spurious onsets, so the audio stage does not need to be perfect.

The aligner is unit-tested. The audio stage is exercised on a synthetic tone.
"""
from __future__ import annotations

import subprocess
import tempfile

import numpy as np
from scipy.signal import find_peaks

from .sync import SyncModel


# --------------------------------------------------------------------------- #
def extract_audio(video_path: str, sr: int = 16000) -> tuple[np.ndarray, int]:
    """Decode mono PCM from a video via ffmpeg. Returns (samples float32, sr)."""
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=True) as tmp:
        cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr),
               "-f", "s16le", "-loglevel", "error", tmp.name]
        subprocess.run(cmd, check=True)
        raw = np.fromfile(tmp.name, dtype=np.int16).astype(np.float32) / 32768.0
    return raw, sr


def detect_onsets(samples: np.ndarray, sr: int, win_ms: float = 25.0,
                  min_gap_sec: float = 8.0, k: float = 4.0) -> list[float]:
    """Short-time-energy onset times (seconds). Peaks are energy jumps that
    stand k sigma above local median. min_gap_sec suppresses double-triggers
    (rounds are >= ~15 s apart)."""
    win = max(1, int(sr * win_ms / 1000))
    n = len(samples) // win
    if n < 2:
        return []
    energy = (samples[:n * win].reshape(n, win) ** 2).mean(1)
    # onset strength = positive change in log-energy
    log_e = np.log(energy + 1e-8)
    strength = np.clip(np.diff(log_e, prepend=log_e[0]), 0, None)
    thr = np.median(strength) + k * (strength.std() + 1e-9)
    peaks, _ = find_peaks(strength, height=thr,
                          distance=max(1, int(min_gap_sec * 1000 / win_ms)))
    return [float(p * win / sr) for p in peaks]


# --------------------------------------------------------------------------- #
def align_events(broadcast_times_sec, demo_ticks, src_fps: float,
                 tickrate: float = 64.0, tol_sec: float = 0.6) -> SyncModel:
    """Recover tick = a*frame + b from unordered, noisy event sets.

    broadcast_times_sec : candidate event times in the VOD (e.g. from onsets)
    demo_ticks          : demo event ticks (e.g. round_start) -- order-free
    """
    if len(broadcast_times_sec) < 2 or len(demo_ticks) < 2:
        raise ValueError("need >= 2 broadcast and >= 2 demo events")
    a0 = tickrate / src_fps                       # known nominal slope
    frames = np.array(broadcast_times_sec, float) * src_fps
    ticks = np.array(demo_ticks, float)
    tol_ticks = tol_sec * tickrate

    # vote on offset b = tick - a0*frame over every pairing
    bs = (ticks[None, :] - a0 * frames[:, None]).ravel()
    bs_sorted = np.sort(bs)
    # densest window of width tol_ticks
    best_b, best_n, j = bs_sorted[0], 0, 0
    for i in range(len(bs_sorted)):
        while bs_sorted[i] - bs_sorted[j] > tol_ticks:
            j += 1
        if i - j + 1 > best_n:
            best_n = i - j + 1
            best_b = float(np.median(bs_sorted[j:i + 1]))

    # collect inlier (frame, tick) pairs consistent with (a0, best_b)
    inl_f, inl_t = [], []
    for f in frames:
        pred = a0 * f + best_b
        d = np.abs(ticks - pred)
        j = int(d.argmin())
        if d[j] <= tol_ticks:
            inl_f.append(f); inl_t.append(ticks[j])
    if len(inl_f) < 2:
        raise RuntimeError("alignment failed: too few consistent events "
                           "(tune onset detection or raise tol_sec)")
    A = np.vstack([inl_f, np.ones(len(inl_f))]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(inl_t), rcond=None)
    resid = np.array(inl_t) - (a * np.array(inl_f) + b)
    return SyncModel(a=float(a), b=float(b),
                     residual_ticks=float(resid.std(ddof=0)),
                     n_anchors=len(inl_f), tickrate=tickrate)


def autosync_video(video_path: str, demo_ticks, src_fps: float,
                   tickrate: float = 64.0, **onset_kw) -> SyncModel:
    """Convenience: audio onsets -> align against demo round_start ticks."""
    samples, sr = extract_audio(video_path)
    onsets = detect_onsets(samples, sr, **onset_kw)
    return align_events(onsets, demo_ticks, src_fps, tickrate)
