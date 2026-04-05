"""14-dimensional semantic vector schema for sonic similarity search.

Owns the mapping from AudioAnalysisData fields to a fixed-size float vector
suitable for USearch ANN indexing. The 14 dimensions are:
  [0-8]  9 scalar features (bpm, energy, danceability, ...)
  [9-11] circular key encoding (sin, cos) + mode
  [12]   RMS energy variance over time
  [13]   Spectral centroid variance over time
"""

from __future__ import annotations

import math

import numpy as np

from music_assistant.models.audio_analysis import AudioAnalysisData

PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

VECTOR_FIELDS = [
    "bpm",
    "energy",
    "danceability",
    "loudness_integrated",
    "loudness_range",
    "brightness",
    "harmonic_complexity",
    "roughness",
    "rhythmic_regularity",
]

VECTOR_DIMENSIONS = 14  # 9 scalars + 2 key encoding + 1 mode + 2 time-series variance

FEATURE_GROUPS = {
    "rhythm": (0, 3),  # bpm, energy, danceability
    "loudness": (3, 5),  # loudness_integrated, loudness_range
    "timbre": (5, 8),  # brightness, harmonic_complexity, roughness
    "regularity": (8, 9),  # rhythmic_regularity
    "tonal": (9, 12),  # key_sin, key_cos, mode
    "dynamics": (12, 14),  # rms_variance, centroid_variance
}


def encode_key_mode(key: str, mode: str) -> tuple[float, float, float]:
    """Encode musical key and mode as three floats for circular and binary representation.

    :param key: Pitch class name (e.g. "C", "F#"). Unknown keys default to pitch class 0.
    :param mode: Tonality string — "major" encodes to 1.0, anything else to 0.0.
    :returns: Tuple of (key_sin, key_cos, mode_float).
    """
    pitch_class = PITCH_CLASS_NAMES.index(key) if key in PITCH_CLASS_NAMES else 0
    angle = 2.0 * math.pi * pitch_class / 12
    key_sin = math.sin(angle)
    key_cos = math.cos(angle)
    mode_float = 1.0 if mode == "major" else 0.0
    return key_sin, key_cos, mode_float


def assemble_vector(analysis: AudioAnalysisData) -> list[float] | None:
    """Assemble a 14-dimensional feature vector from an AudioAnalysisData instance.

    Returns None if any required field (all VECTOR_FIELDS, key, or mode) is None.
    Time-series variance dimensions default to 0.0 when the array is absent or has
    length <= 1.

    :param analysis: Source audio analysis data.
    :returns: 14-element list of floats, or None if required fields are missing.
    """
    # Validate all required scalar fields are present
    for field in VECTOR_FIELDS:
        if getattr(analysis, field) is None:
            return None
    if analysis.key is None or analysis.mode is None:
        return None

    scalars = [float(getattr(analysis, field)) for field in VECTOR_FIELDS]

    key_sin, key_cos, mode_float = encode_key_mode(analysis.key, analysis.mode)

    # Compute time-series variances, defaulting to 0.0 for absent or trivial arrays
    rms = analysis.rms_energy
    rms_var = float(np.var(rms)) if rms is not None and len(rms) > 1 else 0.0

    centroid = analysis.spectral_centroid
    centroid_var = float(np.var(centroid)) if centroid is not None and len(centroid) > 1 else 0.0

    return [*scalars, key_sin, key_cos, mode_float, rms_var, centroid_var]


def normalize_features(
    raw_features: list[float],
    corpus_means: list[float],
    corpus_stds: list[float],
) -> list[float]:
    """Apply z-score then L2 normalization to a raw feature vector.

    Zero standard deviation for a feature produces 0.0 for that dimension.
    If the resulting z-score vector has zero L2 norm, it is returned as-is
    without L2 normalization.

    :param raw_features: Raw feature vector to normalize.
    :param corpus_means: Per-feature means from the corpus.
    :param corpus_stds: Per-feature standard deviations from the corpus.
    :returns: Normalized feature vector as a list of floats.
    """
    # Z-score normalization; zero std → 0.0 for that dimension
    z_scored = [
        (v - m) / s if s != 0.0 else 0.0
        for v, m, s in zip(raw_features, corpus_means, corpus_stds, strict=True)
    ]

    norm = math.sqrt(sum(v * v for v in z_scored))
    if norm == 0.0:
        return [float(v) for v in z_scored]

    return [float(v / norm) for v in z_scored]


def compute_corpus_stats(
    all_features: list[list[float]],
) -> tuple[list[float], list[float]]:
    """Compute per-feature means and standard deviations across a corpus.

    :param all_features: List of feature vectors (all same dimensionality).
    :returns: Tuple of (means, stds) as lists of floats.
    :raises ValueError: If all_features is empty.
    """
    if not all_features:
        msg = "Empty corpus: cannot compute stats from zero feature vectors"
        raise ValueError(msg)

    arr = np.array(all_features, dtype=np.float64)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    return [float(v) for v in means], [float(v) for v in stds]


def compute_weighted_distance(
    sig_a: list[float],
    sig_b: list[float],
    weights: dict[str, float],
) -> float:
    """Compute per-group weighted Euclidean distance between two feature vectors.

    Each feature group (defined in FEATURE_GROUPS) contributes a weighted squared
    distance. Missing group weights default to 1.0. The result is normalized by
    the total weighted dimension count so distances are comparable across weight
    configurations.

    :param sig_a: First feature vector.
    :param sig_b: Second feature vector.
    :param weights: Per-group weight overrides keyed by FEATURE_GROUPS name.
    :returns: Weighted normalized distance as a float.
    """
    a = np.array(sig_a, dtype=np.float64)
    b = np.array(sig_b, dtype=np.float64)

    weighted_sq_sum = 0.0
    total_weighted_dims = 0.0

    for group, (start, end) in FEATURE_GROUPS.items():
        w = weights.get(group, 1.0)
        diff = a[start:end] - b[start:end]
        dim_count = end - start
        weighted_sq_sum += w * float(np.dot(diff, diff))
        total_weighted_dims += w * dim_count

    if total_weighted_dims == 0.0:
        return 0.0

    return math.sqrt(weighted_sq_sum / total_weighted_dims)
