"""Tests for the analyze_file path (background-scan entry point)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from music_assistant.providers.sonic_analysis import SonicAnalysisProvider

# 2 seconds of audio is enough to drive the librosa pipeline through one block
# plus a partial; just over ANALYZE_FILE_MIN_SAMPLES (22050).
_TEST_AUDIO_SECONDS = 2
_TEST_SR = 22050
_FAKE_PATH = "fake-track.mp3"


def _stub_provider(clap_model: Any = None) -> tuple[SonicAnalysisProvider, MagicMock]:
    """Build a SonicAnalysisProvider with mocked MA scaffolding and optional CLAP model."""
    p = SonicAnalysisProvider.__new__(SonicAnalysisProvider)
    fake_logger = MagicMock()
    p.logger = fake_logger
    p.config = MagicMock()
    p.config.get_value = MagicMock(return_value="fast")
    p._clap_model = clap_model
    p._clap_text_embeddings = MagicMock()
    p._clap_prompt_order = []
    return p, fake_logger


def _fake_audio(seconds: float = _TEST_AUDIO_SECONDS) -> np.ndarray:
    """Generate deterministic float32 audio at the analyze_file sample rate."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(int(seconds * _TEST_SR)).astype(np.float32)


@pytest.mark.asyncio
async def test_returns_none_for_invalid_path() -> None:
    """analyze_file rejects None / empty paths before doing any work."""
    p, _ = _stub_provider()
    sd_none = MagicMock(path=None)
    sd_empty = MagicMock(path="")
    assert await p.analyze_file(sd_none) is None
    assert await p.analyze_file(sd_empty) is None


@pytest.mark.asyncio
async def test_returns_none_when_librosa_load_fails() -> None:
    """A librosa.load exception is caught and returns None (graceful skip)."""
    p, fake_logger = _stub_provider()
    sd = MagicMock(path=_FAKE_PATH, provider="filesystem_local", item_id="t1")

    with patch("librosa.load", side_effect=OSError("boom")):
        assert await p.analyze_file(sd) is None
    fake_logger.debug.assert_called()


@pytest.mark.asyncio
async def test_returns_none_for_too_short_audio() -> None:
    """Audio under ANALYZE_FILE_MIN_SAMPLES is skipped."""
    p, _ = _stub_provider()
    sd = MagicMock(path=_FAKE_PATH)
    short_audio = np.zeros(100, dtype=np.float32)

    with patch("librosa.load", return_value=(short_audio, _TEST_SR)):
        assert await p.analyze_file(sd) is None


@pytest.mark.asyncio
async def test_skips_clap_when_model_unloaded() -> None:
    """No CLAP model → librosa fields populated, CLAP scalars left None."""
    p, _ = _stub_provider(clap_model=None)
    sd = MagicMock(path=_FAKE_PATH)

    with patch("librosa.load", return_value=(_fake_audio(), _TEST_SR)):
        result = await p.analyze_file(sd)

    assert result is not None
    assert result.energy is not None
    assert result.danceability is None
    assert result.valence is None
    if result.extra_data is not None:
        assert "clap_embedding" not in result.extra_data


@pytest.mark.asyncio
async def test_populates_clap_scalars_and_embedding_on_success() -> None:
    """Happy path: scalars set on AudioAnalysisData, embedding stored under extra_data."""
    p, _ = _stub_provider(clap_model=MagicMock())

    fake_scalars = {
        "danceability": 0.7,
        "valence": 0.5,
        "arousal": 0.6,
        "instrumentalness": 0.1,
        "acousticness": 0.4,
    }
    fake_emb = np.full(1024, 0.123, dtype=np.float32)
    p._run_clap_inference = MagicMock(return_value=(fake_scalars, fake_emb))  # type: ignore[method-assign]

    sd = MagicMock(path=_FAKE_PATH)
    with patch("librosa.load", return_value=(_fake_audio(), _TEST_SR)):
        result = await p.analyze_file(sd)

    assert result is not None
    assert result.danceability == 0.7
    assert result.valence == 0.5
    assert result.arousal == 0.6
    assert result.instrumentalness == 0.1
    assert result.acousticness == 0.4
    assert result.extra_data is not None
    assert result.extra_data["clap_embedding"] == fake_emb.tolist()


@pytest.mark.asyncio
async def test_swallows_clap_failure_keeps_librosa_fields() -> None:
    """A CLAP exception is caught at debug; librosa-derived fields still populate."""
    p, fake_logger = _stub_provider(clap_model=MagicMock())
    p._run_clap_inference = MagicMock(side_effect=RuntimeError("CLAP failure"))  # type: ignore[method-assign]

    sd = MagicMock(path=_FAKE_PATH)
    with patch("librosa.load", return_value=(_fake_audio(), _TEST_SR)):
        result = await p.analyze_file(sd)

    assert result is not None
    assert result.energy is not None
    assert result.danceability is None
    if result.extra_data is not None:
        assert "clap_embedding" not in result.extra_data
    fake_logger.debug.assert_called()
