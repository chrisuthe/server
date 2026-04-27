"""Tests for sonic_similarity._handle_similar_clap.

The endpoint is intentionally narrow: it must short-circuit cleanly
when the CLAP index is unavailable (text search disabled), and when
the seed isn't in the index. The happy-path orchestration is exercised
through integration with the running provider; these unit tests cover
the guard rails.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from music_assistant.providers.sonic_similarity import SonicSimilarityPlugin


def _stub_plugin(*, sonic_analysis: Any = None) -> SonicSimilarityPlugin:
    """Minimal SonicSimilarityPlugin instance for testing _handle_similar_clap."""
    plugin = SonicSimilarityPlugin.__new__(SonicSimilarityPlugin)
    plugin.logger = MagicMock()
    providers = [sonic_analysis] if sonic_analysis is not None else []
    plugin.mass = SimpleNamespace(get_providers=lambda _t: providers)
    return plugin


@pytest.mark.asyncio
async def test_returns_disabled_when_no_sonic_analysis_provider() -> None:
    """No sonic_analysis provider registered -> clap_index_disabled."""
    plugin = _stub_plugin(sonic_analysis=None)
    result = await plugin._handle_similar_clap(item_id="anything")
    assert result["analyzed"] is False
    assert result["reason"] == "clap_index_disabled"
    assert result["items"] == []


@pytest.mark.asyncio
async def test_returns_disabled_when_clap_index_is_none() -> None:
    """sonic_analysis present but _clap_index is None -> clap_index_disabled."""
    sa = SimpleNamespace(domain="sonic_analysis", _clap_index=None)
    plugin = _stub_plugin(sonic_analysis=sa)
    result = await plugin._handle_similar_clap(item_id="anything")
    assert result["analyzed"] is False
    assert result["reason"] == "clap_index_disabled"


@pytest.mark.asyncio
async def test_returns_seeds_not_in_index_when_lookup_fails() -> None:
    """Seed item_id not present in CLAP index -> seeds_not_in_clap_index."""
    fake_index = MagicMock()
    fake_index.get_embedding_by_item_id = MagicMock(return_value=None)
    sa = SimpleNamespace(domain="sonic_analysis", _clap_index=fake_index)
    plugin = _stub_plugin(sonic_analysis=sa)
    result = await plugin._handle_similar_clap(item_id="missing_track")
    assert result["analyzed"] is False
    assert result["reason"] == "seeds_not_in_clap_index"
    assert result["seed_track_ids"] == ["missing_track"]
