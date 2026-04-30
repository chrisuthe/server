"""Tests for the bulk-read helpers on AudioAnalysisController.

get_audio_analysis_count / get_audio_analysis_rows are the single chokepoint
for "give me everything for this domain" queries that previously lived
inside AudioAnalysisProvider subclasses. These tests pin the SQL/argument
shape they emit so providers can rely on stable behavior across schema
evolution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.controllers.streams.audio_analysis import AudioAnalysisController


def _stub_controller(
    count_result: int = 0,
    list_result: list[dict[str, Any]] | None = None,
) -> tuple[AudioAnalysisController, MagicMock]:
    """Build a bare AudioAnalysisController whose database is mocked.

    :returns: (controller, database_mock).
    """
    c = AudioAnalysisController.__new__(AudioAnalysisController)
    c.logger = MagicMock()
    db = MagicMock()
    db.get_count_from_query = AsyncMock(return_value=count_result)
    db.get_rows = AsyncMock(return_value=list_result or [])
    c.mass = MagicMock()
    c.mass.music = MagicMock()
    c.mass.music.database = db
    return c, db


@pytest.mark.asyncio
async def test_get_audio_analysis_count_returns_helper_result() -> None:
    """The controller forwards whatever get_count_from_query returns."""
    c, _ = _stub_controller(count_result=42)
    assert await c.get_audio_analysis_count("sonic_analysis") == 42


@pytest.mark.asyncio
async def test_get_audio_analysis_count_filters_by_domain_and_track_media_type() -> None:
    """Default count filters on aa_provider_domain AND media_type=track."""
    c, db = _stub_controller(count_result=0)
    await c.get_audio_analysis_count("sonic_analysis")
    sql, params = db.get_count_from_query.await_args.args
    assert "aa_provider_domain = :domain" in sql
    assert "media_type = :media_type" in sql
    assert params == {"domain": "sonic_analysis", "media_type": MediaType.TRACK.value}


@pytest.mark.asyncio
async def test_get_audio_analysis_count_respects_media_type_override() -> None:
    """Caller can count rows for a non-track media type."""
    c, db = _stub_controller(count_result=7)
    result = await c.get_audio_analysis_count(
        "sonic_analysis", media_type=MediaType.PODCAST_EPISODE
    )
    assert result == 7
    params = db.get_count_from_query.await_args.args[1]
    assert params["media_type"] == MediaType.PODCAST_EPISODE.value


@pytest.mark.asyncio
async def test_get_audio_analysis_rows_returns_full_rows() -> None:
    """get_audio_analysis_rows forwards what the DB returns; no filtering or parsing."""
    rows: list[dict[str, Any]] = [
        {"item_id": "a", "provider": "filesystem_local", "analysis_data": "{}"},
        {"item_id": "b", "provider": "filesystem_local", "analysis_data": "{}"},
    ]
    c, _ = _stub_controller(list_result=rows)
    result = await c.get_audio_analysis_rows("sonic_analysis")
    assert result == rows


@pytest.mark.asyncio
async def test_get_audio_analysis_rows_filters_by_domain_and_track_media_type() -> None:
    """Default rows query filters on aa_provider_domain + media_type=track and no row limit."""
    c, db = _stub_controller(list_result=[])
    await c.get_audio_analysis_rows("sonic_analysis")
    call = db.get_rows.await_args
    assert call.args[0] == "audio_analysis"
    assert call.args[1] == {
        "aa_provider_domain": "sonic_analysis",
        "media_type": MediaType.TRACK.value,
    }
    assert call.kwargs["limit"] == 0


@pytest.mark.asyncio
async def test_get_audio_analysis_rows_respects_media_type_override() -> None:
    """Caller can list rows for a non-track media type."""
    c, db = _stub_controller(list_result=[])
    await c.get_audio_analysis_rows("sonic_analysis", media_type=MediaType.PODCAST_EPISODE)
    filters = db.get_rows.await_args.args[1]
    assert filters["media_type"] == MediaType.PODCAST_EPISODE.value


@pytest.mark.asyncio
async def test_get_audio_analysis_rows_passes_limit_and_offset() -> None:
    """Caller-supplied limit/offset are forwarded to the DB layer for paginated reads."""
    c, db = _stub_controller(list_result=[])
    await c.get_audio_analysis_rows("sonic_analysis", limit=50, offset=100)
    call = db.get_rows.await_args
    assert call.kwargs["limit"] == 50
    assert call.kwargs["offset"] == 100
