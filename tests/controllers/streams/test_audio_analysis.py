"""Tests for the new bulk-read helpers on AudioAnalysisController.

count_rows_by_domain / list_rows_by_domain replace the direct DB hits
that previously lived inside AudioAnalysisProvider subclasses, so they
are the single chokepoint for "give me everything for this domain"
queries. These tests pin the SQL/argument shape they emit so providers
can rely on stable behavior across schema evolution.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.controllers.streams.audio_analysis import AudioAnalysisController


def _stub_controller(
    count_query_result: list[dict[str, object]] | None = None,
    list_result: list[dict[str, object]] | None = None,
) -> AudioAnalysisController:
    """Build a bare AudioAnalysisController whose database is mocked."""
    c = AudioAnalysisController.__new__(AudioAnalysisController)
    c.logger = MagicMock()
    c.mass = MagicMock()
    c.mass.music = MagicMock()
    c.mass.music.database = MagicMock()
    c.mass.music.database.get_rows_from_query = AsyncMock(
        return_value=count_query_result if count_query_result is not None else [{"c": 0}]
    )
    c.mass.music.database.get_rows = AsyncMock(return_value=list_result or [])
    return c


@pytest.mark.asyncio
async def test_count_rows_by_domain_returns_count() -> None:
    """count_rows_by_domain parses the c column out of the count-query result."""
    c = _stub_controller(count_query_result=[{"c": 42}])
    result = await c.count_rows_by_domain("sonic_analysis")
    assert result == 42


@pytest.mark.asyncio
async def test_count_rows_by_domain_returns_zero_when_empty() -> None:
    """An empty result set is treated as zero (defensive — sqlite always returns one row though)."""
    c = _stub_controller(count_query_result=[])
    result = await c.count_rows_by_domain("sonic_analysis")
    assert result == 0


@pytest.mark.asyncio
async def test_count_rows_by_domain_filters_by_domain_and_track_media_type() -> None:
    """Default count filters on aa_provider_domain AND media_type=track."""
    c = _stub_controller(count_query_result=[{"c": 0}])
    await c.count_rows_by_domain("sonic_analysis")
    call_args = c.mass.music.database.get_rows_from_query.await_args
    sql, params = call_args.args
    assert "aa_provider_domain = :domain" in sql
    assert "media_type = :media_type" in sql
    assert params == {"domain": "sonic_analysis", "media_type": MediaType.TRACK.value}


@pytest.mark.asyncio
async def test_count_rows_by_domain_respects_media_type_override() -> None:
    """Caller can count rows for a non-track media type."""
    c = _stub_controller(count_query_result=[{"c": 7}])
    result = await c.count_rows_by_domain("sonic_analysis", media_type=MediaType.PODCAST_EPISODE)
    assert result == 7
    params = c.mass.music.database.get_rows_from_query.await_args.args[1]
    assert params["media_type"] == MediaType.PODCAST_EPISODE.value


@pytest.mark.asyncio
async def test_list_rows_by_domain_returns_full_rows() -> None:
    """list_rows_by_domain forwards what the DB returns; no filtering or parsing."""
    rows = [
        {"item_id": "a", "provider": "filesystem_local", "analysis_data": "{}"},
        {"item_id": "b", "provider": "filesystem_local", "analysis_data": "{}"},
    ]
    c = _stub_controller(list_result=rows)
    result = await c.list_rows_by_domain("sonic_analysis")
    assert result == rows


@pytest.mark.asyncio
async def test_list_rows_by_domain_filters_by_domain_and_track_media_type() -> None:
    """Default list filters on aa_provider_domain + media_type=track and no row limit."""
    c = _stub_controller(list_result=[])
    await c.list_rows_by_domain("sonic_analysis")
    call = c.mass.music.database.get_rows.await_args
    table = call.args[0]
    filters = call.args[1]
    assert table == "audio_analysis"
    assert filters == {
        "aa_provider_domain": "sonic_analysis",
        "media_type": MediaType.TRACK.value,
    }
    assert call.kwargs["limit"] == 0


@pytest.mark.asyncio
async def test_list_rows_by_domain_respects_media_type_override() -> None:
    """Caller can list rows for a non-track media type."""
    c = _stub_controller(list_result=[])
    await c.list_rows_by_domain("sonic_analysis", media_type=MediaType.PODCAST_EPISODE)
    filters = c.mass.music.database.get_rows.await_args.args[1]
    assert filters["media_type"] == MediaType.PODCAST_EPISODE.value
