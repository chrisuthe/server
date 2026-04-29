"""Tests for the new bulk-read helpers on AudioAnalysisController.

count_rows_by_domain / list_rows_by_domain replace the direct DB hits
that previously lived inside AudioAnalysisProvider subclasses, so they
are the single chokepoint for "give me everything for this domain"
queries. These tests pin the SQL/argument shape they emit so providers
can rely on stable behavior across schema evolution.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.controllers.streams.audio_analysis import AudioAnalysisController


def _stub_controller(
    count_query_result: list[dict[str, Any]] | None = None,
    list_result: list[dict[str, Any]] | None = None,
) -> tuple[AudioAnalysisController, MagicMock]:
    """Build a bare AudioAnalysisController whose database is mocked.

    :returns: (controller, database_mock) — exposing the mock directly so tests
        can call ``await_args`` without mypy fighting AsyncMock substitution
        on attributes whose real type is a coroutine method.
    """
    c = AudioAnalysisController.__new__(AudioAnalysisController)
    c.logger = MagicMock()
    db = MagicMock()
    db.get_rows_from_query = AsyncMock(
        return_value=count_query_result if count_query_result is not None else [{"c": 0}]
    )
    db.get_rows = AsyncMock(return_value=list_result or [])
    c.mass = MagicMock()
    c.mass.music = MagicMock()
    c.mass.music.database = db
    return c, db


@pytest.mark.asyncio
async def test_count_rows_by_domain_returns_count() -> None:
    """count_rows_by_domain parses the c column out of the count-query result."""
    c, _ = _stub_controller(count_query_result=[{"c": 42}])
    result = await c.count_rows_by_domain("sonic_analysis")
    assert result == 42


@pytest.mark.asyncio
async def test_count_rows_by_domain_returns_zero_when_empty() -> None:
    """An empty result set is treated as zero (defensive — sqlite always returns one row though)."""
    c, _ = _stub_controller(count_query_result=[])
    result = await c.count_rows_by_domain("sonic_analysis")
    assert result == 0


@pytest.mark.asyncio
async def test_count_rows_by_domain_filters_by_domain_and_track_media_type() -> None:
    """Default count filters on aa_provider_domain AND media_type=track."""
    c, db = _stub_controller(count_query_result=[{"c": 0}])
    await c.count_rows_by_domain("sonic_analysis")
    sql, params = db.get_rows_from_query.await_args.args
    assert "aa_provider_domain = :domain" in sql
    assert "media_type = :media_type" in sql
    assert params == {"domain": "sonic_analysis", "media_type": MediaType.TRACK.value}


@pytest.mark.asyncio
async def test_count_rows_by_domain_respects_media_type_override() -> None:
    """Caller can count rows for a non-track media type."""
    c, db = _stub_controller(count_query_result=[{"c": 7}])
    result = await c.count_rows_by_domain("sonic_analysis", media_type=MediaType.PODCAST_EPISODE)
    assert result == 7
    params = db.get_rows_from_query.await_args.args[1]
    assert params["media_type"] == MediaType.PODCAST_EPISODE.value


@pytest.mark.asyncio
async def test_list_rows_by_domain_returns_full_rows() -> None:
    """list_rows_by_domain forwards what the DB returns; no filtering or parsing."""
    rows: list[dict[str, Any]] = [
        {"item_id": "a", "provider": "filesystem_local", "analysis_data": "{}"},
        {"item_id": "b", "provider": "filesystem_local", "analysis_data": "{}"},
    ]
    c, _ = _stub_controller(list_result=rows)
    result = await c.list_rows_by_domain("sonic_analysis")
    assert result == rows


@pytest.mark.asyncio
async def test_list_rows_by_domain_filters_by_domain_and_track_media_type() -> None:
    """Default list filters on aa_provider_domain + media_type=track and no row limit."""
    c, db = _stub_controller(list_result=[])
    await c.list_rows_by_domain("sonic_analysis")
    call = db.get_rows.await_args
    assert call.args[0] == "audio_analysis"
    assert call.args[1] == {
        "aa_provider_domain": "sonic_analysis",
        "media_type": MediaType.TRACK.value,
    }
    assert call.kwargs["limit"] == 0


@pytest.mark.asyncio
async def test_list_rows_by_domain_respects_media_type_override() -> None:
    """Caller can list rows for a non-track media type."""
    c, db = _stub_controller(list_result=[])
    await c.list_rows_by_domain("sonic_analysis", media_type=MediaType.PODCAST_EPISODE)
    filters = db.get_rows.await_args.args[1]
    assert filters["media_type"] == MediaType.PODCAST_EPISODE.value
