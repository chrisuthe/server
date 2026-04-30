"""Tests for AudioAnalysisController's bulk-read helpers."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType

from music_assistant.controllers.streams.audio_analysis import AudioAnalysisController


def _stub_controller(
    count_result: int = 0,
    list_result: list[dict[str, Any]] | None = None,
    rows_from_query_result: list[dict[str, Any]] | None = None,
) -> tuple[AudioAnalysisController, MagicMock]:
    """Build a bare AudioAnalysisController whose database is mocked."""
    c = AudioAnalysisController.__new__(AudioAnalysisController)
    c.logger = MagicMock()
    db = MagicMock()
    db.get_count_from_query = AsyncMock(return_value=count_result)
    db.get_rows = AsyncMock(return_value=list_result or [])
    db.get_rows_from_query = AsyncMock(return_value=rows_from_query_result or [])
    c.mass = MagicMock()
    c.mass.music = MagicMock()
    c.mass.music.database = db
    c.mass.get_providers = MagicMock(return_value=[])
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


def _aa_provider_stub(domain: str, available: bool = True) -> MagicMock:
    """Build a provider stub that satisfies the get_providers().available filter."""
    p = MagicMock()
    p.domain = domain
    p.available = available
    return p


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_merges_within_group() -> None:
    """Two rows for the same (item_id, provider) merge in timestamp order."""
    rows: list[dict[str, Any]] = [
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": '{"bpm": 100.0, "energy": 0.5}',
        },
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "smart_fades",
            "analysis_data": '{"bpm": 120.0, "key": "C"}',
        },
    ]
    c, _ = _stub_controller(rows_from_query_result=rows)
    c.mass.get_providers.return_value = [
        _aa_provider_stub("sonic_analysis"),
        _aa_provider_stub("smart_fades"),
    ]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert len(result) == 1
    item_id, provider, merged = result[0]
    assert (item_id, provider) == ("t1", "filesystem_local")
    assert merged.bpm == 120.0  # smart_fades wins on bpm (later row)
    assert merged.energy == 0.5  # sonic_analysis still wins where smart_fades is None
    assert merged.key == "C"


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_skips_unavailable_providers() -> None:
    """Rows from unavailable AA providers are skipped during merge."""
    rows: list[dict[str, Any]] = [
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": '{"bpm": 100.0}',
        },
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "disabled_provider",
            "analysis_data": '{"bpm": 999.0}',
        },
    ]
    c, _ = _stub_controller(rows_from_query_result=rows)
    c.mass.get_providers.return_value = [_aa_provider_stub("sonic_analysis")]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert len(result) == 1
    assert result[0][2].bpm == 100.0  # disabled_provider's row ignored


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_groups_by_item_provider() -> None:
    """Rows from different (item_id, provider) pairs are emitted as separate entries."""
    rows: list[dict[str, Any]] = [
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": '{"bpm": 100.0}',
        },
        {
            "item_id": "t2",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": '{"bpm": 200.0}',
        },
    ]
    c, _ = _stub_controller(rows_from_query_result=rows)
    c.mass.get_providers.return_value = [_aa_provider_stub("sonic_analysis")]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert len(result) == 2
    assert {r[0] for r in result} == {"t1", "t2"}


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_skips_unparsable_rows() -> None:
    """A row with corrupt JSON is silently skipped without aborting the merge."""
    rows: list[dict[str, Any]] = [
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": "not-json",
        },
        {
            "item_id": "t1",
            "provider": "filesystem_local",
            "aa_provider_domain": "smart_fades",
            "analysis_data": '{"bpm": 120.0}',
        },
    ]
    c, _ = _stub_controller(rows_from_query_result=rows)
    c.mass.get_providers.return_value = [
        _aa_provider_stub("sonic_analysis"),
        _aa_provider_stub("smart_fades"),
    ]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert len(result) == 1
    assert result[0][2].bpm == 120.0


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_empty_db_returns_empty_list() -> None:
    """An empty DB result yields an empty list without flushing a sentinel group."""
    c, _ = _stub_controller(rows_from_query_result=[])
    c.mass.get_providers.return_value = [_aa_provider_stub("sonic_analysis")]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert result == []


@pytest.mark.asyncio
async def test_get_merged_audio_analysis_rows_drops_groups_with_only_corrupt_rows() -> None:
    """A group whose only row has corrupt JSON is not emitted at all."""
    rows: list[dict[str, Any]] = [
        {
            "item_id": "broken",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": "not-json",
        },
        {
            "item_id": "good",
            "provider": "filesystem_local",
            "aa_provider_domain": "sonic_analysis",
            "analysis_data": '{"bpm": 100.0}',
        },
    ]
    c, _ = _stub_controller(rows_from_query_result=rows)
    c.mass.get_providers.return_value = [_aa_provider_stub("sonic_analysis")]

    result = await c.get_merged_audio_analysis_rows("sonic_analysis")
    assert len(result) == 1
    assert result[0][0] == "good"
