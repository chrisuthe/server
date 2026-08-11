"""Tests for the Pandora provider's dynamic-playlist and streaming surface."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import Mock

import pytest
from music_assistant_models.enums import MediaType, StreamType
from music_assistant_models.errors import InvalidDataError, MediaNotFoundError
from music_assistant_models.media_items import SearchResults

from music_assistant.providers.pandora.constants import (
    CATALOG_ANNOTATE_ENDPOINT,
    STATIONS_ENDPOINT,
)
from music_assistant.providers.pandora.fragments import (
    FRAGMENT_STALE_SECONDS,
    FRAGMENT_URL_TTL_SECONDS,
)
from music_assistant.providers.pandora.provider import PandoraProvider

STATION_ID = "4360491625318318161"


def _tracks(count: int = 4, prefix: str = "S") -> list[dict[str, Any]]:
    """Build `count` raw Pandora track dicts with distinct music ids."""
    return [
        {
            "musicId": f"{prefix}{index}",
            "pandoraId": f"TR:{prefix}{index}",
            "stationId": STATION_ID,
            "songTitle": f"Song {index}",
            "artistName": "Some Artist",
            "albumTitle": "Some Album",
            "albumDetailURL": "https://www.pandora.com/artist/album",
            "songDetailURL": "https://www.pandora.com/artist/album/song",
            "trackLength": 180,
            "audioURL": f"https://audio-sv5-t3-2.pandora.com/access/{index}.mp4",
        }
        for index in range(count)
    ]


def _stations(names: list[str]) -> list[dict[str, Any]]:
    """Build raw Pandora station dicts with the given names."""
    return [{"stationId": f"station-{index}", "name": name} for index, name in enumerate(names)]


def _provider(
    payloads: list[list[dict[str, Any]]] | None = None,
    stations: list[dict[str, Any]] | None = None,
) -> PandoraProvider:
    """
    Build a bare provider whose Pandora API calls return canned payloads.

    The stub sits at `_api_request`, not `_fetch_fragment`/`_get_stations`, so the real
    filtering, empty-fragment guard and session retention all execute under test.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider._sessions = {}
    provider._high_quality_available = False
    pending = list(payloads or [_tracks()])
    station_list = stations or []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return the next canned payload instead of calling Pandora."""
        if url == STATIONS_ENDPOINT:
            return {"stations": station_list}
        return {"tracks": pending.pop(0) if pending else _tracks()}

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider


def _annotating_provider(
    annotations: dict[str, Any] | None = None,
    on_demand: bool = True,
) -> tuple[PandoraProvider, list[dict[str, Any]]]:
    """
    Build a provider whose annotateObjects calls return a canned map.

    Returns the provider and a list recording each annotate request body, so a test can
    assert whether the call was made at all rather than only what came back.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider._sessions = {}
    provider._high_quality_available = False
    provider._on_demand_available = on_demand
    annotate_calls: list[dict[str, Any]] = []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return canned fragment/annotate payloads instead of calling Pandora."""
        if url == CATALOG_ANNOTATE_ENDPOINT:
            annotate_calls.append(data or {})
            return dict(annotations or {})
        return {"tracks": _tracks()}

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, annotate_calls


async def test_entitled_account_hydrates_a_fragment() -> None:
    """One batched annotate call per fragment carries the catalogue ids the payload lacks."""
    records = {"TR:S0": {"pandoraId": "TR:S0", "albumId": "AL:1", "artistId": "AR:1"}}
    provider, calls = _annotating_provider(records)
    await provider.get_playlist_tracks(STATION_ID)
    assert len(calls) == 1
    assert calls[0]["pandoraIds"] == [f"TR:S{index}" for index in range(4)]
    assert provider._sessions[STATION_ID].current.annotations == records


async def test_unentitled_account_does_not_hydrate() -> None:
    """Without on-demand, catalogue ids would only offer albums whose tracks cannot play."""
    provider, calls = _annotating_provider({"TR:S0": {}}, on_demand=False)
    await provider.get_playlist_tracks(STATION_ID)
    assert calls == []
    assert provider._sessions[STATION_ID].current.annotations == {}


async def test_failed_hydration_still_serves_the_station() -> None:
    """Hydration is metadata enrichment; losing it must not stop playback."""
    provider, _ = _annotating_provider()

    async def _failing_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        if url == CATALOG_ANNOTATE_ENDPOINT:
            raise InvalidDataError("annotate exploded")
        return {"tracks": _tracks()}

    provider._api_request = _failing_request  # type: ignore[method-assign, assignment]
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert len(tracks) == 4
    assert provider._sessions[STATION_ID].current.annotations == {}


_HYDRATED = {
    "TR:S0": {"pandoraId": "TR:S0", "albumId": "AL:900", "artistId": "AR:800"},
    "AL:900": {"pandoraId": "AL:900", "name": "Some Album"},
    "AR:800": {"pandoraId": "AR:800", "name": "Some Artist"},
}


async def test_hydrated_track_uses_catalogue_album_and_artist_ids() -> None:
    """An entitled account's album and artist are the ids the catalogue uses everywhere."""
    provider, _ = _annotating_provider(_HYDRATED)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert tracks[0].album is not None
    assert tracks[0].album.item_id == "AL:900"
    assert tracks[0].artists[0].item_id == "AR:800"


async def test_unhydrated_track_keeps_todays_album_and_artist() -> None:
    """Without entitlement the album stays track-scoped and the artist name-keyed."""
    provider, _ = _annotating_provider({}, on_demand=False)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert tracks[0].album is not None
    assert tracks[0].album.item_id == "TR:S0"
    assert tracks[0].artists[0].item_id == "Some Artist"


async def test_catalogue_album_is_resolvable_by_id() -> None:
    """An AL: id offered on a track must resolve, or the track offers a dead link."""
    provider, _ = _annotating_provider(_HYDRATED)
    album = await provider.get_album("AL:900")
    assert album.item_id == "AL:900"
    assert album.name == "Some Album"


async def test_catalogue_artist_is_resolvable_by_id() -> None:
    """An AR: id must resolve to the artist's name, not to the id as a name."""
    provider, _ = _annotating_provider(_HYDRATED)
    artist = await provider.get_artist("AR:800")
    assert artist.item_id == "AR:800"
    assert artist.name == "Some Artist"


async def test_unknown_catalogue_id_is_refused() -> None:
    """Pandora returning no record is a missing item, not an empty one."""
    provider, _ = _annotating_provider({})
    with pytest.raises(MediaNotFoundError):
        await provider.get_artist("AR:does-not-exist")


async def test_search_returns_a_matching_station_as_a_playlist() -> None:
    """A station whose name matches the query comes back in the playlist results."""
    provider = _provider(stations=_stations(["Coldplay Radio", "Jazz Radio"]))
    results = await provider.search("Coldplay Radio", [MediaType.PLAYLIST])
    assert [playlist.name for playlist in results.playlists] == ["Coldplay Radio"]


async def test_search_matches_part_of_a_station_name() -> None:
    """A partial query finds the station; whole-name-only matching makes search useless."""
    provider = _provider(stations=_stations(["Classic Rock Radio", "Jazz Radio"]))
    results = await provider.search("rock", [MediaType.PLAYLIST])
    assert [playlist.name for playlist in results.playlists] == ["Classic Rock Radio"]


async def test_search_ignores_case() -> None:
    """Queries match case-insensitively."""
    provider = _provider(stations=_stations(["Classic Rock Radio"]))
    results = await provider.search("CLASSIC rock", [MediaType.PLAYLIST])
    assert len(results.playlists) == 1


async def test_search_honours_the_limit() -> None:
    """A query matching many stations stops at the requested limit."""
    provider = _provider(stations=_stations([f"Rock Radio {index}" for index in range(5)]))
    results = await provider.search("rock", [MediaType.PLAYLIST], limit=2)
    assert len(results.playlists) == 2


async def test_search_finds_nothing_for_a_non_matching_query() -> None:
    """A query that matches no station name returns no playlists."""
    provider = _provider(stations=_stations(["Coldplay Radio"]))
    results = await provider.search("Nonexistent Station", [MediaType.PLAYLIST])
    assert results.playlists == []


async def test_search_without_playlist_media_type_skips_the_station_lookup() -> None:
    """Stations only ever surface as playlists; excluding that type returns nothing at all."""
    provider = _provider(stations=_stations(["Coldplay Radio"]))
    results = await provider.search("Coldplay Radio", [MediaType.TRACK])
    assert results == SearchResults()


async def test_pages_beyond_the_first_terminate_the_loop() -> None:
    """The core pages a playlist until it returns nothing; a station serves one batch."""
    provider = _provider()
    assert await provider.get_playlist_tracks(STATION_ID, page=1) == []


async def test_first_request_returns_a_fragment() -> None:
    """A station with no session yet fetches and returns its first fragment."""
    provider = _provider()
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == [f"TR:S{i}" for i in range(4)]


async def test_track_id_is_the_bare_pandora_id() -> None:
    """A track is identified by Pandora's own catalogue id, not by station and musicId."""
    provider = _provider()
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == [f"TR:S{index}" for index in range(4)]


async def test_the_same_song_from_two_stations_is_one_item() -> None:
    """Station context must not fork a song's identity - that is two library rows."""
    provider = _provider()
    first = await provider.get_playlist_tracks("station-a")
    second = await provider.get_playlist_tracks("station-b")
    assert first[0].item_id == second[0].item_id


async def test_a_track_without_a_pandora_id_is_not_served() -> None:
    """pandoraId is identity now, so a track lacking one cannot be handed to the queue."""
    usable = _tracks(count=2)
    unusable = _tracks(count=2, prefix="X")
    for track in unusable:
        del track["pandoraId"]
    provider = _provider(payloads=[usable + unusable])
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == ["TR:S0", "TR:S1"]


async def test_a_fragment_with_no_identifiable_track_is_refused() -> None:
    """Retaining a fragment nothing can be served from would stall the station."""
    tracks = _tracks()
    for track in tracks:
        del track["pandoraId"]
    provider = _provider(payloads=[tracks])
    with pytest.raises(MediaNotFoundError):
        await provider.get_playlist_tracks(STATION_ID)


async def test_stream_details_resolve_without_station_context() -> None:
    """A bare pandoraId resolves against whichever session holds it."""
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    details = await provider.get_stream_details("TR:S1", MediaType.TRACK)
    assert details.item_id == "TR:S1"
    assert details.stream_type == StreamType.HTTP


async def test_browse_then_play_returns_the_same_batch() -> None:
    """A browse leaves the fragment live, so play must still get its tracks."""
    provider = _provider()
    browsed = await provider.get_playlist_tracks(STATION_ID)
    played = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in played] == [track.item_id for track in browsed]


async def test_refill_serves_the_live_fragment_without_refetching() -> None:
    """
    A refill mid-fragment must not pull a new one, and must not re-serve a played track.

    Returning [] here would read as end-of-playlist; the core de-duplicates the remaining
    repeats via its unplayed-tail check, but a track already handed to the audio pipeline
    must never come back - that check drops it once playback has moved past it.
    """
    provider = _provider([_tracks(prefix="A"), _tracks(prefix="B")])
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A0", MediaType.TRACK)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == [f"TR:A{i}" for i in range(1, 4)]
    assert "TR:A0" not in [track.item_id for track in tracks]


async def test_replay_after_stopping_mid_fragment_still_builds_a_queue() -> None:
    """Stopping after one track and playing again must not yield an empty queue."""
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:S0", MediaType.TRACK)
    # user stops, then presses play again on the same station
    replayed = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in replayed] == [f"TR:S{i}" for i in range(1, 4)]


async def test_empty_fragment_is_not_retained() -> None:
    """
    A fragment with no playable tracks must raise, not become the live fragment.

    Retaining it would make it current with nothing able to spend it, so the station
    would serve nothing until the staleness window elapsed.
    """
    curator_only = [
        {"musicId": "S0", "stationId": STATION_ID, "songTitle": "Curator Message", "audioURL": ""}
    ]
    provider = _provider([curator_only])
    with pytest.raises(MediaNotFoundError):
        await provider.get_playlist_tracks(STATION_ID)
    assert provider._sessions[STATION_ID].current is None


async def test_track_with_null_song_title_gets_a_fallback_name() -> None:
    """A JSON-null songTitle must not crash title parsing; the track still comes through."""
    tracks = _tracks()
    tracks[0]["songTitle"] = None
    provider = _provider([tracks])
    result = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in result] == [f"TR:S{i}" for i in range(4)]
    assert result[0].name == "Unknown Song"


async def test_track_with_null_track_length_gets_zero_duration() -> None:
    """A JSON-null trackLength must not crash int(); it degrades to a zero duration."""
    tracks = _tracks()
    tracks[0]["trackLength"] = None
    provider = _provider([tracks])
    result = await provider.get_playlist_tracks(STATION_ID)
    assert result[0].duration == 0


async def test_track_with_null_album_title_gets_a_fallback_name() -> None:
    """A JSON-null albumTitle must not crash album parsing; it falls back to a default name."""
    tracks = _tracks()
    tracks[0]["albumTitle"] = None
    provider = _provider([tracks])
    result = await provider.get_playlist_tracks(STATION_ID)
    assert result[0].album is not None
    assert result[0].album.name == "Unknown Album"


async def test_track_missing_station_id_is_dropped_not_crashed() -> None:
    """A track without a stationId can't form a track id; drop it instead of raising."""
    tracks = _tracks()
    del tracks[0]["stationId"]
    provider = _provider([tracks])
    result = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in result] == [f"TR:S{i}" for i in range(1, 4)]


async def test_fragment_of_only_malformed_tracks_raises_media_not_found() -> None:
    """If every track in a fragment is dropped, that's the empty-fragment error, not a crash."""
    tracks = _tracks()
    for track in tracks:
        del track["stationId"]
    provider = _provider([tracks])
    with pytest.raises(MediaNotFoundError):
        await provider.get_playlist_tracks(STATION_ID)


async def test_track_with_a_sized_art_entry_missing_url_does_not_crash() -> None:
    """A size-500 art entry without a url key must not raise KeyError while parsing album art."""
    tracks = _tracks()
    tracks[0]["albumArt"] = [{"size": 500}]
    provider = _provider([tracks])
    result = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in result] == [f"TR:S{i}" for i in range(4)]


async def test_refill_advances_once_the_last_track_is_resolved() -> None:
    """Resolving the final track opens the gate for the next fragment."""
    provider = _provider([_tracks(prefix="A"), _tracks(prefix="B")])
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A3", MediaType.TRACK)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == [f"TR:B{i}" for i in range(4)]


async def test_stream_details_point_at_the_pandora_url() -> None:
    """The provider streams by URL and never buffers audio itself."""
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    details = await provider.get_stream_details("TR:S1", MediaType.TRACK)
    assert details.stream_type is StreamType.HTTP
    assert details.path == "https://audio-sv5-t3-2.pandora.com/access/1.mp4"
    assert details.duration == 180
    assert details.can_seek is True
    assert details.allow_seek is True


async def test_stream_details_for_an_evicted_track_raises() -> None:
    """A track outside the live fragment has a dead URL; fail loudly instead of serving it."""
    provider = _provider([_tracks(prefix="A"), _tracks(prefix="B")])
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A3", MediaType.TRACK)
    await provider.get_playlist_tracks(STATION_ID)
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:A0", MediaType.TRACK)


async def test_stream_details_after_the_urls_expire_raises() -> None:
    """
    A pause long enough to outlive the signed URLs must fail by name, not by a CDN 403.

    Nothing refills a paused queue, so the gate in get_playlist_tracks never runs - this path
    is the only thing standing between a resumed track and an expired URL.
    """
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    fragment.fetched_at -= FRAGMENT_URL_TTL_SECONDS + 1
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:S0", MediaType.TRACK)


async def test_stream_details_after_an_ordinary_pause_still_serves() -> None:
    """
    A pause past the staleness window must still resume: those URLs have not expired.

    Staleness decides whether a fragment is worth replacing on the next refill. Using it to
    refuse playback threw away tracks that would have played perfectly well.
    """
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    fragment.last_activity_at -= FRAGMENT_STALE_SECONDS + 1
    assert fragment.is_stale(time.time()) is True
    details = await provider.get_stream_details("TR:S0", MediaType.TRACK)
    assert details.path == "https://audio-sv5-t3-2.pandora.com/access/0.mp4"


async def test_stream_details_rejects_other_media_types() -> None:
    """Stations expose tracks only; radio is gone."""
    provider = _provider()
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:S0", MediaType.RADIO)


async def test_unknown_track_id_is_refused() -> None:
    """An id no retained fragment holds cannot be streamed."""
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:not-a-real-track", MediaType.TRACK)


async def test_get_track_resolves_from_retained_fragments() -> None:
    """Queue history keeps working for tracks whose fragment is no longer live."""
    provider = _provider([_tracks(prefix="A"), _tracks(prefix="B")])
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A3", MediaType.TRACK)
    await provider.get_playlist_tracks(STATION_ID)
    track = await provider.get_track("TR:A0")
    assert track.name == "Song 0"


async def test_get_track_unknown_raises() -> None:
    """An id from no retained fragment is genuinely gone."""
    provider = _provider()
    with pytest.raises(MediaNotFoundError):
        await provider.get_track("TR:nope")
