"""Tests for the Pandora provider's dynamic-playlist and streaming surface."""

from __future__ import annotations

import json
import time
from typing import Any, Self, cast
from unittest.mock import AsyncMock, Mock

import pytest
from music_assistant_models.enums import ContentType, MediaType, StreamType
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    MusicAssistantError,
    ProviderUnavailableError,
    ResourceTemporarilyUnavailable,
)
from music_assistant_models.media_items import Album, SearchResults, Track

from music_assistant.constants import CONF_PASSWORD, CONF_USERNAME
from music_assistant.providers.pandora import provider as provider_module
from music_assistant.providers.pandora.constants import (
    ADD_SEED_ENDPOINT,
    CATALOG_ANNOTATE_ENDPOINT,
    CATALOG_DETAILS_ENDPOINT,
    CONF_DEVICE_UUID,
    CREATE_STATION_ENDPOINT,
    PLAYBACK_SOURCE_ENDPOINT,
    QUALITY_HIGH,
    REMOVE_STATION_ENDPOINT,
    RETRY_REASON_STREAM_VIOLATION,
    SEED_SEARCH_ENDPOINT,
    SOD_SEARCH_ENDPOINT,
    STATIONS_ENDPOINT,
)
from music_assistant.providers.pandora.fragments import (
    FRAGMENT_STALE_SECONDS,
    FRAGMENT_URL_TTL_SECONDS,
    MAX_RETAINED_FRAGMENTS,
)
from music_assistant.providers.pandora.provider import PandoraProvider

STATION_ID = "4360491625318318161"


def _tracks(count: int = 4, prefix: str = "S") -> list[dict[str, Any]]:
    """Build `count` raw Pandora track dicts with distinct Pandora ids."""
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
    provider.http_session = Mock(closed=False)
    provider._sessions = {}
    provider._high_quality_available = False
    provider._on_demand_available = False
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


async def test_unentitled_album_is_addressed_by_its_tracks_id() -> None:
    """Without entitlement this is the only album route there is, and the id must round-trip."""
    provider = _provider()
    await provider.get_playlist_tracks(STATION_ID)
    album = await provider.get_album("TR:S0")
    assert album.item_id == "TR:S0"
    assert album.name == "Some Album"


async def test_unentitled_album_is_gone_once_its_track_ages_out() -> None:
    """A track-keyed album only exists while the fragment naming it is still retained."""
    prefixes = [chr(ord("A") + index) for index in range(MAX_RETAINED_FRAGMENTS + 1)]
    provider = _provider([_tracks(prefix=prefix) for prefix in prefixes])
    for prefix in prefixes:
        await provider.get_playlist_tracks(STATION_ID)
        await provider.get_stream_details(f"TR:{prefix}3", MediaType.TRACK)
    with pytest.raises(MediaNotFoundError):
        await provider.get_album("TR:A0")


async def test_unentitled_artist_is_identified_by_name() -> None:
    """Pandora names a fragment's artist but never identifies it, so the name is the id."""
    provider = _provider()
    artist = await provider.get_artist("Some Artist")
    assert artist.item_id == "Some Artist"
    assert artist.name == "Some Artist"


async def test_get_track_matches_playlist_tracks_identity() -> None:
    """A track resolves to the same album and artist by either entry point."""
    provider = _provider()
    listed = (await provider.get_playlist_tracks(STATION_ID))[0]
    looked_up = await provider.get_track("TR:S0")
    assert looked_up.album is not None
    assert listed.album is not None
    assert looked_up.album.item_id == listed.album.item_id
    assert looked_up.artists[0].item_id == listed.artists[0].item_id


_HYDRATED = {
    "TR:S0": {"pandoraId": "TR:S0", "albumId": "AL:900", "artistId": "AR:800"},
    "AL:900": {"pandoraId": "AL:900", "name": "Some Album"},
    "AR:800": {"pandoraId": "AR:800", "name": "Some Artist"},
}


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
    provider.http_session = Mock(closed=False)
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


def _breaking_provider(error: Exception) -> PandoraProvider:
    """Build an entitled provider whose annotate call raises the given error."""
    provider, _ = _annotating_provider(_HYDRATED)

    async def _failing_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        if url == CATALOG_ANNOTATE_ENDPOINT:
            raise error
        return {"tracks": _tracks()}

    provider._api_request = _failing_request  # type: ignore[method-assign, assignment]
    return provider


async def test_entitled_account_hydrates_a_fragment() -> None:
    """One batched annotate call per fragment carries the catalogue ids the payload lacks."""
    provider, calls = _annotating_provider(_HYDRATED)
    await provider.get_playlist_tracks(STATION_ID)
    assert len(calls) == 1
    assert calls[0] == {
        "pandoraIds": [f"TR:S{index}" for index in range(4)],
        "annotateAlbumTracks": False,
    }
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    assert fragment.annotations == _HYDRATED


async def test_unentitled_account_does_not_hydrate() -> None:
    """Without on-demand, catalogue ids would only offer albums whose tracks cannot play."""
    provider, calls = _annotating_provider(_HYDRATED, on_demand=False)
    await provider.get_playlist_tracks(STATION_ID)
    assert calls == []
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    assert fragment.annotations == {}


async def test_unentitled_hydration_raises_and_logs_nothing() -> None:
    """
    Hydration answers for an unentitled account itself, rather than through the gated path.

    Routing it through the entitlement check would raise once per fragment fetch, and the
    degradation clause would then swallow that raise and log a warning for every station
    an unentitled listener plays.
    """
    provider, calls = _annotating_provider(_HYDRATED, on_demand=False)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert [track.item_id for track in tracks] == [f"TR:S{index}" for index in range(4)]
    assert calls == []
    assert cast("Mock", provider.logger).warning.mock_calls == []


async def test_hydration_drops_a_non_record_value() -> None:
    """The map is keyed by id, but its values are not guaranteed to be records."""
    provider, _ = _annotating_provider({"TR:S0": None, "TR:S1": {"albumId": "AL:900"}})
    await provider.get_playlist_tracks(STATION_ID)
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    assert fragment.annotations == {"TR:S1": {"albumId": "AL:900"}}


async def test_failed_hydration_still_serves_the_station() -> None:
    """Hydration is metadata enrichment; losing it must not stop playback."""
    provider = _breaking_provider(InvalidDataError("annotate exploded"))
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert len(tracks) == 4
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    assert fragment.annotations == {}


@pytest.mark.parametrize(
    "error",
    [
        InvalidDataError("annotate exploded"),
        MediaNotFoundError("Pandora has no record for the id"),
        ProviderUnavailableError("Pandora server error"),
        ResourceTemporarilyUnavailable("Pandora service issue"),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_hydration_degrades_for_each_caught_annotate_error(error: Exception) -> None:
    """
    Every type in _hydrate's except clause must degrade the station, not fail it.

    Only InvalidDataError was ever exercised on this path before, so narrowing the clause
    to drop any of the other three would break no existing test while letting a 500 or a
    rate limit during hydration propagate and stop the station instead of degrading past it.
    """
    provider = _breaking_provider(error)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert len(tracks) == 4
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    assert fragment.annotations == {}


async def test_hydration_failure_over_a_closed_session_is_not_masked() -> None:
    """
    Degrading past a closed transport would report success over a connection that is gone.

    Several _api_request paths close the session before raising; the next call would then
    fail with a bare RuntimeError somewhere unrelated instead of here.
    """
    provider = _breaking_provider(InvalidDataError("annotate exploded after closing"))
    provider.http_session = Mock(closed=True)
    with pytest.raises(InvalidDataError):
        await provider.get_playlist_tracks(STATION_ID)


async def test_hydration_does_not_swallow_an_auth_failure() -> None:
    """A login failure must surface, not degrade into a silently unhydrated station."""
    provider = _breaking_provider(LoginFailed("Pandora authentication failed after retry"))
    with pytest.raises(LoginFailed):
        await provider.get_playlist_tracks(STATION_ID)


async def test_hydration_never_takes_the_stream_over() -> None:
    """Enrichment must not fight the concurrent-stream limit: that stops another device."""
    provider, _ = _annotating_provider(_HYDRATED)
    reasons: list[frozenset[str]] = []

    async def _recording_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        exhausted_retry_reasons: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        if url == CATALOG_ANNOTATE_ENDPOINT:
            reasons.append(exhausted_retry_reasons)
            requested = (data or {}).get("pandoraIds") or []
            return {item_id: {"name": "Some Artist"} for item_id in requested}
        return {"tracks": _tracks()}

    provider._api_request = _recording_request  # type: ignore[method-assign]
    await provider.get_playlist_tracks(STATION_ID)
    # an id no fragment annotated is the only route left that still calls out
    await provider.get_artist("AR:not-in-any-fragment")
    assert reasons == [frozenset({RETRY_REASON_STREAM_VIOLATION})] * 2


async def test_hydrated_track_uses_catalogue_album_and_artist_ids() -> None:
    """An entitled account's album and artist are the ids the catalogue uses everywhere."""
    provider, _ = _annotating_provider(_HYDRATED)
    tracks = await provider.get_playlist_tracks(STATION_ID)
    assert tracks[0].album is not None
    assert tracks[0].album.item_id == "AL:900"
    assert tracks[0].artists[0].item_id == "AR:800"
    # only TR:S0 was annotated: the rest keep the unhydrated fallbacks, track by track
    fallbacks = [track.album.item_id for track in tracks[1:] if track.album]
    assert fallbacks == ["TR:S1", "TR:S2", "TR:S3"]
    assert {track.artists[0].item_id for track in tracks[1:]} == {"Some Artist"}


async def test_unhydrated_track_keeps_todays_album_and_artist() -> None:
    """Without entitlement the album stays track-scoped and the artist name-keyed."""
    provider, _ = _annotating_provider(_HYDRATED, on_demand=False)
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


async def test_unentitled_account_is_refused_a_persisted_catalogue_album() -> None:
    """
    A library row can outlive the entitlement that created it.

    Music Assistant persists library rows, so an `AL:` id minted while an account was
    entitled can still be requested after the subscription lapses. No fragment holds it in
    this fresh provider, so the lookup would otherwise fall through to a live annotate call.
    """
    provider, calls = _annotating_provider(_HYDRATED, on_demand=False)
    with pytest.raises(MediaNotFoundError):
        await provider.get_album("AL:900")
    assert calls == []


async def test_unentitled_account_is_refused_a_persisted_catalogue_artist() -> None:
    """Same as the album case above, but for a persisted `AR:` id."""
    provider, calls = _annotating_provider(_HYDRATED, on_demand=False)
    with pytest.raises(MediaNotFoundError):
        await provider.get_artist("AR:800")
    assert calls == []


async def test_catalogue_album_reuses_a_hydrated_fragments_record() -> None:
    """Hydration already fetched this record; MA resolves albums per item, so do not refetch."""
    provider, calls = _annotating_provider(_HYDRATED)
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    album = await provider.get_album("AL:900")
    assert album.item_id == "AL:900"
    assert album.name == "Some Album"
    assert calls == []


async def test_catalogue_artist_reuses_a_hydrated_fragments_record() -> None:
    """Same for artists: one batched call per fragment must not become one call per item."""
    provider, calls = _annotating_provider(_HYDRATED)
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    artist = await provider.get_artist("AR:800")
    assert artist.item_id == "AR:800"
    assert artist.name == "Some Artist"
    assert calls == []


async def test_hydrated_track_matches_playlist_tracks_identity() -> None:
    """A hydrated track resolves to the same album and artist by either entry point."""
    provider, _ = _annotating_provider(_HYDRATED)
    listed = (await provider.get_playlist_tracks(STATION_ID))[0]
    looked_up = await provider.get_track("TR:S0")
    assert looked_up.album is not None
    assert listed.album is not None
    assert looked_up.album.item_id == listed.album.item_id == "AL:900"
    assert looked_up.artists[0].item_id == listed.artists[0].item_id == "AR:800"


async def test_get_track_from_a_fragment_makes_no_catalogue_call() -> None:
    """The fragment already describes the track, and Music Assistant resolves items one by one."""
    provider, calls = _annotating_provider(_HYDRATED)
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    track = await provider.get_track("TR:S0")
    assert track.name == "Song 0"
    assert calls == []


# A catalogue track and the siblings annotateObjects returns alongside it, unasked: the
# measured response carried 18 records for 6 requested track ids.
_CATALOGUE_TRACK: dict[str, Any] = {
    "TR:1809020": {
        "pandoraId": "TR:1809020",
        "name": "Catalogue Song",
        "duration": 214,
        "albumId": "AL:157378",
        "artistId": "AR:346031",
    },
    "AL:157378": {
        "pandoraId": "AL:157378",
        "name": "Catalogue Album",
        "artistId": "AR:346031",
    },
    "AR:346031": {"pandoraId": "AR:346031", "name": "Catalogue Artist"},
}


async def test_get_track_resolves_a_catalogue_id_no_fragment_holds() -> None:
    """A track found by search is addressed by id alone, so this is its only route to resolve."""
    provider, _ = _annotating_provider(_CATALOGUE_TRACK)
    track = await provider.get_track("TR:1809020")
    assert track.item_id == "TR:1809020"
    assert track.name == "Catalogue Song"
    assert track.duration == 214


async def test_a_looked_up_catalogue_track_carries_its_album_and_artist() -> None:
    """
    Music Assistant refuses to add an artist-less track to the library, so this decides
    whether a Pandora search result can be favourited at all.

    The siblings are already in the response the lookup made, so reading them costs no
    second request.
    """
    provider, calls = _annotating_provider(_CATALOGUE_TRACK)
    track = await provider.get_track("TR:1809020")
    assert track.album is not None
    assert track.album.item_id == "AL:157378"
    assert track.album.name == "Catalogue Album"
    assert [artist.item_id for artist in track.artists] == ["AR:346031"]
    assert track.artists[0].name == "Catalogue Artist"
    assert len(calls) == 1


async def test_a_cached_catalogue_track_carries_its_album_and_artist() -> None:
    """The record already in hand came with its siblings, and must resolve to the same track."""
    provider, calls = _annotating_provider({**_HYDRATED, **_CATALOGUE_TRACK})
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    track = await provider.get_track("TR:1809020")
    assert track.album is not None
    assert track.album.item_id == "AL:157378"
    assert track.album.name == "Catalogue Album"
    assert [artist.item_id for artist in track.artists] == ["AR:346031"]
    assert calls == []


async def test_a_looked_up_catalogue_album_carries_its_artist() -> None:
    """An album with no artists can never match the same album from another provider."""
    provider, _ = _annotating_provider(_CATALOGUE_TRACK)
    album = await provider.get_album("AL:157378")
    assert album.item_id == "AL:157378"
    assert [artist.item_id for artist in album.artists] == ["AR:346031"]
    assert album.artists[0].name == "Catalogue Artist"


async def test_get_track_reuses_an_already_fetched_record() -> None:
    """A record already in hand is not fetched again, as for albums and artists above."""
    record = {"pandoraId": "TR:X9", "name": "Catalogue Song"}
    provider, calls = _annotating_provider({**_HYDRATED, "TR:X9": record})
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    track = await provider.get_track("TR:X9")
    assert track.name == "Catalogue Song"
    assert calls == []


async def test_unentitled_account_is_refused_a_catalogue_track() -> None:
    """A track this account cannot play must be refused by name, not looked up first."""
    provider, calls = _annotating_provider(_HYDRATED, on_demand=False)
    with pytest.raises(MediaNotFoundError, match="not available on this Pandora account"):
        await provider.get_track("TR:1809020")
    assert calls == []


def _two_station_provider() -> tuple[PandoraProvider, dict[str, Any]]:
    """
    Build a provider whose annotate answer can be switched between two station fetches.

    Returns the provider and the mutable map its annotate call reads, so a test can leave
    one station unhydrated and hydrate the next.
    """
    provider, _ = _annotating_provider(_HYDRATED)
    records: dict[str, Any] = {}

    async def _switchable_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,  # noqa: ARG001
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        if url == CATALOG_ANNOTATE_ENDPOINT:
            return dict(records)
        return {"tracks": _tracks()}

    provider._api_request = _switchable_request  # type: ignore[method-assign, assignment]
    return provider, records


_SUPERSEDED = {
    "TR:S0": {"pandoraId": "TR:S0", "albumId": "AL:700", "artistId": "AR:600"},
    "AL:900": {"pandoraId": "AL:900", "name": "Superseded Album"},
}


async def _hydrate_two_stations(
    provider: PandoraProvider,
    records: dict[str, Any],
    first: dict[str, Any],
    second: dict[str, Any],
    older: str,
) -> None:
    """Fetch station-a then station-b with the given records, then age one of them."""
    records.clear()
    records.update(first)
    await provider.get_playlist_tracks("station-a")
    records.clear()
    records.update(second)
    await provider.get_playlist_tracks("station-b")
    fragment = provider._sessions[older].current
    assert fragment is not None
    fragment.fetched_at -= 60


async def test_freshest_annotations_decide_the_album_and_artist() -> None:
    """
    A song must not resolve to two different albums depending on session insertion order.

    Two stations annotating the same song used to be decided by dict order; the freshest
    fetch is Pandora's latest answer for it and decides both the track's identity and the
    record an album lookup reuses.
    """
    provider, records = _two_station_provider()
    await _hydrate_two_stations(provider, records, _SUPERSEDED, _HYDRATED, older="station-a")
    track = await provider.get_track("TR:S0")
    assert track.album is not None
    assert track.album.item_id == "AL:900"
    assert track.artists[0].item_id == "AR:800"
    assert (await provider.get_album("AL:900")).name == "Some Album"


async def test_freshest_annotations_win_regardless_of_session_order() -> None:
    """
    The freshest-annotations rule must not coincide only with insertion order.

    The test above degrades the first-inserted session, so a regression to any
    insertion-order rule would still pass it. Here station-a is inserted first and holds
    the freshest annotations, while station-b, inserted second, is the superseded one.
    """
    provider, records = _two_station_provider()
    await _hydrate_two_stations(provider, records, _HYDRATED, _SUPERSEDED, older="station-b")
    track = await provider.get_track("TR:S0")
    assert track.album is not None
    assert track.album.item_id == "AL:900"
    assert track.artists[0].item_id == "AR:800"
    assert (await provider.get_album("AL:900")).name == "Some Album"


def _creating_provider(
    search_items: list[dict[str, Any]] | None = None,
    created: dict[str, Any] | None = None,
) -> tuple[PandoraProvider, list[dict[str, Any]]]:
    """
    Build a provider whose fullSearch/createStation calls return canned payloads.

    Returns the provider and a list that records each createStation request body, so a
    test can assert which seed was chosen rather than only what was returned.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider._sessions = {}
    provider._high_quality_available = False
    create_calls: list[dict[str, Any]] = []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return canned search/create payloads instead of calling Pandora."""
        if url == SEED_SEARCH_ENDPOINT:
            return {"items": search_items}
        if url == CREATE_STATION_ENDPOINT:
            create_calls.append(data or {})
            return created or {"stationId": "station-new", "name": "Radiohead Radio"}
        raise AssertionError(f"unexpected endpoint {url}")

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, create_calls


async def test_create_playlist_seeds_from_top_seedable_result() -> None:
    """The first result with a seedable prefix wins, even if a non-seedable one ranks higher."""
    provider, create_calls = _creating_provider(
        search_items=[{"pandoraId": "AL:9"}, {"pandoraId": "AR:123"}, {"pandoraId": "TR:456"}]
    )
    await provider.create_playlist("Radiohead", {MediaType.TRACK})
    assert create_calls[0]["pandoraId"] == "AR:123"


async def test_create_playlist_returns_the_station_pandora_made() -> None:
    """Pandora names the station itself; we return what it gives us."""
    provider, _ = _creating_provider(search_items=[{"pandoraId": "AR:123"}])
    playlist = await provider.create_playlist("Radiohead", {MediaType.TRACK})
    assert playlist.item_id == "station-new"
    assert playlist.name == "Radiohead Radio"
    assert playlist.is_dynamic is True


async def test_create_playlist_without_a_seedable_result_raises() -> None:
    """A search that returns only non-seedable types cannot build a station."""
    provider, create_calls = _creating_provider(search_items=[{"pandoraId": "AL:9"}])
    with pytest.raises(MediaNotFoundError):
        await provider.create_playlist("Nothing", {MediaType.TRACK})
    assert create_calls == []


async def test_create_playlist_tolerates_a_null_items_list() -> None:
    """A present-but-null `items` must raise MediaNotFoundError, not TypeError."""
    provider, _ = _creating_provider(search_items=None)
    with pytest.raises(MediaNotFoundError):
        await provider.create_playlist("Nothing", {MediaType.TRACK})


def _removing_provider(
    allow_delete: bool = True,
) -> tuple[PandoraProvider, list[tuple[str, dict[str, Any]]]]:
    """
    Build a provider that records every API call instead of making one.

    :param allow_delete: Value the CONF_ALLOW_STATION_DELETE config entry reports.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    # a bare Mock would return a truthy Mock from get_value, silently passing the gate
    provider.config = Mock(instance_id="pandora--test")
    provider.config.get_value = Mock(return_value=allow_delete)
    provider.logger = Mock()
    provider._sessions = {}
    provider._high_quality_available = False
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Record the call and return an empty success payload."""
        calls.append((url, data or {}))
        return {}

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, calls


async def test_library_remove_deletes_the_station_when_enabled() -> None:
    """With the opt-in toggle on, removing a station playlist calls removeStation."""
    provider, calls = _removing_provider(allow_delete=True)
    assert await provider.library_remove("station-7", MediaType.PLAYLIST) is True
    assert calls == [(REMOVE_STATION_ENDPOINT, {"stationId": "station-7"})]


async def test_library_remove_does_nothing_when_the_toggle_is_off() -> None:
    """The default is off, and off must mean no destructive call reaches Pandora at all."""
    provider, calls = _removing_provider(allow_delete=False)
    assert await provider.library_remove("station-7", MediaType.PLAYLIST) is False
    assert calls == []


async def test_library_remove_ignores_other_media_types() -> None:
    """Only playlists are stations; anything else must not reach a destructive endpoint."""
    provider, calls = _removing_provider(allow_delete=True)
    assert await provider.library_remove("station-7", MediaType.TRACK) is False
    assert calls == []


async def test_library_add_accepts_a_station_without_calling_pandora() -> None:
    """Stations are already in the account library, so adding one is a no-op success."""
    provider, calls = _removing_provider(allow_delete=False)
    assert await provider.library_add(Mock(media_type=MediaType.PLAYLIST)) is True
    assert calls == []


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
    """Stations only ever surface as playlists, and this account has no catalogue to offer."""
    provider = _provider(stations=_stations(["Coldplay Radio"]))
    results = await provider.search("Coldplay Radio", [MediaType.TRACK])
    assert results == SearchResults()


_CATALOGUE_RESULTS = ["TR:100", "AL:157378", "AR:346031", "TR:101"]

_CATALOGUE_ANNOTATIONS: dict[str, Any] = {
    "TR:100": {
        "pandoraId": "TR:100",
        "name": "Playable Song",
        "albumId": "AL:157378",
        "artistId": "AR:346031",
        "duration": 232,
        "isrc": "GBAYE0601498",
        "rightsInfo": {"hasInteractive": True},
    },
    "TR:101": {
        "pandoraId": "TR:101",
        "name": "Station Only Song",
        "albumId": "AL:157378",
        "artistId": "AR:346031",
        "duration": 199,
        "rightsInfo": {"hasInteractive": False},
    },
    "AL:157378": {"pandoraId": "AL:157378", "name": "Some Album", "artistId": "AR:346031"},
    "AR:346031": {"pandoraId": "AR:346031", "name": "Some Artist"},
}


def _searching_provider(
    on_demand: bool = True,
    stations: list[dict[str, Any]] | None = None,
) -> tuple[PandoraProvider, list[dict[str, Any]]]:
    """
    Build a provider whose getStations and sod/search calls return canned payloads.

    Returns the provider and a list recording each catalogue request body, so a test can
    assert the call was never made at all rather than only that nothing came back.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider._sessions = {}
    provider._high_quality_available = False
    provider._on_demand_available = on_demand
    catalogue_calls: list[dict[str, Any]] = []
    station_list = _stations(["Coldplay Radio"]) if stations is None else stations

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return canned station/catalogue payloads instead of calling Pandora."""
        if url == STATIONS_ENDPOINT:
            return {"stations": station_list}
        if url == SOD_SEARCH_ENDPOINT:
            catalogue_calls.append(data or {})
            return {
                "results": list(_CATALOGUE_RESULTS),
                "annotations": dict(_CATALOGUE_ANNOTATIONS),
                "searchToken": "search-token",
            }
        raise AssertionError(f"unexpected endpoint {url}")

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, catalogue_calls


async def test_entitled_search_returns_catalogue_tracks_and_albums() -> None:
    """An account that can play on demand searches Pandora's catalogue, not just its own."""
    provider, calls = _searching_provider()
    results = await provider.search("coldplay", [MediaType.TRACK, MediaType.ALBUM])
    assert len(calls) == 1
    assert [track.item_id for track in results.tracks] == ["TR:100"]
    assert [album.item_id for album in results.albums] == ["AL:157378"]
    track = cast("Track", results.tracks[0])
    assert track.name == "Playable Song"
    assert track.duration == 232
    assert track.album is not None
    assert track.album.name == "Some Album"
    assert track.artists[0].item_id == "AR:346031"


async def test_a_searched_album_carries_its_artist() -> None:
    """The AR: record is already in the search response, and an artist-less album never matches."""
    provider, _ = _searching_provider()
    results = await provider.search("coldplay", [MediaType.ALBUM])
    album = cast("Album", results.albums[0])
    assert [artist.item_id for artist in album.artists] == ["AR:346031"]
    assert album.artists[0].name == "Some Artist"


async def test_search_drops_a_track_the_account_cannot_play() -> None:
    """A result without hasInteractive would fail on click, so it never becomes a result."""
    provider, _ = _searching_provider()
    results = await provider.search("coldplay", [MediaType.TRACK])
    assert "TR:101" not in [track.item_id for track in results.tracks]


async def test_search_does_not_return_artists() -> None:
    """The response carries AR: records, but this provider has no artist page to open."""
    provider, _ = _searching_provider()
    results = await provider.search(
        "coldplay", [MediaType.TRACK, MediaType.ALBUM, MediaType.ARTIST]
    )
    assert results.artists == []


async def test_unentitled_search_returns_stations_and_makes_no_catalogue_call() -> None:
    """A listener who cannot play a catalogue track must not be offered one."""
    provider, calls = _searching_provider(on_demand=False)
    results = await provider.search("coldplay", [MediaType.PLAYLIST, MediaType.TRACK])
    assert [playlist.name for playlist in results.playlists] == ["Coldplay Radio"]
    assert results.tracks == []
    assert results.albums == []
    assert calls == []


async def test_entitled_search_still_returns_stations() -> None:
    """Catalogue results sit alongside the account's own stations, not instead of them."""
    provider, _ = _searching_provider()
    results = await provider.search("coldplay", [MediaType.PLAYLIST, MediaType.TRACK])
    assert [playlist.name for playlist in results.playlists] == ["Coldplay Radio"]
    assert [track.item_id for track in results.tracks] == ["TR:100"]


async def test_playlist_only_search_makes_no_catalogue_call() -> None:
    """Nothing in a playlist search needs the catalogue, entitled or not."""
    provider, calls = _searching_provider()
    results = await provider.search("coldplay", [MediaType.PLAYLIST])
    assert [playlist.name for playlist in results.playlists] == ["Coldplay Radio"]
    assert calls == []


async def test_catalogue_search_sends_the_measured_body() -> None:
    """The body was measured against the live API; extra keys are guesses, not evidence."""
    provider, calls = _searching_provider()
    await provider.search("coldplay", [MediaType.TRACK, MediaType.ALBUM], limit=10)
    assert calls == [{"query": "coldplay", "types": ["TR", "AL"], "count": 10, "annotate": True}]


async def test_catalogue_search_asks_only_for_the_requested_types() -> None:
    """A track-only search must not spend the result count on albums."""
    provider, calls = _searching_provider()
    await provider.search("coldplay", [MediaType.TRACK])
    assert calls[0]["types"] == ["TR"]


_ALBUM_ID = "AL:1728"
# Pandora lists an album's tracks newest-id-first, which is also playing order: the ids
# descend while the track numbers ascend. Anything that sorts this listing shows it wrong.
_ALBUM_TRACK_IDS = ("TR:21356", "TR:21355", "TR:21354")

_ALBUM_TRACKS: dict[str, Any] = {
    track_id: {
        "pandoraId": track_id,
        "name": f"Album Song {number}",
        "trackNumber": number,
        "duration": 160 + number,
        "albumId": _ALBUM_ID,
        "artistId": "AR:6089",
        "rightsInfo": {"hasInteractive": True},
    }
    for number, track_id in enumerate(_ALBUM_TRACK_IDS, start=1)
}

_ALBUM_SIBLINGS: dict[str, Any] = {
    _ALBUM_ID: {
        "pandoraId": _ALBUM_ID,
        "name": "Double Live",
        "artistId": "AR:6089",
        "tracks": list(_ALBUM_TRACK_IDS),
    },
    "AR:6089": {"pandoraId": "AR:6089", "name": "Garth Brooks"},
}


def _detailing_provider(
    details: dict[str, Any],
    annotated: dict[str, Any] | None = None,
    on_demand: bool = True,
) -> tuple[PandoraProvider, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build a provider whose getDetails and annotateObjects answers are canned separately.

    Returns the provider and one recording list per endpoint, so a test can assert how many
    calls an album listing cost as well as what it produced.

    :param details: The annotations map getDetails answers with.
    :param annotated: The records annotateObjects holds; it answers only for ids it is asked
        for, so a test can leave a track unhydrated by both endpoints.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider.http_session = Mock(closed=False)
    provider._sessions = {}
    provider._high_quality_available = False
    provider._on_demand_available = on_demand
    details_calls: list[dict[str, Any]] = []
    annotate_calls: list[dict[str, Any]] = []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return canned catalogue payloads instead of calling Pandora."""
        if url == CATALOG_DETAILS_ENDPOINT:
            details_calls.append(data or {})
            return {"annotations": dict(details)}
        if url == CATALOG_ANNOTATE_ENDPOINT:
            annotate_calls.append(data or {})
            held = annotated or {}
            return {
                item_id: held[item_id]
                for item_id in (data or {}).get("pandoraIds") or []
                if item_id in held
            }
        raise AssertionError(f"unexpected endpoint {url}")

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, details_calls, annotate_calls


async def test_album_tracks_come_from_the_details_response_alone_when_it_hydrates_them() -> None:
    """If getDetails already carries every record, listing the album costs exactly one call."""
    provider, details_calls, annotate_calls = _detailing_provider(
        {**_ALBUM_SIBLINGS, **_ALBUM_TRACKS}
    )
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.item_id for track in tracks] == list(_ALBUM_TRACK_IDS)
    assert len(details_calls) == 1
    assert annotate_calls == []


async def test_album_tracks_pandora_did_not_hydrate_are_fetched_in_one_batch() -> None:
    """
    Whether getDetails hydrates an album's tracks is unmeasured, so the listing must work
    either way - and the way that is not one call must still not be one call per track.
    """
    provider, details_calls, annotate_calls = _detailing_provider(_ALBUM_SIBLINGS, _ALBUM_TRACKS)
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.item_id for track in tracks] == list(_ALBUM_TRACK_IDS)
    assert len(details_calls) == 1
    assert annotate_calls == [{"pandoraIds": list(_ALBUM_TRACK_IDS), "annotateAlbumTracks": False}]


async def test_album_tracks_batch_only_the_records_the_details_response_lacks() -> None:
    """A partly hydrated response must widen the one follow-up call, not repeat the whole album."""
    provider, _, annotate_calls = _detailing_provider(
        {**_ALBUM_SIBLINGS, "TR:21355": _ALBUM_TRACKS["TR:21355"]}, _ALBUM_TRACKS
    )
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.item_id for track in tracks] == list(_ALBUM_TRACK_IDS)
    assert len(annotate_calls) == 1
    assert annotate_calls[0]["pandoraIds"] == ["TR:21356", "TR:21354"]


async def test_album_tracks_keep_pandoras_order() -> None:
    """Pandora's order is the album's order; sorting by id or name would reverse this one."""
    provider, _, _ = _detailing_provider({**_ALBUM_SIBLINGS, **_ALBUM_TRACKS})
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.name for track in tracks] == ["Album Song 1", "Album Song 2", "Album Song 3"]
    assert [track.track_number for track in tracks] == [1, 2, 3]


async def test_album_tracks_carry_their_catalogue_identities() -> None:
    """An album track has to resolve and play by id, and to lead back to its album and artist."""
    provider, _, _ = _detailing_provider({**_ALBUM_SIBLINGS, **_ALBUM_TRACKS})
    track = (await provider.get_album_tracks(_ALBUM_ID))[0]
    assert track.item_id == "TR:21356"
    assert track.duration == 161
    assert track.album is not None
    assert track.album.item_id == _ALBUM_ID
    assert track.album.name == "Double Live"
    assert track.artists[0].item_id == "AR:6089"


async def test_album_tracks_drop_a_track_the_account_cannot_play() -> None:
    """The same rights check search applies: a listing entry that fails on click is not an entry."""
    unplayable = {**_ALBUM_TRACKS["TR:21355"], "rightsInfo": {"hasInteractive": False}}
    provider, _, _ = _detailing_provider(
        {**_ALBUM_SIBLINGS, **_ALBUM_TRACKS, "TR:21355": unplayable}
    )
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.item_id for track in tracks] == ["TR:21356", "TR:21354"]


async def test_album_tracks_drop_a_track_no_record_ever_arrived_for() -> None:
    """Neither endpoint answered for this id, so it cannot be described - or played."""
    listed = [*_ALBUM_TRACK_IDS, "TR:00000"]
    album = {**_ALBUM_SIBLINGS[_ALBUM_ID], "tracks": listed}
    provider, _, annotate_calls = _detailing_provider(
        {**_ALBUM_SIBLINGS, _ALBUM_ID: album, **_ALBUM_TRACKS}
    )
    tracks = await provider.get_album_tracks(_ALBUM_ID)
    assert [track.item_id for track in tracks] == list(_ALBUM_TRACK_IDS)
    assert annotate_calls[0]["pandoraIds"] == ["TR:00000"]


async def test_album_tracks_send_the_measured_body() -> None:
    """This endpoint takes a single `pandoraId`; the batched annotator is the one taking a list."""
    provider, details_calls, _ = _detailing_provider({**_ALBUM_SIBLINGS, **_ALBUM_TRACKS})
    await provider.get_album_tracks(_ALBUM_ID)
    assert details_calls == [{"pandoraId": _ALBUM_ID}]


async def test_album_tracks_never_take_the_stream_over() -> None:
    """Listing an album is metadata: taking the stream over would stop another device."""
    reasons: list[frozenset[str]] = []

    async def _recording_request(
        method: str,  # noqa: ARG001
        url: str,  # noqa: ARG001
        data: dict[str, Any] | None = None,  # noqa: ARG001
        exhausted_retry_reasons: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        reasons.append(exhausted_retry_reasons)
        return {"annotations": {**_ALBUM_SIBLINGS, **_ALBUM_TRACKS}}

    provider, _, _ = _detailing_provider({})
    provider._api_request = _recording_request  # type: ignore[method-assign]
    await provider.get_album_tracks(_ALBUM_ID)
    assert reasons == [frozenset({RETRY_REASON_STREAM_VIOLATION})]


async def test_unknown_album_id_is_refused() -> None:
    """Pandora holding no record for the id is a missing album, not an empty one."""
    provider, _, _ = _detailing_provider({})
    with pytest.raises(MediaNotFoundError):
        await provider.get_album_tracks("AL:does-not-exist")


async def test_a_station_scoped_album_has_no_tracklist() -> None:
    """A fragment album is keyed by one of its tracks and Pandora holds no album record for it."""
    provider, details_calls, annotate_calls = _detailing_provider({**_ALBUM_SIBLINGS})
    with pytest.raises(MediaNotFoundError):
        await provider.get_album_tracks("TR:S0")
    assert details_calls == []
    assert annotate_calls == []


async def test_unentitled_account_is_refused_an_album_tracklist() -> None:
    """A library row outlives the entitlement that made it; its tracks still cannot play."""
    provider, details_calls, annotate_calls = _detailing_provider(
        {**_ALBUM_SIBLINGS, **_ALBUM_TRACKS}, on_demand=False
    )
    with pytest.raises(MediaNotFoundError, match="not available on this Pandora account"):
        await provider.get_album_tracks(_ALBUM_ID)
    assert details_calls == []
    assert annotate_calls == []


async def test_station_is_editable_only_when_pandora_allows_seeding() -> None:
    """Pandora reports per-station whether it accepts new seeds; mirror that in is_editable."""
    provider = _provider(
        stations=[
            {"stationId": "s1", "name": "Artist Radio", "allowAddSeed": True},
            {"stationId": "s2", "name": "Genre Radio", "allowAddSeed": False},
        ]
    )
    playlists = {playlist.item_id: playlist async for playlist in provider.get_library_playlists()}
    assert playlists["s1"].is_editable is True
    assert playlists["s2"].is_editable is False


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
    """A track lacking a pandoraId cannot be handed to the queue: the id is identity now."""
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


async def test_a_fresher_session_serves_a_track_another_holds_expired() -> None:
    """
    Stations overlap: one station's expired copy must not fail a track another can still play.

    Refusing on the first match made playback failure depend on which station was browsed
    first, which is not something the user can see or influence.
    """
    provider = _provider()
    await provider.get_playlist_tracks("station-a")
    await provider.get_playlist_tracks("station-b")
    stale = provider._sessions["station-a"].current
    assert stale is not None
    stale.fetched_at -= FRAGMENT_URL_TTL_SECONDS + 1
    details = await provider.get_stream_details("TR:S0", MediaType.TRACK)
    assert details.item_id == "TR:S0"
    assert details.path == "https://audio-sv5-t3-2.pandora.com/access/0.mp4"


async def test_every_copy_expired_still_raises_the_named_error() -> None:
    """With no session able to serve it, the failure is still the paused-too-long one."""
    provider = _provider()
    await provider.get_playlist_tracks("station-a")
    await provider.get_playlist_tracks("station-b")
    for station in ("station-a", "station-b"):
        fragment = provider._sessions[station].current
        assert fragment is not None
        fragment.fetched_at -= FRAGMENT_URL_TTL_SECONDS + 1
    with pytest.raises(MediaNotFoundError, match="expired while playback was stopped"):
        await provider.get_stream_details("TR:S0", MediaType.TRACK)


async def test_the_serving_session_is_the_one_marked_as_having_served() -> None:
    """
    Recording the hand-out on another station's fragment corrupts both stations' refills.

    The served track stays pending where it played and is re-offered, while the fragment
    that never served it is driven towards spent.
    """
    provider = _provider()
    await provider.get_playlist_tracks("station-a")
    await provider.get_playlist_tracks("station-b")
    older = provider._sessions["station-a"].current
    newer = provider._sessions["station-b"].current
    assert older is not None
    assert newer is not None
    older.fetched_at -= 60
    await provider.get_stream_details("TR:S0", MediaType.TRACK)
    assert newer.served == {"TR:S0"}
    assert older.served == set()


async def test_get_track_uses_the_freshest_fragment() -> None:
    """
    A song must not resolve differently depending on session insertion order.

    Two stations holding the same song used to be decided by dict order; the freshest fetch
    is Pandora's latest answer for it and decides it now.
    """
    stale_tracks = _tracks()
    stale_tracks[0] = {**stale_tracks[0], "songTitle": "Stale Song 0"}
    provider = _provider(payloads=[stale_tracks, _tracks()])
    await provider.get_playlist_tracks("station-a")
    await provider.get_playlist_tracks("station-b")
    stale = provider._sessions["station-a"].current
    assert stale is not None
    stale.fetched_at -= 60
    track = await provider.get_track("TR:S0")
    assert track.name == "Song 0"


async def test_freshest_fragment_wins_regardless_of_session_order() -> None:
    """
    Freshest-fragment selection must not coincide only with insertion order.

    The tests above always make the first-inserted session, station-a, the degraded one,
    so a regression to any insertion-order rule would still pass them. Here station-a is
    inserted first but holds the freshest fragment - a station refetching after another
    was opened - while station-b, inserted second, is the one that has gone stale.
    """
    stale_tracks = _tracks()
    stale_tracks[0] = {**stale_tracks[0], "songTitle": "Stale Song 0"}
    provider = _provider(payloads=[_tracks(), stale_tracks])
    await provider.get_playlist_tracks("station-a")
    await provider.get_playlist_tracks("station-b")
    stale = provider._sessions["station-b"].current
    assert stale is not None
    stale.fetched_at -= 60
    track = await provider.get_track("TR:S0")
    assert track.name == "Song 0"


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


MINTED_URL = "https://audio-dc6-t3-1.pandora.com/access/minted.mp4"
DEVICE_UUID = "1f1c5d0e-2a3b-4c5d-8e9f-0a1b2c3d4e5f"

_MINTED_ITEM = {
    "pandoraId": "TR:S0",
    "audioUrl": MINTED_URL,
    "duration": 214,
    "encoding": "aacplus",
    "fileGain": "2.45",
    "interactions": ["SKIP", "SEEK"],
    "playerStyle": "on_demand",
    "trackToken": "a-track-token",
}


def _minting_provider(
    item: dict[str, Any] | None = None,
    on_demand: bool = True,
    payloads: list[list[dict[str, Any]]] | None = None,
) -> tuple[PandoraProvider, list[tuple[str, dict[str, Any]]]]:
    """
    Build a provider whose playback/source calls return a canned minted item.

    Returns the provider and a list recording every API call it made, so a test can assert
    that a route which must not call Pandora made no call at all.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider.logger = Mock()
    provider.http_session = Mock(closed=False)
    provider._sessions = {}
    provider._high_quality_available = False
    provider._on_demand_available = on_demand
    provider._device_uuid = DEVICE_UUID
    pending = list(payloads or [])
    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_api_request(
        method: str,  # noqa: ARG001
        url: str,
        data: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG001
    ) -> dict[str, Any]:
        """Return canned fragment/mint payloads instead of calling Pandora."""
        calls.append((url, data or {}))
        if url == PLAYBACK_SOURCE_ENDPOINT:
            return {"item": _MINTED_ITEM if item is None else item}
        if url == CATALOG_ANNOTATE_ENDPOINT:
            return {}
        return {"tracks": pending.pop(0) if pending else _tracks()}

    provider._api_request = _fake_api_request  # type: ignore[method-assign, assignment]
    return provider, calls


async def test_a_live_fragment_plays_without_minting() -> None:
    """The URL a fragment already carries is free; minting one instead would cost a play."""
    provider, calls = _minting_provider()
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    details = await provider.get_stream_details("TR:S0", MediaType.TRACK)
    assert details.path == "https://audio-sv5-t3-2.pandora.com/access/0.mp4"
    assert calls == []


async def test_a_catalogue_track_is_minted_per_play() -> None:
    """A track no fragment holds is playable, and the mint body is what Pandora was measured to want."""
    provider, calls = _minting_provider()
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.stream_type is StreamType.HTTP
    assert details.path == MINTED_URL
    assert details.duration == 214
    assert [url for url, _ in calls] == [PLAYBACK_SOURCE_ENDPOINT]
    assert calls[0][1] == {
        "sourceId": "TR:1809020",
        "includeItem": True,
        "includeSource": True,
        "deviceUuid": DEVICE_UUID,
    }


async def test_the_mint_names_its_own_encoding_not_the_account_preference() -> None:
    """
    An account whose fragments arrive as MP3 still gets the encoding the mint actually made.

    Describing an AAC stream as MP3 hands the player the wrong container for the bytes, so
    the account's quality preference must not reach a minted stream at all.
    """
    provider, _ = _minting_provider()
    provider._high_quality_available = True
    provider.config = Mock(instance_id="pandora--test", get_value=Mock(return_value=QUALITY_HIGH))
    assert provider._audio_format().content_type is ContentType.MP3
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.audio_format.content_type is ContentType.AAC


async def test_an_mp3_mint_is_described_as_mp3() -> None:
    """Pandora mints MP3 too, and a standard-quality account must not force AAC onto it."""
    provider, _ = _minting_provider({**_MINTED_ITEM, "encoding": "mp3-hifi"})
    assert provider._audio_format().content_type is ContentType.AAC
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.audio_format.content_type is ContentType.MP3


async def test_a_mint_that_names_no_encoding_falls_back_to_aac() -> None:
    """Every measured mint named one, but an unnamed encoding must still play rather than fail."""
    item = {key: value for key, value in _MINTED_ITEM.items() if key != "encoding"}
    provider, _ = _minting_provider(item)
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.audio_format.content_type is ContentType.AAC


async def test_a_minted_stream_carries_pandoras_loudness_and_seekability() -> None:
    """Both are things the fragment path cannot know and Music Assistant would otherwise guess."""
    provider, _ = _minting_provider()
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.loudness == pytest.approx(-20.45)
    assert details.can_seek is True
    assert details.allow_seek is True


async def test_a_source_that_may_not_be_seeked_is_not_seekable() -> None:
    """Seekability follows Pandora's answer, not the presence of a duration."""
    provider, _ = _minting_provider({**_MINTED_ITEM, "interactions": ["SKIP"]})
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.duration == 214
    assert details.can_seek is False
    assert details.allow_seek is False


@pytest.mark.parametrize("file_gain", [None, "", "loud", "NaN", {}, []], ids=repr)
async def test_an_unusable_file_gain_yields_no_loudness(file_gain: Any) -> None:
    """A gain Pandora sent in a shape we cannot read must leave the track unmeasured, not crash."""
    provider, _ = _minting_provider({**_MINTED_ITEM, "fileGain": file_gain})
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.loudness is None
    assert details.path == MINTED_URL


async def test_a_missing_file_gain_yields_no_loudness() -> None:
    """Not every item carries a gain, and its absence is not a playback failure."""
    item = {key: value for key, value in _MINTED_ITEM.items() if key != "fileGain"}
    provider, _ = _minting_provider(item)
    details = await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert details.loudness is None


async def test_a_mint_without_an_audio_url_is_refused() -> None:
    """A 200 naming no URL must fail here, not hand ffmpeg a None to open."""
    provider, _ = _minting_provider({**_MINTED_ITEM, "audioUrl": None})
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:1809020", MediaType.TRACK)


async def test_an_unentitled_account_is_refused_before_the_mint() -> None:
    """The refusal is certain, so it costs no round trip and no NO_ENTITLEMENTS answer."""
    provider, calls = _minting_provider(on_demand=False)
    with pytest.raises(MediaNotFoundError, match="not available on this Pandora account"):
        await provider.get_stream_details("TR:1809020", MediaType.TRACK)
    assert calls == []


async def test_an_aged_out_station_track_is_minted_again_when_entitled() -> None:
    """
    A track that has scrolled out of the live fragment plays again, deliberately.

    It used to raise, which made every walk of recently-played history noisy. Re-minting
    spends an interactive play rather than a radio play, so it is Premium-only.
    """
    provider, calls = _minting_provider(payloads=[_tracks(prefix="A"), _tracks(prefix="B")])
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A3", MediaType.TRACK)
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    details = await provider.get_stream_details("TR:A0", MediaType.TRACK)
    assert details.path == MINTED_URL
    assert [url for url, _ in calls] == [PLAYBACK_SOURCE_ENDPOINT]


async def test_an_aged_out_station_track_still_raises_without_entitlement() -> None:
    """Today's behaviour is unchanged for an account that cannot play on demand."""
    provider, calls = _minting_provider(
        on_demand=False, payloads=[_tracks(prefix="A"), _tracks(prefix="B")]
    )
    await provider.get_playlist_tracks(STATION_ID)
    await provider.get_stream_details("TR:A3", MediaType.TRACK)
    await provider.get_playlist_tracks(STATION_ID)
    calls.clear()
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("TR:A0", MediaType.TRACK)
    assert calls == []


async def test_an_expired_fragment_track_is_minted_again_when_entitled() -> None:
    """A pause long enough to outlive the signed URLs resumes by minting a fresh one."""
    provider, calls = _minting_provider()
    await provider.get_playlist_tracks(STATION_ID)
    fragment = provider._sessions[STATION_ID].current
    assert fragment is not None
    fragment.fetched_at -= FRAGMENT_URL_TTL_SECONDS + 1
    calls.clear()
    details = await provider.get_stream_details("TR:S0", MediaType.TRACK)
    assert details.path == MINTED_URL
    assert [url for url, _ in calls] == [PLAYBACK_SOURCE_ENDPOINT]


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


class _LoginResponse:
    """Stand-in for the aiohttp POST `_authenticate` reads the login payload from."""

    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def post(self, *args: Any, **kwargs: Any) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload


async def _login(monkeypatch: pytest.MonkeyPatch, flags: list[str]) -> PandoraProvider:
    """Run a real _authenticate against a canned login payload carrying the given flags."""
    provider = _provider()
    provider.http_session = _LoginResponse(  # type: ignore[assignment]
        {"authToken": "token", "listenerId": "listener", "config": {"flags": flags}}
    )
    monkeypatch.setattr(provider_module, "get_csrf_token", AsyncMock(return_value="csrf"))
    await provider._authenticate("user", "secret")
    return provider


async def test_authentication_records_the_high_quality_entitlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requested audio format hangs on this one assignment out of the login payload."""
    provider = await _login(monkeypatch, ["highQualityStreamingAvailable"])
    assert provider._high_quality_available is True


async def test_authentication_leaves_a_free_account_unentitled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A login without the flag must not leave it set from a previous account."""
    provider = await _login(monkeypatch, ["adSupportedSkip"])
    assert provider._high_quality_available is False


async def test_authentication_records_the_on_demand_entitlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured Premium flag set must set the on-demand playback gate."""
    provider = await _login(
        monkeypatch,
        [
            "adFreeReplay",
            "adFreeSkip",
            "highQualityStreamingAvailable",
            "onDemand",
            "seenWebPremiumWelcome",
        ],
    )
    assert provider._on_demand_available is True


async def test_authentication_leaves_a_free_account_without_on_demand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The measured free-tier flag set carries no onDemand flag and must stay unentitled."""
    provider = await _login(monkeypatch, ["adSupportedReplay", "adSupportedSkip"])
    assert provider._on_demand_available is False


def _init_provider(
    stored_setup: dict[str, Any] | None = None,
) -> tuple[PandoraProvider, dict[str, Any]]:
    """
    Build a provider ready for handle_async_init, with auth and setup storage stubbed.

    setup_data starts pre-seeded with credentials so the login guard passes; a caller
    seeding CONF_DEVICE_UUID simulates a provider reloading with an identity already stored.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.mass = Mock()
    provider.config = Mock(instance_id="pandora--test")
    provider.config.get_value = Mock(return_value="")
    provider.logger = Mock()
    provider._authenticate = AsyncMock()  # type: ignore[method-assign]
    setup_data: dict[str, Any] = {CONF_USERNAME: "user", CONF_PASSWORD: "secret"}
    setup_data.update(stored_setup or {})

    def _get_setup_value(key: str, default: Any = None) -> Any:
        return setup_data.get(key, default)

    def _update_setup_data(key: str, value: Any, immediate: bool = True) -> None:  # noqa: ARG001
        setup_data[key] = value

    provider.get_setup_value = _get_setup_value  # type: ignore[method-assign]
    provider._update_setup_data = _update_setup_data  # type: ignore[method-assign]
    return provider, setup_data


async def test_device_uuid_is_generated_once_and_reused_across_loads() -> None:
    """A restart must not look like a new device to an account limited to one stream."""
    provider, setup_data = _init_provider()
    await provider.handle_async_init()
    first_uuid = provider._device_uuid
    assert setup_data[CONF_DEVICE_UUID] == first_uuid

    reloaded, _ = _init_provider(stored_setup={CONF_DEVICE_UUID: first_uuid})
    await reloaded.handle_async_init()
    assert reloaded._device_uuid == first_uuid


async def test_device_uuid_never_appears_in_a_log_call() -> None:
    """The device identity must never be logged, even incidentally."""
    provider, _ = _init_provider()
    await provider.handle_async_init()
    device_uuid = provider._device_uuid
    for call in cast("Mock", provider.logger).mock_calls:
        assert device_uuid not in str(call)


async def test_add_playlist_tracks_seeds_the_station_with_each_track() -> None:
    """A track id is already the seed id addSeed wants, so each one goes straight through."""
    provider, calls = _removing_provider(allow_delete=False)
    await provider.add_playlist_tracks(STATION_ID, ["TR:1", "TR:2"])
    assert calls == [
        (ADD_SEED_ENDPOINT, {"stationId": STATION_ID, "pandoraId": "TR:1"}),
        (ADD_SEED_ENDPOINT, {"stationId": STATION_ID, "pandoraId": "TR:2"}),
    ]


async def test_remove_playlist_tracks_refuses_positional_removal() -> None:
    """Positions address the live fragment, not the seed list, so removal cannot be honoured."""
    provider, calls = _removing_provider(allow_delete=False)
    with pytest.raises(MusicAssistantError):
        await provider.remove_playlist_tracks(STATION_ID, (0, 2))
    assert calls == []


class _ApiResponse:
    """Stand-in for the aiohttp session+response `_api_request` reads status and body from."""

    def __init__(self, status: int, payload: Any = None, *, bad_json: bool = False) -> None:
        self.status = status
        self._payload = payload
        self._bad_json = bad_json
        self.close = AsyncMock()

    def request(self, *args: Any, **kwargs: Any) -> Self:
        return self

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def json(self) -> Any:
        if self._bad_json:
            raise json.JSONDecodeError("bad json", "", 0)
        return self._payload


def _api_provider(response: _ApiResponse) -> PandoraProvider:
    """Build a bare provider whose http_session is the given canned response stand-in."""
    provider = PandoraProvider.__new__(PandoraProvider)
    provider._csrf_token = "csrf"
    provider._auth_token = "auth"
    provider._socks_proxy = True
    provider.http_session = response  # type: ignore[assignment]
    return provider


async def test_no_entitlements_400_names_the_refusal_and_leaves_session_open() -> None:
    """A free account's on-demand refusal must be legible, not a generic close-and-raise."""
    response = _ApiResponse(
        400,
        {
            "message": "Listener does not have rights to play source AP:16722:15160249",
            "errorCode": 0,
            "errorString": "NO_ENTITLEMENTS",
        },
    )
    provider = _api_provider(response)
    with pytest.raises(MediaNotFoundError, match="not available"):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_not_called()


async def test_no_playable_content_400_names_the_refusal_and_leaves_session_open() -> None:
    """An empty source's refusal must be legible, not a generic close-and-raise."""
    response = _ApiResponse(
        400,
        {
            "message": "Source does not have any playable tracks",
            "errorCode": 0,
            "errorString": "NO_PLAYABLE_CONTENT",
        },
    )
    provider = _api_provider(response)
    with pytest.raises(MediaNotFoundError, match="nothing playable"):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_not_called()


async def test_other_400_body_still_raises_the_generic_api_error() -> None:
    """A 400 that isn't the entitlement refusal keeps the pre-existing behaviour."""
    response = _ApiResponse(400, {"errorString": "SOME_OTHER_ERROR"})
    provider = _api_provider(response)
    with pytest.raises(InvalidDataError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()


async def test_non_json_400_body_does_not_crash() -> None:
    """A 400 whose body isn't JSON must still raise cleanly, not a parse error."""
    response = _ApiResponse(400, bad_json=True)
    provider = _api_provider(response)
    with pytest.raises(InvalidDataError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()


async def test_null_json_400_body_does_not_crash() -> None:
    """A 400 whose body is JSON null must raise the generic error, not AttributeError."""
    response = _ApiResponse(400, payload=None)
    provider = _api_provider(response)
    with pytest.raises(InvalidDataError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()


async def test_list_json_400_body_does_not_crash() -> None:
    """A 400 whose body is a JSON list must raise the generic error, not AttributeError."""
    response = _ApiResponse(400, payload=[])
    provider = _api_provider(response)
    with pytest.raises(InvalidDataError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()


async def test_string_json_400_body_does_not_crash() -> None:
    """A 400 whose body is a JSON string must raise the generic error, not AttributeError."""
    response = _ApiResponse(400, payload="error message")
    provider = _api_provider(response)
    with pytest.raises(InvalidDataError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()


async def test_404_still_closes_the_session() -> None:
    """Other status branches keep closing the session; only the 400 refusal path changed."""
    response = _ApiResponse(404)
    provider = _api_provider(response)
    with pytest.raises(MediaNotFoundError):
        await provider._api_request("GET", "https://example.com/x")
    response.close.assert_called_once()
