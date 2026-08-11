"""Tests for the Pandora provider's parse layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from music_assistant.providers.pandora.parsers import (
    parse_album,
    parse_artist,
    parse_station,
    parse_track,
)
from music_assistant.providers.pandora.provider import PandoraProvider


def _provider() -> PandoraProvider:
    """
    Build a provider bare enough to prove the parsers never reach the network.

    Nothing here can make an API call: `_api_request` is left unbound and there is no
    http_session, so a parser that tried to fetch would raise rather than pass.
    """
    provider = PandoraProvider.__new__(PandoraProvider)
    provider.manifest = Mock(domain="pandora")
    provider.config = Mock(instance_id="pandora--test")
    provider._high_quality_available = False
    return provider


def _track(**overrides: Any) -> dict[str, Any]:
    """Build one raw Pandora fragment track, with the given fields replaced."""
    return {
        "musicId": "S0",
        "pandoraId": "TR:S0",
        "songTitle": "Some Song",
        "artistName": "Some Artist",
        "albumTitle": "Some Album",
        "albumDetailURL": "https://www.pandora.com/artist/album",
        "songDetailURL": "https://www.pandora.com/artist/album/song",
        "trackLength": 180,
        "audioURL": "https://audio-sv5-t3-2.pandora.com/access/0.mp4",
    } | overrides


def test_track_carries_its_pandora_id_and_provider_mapping() -> None:
    """A track is identified by Pandora's own catalogue id on both the item and its mapping."""
    track = parse_track(_provider(), _track())
    assert track.item_id == "TR:S0"
    assert track.name == "Some Song"
    assert track.duration == 180
    mapping = next(iter(track.provider_mappings))
    assert mapping.item_id == "TR:S0"
    assert mapping.provider_domain == "pandora"
    assert mapping.url == "https://www.pandora.com/artist/album/song"


def test_track_prefers_the_size_500_album_art() -> None:
    """Pandora offers several art sizes; the parser picks the one the UI wants."""
    track = parse_track(
        _provider(),
        _track(
            albumArt=[
                {"size": 90, "url": "https://art/90.jpg"},
                {"size": 500, "url": "https://art/500.jpg"},
                {"size": 1080, "url": "https://art/1080.jpg"},
            ]
        ),
    )
    assert track.metadata.images is not None
    assert [image.path for image in track.metadata.images] == ["https://art/500.jpg"]


def test_track_falls_back_to_the_last_art_entry() -> None:
    """Without a size-500 entry the last one still gives the track a thumbnail."""
    track = parse_track(_provider(), _track(albumArt=[{"size": 90, "url": "https://art/90.jpg"}]))
    assert track.metadata.images is not None
    assert [image.path for image in track.metadata.images] == ["https://art/90.jpg"]


def test_track_with_a_sized_art_entry_missing_its_url_still_parses() -> None:
    """A size-500 entry without a url must not raise KeyError while parsing album art."""
    track = parse_track(_provider(), _track(albumArt=[{"size": 500}]))
    assert track.item_id == "TR:S0"
    assert not track.metadata.images


def test_track_tolerates_present_but_null_fields() -> None:
    """Pandora really does send JSON nulls; every read must degrade rather than crash."""
    track = parse_track(
        _provider(), _track(songTitle=None, trackLength=None, albumTitle=None, artistName=None)
    )
    assert track.name == "Unknown Song"
    assert track.duration == 0
    assert track.album is not None
    assert track.album.name == "Unknown Album"
    assert list(track.artists) == []


def test_album_is_addressed_by_the_tracks_own_id() -> None:
    """A fragment names no album of its own, so the track's id stands in for it."""
    album = parse_album(_provider(), _track(), "TR:S0")
    assert album is not None
    assert album.item_id == "TR:S0"
    assert album.name == "Some Album"


def test_album_is_omitted_when_pandora_named_none() -> None:
    """Without an album detail URL there is no album to offer, so the track carries none."""
    assert parse_album(_provider(), _track(albumDetailURL=None), "TR:S0") is None
    assert parse_track(_provider(), _track(albumDetailURL=None)).album is None


def test_artist_is_keyed_by_name() -> None:
    """A fragment names its artist but never identifies it, so the name is the id."""
    artist = parse_artist(_provider(), "Some Artist")
    assert artist.item_id == "Some Artist"
    assert artist.name == "Some Artist"


def test_station_is_a_dynamic_playlist() -> None:
    """Stations are endless, so they surface as dynamic playlists rather than fixed ones."""
    playlist = parse_station(_provider(), {"stationId": "station-1", "name": "Coldplay Radio"})
    assert playlist.item_id == "station-1"
    assert playlist.name == "Coldplay Radio"
    assert playlist.is_dynamic is True
    assert playlist.is_editable is False


def test_station_is_editable_only_when_pandora_allows_seeding() -> None:
    """Pandora reports per-station whether it accepts new seeds; mirror that in is_editable."""
    station = {"stationId": "station-1", "name": "Coldplay Radio", "allowAddSeed": True}
    assert parse_station(_provider(), station).is_editable is True


def test_station_prefers_the_size_500_art() -> None:
    """A station's thumbnail follows the same size preference as a track's."""
    playlist = parse_station(
        _provider(),
        {
            "stationId": "station-1",
            "name": "Coldplay Radio",
            "art": [
                {"size": 90, "url": "https://art/90.jpg"},
                {"size": 500, "url": "https://art/500.jpg"},
            ],
        },
    )
    assert playlist.metadata.images is not None
    assert [image.path for image in playlist.metadata.images] == ["https://art/500.jpg"]
