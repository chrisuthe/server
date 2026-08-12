"""Tests for the Pandora provider's parse layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from music_assistant_models.enums import ExternalID

from music_assistant.providers.pandora.parsers import (
    parse_album,
    parse_album_record,
    parse_artist,
    parse_artist_record,
    parse_station,
    parse_track,
    parse_track_record,
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


def test_annotated_track_takes_its_album_and_artist_ids_from_the_record() -> None:
    """A record names the album and artist in Pandora's catalogue; those are the real ids."""
    annotations = {"TR:S0": {"albumId": "AL:900", "artistId": "AR:800"}}
    track = parse_track(_provider(), _track(), annotations)
    assert track.album is not None
    assert track.album.item_id == "AL:900"
    assert track.artists[0].item_id == "AR:800"
    # the record names the ids, but the fragment still names the album and the artist
    assert track.album.name == "Some Album"
    assert track.artists[0].name == "Some Artist"


def test_track_without_an_annotation_keeps_the_fallback_ids() -> None:
    """
    An unannotated track keeps a track-scoped album and a name-keyed artist.

    An omitted map, an empty one and a map annotating some other track must all land here:
    only the ids for this track can change what it resolves to.
    """
    for annotated in (None, {}, {"TR:OTHER": {"albumId": "AL:900"}}):
        track = parse_track(_provider(), _track(), annotated)
        assert track.album is not None
        assert track.album.item_id == "TR:S0"
        assert track.artists[0].item_id == "Some Artist"


def test_track_tolerates_a_present_but_null_annotation() -> None:
    """A key present with a null value must degrade to the fallbacks, not crash."""
    track = parse_track(_provider(), _track(), {"TR:S0": None})
    assert track.album is not None
    assert track.album.item_id == "TR:S0"


_CATALOGUE: dict[str, dict[str, Any]] = {
    "TR:100": {
        "pandoraId": "TR:100",
        "name": "Some Song",
        "albumId": "AL:157378",
        "artistId": "AR:346031",
        "duration": 232,
        "isrc": "GBAYE0601498",
        "rightsInfo": {"hasInteractive": True},
    },
    "AL:157378": {"pandoraId": "AL:157378", "name": "Some Album"},
    "AR:346031": {"pandoraId": "AR:346031", "name": "Some Artist"},
}


def test_track_record_builds_its_album_and_artist_from_the_siblings() -> None:
    """A catalogue track names its album and artist by id; their names come from the map."""
    track = parse_track_record(_provider(), _CATALOGUE["TR:100"], "TR:100", _CATALOGUE)
    assert track.item_id == "TR:100"
    assert track.name == "Some Song"
    assert track.album is not None
    assert track.album.item_id == "AL:157378"
    assert track.album.name == "Some Album"
    assert track.artists[0].item_id == "AR:346031"
    assert track.artists[0].name == "Some Artist"


def test_track_record_duration_is_seconds() -> None:
    """Pandora reports a catalogue track's length in seconds, as MA expects it."""
    assert parse_track_record(_provider(), _CATALOGUE["TR:100"], "TR:100").duration == 232


def test_track_record_carries_its_isrc() -> None:
    """The ISRC is what lets the same recording match across providers."""
    track = parse_track_record(_provider(), _CATALOGUE["TR:100"], "TR:100")
    assert track.get_external_id(ExternalID.ISRC) == "GBAYE0601498"


def test_track_record_without_siblings_is_still_usable() -> None:
    """
    A record whose album and artist are absent from the map still plays.

    Omitted, empty, and a map holding some other id's records must all land here: an album
    built from a missing sibling would carry a real id under the name "Unknown Album".
    """
    unusable: list[dict[str, Any] | None] = [
        None,
        {},
        {"AL:OTHER": {"name": "Some Other Album"}},
        {"AL:157378": None},
    ]
    for annotations in unusable:
        track = parse_track_record(_provider(), _CATALOGUE["TR:100"], "TR:100", annotations)
        assert track.item_id == "TR:100"
        assert track.name == "Some Song"
        assert track.album is None
        assert list(track.artists) == []


def test_track_record_tolerates_present_but_null_fields() -> None:
    """Pandora really does send JSON nulls; every read must degrade rather than crash."""
    record = {"pandoraId": "TR:100", "name": None, "duration": None, "isrc": None}
    track = parse_track_record(_provider(), record, "TR:100", _CATALOGUE)
    assert track.name == "Unknown Track"
    assert track.duration == 0
    assert track.get_external_id(ExternalID.ISRC) is None


def test_album_record_keeps_the_id_it_was_requested_by() -> None:
    """A record without a pandoraId must not break the lookup or rename the album."""
    album = parse_album_record(_provider(), {"name": "Some Album"}, "AL:900")
    assert album.item_id == "AL:900"
    assert album.name == "Some Album"
    assert next(iter(album.provider_mappings)).item_id == "AL:900"


def test_album_record_carries_no_image() -> None:
    """A record's icon.artUrl is a relative path whose CDN base is unmeasured."""
    record = {"name": "Some Album", "icon": {"artUrl": "images/abc/500W_500H.jpg"}}
    assert not parse_album_record(_provider(), record, "AL:900").metadata.images


def test_album_record_without_a_name_falls_back() -> None:
    """A nameless record must still produce a usable album rather than crash."""
    assert parse_album_record(_provider(), {}, "AL:900").name == "Unknown Album"


def test_artist_record_keeps_the_id_and_takes_the_name() -> None:
    """An AR: id must resolve to the artist's name, not to the id as a name."""
    artist = parse_artist_record(_provider(), {"name": "Some Artist"}, "AR:800")
    assert artist.item_id == "AR:800"
    assert artist.name == "Some Artist"


def test_artist_record_without_a_name_falls_back_to_the_id() -> None:
    """A nameless record leaves nothing else to show, so the id has to do."""
    assert parse_artist_record(_provider(), {}, "AR:800").name == "AR:800"


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
