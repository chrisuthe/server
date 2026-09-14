"""Synchronous, IO-free parsers turning Pandora payloads into Music Assistant media items."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, ImageType
from music_assistant_models.media_items import (
    Album,
    Artist,
    MediaItemImage,
    ProviderMapping,
    Radio,
    Track,
)
from music_assistant_models.unique_list import UniqueList

from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from .provider import PandoraProvider


def _parse_art(provider: PandoraProvider, art: list[dict[str, Any]]) -> MediaItemImage | None:
    """Parse the thumbnail for a Pandora art list, if any entry names a URL."""
    art_url = next(
        (entry.get("url") for entry in art if entry.get("size") == 500), art[-1].get("url")
    )
    if not art_url:
        return None
    return MediaItemImage(
        type=ImageType.THUMB,
        path=str(art_url),
        provider=provider.instance_id,
        remotely_accessible=True,
    )


def parse_station(provider: PandoraProvider, station: dict[str, Any]) -> Radio:
    """Parse a station object into a dynamic radio station."""
    radio = Radio(
        item_id=station["stationId"],
        provider=provider.instance_id,
        name=station["name"],
        is_dynamic=True,
        provider_mappings={
            ProviderMapping(
                item_id=station["stationId"],
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )
    if (art := station.get("art")) and (image := _parse_art(provider, art)):
        radio.metadata.add_image(image)
    return radio


def parse_track(
    provider: PandoraProvider,
    obj: dict[str, Any],
    annotations: dict[str, Any] | None = None,
    available: bool = True,
) -> Track:
    """
    Parse a raw fragment track into a Track.

    :param obj: One raw track from a Pandora fragment.
    :param annotations: Catalogue records keyed by pandoraId; empty when nothing was annotated.
    :param available: Whether the track can be played from where it is listed.
    """
    name, version = parse_title_and_version(obj.get("songTitle") or "Unknown Song")
    track_id = obj["pandoraId"]
    record = (annotations or {}).get(track_id) or {}
    track = Track(
        item_id=track_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        duration=int(obj.get("trackLength") or 0),
        provider_mappings={
            ProviderMapping(
                item_id=track_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
                audio_format=provider._audio_format(),
                url=obj.get("songDetailURL"),
                available=available,
            )
        },
    )
    if (album_art := obj.get("albumArt")) and (image := _parse_art(provider, album_art)):
        track.metadata.add_image(image)
    if artist_name := obj.get("artistName"):
        track.artists = UniqueList([parse_artist(provider, artist_name, record.get("artistId"))])
    track.album = parse_album(provider, obj, track_id, record)
    return track


def parse_album(
    provider: PandoraProvider,
    obj: dict[str, Any],
    track_id: str,
    record: dict[str, Any] | None = None,
) -> Album | None:
    """
    Parse the album a fragment track belongs to, if the API named one.

    :param obj: One raw track from a Pandora fragment.
    :param track_id: The track's own Pandora id, the album id when no record names one.
    :param record: The track's catalogue record, if one has been fetched.
    """
    if not (url := obj.get("albumDetailURL")):
        return None
    album_id = str((record or {}).get("albumId") or track_id)
    name, version = parse_title_and_version(obj.get("albumTitle") or "Unknown Album")
    return Album(
        item_id=album_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        provider_mappings={
            ProviderMapping(
                item_id=album_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
                url=url,
            )
        },
    )


def parse_artist(provider: PandoraProvider, name: str, artist_id: str | None = None) -> Artist:
    """
    Parse an artist.

    :param name: The artist's name, also its id when no catalogue id is given.
    :param artist_id: The artist's catalogue id, if one has been fetched.
    """
    item_id = artist_id or name
    return Artist(
        item_id=item_id,
        name=name,
        provider=provider.instance_id,
        provider_mappings={
            ProviderMapping(
                item_id=item_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )


def parse_track_record(
    provider: PandoraProvider,
    record: dict[str, Any],
    track_id: str,
    annotations: dict[str, Any] | None = None,
) -> Track:
    """
    Parse a track from a Pandora catalogue record.

    :param record: The catalogue record Pandora returned for the track.
    :param track_id: The id the track was requested by, which it keeps.
    :param annotations: The records returned alongside this one, keyed by pandoraId, which
        supply the track's album and artist.
    """
    name, version = parse_title_and_version(str(record.get("name") or "Unknown Track"))
    track = Track(
        item_id=track_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        duration=int(record.get("duration") or 0),
        track_number=int(record.get("trackNumber") or 0),
        provider_mappings={
            ProviderMapping(
                item_id=track_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )
    if isrc := record.get("isrc"):
        track.external_ids.add((ExternalID.ISRC, str(isrc)))
    siblings = annotations or {}
    album_id = record.get("albumId")
    if album_id and isinstance(album_record := siblings.get(album_id), dict):
        track.album = parse_album_record(provider, album_record, str(album_id), siblings)
    artist_id = record.get("artistId")
    if artist_id and isinstance(artist_record := siblings.get(artist_id), dict):
        track.artists = UniqueList([parse_artist_record(provider, artist_record, str(artist_id))])
    return track


def parse_album_record(
    provider: PandoraProvider,
    record: dict[str, Any],
    album_id: str,
    annotations: dict[str, Any] | None = None,
) -> Album:
    """
    Parse an album from a Pandora catalogue record.

    :param record: The catalogue record Pandora returned for the album.
    :param album_id: The id the album was requested by, which it keeps.
    :param annotations: The records returned alongside this one, keyed by pandoraId, which
        supply the album's artist.
    """
    name, version = parse_title_and_version(str(record.get("name") or "Unknown Album"))
    album = Album(
        item_id=album_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        provider_mappings={
            ProviderMapping(
                item_id=album_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )
    siblings = annotations or {}
    artist_id = record.get("artistId")
    if artist_id and isinstance(artist_record := siblings.get(artist_id), dict):
        album.artists = UniqueList([parse_artist_record(provider, artist_record, str(artist_id))])
    return album


def parse_artist_record(
    provider: PandoraProvider, record: dict[str, Any], artist_id: str
) -> Artist:
    """
    Parse an artist from a Pandora catalogue record.

    :param record: The catalogue record Pandora returned for the artist.
    :param artist_id: The id the artist was requested by, which it keeps.
    """
    return parse_artist(provider, str(record.get("name") or artist_id), artist_id)
