"""
Parsers for the payloads Pandora returns.

Every function here is synchronous and free of IO: it turns a payload the caller already
holds into a Music Assistant media item, so fetching stays in the provider and a parse can
never become a network call per item in a listing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, ImageType
from music_assistant_models.media_items import (
    Album,
    Artist,
    MediaItemImage,
    Playlist,
    ProviderMapping,
    Track,
)
from music_assistant_models.unique_list import UniqueList

from music_assistant.helpers.util import parse_title_and_version

if TYPE_CHECKING:
    from .provider import PandoraProvider


def parse_station(provider: PandoraProvider, station: dict[str, Any]) -> Playlist:
    """Parse a station object into a dynamic playlist."""
    playlist = Playlist(
        item_id=station["stationId"],
        provider=provider.instance_id,
        name=station["name"],
        is_dynamic=True,
        is_editable=bool(station.get("allowAddSeed")),
        provider_mappings={
            ProviderMapping(
                item_id=station["stationId"],
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )
    if art := station.get("art"):
        art_url = next(
            (item.get("url") for item in art if item.get("size") == 500), art[-1].get("url")
        )
        if art_url:
            playlist.metadata.add_image(
                MediaItemImage(
                    type=ImageType.THUMB,
                    path=art_url,
                    provider=provider.instance_id,
                    remotely_accessible=True,
                )
            )
    return playlist


def parse_track(
    provider: PandoraProvider,
    obj: dict[str, Any],
    annotations: dict[str, Any] | None = None,
) -> Track:
    """
    Parse a raw fragment track into a Track.

    :param obj: One raw track from a Pandora fragment.
    :param annotations: Catalogue records keyed by pandoraId, from whichever endpoint
        produced them. Omitted or empty when nothing has been annotated, which is the case
        for an account without on-demand entitlement.
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
            )
        },
    )
    if album_art := obj.get("albumArt"):
        art_url = next(
            (art.get("url") for art in album_art if art.get("size") == 500),
            album_art[-1].get("url"),
        )
        if art_url:
            track.metadata.add_image(
                MediaItemImage(
                    provider=provider.instance_id,
                    type=ImageType.THUMB,
                    path=art_url,
                    remotely_accessible=True,
                )
            )
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

    An annotated track names its album in Pandora's catalogue, which is the id that album
    carries everywhere else. A fragment on its own names no album at all, so the track's own
    id stands in - the two cannot be confused, since they carry different prefixes.

    :param obj: One raw track from a Pandora fragment.
    :param track_id: The track's own Pandora id, used as the album id when there is no other.
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

    :param name: The artist's name, which is also its id when the catalogue gave none.
    :param artist_id: The artist's catalogue id, if one has been fetched. Without it a
        Pandora fragment identifies its artist by name only.
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

    No image is set: a record's `icon.artUrl` is a relative path whose CDN base is unmeasured.

    :param record: The catalogue record Pandora returned for the track.
    :param track_id: The id the track was requested by, which it keeps.
    :param annotations: The records that came back alongside this one, keyed by pandoraId.
        The track's album and artist are read from those siblings; a record whose siblings
        are absent still yields a playable track, carrying neither.
    """
    name, version = parse_title_and_version(str(record.get("name") or "Unknown Track"))
    track = Track(
        item_id=track_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        duration=int(record.get("duration") or 0),
        provider_mappings={
            ProviderMapping(
                item_id=track_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
                # no audio format: a catalogue track is minted per play and Pandora names the
                # encoding it made then, which the fragment quality preference does not describe
            )
        },
    )
    if isrc := record.get("isrc"):
        track.external_ids.add((ExternalID.ISRC, str(isrc)))
    siblings = annotations or {}
    album_id = record.get("albumId")
    if album_id and isinstance(album_record := siblings.get(album_id), dict):
        track.album = parse_album_record(provider, album_record, str(album_id))
    artist_id = record.get("artistId")
    if artist_id and isinstance(artist_record := siblings.get(artist_id), dict):
        track.artists = UniqueList([parse_artist_record(provider, artist_record, str(artist_id))])
    return track


def parse_album_record(provider: PandoraProvider, record: dict[str, Any], album_id: str) -> Album:
    """
    Parse an album from a Pandora catalogue record.

    No image is set: a record's `icon.artUrl` is a relative path whose CDN base is unmeasured.

    :param record: The catalogue record Pandora returned for the album.
    :param album_id: The id the album was requested by, which it keeps.
    """
    name, version = parse_title_and_version(str(record.get("name") or "Unknown Album"))
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
            )
        },
    )


def parse_artist_record(
    provider: PandoraProvider, record: dict[str, Any], artist_id: str
) -> Artist:
    """
    Parse an artist from a Pandora catalogue record.

    :param record: The catalogue record Pandora returned for the artist.
    :param artist_id: The id the artist was requested by, which it keeps.
    """
    return parse_artist(provider, str(record.get("name") or artist_id), artist_id)
