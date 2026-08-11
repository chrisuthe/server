"""
Parsers for the payloads Pandora returns.

Every function here is synchronous and free of IO: it turns a payload the caller already
holds into a Music Assistant media item, so fetching stays in the provider and a parse can
never become a network call per item in a listing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ImageType
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


def parse_track(provider: PandoraProvider, obj: dict[str, Any]) -> Track:
    """Parse a raw fragment track into a Track."""
    name, version = parse_title_and_version(obj.get("songTitle") or "Unknown Song")
    track_id = obj["pandoraId"]
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
        track.artists = UniqueList([parse_artist(provider, artist_name)])
    track.album = parse_album(provider, obj, track_id)
    return track


def parse_album(provider: PandoraProvider, obj: dict[str, Any], track_id: str) -> Album | None:
    """Parse the album a fragment track belongs to, if the API named one."""
    if not (url := obj.get("albumDetailURL")):
        return None
    name, version = parse_title_and_version(obj.get("albumTitle") or "Unknown Album")
    return Album(
        item_id=track_id,
        provider=provider.instance_id,
        name=name,
        version=version,
        provider_mappings={
            ProviderMapping(
                item_id=track_id,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
                url=url,
            )
        },
    )


def parse_artist(provider: PandoraProvider, artist_name: str) -> Artist:
    """Parse an artist; Pandora fragments identify artists by name only."""
    return Artist(
        item_id=artist_name,
        name=artist_name,
        provider=provider.instance_id,
        provider_mappings={
            ProviderMapping(
                item_id=artist_name,
                provider_domain=provider.domain,
                provider_instance=provider.instance_id,
            )
        },
    )
