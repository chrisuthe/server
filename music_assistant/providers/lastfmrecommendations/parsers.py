"""Parsers to convert Last.fm API responses to Music Assistant media items."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID
from music_assistant_models.media_items import Artist, ItemMapping, Track

if TYPE_CHECKING:
    from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver


def parse_artist(lastfm_artist: dict[str, Any], provider_instance_id: str) -> Artist:
    """Parse Last.fm artist to Music Assistant Artist.

    :param lastfm_artist: Raw Last.fm artist dict.
    :param provider_instance_id: Provider instance ID for this artist.
    """
    name = lastfm_artist.get("name", "Unknown Artist")
    mbid = lastfm_artist.get("mbid")

    # Build external IDs
    external_ids = set()
    if mbid:
        external_ids.add((ExternalID.MUSICBRAINZ_ARTIST, mbid))

    return Artist(
        item_id=mbid or name,  # Use MBID as item_id if available, fallback to name
        provider=provider_instance_id,
        name=name,
        external_ids=external_ids,
    )


async def parse_track(
    lastfm_track: dict[str, Any],
    provider_instance_id: str,
    mbid_resolver: MBIDResolver,
) -> Track:
    """Parse Last.fm track to Music Assistant Track.

    This async function resolves MBIDs to ISRCs via MusicBrainz for better
    matching accuracy with streaming providers.

    :param lastfm_track: Raw Last.fm track dict.
    :param provider_instance_id: Provider instance ID for this track.
    :param mbid_resolver: MBID resolver instance for ISRC lookups.
    """
    name = lastfm_track.get("name", "Unknown Track")
    mbid = lastfm_track.get("mbid")

    # Parse artist info
    artist_data = lastfm_track.get("artist", {})
    if isinstance(artist_data, str):
        # Sometimes artist is just a string
        artist_name = artist_data
        artist_mbid = None
    else:
        artist_name = artist_data.get("name", "Unknown Artist")
        artist_mbid = artist_data.get("mbid")

    # Build external IDs
    external_ids = set()

    if mbid:
        # Add MusicBrainz recording ID
        external_ids.add((ExternalID.MUSICBRAINZ_RECORDING, mbid))

        # Resolve MBID to ISRCs via MusicBrainz (with 90-day cache)
        isrcs = await mbid_resolver.get_isrcs_for_recording(mbid)
        for isrc in isrcs:
            external_ids.add((ExternalID.ISRC, isrc))

    # Create artist as ItemMapping
    artist_item = ItemMapping(
        item_id=artist_mbid or artist_name,
        provider=provider_instance_id,
        name=artist_name,
        media_type="artist",
    )

    # Build track
    track = Track(
        item_id=mbid or f"{artist_name}_{name}",  # Use MBID or fallback to compound ID
        provider=provider_instance_id,
        name=name,
        artists=[artist_item],
        external_ids=external_ids,
    )

    # Add duration if available (Last.fm provides it in seconds)
    if duration := lastfm_track.get("duration"):
        with contextlib.suppress(ValueError, TypeError):
            track.duration = int(duration)

    return track
