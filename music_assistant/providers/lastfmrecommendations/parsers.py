"""Parsers to convert Last.fm API responses to Music Assistant media items."""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID
from music_assistant_models.media_items import Artist, ProviderMapping, Track
from music_assistant_models.unique_list import UniqueList

from music_assistant.constants import MASS_LOGGER_NAME

if TYPE_CHECKING:
    from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.lastfmrecommendations")


def parse_artist(
    lastfm_artist: dict[str, Any], provider_instance_id: str, provider_domain: str
) -> Artist:
    """Parse Last.fm artist to Music Assistant Artist.

    Extracts artist name and MusicBrainz ID from Last.fm response and
    creates an Artist object with external IDs for matching.

    :param lastfm_artist: Raw Last.fm artist dict with 'name' and 'mbid' fields.
    :param provider_instance_id: Provider instance ID for this artist.
    :param provider_domain: Provider domain for provider mappings.
    :return: Artist object with name and external IDs populated.
    """
    LOGGER.debug("Last.fm artist data: %s", lastfm_artist)

    name = lastfm_artist.get("name", "Unknown Artist")
    mbid = lastfm_artist.get("mbid")

    # Build external IDs
    external_ids = set()
    if mbid:
        external_ids.add((ExternalID.MB_ARTIST, mbid))

    return Artist(
        item_id=mbid or name,  # Use MBID as item_id if available, fallback to name
        provider=provider_instance_id,
        name=name,
        external_ids=external_ids,
        provider_mappings={
            ProviderMapping(
                item_id=mbid or name,
                provider_domain=provider_domain,
                provider_instance=provider_instance_id,
            )
        },
    )


async def parse_track(
    lastfm_track: dict[str, Any],
    provider_instance_id: str,
    provider_domain: str,
    mbid_resolver: MBIDResolver,
) -> Track:
    """Parse Last.fm track to Music Assistant Track.

    This async function resolves MBIDs to ISRCs via MusicBrainz for better
    matching accuracy with streaming providers. Extracts track name, artist,
    duration, and MBID from Last.fm response.

    :param lastfm_track: Raw Last.fm track dict with 'name', 'artist', 'mbid', 'duration'.
    :param provider_instance_id: Provider instance ID for this track.
    :param provider_domain: Provider domain for provider mappings.
    :param mbid_resolver: MBID resolver instance for ISRC lookups.
    :return: Track object with name, artists, duration, and external IDs (MBID + ISRCs).
    """
    LOGGER.debug("Last.fm track data: %s", lastfm_track)

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
        external_ids.add((ExternalID.MB_RECORDING, mbid))

        # Resolve MBID to ISRCs via MusicBrainz (with 90-day cache)
        isrcs = await mbid_resolver.get_isrcs_for_recording(mbid)
        for isrc in isrcs:
            external_ids.add((ExternalID.ISRC, isrc))

    # Create artist as full Artist object (not ItemMapping)
    artist = Artist(
        item_id=artist_mbid or artist_name,
        provider=provider_instance_id,
        name=artist_name,
        provider_mappings={
            ProviderMapping(
                item_id=artist_mbid or artist_name,
                provider_domain=provider_domain,
                provider_instance=provider_instance_id,
            )
        },
    )
    if artist_mbid:
        artist.external_ids.add((ExternalID.MB_ARTIST, artist_mbid))

    # Build track
    track = Track(
        item_id=mbid or f"{artist_name}_{name}",  # Use MBID or fallback to compound ID
        provider=provider_instance_id,
        name=name,
        artists=UniqueList([artist]),
        external_ids=external_ids,
        provider_mappings={
            ProviderMapping(
                item_id=mbid or f"{artist_name}_{name}",
                provider_domain=provider_domain,
                provider_instance=provider_instance_id,
            )
        },
    )

    # Add duration if available (Last.fm provides it in seconds)
    if duration := lastfm_track.get("duration"):
        with contextlib.suppress(ValueError, TypeError):
            track.duration = int(duration)

    return track
