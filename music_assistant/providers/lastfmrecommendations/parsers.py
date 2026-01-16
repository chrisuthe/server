"""Parsers to convert Last.fm API responses to Music Assistant media items."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.media_items import ItemMapping

from music_assistant.constants import MASS_LOGGER_NAME

if TYPE_CHECKING:
    from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.lastfmrecommendations")


def parse_artist(lastfm_artist: dict[str, Any]) -> ItemMapping:
    """Parse Last.fm artist to ItemMapping.

    Creates a lightweight reference that MA will use to find the artist
    in other providers (Spotify, Tidal, etc.) using external IDs.

    :param lastfm_artist: Raw Last.fm artist dict with 'name' and 'mbid' fields.
    :return: ItemMapping with name and external IDs for matching.
    """
    LOGGER.debug("Last.fm artist data: %s", lastfm_artist)

    name = lastfm_artist.get("name", "Unknown Artist")
    mbid = lastfm_artist.get("mbid")

    # Build external IDs
    external_ids = set()
    if mbid:
        external_ids.add((ExternalID.MB_ARTIST, mbid))

    return ItemMapping(
        media_type=MediaType.ARTIST,
        item_id=mbid or name,  # Use MBID if available, fallback to name
        provider="library",  # Library provider - MA will try to find this in library
        name=name,
        external_ids=external_ids,
    )


async def parse_track(
    lastfm_track: dict[str, Any],
    mbid_resolver: MBIDResolver,
) -> ItemMapping:
    """Parse Last.fm track to ItemMapping.

    Creates a lightweight reference that MA will use to find the track
    in other providers (Spotify, Tidal, etc.) using external IDs.
    Resolves MBIDs to ISRCs via MusicBrainz for better matching accuracy.

    :param lastfm_track: Raw Last.fm track dict with 'name', 'artist', 'mbid', 'duration'.
    :param mbid_resolver: MBID resolver instance for ISRC lookups.
    :return: ItemMapping with name, artist, and external IDs (MBID + ISRCs) for matching.
    """
    LOGGER.debug("Last.fm track data: %s", lastfm_track)

    name = lastfm_track.get("name", "Unknown Track")
    mbid = lastfm_track.get("mbid")

    # Parse artist info
    artist_data = lastfm_track.get("artist", {})
    if isinstance(artist_data, str):
        # Sometimes artist is just a string
        artist_name = artist_data
    else:
        artist_name = artist_data.get("name", "Unknown Artist")

    # Build external IDs
    external_ids = set()

    if mbid:
        # Add MusicBrainz recording ID
        external_ids.add((ExternalID.MB_RECORDING, mbid))

        # Resolve MBID to ISRCs via MusicBrainz (with 90-day cache)
        isrcs = await mbid_resolver.get_isrcs_for_recording(mbid)
        for isrc in isrcs:
            external_ids.add((ExternalID.ISRC, isrc))

    return ItemMapping(
        media_type=MediaType.TRACK,
        item_id=mbid or f"{artist_name}_{name}",  # Use MBID if available
        provider="library",  # Library provider - MA will try to find this in library
        name=f"{artist_name} - {name}",  # Include artist in name for display
        external_ids=external_ids,
    )
