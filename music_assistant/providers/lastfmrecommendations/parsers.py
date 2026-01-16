"""Parsers to convert Last.fm API responses to Music Assistant media items."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ExternalID, ImageType, MediaType
from music_assistant_models.media_items import Artist, ItemMapping, MediaItemImage, Track

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.helpers.compare import compare_media_item

if TYPE_CHECKING:
    from music_assistant import MusicAssistant
    from music_assistant.controllers.media.artists import ArtistsController
    from music_assistant.controllers.media.tracks import TracksController
    from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.lastfmrecommendations")


def _extract_image_url(image_array: list[dict[str, Any]]) -> str | None:
    """Extract the best quality image URL from Last.fm's image array.

    Last.fm returns images in multiple sizes: small, medium, large, extralarge, mega.
    This function returns the largest available image URL.

    :param image_array: List of image dicts from Last.fm API.
    :return: URL string of the best quality image, or None if no valid images.
    """
    if not image_array:
        return None

    # Prefer larger sizes first
    size_priority = ["mega", "extralarge", "large", "medium", "small"]

    for size in size_priority:
        for img in image_array:
            if img.get("size") == size and img.get("#text"):
                url = str(img["#text"]).strip()
                # Filter out placeholder/empty images
                if url and not url.endswith("/default.png"):
                    return url

    return None


async def _resolve_item(
    item_mapping: ItemMapping, mass: MusicAssistant, provider_instance_to_skip: str
) -> Artist | Track | None:
    """Resolve an ItemMapping to an actual library or provider item.

    :param item_mapping: ItemMapping with metadata and external IDs from Last.fm.
    :param mass: MusicAssistant instance.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    :return: Resolved Artist/Track or None if not found.
    """
    # Get the appropriate controller
    ctrl: ArtistsController | TracksController
    if item_mapping.media_type == MediaType.ARTIST:
        ctrl = mass.music.artists
    elif item_mapping.media_type == MediaType.TRACK:
        ctrl = mass.music.tracks
    else:
        return None

    # First, check library by external IDs
    if library_item := await ctrl.get_library_item_by_external_ids(item_mapping.external_ids):
        LOGGER.debug("Found %s in library: %s", item_mapping.media_type.value, library_item.name)
        return library_item

    # Search streaming providers with external ID matching
    for provider in mass.music.providers:
        if provider.instance_id == provider_instance_to_skip:
            continue
        if not provider.is_streaming_provider:
            continue

        try:
            search_results = await ctrl.search(item_mapping.name, provider.instance_id, limit=1)
            for result in search_results:
                if compare_media_item(item_mapping, result, strict=False):
                    LOGGER.debug(
                        "Found %s on provider %s via external IDs: %s",
                        item_mapping.media_type.value,
                        provider.name,
                        result.name,
                    )
                    return result
        except Exception as err:
            LOGGER.debug("Provider %s search failed: %s", provider.name, type(err).__name__)
            continue

    # Fallback: name-only search
    for provider in mass.music.providers:
        if provider.instance_id == provider_instance_to_skip:
            continue
        if not provider.is_streaming_provider:
            continue

        try:
            search_results = await ctrl.search(item_mapping.name, provider.instance_id, limit=1)
            if search_results:
                result = search_results[0]
                LOGGER.debug(
                    "Found %s on provider %s via name search: %s",
                    item_mapping.media_type.value,
                    provider.name,
                    result.name,
                )
                return result
        except Exception as err:
            LOGGER.debug(
                "Provider %s fallback search failed: %s", provider.name, type(err).__name__
            )
            continue

    # No match found
    LOGGER.debug("Could not resolve %s: %s", item_mapping.media_type.value, item_mapping.name)
    return None


async def parse_artist(
    lastfm_artist: dict[str, Any], mass: MusicAssistant, provider_instance: str
) -> Artist | None:
    """Parse Last.fm artist and resolve to a full Artist object.

    Resolves the Last.fm artist to an actual provider item using external IDs and name matching.

    :param lastfm_artist: Raw Last.fm artist dict with 'name' and 'mbid' fields.
    :param mass: MusicAssistant instance for accessing library and providers.
    :param provider_instance: Provider instance ID to skip when searching.
    :return: Resolved Artist object or None if not found.
    """
    LOGGER.debug("Last.fm artist data: %s", lastfm_artist)

    name = lastfm_artist.get("name", "Unknown Artist")
    mbid = lastfm_artist.get("mbid")

    # Build external IDs
    external_ids = set()
    if mbid:
        external_ids.add((ExternalID.MB_ARTIST, mbid))

    # Extract image
    image = None
    if image_url := _extract_image_url(lastfm_artist.get("image", [])):
        image = MediaItemImage(
            type=ImageType.THUMB,
            path=image_url,
            provider="lastfm",
        )

    # Create temporary ItemMapping for matching
    item_mapping = ItemMapping(
        media_type=MediaType.ARTIST,
        item_id="temp",  # Temporary ID, not used
        provider="lastfmrecommendations",  # Temporary provider
        name=name,
        external_ids=external_ids,
        image=image,
    )

    # Resolve to actual Artist object
    return cast("Artist | None", await _resolve_item(item_mapping, mass, provider_instance))


async def parse_track(
    lastfm_track: dict[str, Any],
    mbid_resolver: MBIDResolver,
    mass: MusicAssistant,
    provider_instance: str,
) -> Track | None:
    """Parse Last.fm track and resolve to a full Track object.

    Resolves the Last.fm track to an actual provider item using external IDs
    (MBIDs and ISRCs) and name matching.

    :param lastfm_track: Raw Last.fm track dict with 'name', 'artist', 'mbid', 'duration'.
    :param mbid_resolver: MBID resolver instance for ISRC lookups.
    :param mass: MusicAssistant instance for accessing library and providers.
    :param provider_instance: Provider instance ID to skip when searching.
    :return: Resolved Track object or None if not found.
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

    # Extract image
    image = None
    if image_url := _extract_image_url(lastfm_track.get("image", [])):
        image = MediaItemImage(
            type=ImageType.THUMB,
            path=image_url,
            provider="lastfm",
        )

    # Create temporary ItemMapping for matching
    item_mapping = ItemMapping(
        media_type=MediaType.TRACK,
        item_id="temp",  # Temporary ID, not used
        provider="lastfmrecommendations",  # Temporary provider
        name=f"{artist_name} - {name}",  # Include artist in name for display
        external_ids=external_ids,
        image=image,
    )

    # Resolve to actual Track object
    return cast("Track | None", await _resolve_item(item_mapping, mass, provider_instance))
