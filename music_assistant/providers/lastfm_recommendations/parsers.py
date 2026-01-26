"""Parsers to convert Last.fm API responses to Music Assistant media items."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ExternalID, ImageType, MediaType, ProviderFeature
from music_assistant_models.errors import MusicAssistantError
from music_assistant_models.media_items import Album, Artist, ItemMapping, MediaItemImage, Track

from music_assistant.constants import MASS_LOGGER_NAME
from music_assistant.helpers.compare import compare_strings

if TYPE_CHECKING:
    from music_assistant import MusicAssistant
    from music_assistant.controllers.media.albums import AlbumsController
    from music_assistant.controllers.media.artists import ArtistsController
    from music_assistant.controllers.media.tracks import TracksController
    from music_assistant.providers.lastfm_recommendations.mbid_resolver import MBIDResolver

LOGGER = logging.getLogger(f"{MASS_LOGGER_NAME}.lastfm_recommendations")

# Semaphore to limit concurrent provider searches (prevents overwhelming Spotify API)
_SEARCH_SEMAPHORE = asyncio.Semaphore(5)


def _has_matching_external_ids(
    item_mapping: ItemMapping, media_item: Artist | Album | Track
) -> bool:
    """Check if an ItemMapping has any matching external IDs with a media item.

    :param item_mapping: ItemMapping with external IDs from Last.fm.
    :param media_item: Artist or Track to compare against.
    :return: True if any external IDs match, False otherwise.
    """
    if not item_mapping.external_ids:
        return False

    # external_ids is a set of tuples (ExternalID, str)
    # Check if any external IDs overlap
    return bool(item_mapping.external_ids & media_item.external_ids)


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


def _get_streaming_providers(
    mass: MusicAssistant, item_mapping: ItemMapping, provider_instance_to_skip: str
) -> list[Any]:
    """Get list of streaming providers that support the media type we're searching for.

    :param mass: MusicAssistant instance.
    :param item_mapping: ItemMapping with the media type to search for.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    :return: List of streaming providers that support the media type.
    """
    streaming_providers = []
    for p in mass.music.providers:
        if p.instance_id == provider_instance_to_skip:
            continue
        if not p.is_streaming_provider:
            continue

        # Check if provider supports the media type we're searching for
        if item_mapping.media_type == MediaType.ARTIST:
            if ProviderFeature.LIBRARY_ARTISTS not in p.supported_features:
                continue
        elif item_mapping.media_type == MediaType.ALBUM:
            if ProviderFeature.LIBRARY_ALBUMS not in p.supported_features:
                continue
        elif item_mapping.media_type == MediaType.TRACK:
            if ProviderFeature.LIBRARY_TRACKS not in p.supported_features:
                continue

        streaming_providers.append(p)
    return streaming_providers


async def _search_provider(
    ctrl: ArtistsController | AlbumsController | TracksController,
    item_mapping: ItemMapping,
    provider: Any,
) -> Artist | Album | Track | None:
    """Search a single provider for a matching item.

    Uses semaphore to limit concurrent searches and prevent overwhelming provider APIs.

    :param ctrl: Controller for the media type.
    :param item_mapping: ItemMapping to search for.
    :param provider: Provider instance to search.
    :return: Matched item or None.
    """
    async with _SEARCH_SEMAPHORE:
        try:
            LOGGER.debug(
                "Searching %s on %s for: %s",
                item_mapping.media_type.value,
                provider.name,
                item_mapping.name,
            )
            search_results = await ctrl.search(item_mapping.name, provider.instance_id, limit=1)
            if not search_results:
                LOGGER.debug("No search results from %s", provider.name)
                return None

            result = search_results[0]
            LOGGER.debug(
                "Found %s on provider %s: %s",
                item_mapping.media_type.value,
                provider.name,
                result.name,
            )
            return result
        except MusicAssistantError as err:
            # Expected errors from provider searches (e.g., provider unavailable, timeout, etc.)
            LOGGER.debug("Provider %s search failed: %s", provider.name, type(err).__name__)
            return None


async def _search_providers_concurrent(
    ctrl: ArtistsController | AlbumsController | TracksController,
    item_mapping: ItemMapping,
    providers: list[Any],
    require_external_id_match: bool,
) -> Artist | Album | Track | None:
    """Search multiple providers concurrently with smart result prioritization.

    Optimized to make only ONE API call per provider by intelligently handling results:
    - If we require external ID matching (have ISRCs/MBIDs):
      1. Return immediately on external ID match
      2. Reject results with non-matching external IDs
      3. Save results without external IDs as fallback
      4. Return fallback if no external ID match found
    - If we don't require external ID matching:
      1. Return first result

    :param ctrl: Controller for the media type.
    :param item_mapping: ItemMapping to search for.
    :param providers: List of providers to search.
    :param require_external_id_match: If True, try to match on external IDs.
    :return: Best matched item or None if not found.
    """
    tasks = [
        asyncio.create_task(_search_provider(ctrl, item_mapping, provider))
        for provider in providers
    ]

    fallback_result = None

    # Process results as they complete
    for task in asyncio.as_completed(tasks):
        result = await task
        if result is None:
            continue

        if not require_external_id_match:
            # No external IDs to match - verify name similarity before accepting
            if compare_strings(item_mapping.name, result.name, strict=False):
                LOGGER.debug(
                    "Name match on %s: %s (searched: %s)",
                    result.provider,
                    result.name,
                    item_mapping.name,
                )
                for t in tasks:
                    if not t.done():
                        t.cancel()
                return result

            # Name doesn't match well - save as fallback but keep searching
            LOGGER.debug(
                "Rejecting %s from %s: name mismatch (searched: %s)",
                result.name,
                result.provider,
                item_mapping.name,
            )
            if not fallback_result:
                fallback_result = result
            continue

        # We have external IDs - try to match them
        if _has_matching_external_ids(item_mapping, result):
            # Perfect match! Return immediately
            LOGGER.debug(
                "External ID match on %s: %s",
                result.provider,
                result.name,
            )
            for t in tasks:
                if not t.done():
                    t.cancel()
            return result

        # Check if result has any of the external ID types we're looking for
        result_has_external_ids = any(
            ext_id[0] in {ext_id_check[0] for ext_id_check in item_mapping.external_ids}
            for ext_id in result.external_ids
        )

        if result_has_external_ids:
            # Has external IDs but they don't match - reject this result
            LOGGER.debug(
                "Rejecting %s from %s: has external IDs but they don't match",
                result.name,
                result.provider,
            )
        elif not fallback_result:
            # No external IDs to compare - save as fallback
            LOGGER.debug(
                "Saving %s from %s as fallback (no external IDs to verify)",
                result.name,
                result.provider,
            )
            fallback_result = result

    # All providers returned - use fallback if we have one
    if fallback_result:
        LOGGER.debug("No external ID matches found, using fallback result")
    return fallback_result


async def _resolve_item(
    item_mapping: ItemMapping, mass: MusicAssistant, provider_instance_to_skip: str
) -> Artist | Album | Track | None:
    """Resolve an ItemMapping to an actual library or provider item.

    Searches all providers concurrently and returns the first match found.
    This is much faster than sequential search, especially with rate-limited providers.

    :param item_mapping: ItemMapping with metadata and external IDs from Last.fm.
    :param mass: MusicAssistant instance.
    :param provider_instance_to_skip: Provider instance to skip (ourselves).
    :return: Resolved Artist/Album/Track or None if not found.
    """
    # Get the appropriate controller
    ctrl: ArtistsController | AlbumsController | TracksController
    if item_mapping.media_type == MediaType.ARTIST:
        ctrl = mass.music.artists
    elif item_mapping.media_type == MediaType.ALBUM:
        ctrl = mass.music.albums
    elif item_mapping.media_type == MediaType.TRACK:
        ctrl = mass.music.tracks
    else:
        return None

    LOGGER.debug(
        "Resolving %s: %s (external IDs: %s)",
        item_mapping.media_type.value,
        item_mapping.name,
        item_mapping.external_ids or "none",
    )

    # First, check library by external IDs
    if library_item := await ctrl.get_library_item_by_external_ids(item_mapping.external_ids):
        LOGGER.debug("Found %s in library: %s", item_mapping.media_type.value, library_item.name)
        return library_item

    # Get list of streaming providers that support the media type
    streaming_providers = _get_streaming_providers(mass, item_mapping, provider_instance_to_skip)
    if not streaming_providers:
        LOGGER.debug("No streaming providers available for resolution")
        return None

    provider_names = [p.name for p in streaming_providers]
    LOGGER.debug("Searching %d providers: %s", len(streaming_providers), ", ".join(provider_names))

    # Determine if we should try to match on external IDs
    # For tracks: only if we have ISRCs (streaming providers support ISRC matching)
    # For artists/albums: streaming providers don't expose MBIDs, so always use name matching
    require_external_id_match = False
    if item_mapping.media_type == MediaType.TRACK:
        # For tracks, only match on external IDs if we have ISRCs
        has_isrc = any(ext_id[0] == ExternalID.ISRC for ext_id in item_mapping.external_ids)
        if has_isrc:
            LOGGER.debug("Have ISRCs, will prioritize ISRC matches")
            require_external_id_match = True
        else:
            LOGGER.debug("No ISRCs available, accepting any name match")
    else:
        # Artists and albums: streaming providers don't expose MBIDs, use name matching
        LOGGER.debug("Using name-based matching for %s", item_mapping.media_type.value)

    # Search all providers once with smart prioritization
    # This makes only ONE API call per provider (instead of two)
    result = await _search_providers_concurrent(
        ctrl, item_mapping, streaming_providers, require_external_id_match
    )
    if result is None:
        LOGGER.debug("Could not resolve %s: %s", item_mapping.media_type.value, item_mapping.name)
    return result


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
        provider="lastfm_recommendations",  # Temporary provider
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
        LOGGER.debug("Resolving MBID %s to ISRCs via MusicBrainz", mbid)
        isrcs = await mbid_resolver.get_isrcs_for_recording(mbid)
        if isrcs:
            LOGGER.debug("Found %d ISRCs for MBID %s: %s", len(isrcs), mbid, isrcs)
            for isrc in isrcs:
                external_ids.add((ExternalID.ISRC, isrc))
        else:
            LOGGER.debug("No ISRCs found for MBID %s", mbid)
    else:
        LOGGER.debug("Track has no MBID, cannot resolve ISRCs")

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
        provider="lastfm_recommendations",  # Temporary provider
        name=f"{artist_name} - {name}",  # Include artist in name for display
        external_ids=external_ids,
        image=image,
    )

    # Resolve to actual Track object
    return cast("Track | None", await _resolve_item(item_mapping, mass, provider_instance))


async def parse_album(
    lastfm_album: dict[str, Any], mass: MusicAssistant, provider_instance: str
) -> Album | None:
    """Parse Last.fm album and resolve to a full Album object.

    Resolves the Last.fm album to an actual provider item using external IDs and name matching.

    :param lastfm_album: Raw Last.fm album dict with 'name', 'artist', 'mbid'.
    :param mass: MusicAssistant instance for accessing library and providers.
    :param provider_instance: Provider instance ID to skip when searching.
    :return: Resolved Album object or None if not found.
    """
    name = lastfm_album.get("name", "Unknown Album")
    mbid = lastfm_album.get("mbid")

    # Parse artist info
    artist_data = lastfm_album.get("artist", {})
    if isinstance(artist_data, str):
        # Sometimes artist is just a string
        artist_name = artist_data
    else:
        artist_name = artist_data.get("name", "Unknown Artist")

    # Build external IDs
    external_ids = set()
    if mbid:
        external_ids.add((ExternalID.MB_ALBUM, mbid))

    # Extract image
    image = None
    if image_url := _extract_image_url(lastfm_album.get("image", [])):
        image = MediaItemImage(
            type=ImageType.THUMB,
            path=image_url,
            provider="lastfm",
        )

    # Create temporary ItemMapping for matching
    item_mapping = ItemMapping(
        media_type=MediaType.ALBUM,
        item_id="temp",  # Temporary ID, not used
        provider="lastfm_recommendations",  # Temporary provider
        name=f"{artist_name} - {name}",  # Include artist in name for display
        external_ids=external_ids,
        image=image,
    )

    # Resolve to actual Album object
    return cast("Album | None", await _resolve_item(item_mapping, mass, provider_instance))
