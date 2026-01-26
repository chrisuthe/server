"""MBID to ISRC resolver using MusicBrainz provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from aiohttp import ClientError
from music_assistant_models.errors import InvalidDataError, ProviderUnavailableError

from music_assistant.providers.lastfm_recommendations.constants import CACHE_CATEGORY_MBID_ISRC

if TYPE_CHECKING:
    from music_assistant.providers.lastfm_recommendations import LastFMRecommendationsProvider
    from music_assistant.providers.musicbrainz import MusicbrainzProvider


class MBIDResolver:
    """Resolves MusicBrainz IDs to ISRCs with extended caching.

    This class queries the MusicBrainz provider to resolve MusicBrainz Recording IDs
    (MBIDs) to International Standard Recording Codes (ISRCs). ISRCs are used by
    streaming providers for accurate track matching.

    A 90-day cache is implemented on top of MusicBrainz's own 30-day cache to
    minimize API load, as ISRC mappings tend not to change.
    """

    CACHE_EXPIRATION = 86400 * 90  # 90 days

    def __init__(self, provider: LastFMRecommendationsProvider) -> None:
        """Initialize MBID resolver.

        :param provider: The Last.fm recommendations provider instance.
        """
        self.provider = provider
        self.mass = provider.mass
        self.logger = provider.logger

    async def get_isrcs_for_recording(self, mbid: str) -> list[str]:
        """Get ISRCs for a recording MBID via MusicBrainz.

        This method implements a 90-day cache on top of MusicBrainz's own caching
        to minimize API calls to MusicBrainz. ISRC mappings tend not to change,
        so once we've looked up an MBID->ISRC mapping, it should remain valid.

        :param mbid: MusicBrainz recording ID.
        :return: List of ISRCs for the recording (may be empty if none found).
        """
        cache_key = f"recording_{mbid}"

        # Check 90-day cache first
        cached = await self.mass.cache.get(
            key=cache_key,
            category=CACHE_CATEGORY_MBID_ISRC,
        )

        if cached is not None:
            return cast("list[str]", cached.get("isrcs", []))

        # Get MusicBrainz provider (built-in metadata provider)
        mb_provider = self.mass.get_provider("musicbrainz")
        if not mb_provider:
            msg = "MusicBrainz provider not available"
            raise ProviderUnavailableError(msg)

        try:
            # Query MusicBrainz for recording details
            # Note: MusicBrainz provider has its own 30-day cache and rate limiting
            recording = await cast("MusicbrainzProvider", mb_provider).get_recording_details(mbid)

            isrcs = recording.isrcs if recording and recording.isrcs else []

            # Cache result for 90 days (even if empty to avoid repeated lookups)
            await self.mass.cache.set(
                key=cache_key,
                data={"isrcs": isrcs},
                category=CACHE_CATEGORY_MBID_ISRC,
                expiration=self.CACHE_EXPIRATION,
            )

            return isrcs

        except (TimeoutError, ClientError, InvalidDataError) as err:
            self.logger.debug("Failed to get ISRCs for MBID %s: %s", mbid, type(err).__name__)

            # Cache the failure (empty list) to avoid repeated failed lookups
            await self.mass.cache.set(
                key=cache_key,
                data={"isrcs": []},
                category=CACHE_CATEGORY_MBID_ISRC,
                expiration=self.CACHE_EXPIRATION,
            )

            return []
