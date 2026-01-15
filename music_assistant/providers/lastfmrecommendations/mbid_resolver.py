"""MBID to ISRC resolver using MusicBrainz provider."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_assistant.providers.lastfmrecommendations import LastFMRecommendationsProvider


class MBIDResolver:
    """Resolves MusicBrainz IDs to ISRCs with extended caching."""

    CACHE_CATEGORY = "lastfm_mbid_isrc"
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
        to minimize API calls to MusicBrainz. ISRCs are permanent identifiers,
        so once we've looked up an MBID->ISRC mapping, it won't change.

        :param mbid: MusicBrainz recording ID.
        """
        cache_key = f"recording_{mbid}"

        # Check 90-day cache first
        cached = await self.mass.cache.get(
            key=cache_key,
            category=self.CACHE_CATEGORY,
        )

        if cached is not None:
            self.logger.debug("ISRC cache hit for MBID %s", mbid)
            return cached.get("isrcs", [])

        # Cache miss - need to query MusicBrainz
        self.logger.debug("ISRC cache miss for MBID %s - querying MusicBrainz", mbid)

        try:
            # Get MusicBrainz provider (built-in metadata provider)
            mb_provider = self.mass.get_provider("musicbrainz")
            if not mb_provider:
                self.logger.warning("MusicBrainz provider not available")
                return []

            # Query MusicBrainz for recording details
            # Note: MusicBrainz provider has its own 30-day cache and rate limiting
            recording = await mb_provider.get_recording_details(mbid)

            isrcs = recording.isrcs if recording and recording.isrcs else []

            # Cache result for 90 days (even if empty to avoid repeated lookups)
            await self.mass.cache.set(
                key=cache_key,
                data={"isrcs": isrcs},
                category=self.CACHE_CATEGORY,
                expiration=self.CACHE_EXPIRATION,
            )

            if isrcs:
                self.logger.debug("Found %d ISRC(s) for MBID %s", len(isrcs), mbid)
            else:
                self.logger.debug("No ISRCs found for MBID %s", mbid)

            return isrcs

        except Exception as err:
            self.logger.warning("Failed to get ISRCs for MBID %s: %s", mbid, err, exc_info=err)

            # Cache the failure (empty list) to avoid repeated failed lookups
            await self.mass.cache.set(
                key=cache_key,
                data={"isrcs": []},
                category=self.CACHE_CATEGORY,
                expiration=self.CACHE_EXPIRATION,
            )

            return []
