"""Last.fm Recommendations music provider for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.enums import ProviderFeature
from music_assistant_models.errors import SetupFailedError

from music_assistant.controllers.cache import use_cache
from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.lastfmrecommendations.api_client import LastFMAPIClient
from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver
from music_assistant.providers.lastfmrecommendations.recommendations import (
    LastFMRecommendationManager,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.media_items import RecommendationFolder
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> LastFMRecommendationsProvider:
    """Initialize provider(instance) with given configuration."""
    return LastFMRecommendationsProvider(mass, manifest, config)


class LastFMRecommendationsProvider(MusicProvider):
    """Last.fm Recommendations Provider for Music Assistant.

    This provider delivers music recommendations from Last.fm based on the user's
    listening history in Music Assistant. It provides both personalized recommendations
    (similar artists/tracks to what you listen to) and global discovery recommendations
    (Last.fm's worldwide top charts).
    """

    def __init__(
        self, mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
    ) -> None:
        """Initialize the Last.fm Recommendations provider.

        :param mass: MusicAssistant instance.
        :param manifest: Provider manifest.
        :param config: Provider configuration.
        """
        super().__init__(mass, manifest, config, SUPPORTED_FEATURES)
        self.api = LastFMAPIClient(self)
        self.mbid_resolver = MBIDResolver(self)
        self.recommendations_manager = LastFMRecommendationManager(self)

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        # Validate API key is configured
        api_key = self.config.get_value("api_key")
        if not api_key:
            raise SetupFailedError("Last.fm API key is required")

        # Test API key by making a simple request
        try:
            # Try to get top artists chart (doesn't require authentication)
            await self.api.get_chart_top_artists(limit=1)
            self.logger.info("Last.fm API key validated successfully")
        except Exception as err:
            raise SetupFailedError(f"Failed to validate Last.fm API key: {err}") from err

    @use_cache(3600)  # Cache recommendations for 1 hour
    async def recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

        Returns up to 4 recommendation folders:
        1. Discover Similar Artists (personalized, based on your listening)
        2. Discover Similar Tracks (personalized, based on your listening)
        3. Last.fm Top Artists (global chart)
        4. Last.fm Top Tracks (global chart)

        Personalized folders only appear if the user has listening history.
        If the library is empty, only global charts will be shown.
        """
        return await self.recommendations_manager.get_recommendations()
