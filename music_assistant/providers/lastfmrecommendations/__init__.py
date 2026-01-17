"""Last.fm Recommendations music provider for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiohttp import ClientError
from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import ConfigEntryType, ProviderFeature
from music_assistant_models.errors import InvalidDataError, SetupFailedError
from music_assistant_models.media_items import RecommendationFolder  # noqa: TC002

from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.lastfmrecommendations.api_client import LastFMAPIClient
from music_assistant.providers.lastfmrecommendations.mbid_resolver import MBIDResolver
from music_assistant.providers.lastfmrecommendations.recommendations import (
    LastFMRecommendationManager,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> LastFMRecommendationsProvider:
    """Initialize provider(instance) with given configuration."""
    return LastFMRecommendationsProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    return (
        ConfigEntry(
            key="api_key",
            type=ConfigEntryType.SECURE_STRING,
            label="Last.fm API Key",
            required=True,
            description="Get your API key from https://www.last.fm/api/account/create",
            value=values.get("api_key") if values else None,
        ),
        ConfigEntry(
            key="refresh_interval",
            type=ConfigEntryType.INTEGER,
            label="Refresh Interval (hours)",
            default_value=6,
            description="How often to refresh recommendations (0 to disable automatic refresh)",
            category="recommendations",
            range=(0, 168),  # 0 to 1 week
        ),
        ConfigEntry(
            key="enable_similar_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Similar Artists (Personalized)",
            default_value=False,
            description="Show similar artists based on your listening history",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_similar_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Similar Tracks (Personalized)",
            default_value=False,
            description="Show similar tracks based on your listening history",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_top_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Last.fm Top Artists (Global)",
            default_value=False,
            description="Show Last.fm's worldwide top artists chart",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_top_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Last.fm Top Tracks (Global)",
            default_value=False,
            description="Show Last.fm's worldwide top tracks chart",
            category="recommendations",
        ),
    )


class LastFMRecommendationsProvider(MusicProvider):
    """Last.fm Recommendations Provider for Music Assistant.

    This provider delivers music recommendations from Last.fm based on the user's
    listening history in Music Assistant. It provides both personalized recommendations
    (similar artists/tracks to what you listen to) and global discovery recommendations
    (Last.fm's worldwide top charts).
    """

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self.api = LastFMAPIClient(self)
        self.mbid_resolver = MBIDResolver(self)
        self.recommendations_manager = LastFMRecommendationManager(self)

        # Initialize empty recommendation folders (will be populated progressively)
        self._recommendation_folders: list[RecommendationFolder] = []
        self._recommendations_populated = False

        # Test API key by making a simple request
        try:
            await self.api.get_chart_top_artists(limit=1)
            self.logger.info("Last.fm API key validated successfully")
        except (TimeoutError, ClientError, InvalidDataError, KeyError) as err:
            msg = f"Failed to validate Last.fm API key: {type(err).__name__}"
            raise SetupFailedError(msg) from err

        # Start background task to populate recommendations immediately
        self.mass.create_task(self._populate_recommendations())

        # Schedule periodic refresh using MA's scheduler
        self._schedule_refresh()

    async def _populate_recommendations(self) -> None:
        """Populate recommendation folders in the background.

        This runs immediately after initialization, building recommendation
        folders progressively. The instance variable is updated as folders
        are populated, so each homepage visit shows more items.
        """
        try:
            self.logger.info("Starting background population of recommendations")
            # Fetch and store recommendation folders
            folders = await self.recommendations_manager.get_recommendations()
            self._recommendation_folders = folders
            self._recommendations_populated = True
            self.logger.info("Recommendations populated with %d folders", len(folders))
        except Exception as err:
            self.logger.warning("Failed to populate recommendations: %s", err)

    def _schedule_refresh(self) -> None:
        """Schedule periodic refresh of recommendations using MA's scheduler.

        Uses the configured refresh interval (default: 6 hours). Set to 0 to disable.
        """
        refresh_interval_value = self.config.get_value("refresh_interval")
        if isinstance(refresh_interval_value, (int, float)):
            refresh_interval_hours = int(refresh_interval_value)
        else:
            refresh_interval_hours = 6  # Default
        if refresh_interval_hours <= 0:
            self.logger.info("Automatic refresh disabled (interval set to 0)")
            return

        # Convert hours to seconds
        refresh_interval_seconds = float(refresh_interval_hours * 3600)

        # Schedule next refresh using MA's call_later
        self.mass.call_later(
            refresh_interval_seconds,
            self._refresh_recommendations,
            task_id=f"lastfm_recommendations_refresh_{self.instance_id}",
        )
        self.logger.info(
            "Scheduled next recommendations refresh in %d hours", refresh_interval_hours
        )

    async def _refresh_recommendations(self) -> None:
        """Refresh recommendations (called by scheduler).

        Re-populates recommendations and reschedules next refresh.
        """
        try:
            self.logger.info("Refreshing Last.fm recommendations (scheduled)")
            await self._populate_recommendations()
        except Exception as err:
            self.logger.warning("Failed to refresh recommendations: %s", err)
        finally:
            # Reschedule next refresh
            self._schedule_refresh()

    async def recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

        Returns the current state of recommendation folders. On first call (before
        background population completes), this returns an empty list. Each subsequent
        call returns progressively more populated folders as the background task
        resolves items.

        Returns up to 4 recommendation folders:
        1. Discover Similar Artists (personalized, based on your listening)
        2. Discover Similar Tracks (personalized, based on your listening)
        3. Last.fm Top Artists (global chart)
        4. Last.fm Top Tracks (global chart)

        Personalized folders only appear if the user has listening history.
        If the library is empty, only global charts will be shown.
        """
        # Return current state (empty on first call, populated later)
        return self._recommendation_folders
