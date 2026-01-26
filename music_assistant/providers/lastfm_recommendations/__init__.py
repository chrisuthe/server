"""Last.fm Recommendations music provider for Music Assistant."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import (
    ConfigEntry,
    ConfigValueOption,
    ConfigValueType,
)
from music_assistant_models.enums import ConfigEntryType, ProviderFeature
from music_assistant_models.errors import MusicAssistantError
from music_assistant_models.media_items import RecommendationFolder  # noqa: TC002

from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.lastfm_recommendations.api_client import LastFMAPIClient
from music_assistant.providers.lastfm_recommendations.mbid_resolver import MBIDResolver
from music_assistant.providers.lastfm_recommendations.recommendations import (
    LastFMRecommendationManager,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant


SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}

# Config action constants
CONF_ACTION_CLEAR_CACHE = "clear_cache"

# Curated list of popular countries for Last.fm geo charts
# Last.fm API expects full country names (not ISO codes)
# This list covers major music markets and can be expanded based on user requests
GEO_COUNTRIES = [
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Brazil",
    "Canada",
    "China",
    "Czech Republic",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "India",
    "Ireland",
    "Israel",
    "Italy",
    "Japan",
    "Lithuania",
    "Mexico",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Philippines",
    "Poland",
    "Portugal",
    "Serbia",
    "Singapore",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
]


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> LastFMRecommendationsProvider:
    """Initialize provider(instance) with given configuration."""
    return LastFMRecommendationsProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider."""
    # Handle clear cache action
    if action == CONF_ACTION_CLEAR_CACHE and instance_id:
        # Get the provider instance and clear its cache
        provider = mass.get_provider(instance_id)
        if isinstance(provider, LastFMRecommendationsProvider):
            await provider.recommendations_manager.clear_cache()
            # Trigger re-population after clearing
            mass.create_task(provider._populate_recommendations())

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
            key="username",
            type=ConfigEntryType.STRING,
            label="Last.fm Username",
            required=False,
            description="Your Last.fm username for genre-based recommendations (optional)",
            value=values.get("username") if values else None,
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
            label="Enable Global Top Artists",
            default_value=False,
            description="Show worldwide top artists chart from Last.fm",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_top_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Global Top Tracks",
            default_value=False,
            description="Show worldwide top tracks chart from Last.fm",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Artists",
            default_value=False,
            description="Show top artists from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_albums",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Albums",
            default_value=False,
            description="Show top albums from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_genre_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Genre Tracks",
            default_value=False,
            description="Show top tracks from your most played genre (requires username)",
            category="recommendations",
        ),
        ConfigEntry(
            key="geo_country",
            type=ConfigEntryType.STRING,
            label="Country for Geographic Charts",
            default_value="Argentina",
            description="Select country for geography-based top artists and tracks",
            options=[ConfigValueOption(country, country) for country in GEO_COUNTRIES],
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_geo_artists",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Geographic Top Artists",
            default_value=False,
            description="Show top artists from selected country",
            category="recommendations",
        ),
        ConfigEntry(
            key="enable_geo_tracks",
            type=ConfigEntryType.BOOLEAN,
            label="Enable Geographic Top Tracks",
            default_value=False,
            description="Show top tracks from selected country",
            category="recommendations",
        ),
        ConfigEntry(
            key=CONF_ACTION_CLEAR_CACHE,
            type=ConfigEntryType.ACTION,
            label="Clear Recommendation Cache",
            description=(
                "Clear all cached recommendations. "
                "Use if a provider was removed or recommendations are stale."
            ),
            action=CONF_ACTION_CLEAR_CACHE,
            action_label="Clear Cache",
            category="advanced",
            required=False,
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

        # Try to load cached recommendation folders from persistent storage
        # This prevents losing recommendations on provider reload
        cache_key = f"recommendation_folders_{self.instance_id}"
        cached_folders = await self.mass.cache.get(cache_key)

        if cached_folders and isinstance(cached_folders, list):
            self._recommendation_folders: list[RecommendationFolder] = cached_folders
            self._recommendations_populated = True
            self.logger.info("Loaded %d recommendation folders from cache", len(cached_folders))
        else:
            # Initialize empty recommendation folders (will be populated progressively)
            self._recommendation_folders = []
            self._recommendations_populated = False

            # Start background task to populate recommendations immediately
            # API key validation happens during first populate attempt
            self.mass.create_task(self._populate_recommendations())

        # Schedule periodic refresh using MA's scheduler
        self._schedule_refresh()

    async def _populate_recommendations(self) -> None:
        """Populate recommendation folders in the background.

        This runs immediately after initialization, building recommendation
        folders progressively. Each folder appears as soon as it's populated,
        allowing the frontend to display results incrementally.
        """
        try:
            # Wait 20 seconds for other providers (e.g., Spotify) to finish loading
            # This prevents resolution failures due to no streaming providers being available yet
            self.logger.info(
                "Waiting 20 seconds for other providers to load before building recommendations..."
            )
            await asyncio.sleep(20)

            self.logger.info("Starting background population of recommendations")

            # Build folders incrementally - each category appears as soon as it's ready
            # Get personalized recommendations based on user's library
            personalized_folders = (
                await self.recommendations_manager._get_personalized_recommendations()
            )
            if personalized_folders:
                self._recommendation_folders.extend(personalized_folders)
                self.logger.info(
                    "Added %d personalized recommendation folder(s)", len(personalized_folders)
                )

            # Get global discovery recommendations
            global_folders = await self.recommendations_manager._get_global_recommendations()
            if global_folders:
                self._recommendation_folders.extend(global_folders)
                self.logger.info("Added %d global recommendation folder(s)", len(global_folders))

            # Get genre-based recommendations (requires username)
            genre_folders = await self.recommendations_manager._get_genre_based_recommendations()
            if genre_folders:
                self._recommendation_folders.extend(genre_folders)
                self.logger.info(
                    "Added %d genre-based recommendation folder(s)", len(genre_folders)
                )

            # Get geography-based recommendations
            geo_folders = await self.recommendations_manager._get_geo_based_recommendations()
            if geo_folders:
                self._recommendation_folders.extend(geo_folders)
                self.logger.info(
                    "Added %d geography-based recommendation folder(s)", len(geo_folders)
                )

            self._recommendations_populated = True
            self.logger.info(
                "Recommendations fully populated with %d total folders",
                len(self._recommendation_folders),
            )

            # Save to persistent cache so recommendations survive provider reloads
            cache_key = f"recommendation_folders_{self.instance_id}"
            await self.mass.cache.set(
                cache_key,
                self._recommendation_folders,
                expiration=60 * 60 * 24,  # 24 hours
            )
        except MusicAssistantError as err:
            # Expected MA errors (provider unavailable, database errors, etc.)
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
        except MusicAssistantError as err:
            # Expected MA errors (provider unavailable, database errors, etc.)
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
        3. Global Top Artists (worldwide chart from Last.fm)
        4. Global Top Tracks (worldwide chart from Last.fm)

        Personalized folders only appear if the user has listening history.
        If the library is empty, only global charts will be shown.
        """
        # Return current state (empty on first call, populated later)
        return self._recommendation_folders
