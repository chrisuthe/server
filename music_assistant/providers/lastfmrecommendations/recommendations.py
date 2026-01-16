"""Recommendation logic for Last.fm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.enums import ExternalID
from music_assistant_models.media_items import (
    Artist,
    ItemMapping,
    RecommendationFolder,
    Track,
    UniqueList,
)

from music_assistant.helpers.compare import compare_media_item
from music_assistant.providers.lastfmrecommendations.parsers import parse_artist, parse_track

if TYPE_CHECKING:
    from music_assistant.controllers.media.artists import ArtistsController
    from music_assistant.controllers.media.tracks import TracksController
    from music_assistant.providers.lastfmrecommendations import LastFMRecommendationsProvider


class LastFMRecommendationManager:
    """Manages Last.fm recommendations.

    This class orchestrates the generation of recommendation folders by:
    1. Querying user's listening history from Music Assistant
    2. Fetching similar items from Last.fm API
    3. Resolving MBIDs to ISRCs for accurate matching
    4. Organizing results into RecommendationFolder objects
    """

    def __init__(self, provider: LastFMRecommendationsProvider) -> None:
        """Initialize recommendation manager.

        :param provider: The Last.fm recommendations provider instance.
        """
        self.provider = provider
        self.api = provider.api
        self.mbid_resolver = provider.mbid_resolver
        self.logger = provider.logger
        self.mass = provider.mass

    async def resolve_item_mapping(
        self, item_mapping: ItemMapping
    ) -> Artist | Track | ItemMapping | None:
        """Resolve an ItemMapping to an actual library or provider item.

        Tries to find a matching item in the library or streaming providers
        using external IDs, then falls back to name-only search.

        :param item_mapping: ItemMapping to resolve.
        :return: Resolved Artist/Track, or None if no match found.
        """
        # First, try to find in library by external IDs
        ctrl: ArtistsController | TracksController
        if item_mapping.media_type.value == "artist":
            ctrl = self.mass.music.artists
        elif item_mapping.media_type.value == "track":
            ctrl = self.mass.music.tracks
        else:
            return item_mapping

        # Check library first
        if library_item := await ctrl.get_library_item_by_external_ids(item_mapping.external_ids):
            self.logger.debug(
                "Found %s in library: %s", item_mapping.media_type.value, library_item.name
            )
            return library_item

        # Not in library - search streaming providers with external ID matching
        for provider in self.mass.music.providers:
            if provider.domain == self.provider.domain:
                continue  # Skip ourselves
            if not provider.is_streaming_provider:
                continue

            try:
                # Search using the name (limit=1 since we only need first match)
                search_results = await ctrl.search(item_mapping.name, provider.instance_id, limit=1)

                # Find first match using external IDs
                for result in search_results:
                    if compare_media_item(item_mapping, result, strict=False):
                        self.logger.debug(
                            "Found %s on provider %s via external IDs: %s",
                            item_mapping.media_type.value,
                            provider.name,
                            result.name,
                        )
                        return result
            except Exception as err:
                self.logger.debug(
                    "Provider %s search failed: %s", provider.name, type(err).__name__
                )
                continue

        # Fallback: Try name-only search without external ID matching
        for provider in self.mass.music.providers:
            if provider.domain == self.provider.domain:
                continue
            if not provider.is_streaming_provider:
                continue

            try:
                search_results = await ctrl.search(item_mapping.name, provider.instance_id, limit=1)
                if search_results:
                    # Return first result even without external ID match
                    result = search_results[0]
                    self.logger.debug(
                        "Found %s on provider %s via name search: %s",
                        item_mapping.media_type.value,
                        provider.name,
                        result.name,
                    )
                    return result
            except Exception as err:
                self.logger.debug(
                    "Provider %s fallback search failed: %s", provider.name, type(err).__name__
                )
                continue

        # No match found anywhere - filter out this item
        self.logger.debug(
            "Could not resolve %s: %s", item_mapping.media_type.value, item_mapping.name
        )
        return None

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

        Generates up to 4 recommendation folders:
        - Discover Similar Artists (personalized)
        - Discover Similar Tracks (personalized)
        - Last.fm Top Artists (global)
        - Last.fm Top Tracks (global)

        Personalized folders only appear if user has listening history.

        :return: List of recommendation folders (may be empty if no data available).

        Note: Individual recommendation methods handle their own errors and
        return empty lists on failure, so errors should not bubble up here.
        If they do, it indicates a programming error that should be fixed.
        """
        folders: list[RecommendationFolder] = []

        # Get personalized recommendations based on user's library
        folders.extend(await self._get_personalized_recommendations())

        # Get global discovery recommendations
        folders.extend(await self._get_global_recommendations())

        return folders

    async def _get_personalized_recommendations(self) -> list[RecommendationFolder]:
        """Get personalized recommendations based on user's listening history.

        Queries MA's library for top played artists/tracks and fetches similar
        items from Last.fm. Returns up to 2 folders (similar artists, similar tracks).

        :return: List of personalized recommendation folders (empty if no play history).
        """
        folders: list[RecommendationFolder] = []

        # TODO: Consider if users want all-time top items or recent top items (e.g., last week)
        # for more current recommendations. Current implementation uses all-time play_count.
        # Possible alternatives:
        # - Filter by timestamp: WHERE timestamp_added > date('now', '-7 days')
        # - Use last_played column: ORDER BY last_played DESC
        # - Weighted combination: play recent items more heavily
        # Need user feedback to determine best approach.

        # Get top 5 most played artists from library
        top_artists = await self.mass.music.artists.library_items(
            limit=5, order_by="play_count_desc"
        )

        if top_artists:
            # Get similar artists based on user's favorites
            similar_artists = await self._get_similar_artists_from_seeds(top_artists)

            if similar_artists:
                folders.append(
                    RecommendationFolder(
                        item_id=f"{self.provider.instance_id}_similar_artists",
                        name="Discover Similar Artists",
                        provider=self.provider.instance_id,
                        items=UniqueList(similar_artists[:10]),
                        subtitle=f"Based on your top {len(top_artists)} artists",
                        icon="mdi-account-music-outline",
                    )
                )

        # Get top 5 most played tracks from library
        top_tracks = await self.mass.music.tracks.library_items(limit=5, order_by="play_count_desc")

        if top_tracks:
            # Get similar tracks based on user's favorites
            similar_tracks = await self._get_similar_tracks_from_seeds(top_tracks)

            if similar_tracks:
                folders.append(
                    RecommendationFolder(
                        item_id=f"{self.provider.instance_id}_similar_tracks",
                        name="Discover Similar Tracks",
                        provider=self.provider.instance_id,
                        items=UniqueList(similar_tracks[:10]),
                        subtitle=f"Based on your top {len(top_tracks)} tracks",
                        icon="mdi-music-note-outline",
                    )
                )

        return folders

    async def _get_global_recommendations(self) -> list[RecommendationFolder]:
        """Get global discovery recommendations from Last.fm charts.

        Fetches Last.fm's worldwide top artists and tracks charts.
        Returns up to 2 folders (top artists, top tracks).

        :return: List of global chart recommendation folders (empty if API fails).
        """
        folders: list[RecommendationFolder] = []

        # Global top artists
        top_artists_raw = await self.api.get_chart_top_artists(limit=10)
        if top_artists_raw:
            top_artists_mappings = [parse_artist(artist_data) for artist_data in top_artists_raw]
            # Resolve ItemMappings to actual items (filter out None for unresolved items)
            top_artists = [
                item
                for item in [await self.resolve_item_mapping(m) for m in top_artists_mappings]
                if item is not None
            ]

            if top_artists:
                folders.append(
                    RecommendationFolder(
                        item_id=f"{self.provider.instance_id}_chart_top_artists",
                        name="Last.fm Top Artists",
                        provider=self.provider.instance_id,
                        items=UniqueList(top_artists),
                        subtitle="Most popular artists worldwide",
                        icon="mdi-chart-line",
                    )
                )

        # Global top tracks
        top_tracks_raw = await self.api.get_chart_top_tracks(limit=10)
        if top_tracks_raw:
            top_tracks_mappings = [
                await parse_track(track_data, self.mbid_resolver) for track_data in top_tracks_raw
            ]
            # Resolve ItemMappings to actual items (filter out None for unresolved items)
            top_tracks = [
                item
                for item in [await self.resolve_item_mapping(m) for m in top_tracks_mappings]
                if item is not None
            ]

            if top_tracks:
                folders.append(
                    RecommendationFolder(
                        item_id=f"{self.provider.instance_id}_chart_top_tracks",
                        name="Last.fm Top Tracks",
                        provider=self.provider.instance_id,
                        items=UniqueList(top_tracks),
                        subtitle="Most popular tracks worldwide",
                        icon="mdi-chart-box",
                    )
                )

        return folders

    async def _get_similar_artists_from_seeds(
        self, seed_artists: list[Artist]
    ) -> list[Artist | Track | ItemMapping]:
        """Get similar artists based on seed artists.

        For each seed artist, fetches 3 similar artists from Last.fm,
        deduplicates, resolves to actual provider items, and returns top 12.

        :param seed_artists: List of seed artists from user's library.
        :return: List of up to 12 resolved media items or ItemMappings.
        """
        all_similar: list[dict[str, Any]] = []

        # Get 3 similar artists for each seed
        for seed_artist in seed_artists:
            # Extract MBID if available using get_external_id helper
            mbid = seed_artist.get_external_id(ExternalID.MB_ARTIST)

            similar = await self.api.get_similar_artists(
                artist_name=seed_artist.name, artist_mbid=mbid, limit=3
            )
            all_similar.extend(similar)

        # Deduplicate by MBID or name
        seen = set()
        unique_similar: list[dict[str, Any]] = []
        for artist_data in all_similar:
            # Use MBID for deduplication if available, otherwise use name
            unique_key = artist_data.get("mbid") or artist_data.get("name", "").lower()
            if unique_key and unique_key not in seen:
                seen.add(unique_key)
                unique_similar.append(artist_data)

        # Sort by match score (similarity) and take top results
        unique_similar.sort(key=lambda x: float(x.get("match", 0)), reverse=True)

        # Parse to ItemMapping objects and resolve them
        artist_mappings = [
            parse_artist(artist_data)
            for artist_data in unique_similar[:12]  # Get 12 to ensure we have 10 after filtering
        ]
        # Resolve ItemMappings to actual items (filter out None for unresolved items)
        return [
            item
            for item in [await self.resolve_item_mapping(m) for m in artist_mappings]
            if item is not None
        ]

    async def _get_similar_tracks_from_seeds(
        self, seed_tracks: list[Track]
    ) -> list[Artist | Track | ItemMapping]:
        """Get similar tracks based on seed tracks.

        For each seed track, fetches 3 similar tracks from Last.fm,
        deduplicates, resolves ISRCs via MusicBrainz, and returns top 10.

        :param seed_tracks: List of seed tracks from user's library.
        :return: List of up to 10 resolved media items or ItemMappings.
        """
        all_similar: list[dict[str, Any]] = []

        # Get 3 similar tracks for each seed
        for seed_track in seed_tracks:
            # Extract MBID if available using get_external_id helper
            mbid = seed_track.get_external_id(ExternalID.MB_RECORDING)

            # Get artist name (first artist)
            artist_name = seed_track.artists[0].name if seed_track.artists else "Unknown Artist"

            similar = await self.api.get_similar_tracks(
                artist_name=artist_name,
                track_name=seed_track.name,
                track_mbid=mbid,
                limit=3,
            )
            all_similar.extend(similar)

        # Deduplicate by MBID or name+artist combination
        seen = set()
        unique_similar: list[dict[str, Any]] = []
        for track_data in all_similar:
            # Use MBID for deduplication if available
            if mbid := track_data.get("mbid"):
                unique_key = mbid
            else:
                # Fallback to name+artist
                artist_info = track_data.get("artist", {})
                if isinstance(artist_info, str):
                    artist_name = artist_info
                else:
                    artist_name = artist_info.get("name", "")
                track_name = track_data.get("name", "")
                unique_key = f"{artist_name}_{track_name}".lower()

            if unique_key and unique_key not in seen:
                seen.add(unique_key)
                unique_similar.append(track_data)

        # Sort by match score (similarity) and take top results
        unique_similar.sort(key=lambda x: float(x.get("match", 0)), reverse=True)

        # Only resolve ISRCs for top 10 tracks (optimization)
        top_tracks_data = unique_similar[:10]

        # Parse to ItemMapping objects (this includes ISRC resolution via MusicBrainz)
        track_mappings = [
            await parse_track(track_data, self.mbid_resolver) for track_data in top_tracks_data
        ]
        # Resolve ItemMappings to actual items (filter out None for unresolved items)
        return [
            item
            for item in [await self.resolve_item_mapping(m) for m in track_mappings]
            if item is not None
        ]
