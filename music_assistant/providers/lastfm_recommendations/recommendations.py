"""Recommendation logic for Last.fm."""

from __future__ import annotations

import asyncio
import datetime
import random
from typing import TYPE_CHECKING, Any, cast

from music_assistant_models.enums import ExternalID, MediaType
from music_assistant_models.media_items import (
    Album,
    Artist,
    RecommendationFolder,
    Track,
    UniqueList,
)

from music_assistant.providers.lastfm_recommendations.constants import (
    CACHE_CATEGORY_RESOLVED_ITEMS,
    CACHE_EXPIRATION_SECONDS,
    RESOLUTION_BUFFER_LARGE,
    RESOLUTION_BUFFER_SMALL,
    SIMILAR_ITEMS_BUFFER,
    SIMILAR_ITEMS_PER_SEED,
    TARGET_ITEM_COUNT,
    TOP_ARTISTS_LIMIT,
    TOP_ITEMS_TO_TAKE,
    TOP_TAGS_LIMIT,
    TOP_TRACKS_LIMIT,
)
from music_assistant.providers.lastfm_recommendations.parsers import (
    parse_album,
    parse_artist,
    parse_track,
)

if TYPE_CHECKING:
    from music_assistant.providers.lastfm_recommendations import LastFMRecommendationsProvider


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

        # Cache resolved items by their unique key (MBID or name) to avoid re-resolving
        self._resolved_cache: dict[str, Artist | Album | Track] = {}

    async def clear_cache(self) -> None:
        """Clear both in-memory and persistent caches.

        Useful when a streaming provider is removed or when recommendations
        are returning stale/incorrect results.
        """
        # Clear in-memory cache
        self._resolved_cache.clear()

        # Clear persistent cache for resolved items
        await self.mass.cache.clear(category_filter=CACHE_CATEGORY_RESOLVED_ITEMS)

        # Clear cached recommendation folders
        cache_key = f"recommendation_folders_{self.provider.instance_id}"
        await self.mass.cache.delete(cache_key)

        # Clear the provider's in-memory folders too
        self.provider._recommendation_folders.clear()
        self.provider._recommendations_populated = False

        self.logger.info("Cleared all recommendation caches (in-memory and persistent)")

    async def _is_in_library(self, item_data: dict[str, Any], media_type: MediaType) -> bool:
        """Check if an item is already in the library using a cheap database query.

        This avoids expensive MusicBrainz lookups and provider searches for items
        the user already has.

        :param item_data: Raw Last.fm item data (artist, album, or track dict).
        :param media_type: Type of media item to check.
        :return: True if item is in library, False otherwise.
        """
        # Try MBID lookup first (most reliable)
        mbid = item_data.get("mbid")
        if mbid:
            # Check if library has this external ID using the appropriate controller
            if media_type == MediaType.ARTIST:
                if await self.mass.music.artists.get_library_item_by_external_id(
                    mbid, ExternalID.MB_ARTIST
                ):
                    return True
            elif media_type == MediaType.ALBUM:
                if await self.mass.music.albums.get_library_item_by_external_id(
                    mbid, ExternalID.MB_ALBUM
                ):
                    return True
            elif media_type == MediaType.TRACK:
                if await self.mass.music.tracks.get_library_item_by_external_id(
                    mbid, ExternalID.MB_RECORDING
                ):
                    return True

        # Fallback to name search (less reliable but catches items without MBID)
        if media_type == MediaType.ARTIST:
            name = item_data.get("name", "")
            if name:
                artist_results = await self.mass.music.artists.library_items(search=name, limit=1)
                return len(artist_results) > 0

        elif media_type == MediaType.ALBUM:
            name = item_data.get("name", "")
            if name:
                album_results = await self.mass.music.albums.library_items(search=name, limit=1)
                return len(album_results) > 0

        elif media_type == MediaType.TRACK:
            # For tracks, need both track name and artist name
            artist_info = item_data.get("artist", {})
            artist_name = (
                artist_info if isinstance(artist_info, str) else artist_info.get("name", "")
            )
            track_name = item_data.get("name", "")
            if track_name and artist_name:
                search_query = f"{artist_name} {track_name}"
                track_results = await self.mass.music.tracks.library_items(
                    search=search_query, limit=1
                )
                return len(track_results) > 0

        return False

    def _sample_items(
        self, items: list[dict[str, Any]], seed_suffix: str, target_count: int = TARGET_ITEM_COUNT
    ) -> list[dict[str, Any]]:
        """Sample items using 'top N + random' strategy.

        Takes the top N items and random items from the remainder to reach target_count.
        Uses a daily-based random seed for consistency within the day.

        :param items: List of items to sample from (already filtered).
        :param seed_suffix: Unique suffix for random seed (to vary between recommendation types).
        :param target_count: Target number of items to return.
        :return: List of sampled items (up to target_count).
        """
        if len(items) <= target_count:
            # Not enough items to sample, return all
            return items

        # Take top items
        top_items = items[:TOP_ITEMS_TO_TAKE]

        # Random sample from the rest
        remaining = items[TOP_ITEMS_TO_TAKE:]
        random_count = target_count - TOP_ITEMS_TO_TAKE

        # Use daily seed for consistency (same results all day, changes daily)
        seed = f"{datetime.datetime.now(tz=datetime.UTC).date().isoformat()}_{seed_suffix}"
        random.seed(seed)
        random_items = random.sample(remaining, min(random_count, len(remaining)))

        return top_items + random_items

    async def _get_or_resolve_artist(self, lastfm_artist: dict[str, Any]) -> Artist | None:
        """Get artist from cache or resolve it.

        Uses a two-tier caching strategy:
        1. In-memory cache (fast, cleared on restart)
        2. Persistent cache (survives restarts, 90-day expiry)

        :param lastfm_artist: Raw Last.fm artist dict.
        :return: Resolved Artist or None if not found.
        """
        # Create cache key (prefer MBID, fallback to name)
        cache_key = lastfm_artist.get("mbid") or lastfm_artist.get("name", "")
        if not cache_key:
            return None

        # Check in-memory cache first (fastest)
        if cache_key in self._resolved_cache:
            cached = self._resolved_cache[cache_key]
            if isinstance(cached, Artist):
                return cached

        # Check persistent cache (survives restarts)
        persistent_cache_key = f"artist_{cache_key}"
        cached_artist = await self.mass.cache.get(
            key=persistent_cache_key, category=CACHE_CATEGORY_RESOLVED_ITEMS
        )
        if cached_artist is not None and isinstance(cached_artist, Artist):
            # Store in memory cache for faster subsequent access
            artist_obj = cast("Artist", cached_artist)
            self._resolved_cache[cache_key] = artist_obj
            return artist_obj

        # Not in any cache - resolve it
        artist = await parse_artist(lastfm_artist, self.mass, self.provider.instance_id)
        if artist:
            # Store in both caches
            self._resolved_cache[cache_key] = artist
            await self.mass.cache.set(
                persistent_cache_key,
                artist,
                category=CACHE_CATEGORY_RESOLVED_ITEMS,
                expiration=CACHE_EXPIRATION_SECONDS,
            )
        return artist

    async def _get_or_resolve_track(self, lastfm_track: dict[str, Any]) -> Track | None:
        """Get track from cache or resolve it.

        Uses a two-tier caching strategy:
        1. In-memory cache (fast, cleared on restart)
        2. Persistent cache (survives restarts, 90-day expiry)

        :param lastfm_track: Raw Last.fm track dict.
        :return: Resolved Track or None if not found.
        """
        # Create cache key (prefer MBID, fallback to artist_name)
        cache_key = lastfm_track.get("mbid")
        if not cache_key:
            artist_data = lastfm_track.get("artist", {})
            artist_name = (
                artist_data if isinstance(artist_data, str) else artist_data.get("name", "")
            )
            track_name = lastfm_track.get("name", "")
            cache_key = f"{artist_name}_{track_name}" if artist_name and track_name else ""

        if not cache_key:
            return None

        # Check in-memory cache first (fastest)
        if cache_key in self._resolved_cache:
            cached = self._resolved_cache[cache_key]
            if isinstance(cached, Track):
                return cached

        # Check persistent cache (survives restarts)
        persistent_cache_key = f"track_{cache_key}"
        cached_track = await self.mass.cache.get(
            key=persistent_cache_key, category=CACHE_CATEGORY_RESOLVED_ITEMS
        )
        if cached_track is not None and isinstance(cached_track, Track):
            # Store in memory cache for faster subsequent access
            track_obj = cast("Track", cached_track)
            self._resolved_cache[cache_key] = track_obj
            return track_obj

        # Not in any cache - resolve it
        track = await parse_track(
            lastfm_track, self.mbid_resolver, self.mass, self.provider.instance_id
        )
        if track:
            # Store in both caches
            self._resolved_cache[cache_key] = track
            await self.mass.cache.set(
                persistent_cache_key,
                track,
                category=CACHE_CATEGORY_RESOLVED_ITEMS,
                expiration=CACHE_EXPIRATION_SECONDS,
            )
        return track

    async def _get_or_resolve_album(self, lastfm_album: dict[str, Any]) -> Album | None:
        """Get album from cache or resolve it.

        Uses a two-tier caching strategy:
        1. In-memory cache (fast, cleared on restart)
        2. Persistent cache (survives restarts, 90-day expiry)

        :param lastfm_album: Raw Last.fm album dict.
        :return: Resolved Album or None if not found.
        """
        # Create cache key (prefer MBID, fallback to artist+album name)
        cache_key = lastfm_album.get("mbid")
        if not cache_key:
            artist_data = lastfm_album.get("artist", {})
            artist_name = (
                artist_data if isinstance(artist_data, str) else artist_data.get("name", "")
            )
            album_name = lastfm_album.get("name", "")
            cache_key = f"{artist_name}_{album_name}" if artist_name and album_name else ""

        if not cache_key:
            return None

        # Check in-memory cache first (fastest)
        if cache_key in self._resolved_cache:
            cached = self._resolved_cache[cache_key]
            if isinstance(cached, Album):
                return cached

        # Check persistent cache (survives restarts)
        persistent_cache_key = f"album_{cache_key}"
        cached_album = await self.mass.cache.get(
            key=persistent_cache_key, category=CACHE_CATEGORY_RESOLVED_ITEMS
        )
        if cached_album is not None and isinstance(cached_album, Album):
            # Store in memory cache for faster subsequent access
            album_obj = cast("Album", cached_album)
            self._resolved_cache[cache_key] = album_obj
            return album_obj

        # Not in any cache - resolve it
        album = await parse_album(lastfm_album, self.mass, self.provider.instance_id)
        if album:
            # Store in both caches
            self._resolved_cache[cache_key] = album
            await self.mass.cache.set(
                persistent_cache_key,
                album,
                category=CACHE_CATEGORY_RESOLVED_ITEMS,
                expiration=CACHE_EXPIRATION_SECONDS,
            )
        return album

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

        Generates up to 9 recommendation folders:
        - Discover Similar Artists (personalized)
        - Discover Similar Tracks (personalized)
        - Global Top Artists
        - Global Top Tracks
        - Discover <Genre> Artists (based on user's top tag)
        - Discover <Genre> Albums (based on user's top tag)
        - Discover <Genre> Tracks (based on user's top tag)
        - Top Artists in <Country> (geography-based)
        - Top Tracks in <Country> (geography-based)

        Personalized folders only appear if user has listening history.
        Genre folders only appear if username is configured.
        Geography folders only appear if enabled in config.

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

        # Get genre-based recommendations (requires username)
        folders.extend(await self._get_genre_based_recommendations())

        # Get geography-based recommendations
        folders.extend(await self._get_geo_based_recommendations())

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

        # Similar Artists (only if enabled)
        if self.provider.config.get_value("enable_similar_artists"):
            # Get top most played artists from library
            top_artists = await self.mass.music.artists.library_items(
                limit=TOP_ARTISTS_LIMIT, order_by="play_count_desc"
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
                            items=UniqueList(similar_artists[:TARGET_ITEM_COUNT]),
                            subtitle=f"Based on your top {len(top_artists)} artists",
                            icon="mdi-account-music-outline",
                        )
                    )

        # Similar Tracks (only if enabled)
        if self.provider.config.get_value("enable_similar_tracks"):
            # Get top most played tracks from library
            top_tracks = await self.mass.music.tracks.library_items(
                limit=TOP_TRACKS_LIMIT, order_by="play_count_desc"
            )

            if top_tracks:
                # Get similar tracks based on user's favorites
                similar_tracks = await self._get_similar_tracks_from_seeds(top_tracks)

                if similar_tracks:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_similar_tracks",
                            name="Discover Similar Tracks",
                            provider=self.provider.instance_id,
                            items=UniqueList(similar_tracks[:TARGET_ITEM_COUNT]),
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

        # Global Top Artists (only if enabled)
        if self.provider.config.get_value("enable_top_artists"):
            # Request extra items to account for resolution failures
            top_artists_raw = await self.api.get_chart_top_artists(limit=RESOLUTION_BUFFER_SMALL)
            if top_artists_raw:
                # Parse and resolve artists (uses cache to avoid re-resolving)
                resolved_artists = await asyncio.gather(
                    *[self._get_or_resolve_artist(artist_data) for artist_data in top_artists_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved = [artist for artist in resolved_artists if artist is not None]
                deduplicated = list(UniqueList(all_resolved))[:TARGET_ITEM_COUNT]
                top_artists = UniqueList(deduplicated)

                if len(top_artists) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Global Top Artists: only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        len(top_artists),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len([a for a in resolved_artists if a is not None]),
                    )

                if top_artists:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_chart_top_artists",
                            name="Global Top Artists",
                            provider=self.provider.instance_id,
                            items=UniqueList(top_artists),
                            subtitle="Most popular artists worldwide",
                            icon="mdi-chart-line",
                        )
                    )

        # Global Top Tracks (only if enabled)
        if self.provider.config.get_value("enable_top_tracks"):
            # Request extra items to account for resolution failures
            top_tracks_raw = await self.api.get_chart_top_tracks(limit=RESOLUTION_BUFFER_SMALL)
            if top_tracks_raw:
                # Parse and resolve tracks (uses cache to avoid re-resolving)
                resolved_tracks = await asyncio.gather(
                    *[self._get_or_resolve_track(track_data) for track_data in top_tracks_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved_tracks = [track for track in resolved_tracks if track is not None]
                deduplicated_tracks = list(UniqueList(all_resolved_tracks))[:TARGET_ITEM_COUNT]
                top_tracks = UniqueList(deduplicated_tracks)

                if len(top_tracks) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Global Top Tracks: only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        len(top_tracks),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len(all_resolved_tracks),
                    )

                if top_tracks:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_chart_top_tracks",
                            name="Global Top Tracks",
                            provider=self.provider.instance_id,
                            items=UniqueList(top_tracks),
                            subtitle="Most popular tracks worldwide",
                            icon="mdi-chart-box",
                        )
                    )

        return folders

    async def _get_genre_based_recommendations(self) -> list[RecommendationFolder]:
        """Get genre-based recommendations using user's top tag.

        Fetches the user's most played genre/tag from Last.fm and uses it to fetch
        top albums, artists, and tracks for that genre.
        Returns up to 3 folders (genre albums, artists, tracks).

        Requires username to be configured.

        :return: List of genre-based recommendation folders (empty if no username or API fails).
        """
        folders: list[RecommendationFolder] = []

        # Check if username is configured
        username = self.provider.config.get_value("username")
        if not username or not isinstance(username, str):
            return folders

        # Get user's top tag (most played genre)
        top_tags = await self.api.get_user_top_tags(username, limit=TOP_TAGS_LIMIT)
        if not top_tags:
            return folders

        tag_name = top_tags[0].get("name")
        if not tag_name:
            return folders

        # Genre Artists (only if enabled)
        if self.provider.config.get_value("enable_genre_artists"):
            # Request many items to have enough after filtering library items
            genre_artists_raw = await self.api.get_tag_top_artists(
                tag_name, limit=RESOLUTION_BUFFER_LARGE
            )
            if genre_artists_raw:
                # Filter out library items using cheap database query (no expensive resolution yet)
                non_library_artists_raw = [
                    artist_data
                    for artist_data in genre_artists_raw
                    if not await self._is_in_library(artist_data, MediaType.ARTIST)
                ]

                # Sample items to account for resolution failures
                sampled_artists_raw = self._sample_items(
                    non_library_artists_raw,
                    seed_suffix="genre_artists",
                    target_count=RESOLUTION_BUFFER_SMALL,
                )

                # Resolve items (expensive MusicBrainz + provider search)
                resolved_artists = await asyncio.gather(
                    *[
                        self._get_or_resolve_artist(artist_data)
                        for artist_data in sampled_artists_raw
                    ]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved = [artist for artist in resolved_artists if artist is not None]
                deduplicated = list(UniqueList(all_resolved))[:TARGET_ITEM_COUNT]
                genre_artists = UniqueList(deduplicated)

                if len(genre_artists) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Genre Artists (%s): only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        tag_name,
                        len(genre_artists),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len([a for a in resolved_artists if a is not None]),
                    )

                if genre_artists:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_genre_artists",
                            name=f"Discover {tag_name.title()} Artists",
                            provider=self.provider.instance_id,
                            items=UniqueList(genre_artists),
                            subtitle="Top artists in your most played genre",
                            icon="mdi-account-music",
                        )
                    )

        # Genre Albums (only if enabled)
        if self.provider.config.get_value("enable_genre_albums"):
            # Request many items to have enough after filtering library items
            genre_albums_raw = await self.api.get_tag_top_albums(
                tag_name, limit=RESOLUTION_BUFFER_LARGE
            )
            if genre_albums_raw:
                # Filter out library items using cheap database query (no expensive resolution yet)
                non_library_albums_raw = [
                    album_data
                    for album_data in genre_albums_raw
                    if not await self._is_in_library(album_data, MediaType.ALBUM)
                ]

                # Sample items to account for resolution failures
                sampled_albums_raw = self._sample_items(
                    non_library_albums_raw,
                    seed_suffix="genre_albums",
                    target_count=RESOLUTION_BUFFER_SMALL,
                )

                # Resolve items (expensive MusicBrainz + provider search)
                resolved_albums = await asyncio.gather(
                    *[self._get_or_resolve_album(album_data) for album_data in sampled_albums_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved_albums = [album for album in resolved_albums if album is not None]
                genre_albums = list(UniqueList(all_resolved_albums))[:TARGET_ITEM_COUNT]

                if len(genre_albums) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Genre Albums (%s): only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        tag_name,
                        len(genre_albums),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len(all_resolved_albums),
                    )

                if genre_albums:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_genre_albums",
                            name=f"Discover {tag_name.title()} Albums",
                            provider=self.provider.instance_id,
                            items=UniqueList(genre_albums),
                            subtitle="Top albums in your most played genre",
                            icon="mdi-album",
                        )
                    )

        # Genre Tracks (only if enabled)
        if self.provider.config.get_value("enable_genre_tracks"):
            # Request many items to have enough after filtering library items
            genre_tracks_raw = await self.api.get_tag_top_tracks(
                tag_name, limit=RESOLUTION_BUFFER_LARGE
            )
            if genre_tracks_raw:
                # Filter out library items using cheap database query (no expensive resolution yet)
                non_library_tracks_raw = [
                    track_data
                    for track_data in genre_tracks_raw
                    if not await self._is_in_library(track_data, MediaType.TRACK)
                ]

                # Sample items to account for resolution failures
                sampled_tracks_raw = self._sample_items(
                    non_library_tracks_raw,
                    seed_suffix="genre_tracks",
                    target_count=RESOLUTION_BUFFER_SMALL,
                )

                # Resolve items (expensive MusicBrainz + provider search)
                resolved_tracks = await asyncio.gather(
                    *[self._get_or_resolve_track(track_data) for track_data in sampled_tracks_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved_genre_tracks = [
                    track for track in resolved_tracks if track is not None
                ]
                genre_tracks = list(UniqueList(all_resolved_genre_tracks))[:TARGET_ITEM_COUNT]

                if len(genre_tracks) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Genre Tracks (%s): only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        tag_name,
                        len(genre_tracks),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len(all_resolved_genre_tracks),
                    )

                if genre_tracks:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_genre_tracks",
                            name=f"Discover {tag_name.title()} Tracks",
                            provider=self.provider.instance_id,
                            items=UniqueList(genre_tracks),
                            subtitle="Top tracks in your most played genre",
                            icon="mdi-music",
                        )
                    )

        return folders

    async def _get_geo_based_recommendations(self) -> list[RecommendationFolder]:
        """Get geography-based recommendations using selected country.

        Fetches top artists and tracks for the configured country from Last.fm.
        Returns up to 2 folders (geo artists, geo tracks).

        Requires country to be configured.

        :return: List of geo-based recommendation folders.
            Returns empty list if country not configured or API fails.
        """
        folders: list[RecommendationFolder] = []

        # Get configured country
        country = self.provider.config.get_value("geo_country")
        if not country or not isinstance(country, str):
            return folders

        # Geo Top Artists (only if enabled)
        if self.provider.config.get_value("enable_geo_artists"):
            # Request extra items to account for resolution failures
            geo_artists_raw = await self.api.get_geo_top_artists(
                country, limit=RESOLUTION_BUFFER_SMALL
            )
            if geo_artists_raw:
                # Resolve all artists (no library filtering for geographic charts)
                resolved_artists = await asyncio.gather(
                    *[self._get_or_resolve_artist(artist_data) for artist_data in geo_artists_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved = [artist for artist in resolved_artists if artist is not None]
                geo_artists = list(UniqueList(all_resolved))[:TARGET_ITEM_COUNT]

                if len(geo_artists) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Geo Top Artists (%s): only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        country,
                        len(geo_artists),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len([a for a in resolved_artists if a is not None]),
                    )

                if geo_artists:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_geo_artists",
                            name=f"Top artists for {country}",
                            provider=self.provider.instance_id,
                            items=UniqueList(geo_artists),
                            subtitle=f"Most popular artists in {country}",
                            icon="mdi-earth",
                        )
                    )

        # Geo Top Tracks (only if enabled)
        if self.provider.config.get_value("enable_geo_tracks"):
            # Request extra items to account for resolution failures
            geo_tracks_raw = await self.api.get_geo_top_tracks(
                country, limit=RESOLUTION_BUFFER_SMALL
            )
            if geo_tracks_raw:
                # Resolve all tracks (no library filtering for geographic charts)
                resolved_tracks = await asyncio.gather(
                    *[self._get_or_resolve_track(track_data) for track_data in geo_tracks_raw]
                )
                # Filter out failed resolutions, deduplicate, then take target count
                all_resolved_geo_tracks = [track for track in resolved_tracks if track is not None]
                geo_tracks = list(UniqueList(all_resolved_geo_tracks))[:TARGET_ITEM_COUNT]

                if len(geo_tracks) < TARGET_ITEM_COUNT:
                    self.logger.info(
                        "Geo Top Tracks (%s): only %d/%d items after resolution "
                        "(requested %d, resolved %d)",
                        country,
                        len(geo_tracks),
                        TARGET_ITEM_COUNT,
                        RESOLUTION_BUFFER_SMALL,
                        len(all_resolved_geo_tracks),
                    )

                if geo_tracks:
                    folders.append(
                        RecommendationFolder(
                            item_id=f"{self.provider.instance_id}_geo_tracks",
                            name=f"Top tracks for {country}",
                            provider=self.provider.instance_id,
                            items=UniqueList(geo_tracks),
                            subtitle=f"Most popular tracks in {country}",
                            icon="mdi-earth",
                        )
                    )

        return folders

    async def _get_similar_artists_from_seeds(self, seed_artists: list[Artist]) -> list[Artist]:
        """Get similar artists based on seed artists.

        For each seed artist, fetches similar artists from Last.fm,
        deduplicates, resolves to actual provider items, and returns results.

        :param seed_artists: List of seed artists from user's library.
        :return: List of resolved media items or ItemMappings.
        """
        all_similar: list[dict[str, Any]] = []

        # Build set of seed artist MBIDs and names to exclude from results
        seed_mbids = {
            seed_artist.get_external_id(ExternalID.MB_ARTIST)
            for seed_artist in seed_artists
            if seed_artist.get_external_id(ExternalID.MB_ARTIST)
        }
        seed_names = {seed_artist.name.lower() for seed_artist in seed_artists}

        # Get similar artists for each seed
        for seed_artist in seed_artists:
            # Extract MBID if available using get_external_id helper
            mbid = seed_artist.get_external_id(ExternalID.MB_ARTIST)

            similar = await self.api.get_similar_artists(
                artist_name=seed_artist.name, artist_mbid=mbid, limit=SIMILAR_ITEMS_PER_SEED
            )
            all_similar.extend(similar)

        # Deduplicate by both MBID and name to prevent duplicates when
        # Last.fm returns same artist with/without MBID
        # Also exclude seed artists to prevent showing them in their own recommendations
        seen_mbids = set()
        seen_names = set()
        unique_similar: list[dict[str, Any]] = []
        for artist_data in all_similar:
            mbid = artist_data.get("mbid")
            name = artist_data.get("name", "").lower()

            # Skip if this is a seed artist (prevent showing artist in its own recommendations)
            if mbid and mbid in seed_mbids:
                continue
            if name and name in seed_names:
                continue

            # Skip if we've already seen this MBID or name
            if mbid and mbid in seen_mbids:
                continue
            if name and name in seen_names:
                continue

            # Add to unique list and mark as seen
            unique_similar.append(artist_data)
            if mbid:
                seen_mbids.add(mbid)
            if name:
                seen_names.add(name)

        # Sort by match score (similarity) and take top results
        unique_similar.sort(key=lambda x: float(x.get("match", 0)), reverse=True)

        # Parse and resolve artists (uses cache to avoid re-resolving)
        # Use asyncio.gather to ensure proper concurrent execution without race conditions
        resolved_artists = await asyncio.gather(
            *[
                self._get_or_resolve_artist(artist_data)
                for artist_data in unique_similar[:SIMILAR_ITEMS_BUFFER]
            ]
        )
        result = [artist for artist in resolved_artists if artist is not None]

        if len(result) < len(resolved_artists):
            self.logger.info(
                "Similar artists: %d/%d resolved successfully (%d failed)",
                len(result),
                len(resolved_artists),
                len(resolved_artists) - len(result),
            )

        return result

    async def _get_similar_tracks_from_seeds(self, seed_tracks: list[Track]) -> list[Track]:
        """Get similar tracks based on seed tracks.

        For each seed track, fetches similar tracks from Last.fm,
        deduplicates, resolves ISRCs via MusicBrainz, and returns results.

        :param seed_tracks: List of seed tracks from user's library.
        :return: List of resolved media items or ItemMappings.
        """
        all_similar: list[dict[str, Any]] = []

        # Build set of seed track MBIDs and name keys to exclude from results
        seed_mbids = {
            seed_track.get_external_id(ExternalID.MB_RECORDING)
            for seed_track in seed_tracks
            if seed_track.get_external_id(ExternalID.MB_RECORDING)
        }
        seed_name_keys = {
            f"{seed_track.artists[0].name if seed_track.artists else ''}_{seed_track.name}".lower()
            for seed_track in seed_tracks
        }

        # Get similar tracks for each seed
        for seed_track in seed_tracks:
            # Extract MBID if available using get_external_id helper
            mbid = seed_track.get_external_id(ExternalID.MB_RECORDING)

            # Get artist name (first artist)
            artist_name = seed_track.artists[0].name if seed_track.artists else "Unknown Artist"

            similar = await self.api.get_similar_tracks(
                artist_name=artist_name,
                track_name=seed_track.name,
                track_mbid=mbid,
                limit=SIMILAR_ITEMS_PER_SEED,
            )
            all_similar.extend(similar)

        # Deduplicate by both MBID and name+artist to prevent duplicates when
        # Last.fm returns same track with/without MBID
        # Also exclude seed tracks to prevent showing them in their own recommendations
        seen_mbids = set()
        seen_names = set()
        unique_similar: list[dict[str, Any]] = []
        for track_data in all_similar:
            mbid = track_data.get("mbid")

            # Build name-based key for deduplication
            artist_info = track_data.get("artist", {})
            if isinstance(artist_info, str):
                artist_name = artist_info
            else:
                artist_name = artist_info.get("name", "")
            track_name = track_data.get("name", "")
            name_key = f"{artist_name}_{track_name}".lower() if artist_name and track_name else ""

            # Skip if this is a seed track (prevent showing track in its own recommendations)
            if mbid and mbid in seed_mbids:
                continue
            if name_key and name_key in seed_name_keys:
                continue

            # Skip if we've already seen this MBID or name combination
            if mbid and mbid in seen_mbids:
                continue
            if name_key and name_key in seen_names:
                continue

            # Add to unique list and mark as seen
            unique_similar.append(track_data)
            if mbid:
                seen_mbids.add(mbid)
            if name_key:
                seen_names.add(name_key)

        # Sort by match score (similarity) and take top results
        unique_similar.sort(key=lambda x: float(x.get("match", 0)), reverse=True)

        # Only resolve ISRCs for top tracks (optimization)
        top_tracks_data = unique_similar[:TARGET_ITEM_COUNT]

        # Parse and resolve tracks (uses cache to avoid re-resolving)
        # Use asyncio.gather to ensure proper concurrent execution without race conditions
        resolved_tracks = await asyncio.gather(
            *[self._get_or_resolve_track(track_data) for track_data in top_tracks_data]
        )
        return [track for track in resolved_tracks if track is not None]
