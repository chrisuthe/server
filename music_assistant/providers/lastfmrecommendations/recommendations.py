"""Recommendation logic for Last.fm."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from music_assistant_models.media_items import (
    Artist,
    MediaItemType,
    RecommendationFolder,
    Track,
    UniqueList,
)

from music_assistant.providers.lastfmrecommendations.parsers import parse_artist, parse_track

if TYPE_CHECKING:
    from music_assistant.providers.lastfmrecommendations import LastFMRecommendationsProvider


class LastFMRecommendationManager:
    """Manages Last.fm recommendations."""

    def __init__(self, provider: LastFMRecommendationsProvider) -> None:
        """Initialize recommendation manager.

        :param provider: The Last.fm recommendations provider instance.
        """
        self.provider = provider
        self.api = provider.api
        self.mbid_resolver = provider.mbid_resolver
        self.logger = provider.logger
        self.mass = provider.mass

    async def get_recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

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
        """Get personalized recommendations based on user's listening history."""
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
                        items=UniqueList[MediaItemType](similar_artists[:10]),
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
                        items=UniqueList[MediaItemType](similar_tracks[:10]),
                        subtitle=f"Based on your top {len(top_tracks)} tracks",
                        icon="mdi-music-note-outline",
                    )
                )

        return folders

    async def _get_global_recommendations(self) -> list[RecommendationFolder]:
        """Get global discovery recommendations from Last.fm charts."""
        folders: list[RecommendationFolder] = []

        # Global top artists
        top_artists_raw = await self.api.get_chart_top_artists(limit=10)
        if top_artists_raw:
            top_artists = [
                parse_artist(artist_data, self.provider.instance_id)
                for artist_data in top_artists_raw
            ]

            folders.append(
                RecommendationFolder(
                    item_id=f"{self.provider.instance_id}_chart_top_artists",
                    name="Last.fm Top Artists",
                    provider=self.provider.instance_id,
                    items=UniqueList[MediaItemType](top_artists),
                    subtitle="Most popular artists worldwide",
                    icon="mdi-chart-line",
                )
            )

        # Global top tracks
        top_tracks_raw = await self.api.get_chart_top_tracks(limit=10)
        if top_tracks_raw:
            top_tracks = [
                await parse_track(track_data, self.provider.instance_id, self.mbid_resolver)
                for track_data in top_tracks_raw
            ]

            folders.append(
                RecommendationFolder(
                    item_id=f"{self.provider.instance_id}_chart_top_tracks",
                    name="Last.fm Top Tracks",
                    provider=self.provider.instance_id,
                    items=UniqueList[MediaItemType](top_tracks),
                    subtitle="Most popular tracks worldwide",
                    icon="mdi-chart-box",
                )
            )

        return folders

    async def _get_similar_artists_from_seeds(self, seed_artists: list[Artist]) -> list[Artist]:
        """Get similar artists based on seed artists.

        :param seed_artists: List of seed artists from user's library.
        """
        all_similar: list[dict[str, Any]] = []

        # Get 3 similar artists for each seed
        for seed_artist in seed_artists:
            # Extract MBID if available from external_ids
            mbid = None
            for ext_id_type, ext_id_value in seed_artist.external_ids:
                if ext_id_type == "musicbrainz_artist":
                    mbid = ext_id_value
                    break

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

        # Parse to Artist objects
        return [
            parse_artist(artist_data, self.provider.instance_id)
            for artist_data in unique_similar[:12]  # Get 12 to ensure we have 10 after filtering
        ]

    async def _get_similar_tracks_from_seeds(self, seed_tracks: list[Track]) -> list[Track]:
        """Get similar tracks based on seed tracks.

        :param seed_tracks: List of seed tracks from user's library.
        """
        all_similar: list[dict[str, Any]] = []

        # Get 3 similar tracks for each seed
        for seed_track in seed_tracks:
            # Extract MBID if available
            mbid = None
            for ext_id_type, ext_id_value in seed_track.external_ids:
                if ext_id_type == "musicbrainz_recording":
                    mbid = ext_id_value
                    break

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

        # Parse to Track objects (this includes ISRC resolution via MusicBrainz)
        return [
            await parse_track(track_data, self.provider.instance_id, self.mbid_resolver)
            for track_data in top_tracks_data
        ]
