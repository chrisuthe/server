"""Last.fm API client for recommendations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiohttp import ClientError
from music_assistant_models.errors import InvalidDataError

from music_assistant.helpers.throttle_retry import ThrottlerManager

if TYPE_CHECKING:
    from aiohttp import ClientSession

    from music_assistant.providers.lastfmrecommendations import LastFMRecommendationsProvider


class LastFMAPIClient:
    """Last.fm API client for fetching recommendations."""

    BASE_URL = "https://ws.audioscrobbler.com/2.0/"
    throttler = ThrottlerManager(rate_limit=5, period=1)  # 5 requests per second

    def __init__(self, provider: LastFMRecommendationsProvider) -> None:
        """Initialize Last.fm API client.

        :param provider: The Last.fm recommendations provider instance.
        """
        self.provider = provider
        self.logger = provider.logger
        self.http_session: ClientSession = provider.mass.http_session

    async def _get_data(self, method: str, **params: Any) -> dict[str, Any]:
        """Make a request to the Last.fm API.

        :param method: The Last.fm API method to call.
        :param params: Additional query parameters.
        :raises ClientError: If HTTP request fails.
        :raises InvalidDataError: If Last.fm returns an error or invalid JSON.
        :raises asyncio.TimeoutError: If request times out.
        """
        async with self.throttler.acquire():
            params.update(
                {
                    "method": method,
                    "api_key": self.provider.config.get_value("api_key"),
                    "format": "json",
                }
            )

            async with self.http_session.get(self.BASE_URL, params=params) as response:
                response.raise_for_status()
                data: dict[str, Any] = await response.json()

                # Last.fm returns errors in the response body
                if "error" in data:
                    error_msg = data.get("message", "Unknown error")
                    msg = f"Last.fm API error: {error_msg}"
                    raise InvalidDataError(msg)

                return data

    async def get_similar_artists(
        self, artist_name: str, artist_mbid: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Get similar artists from Last.fm.

        :param artist_name: Name of the artist.
        :param artist_mbid: Optional MusicBrainz ID for more accurate matching.
        :param limit: Maximum number of similar artists to return.
        """
        params: dict[str, Any] = {"limit": limit}

        # Prefer MBID if available for more accurate results
        if artist_mbid:
            params["mbid"] = artist_mbid
        else:
            params["artist"] = artist_name

        try:
            data = await self._get_data("artist.getSimilar", **params)
            similar_artists: list[dict[str, Any]] | dict[str, Any] = data.get(
                "similarartists", {}
            ).get("artist", [])

            # Normalize response (can be single dict or list)
            if isinstance(similar_artists, dict):
                return [similar_artists]

            return similar_artists

        except (TimeoutError, ClientError, InvalidDataError, KeyError) as err:
            # Don't log full error to avoid exposing API key in URLs
            self.logger.debug(
                "Failed to get similar artists for %s: %s", artist_name, type(err).__name__
            )
            return []

    async def get_similar_tracks(
        self,
        artist_name: str,
        track_name: str,
        track_mbid: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Get similar tracks from Last.fm.

        :param artist_name: Name of the track's artist.
        :param track_name: Name of the track.
        :param track_mbid: Optional MusicBrainz ID for more accurate matching.
        :param limit: Maximum number of similar tracks to return.
        """
        params: dict[str, Any] = {"limit": limit}

        # Prefer MBID if available for more accurate results
        if track_mbid:
            params["mbid"] = track_mbid
        else:
            params["artist"] = artist_name
            params["track"] = track_name

        try:
            data = await self._get_data("track.getSimilar", **params)
            similar_tracks: list[dict[str, Any]] | dict[str, Any] = data.get(
                "similartracks", {}
            ).get("track", [])

            # Normalize response (can be single dict or list)
            if isinstance(similar_tracks, dict):
                return [similar_tracks]

            return similar_tracks

        except (TimeoutError, ClientError, InvalidDataError, KeyError) as err:
            # Don't log full error to avoid exposing API key in URLs
            self.logger.debug(
                "Failed to get similar tracks for %s - %s: %s",
                artist_name,
                track_name,
                type(err).__name__,
            )
            return []

    async def get_chart_top_artists(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get global top artists chart from Last.fm.

        :param limit: Maximum number of artists to return.
        """
        try:
            data = await self._get_data("chart.getTopArtists", limit=limit)
            artists: list[dict[str, Any]] | dict[str, Any] = data.get("artists", {}).get(
                "artist", []
            )

            # Normalize response
            if isinstance(artists, dict):
                return [artists]

            return artists

        except (TimeoutError, ClientError, InvalidDataError, KeyError) as err:
            # Don't log full error to avoid exposing API key in URLs
            self.logger.warning("Failed to get top artists chart: %s", type(err).__name__)
            return []

    async def get_chart_top_tracks(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get global top tracks chart from Last.fm.

        :param limit: Maximum number of tracks to return.
        """
        try:
            data = await self._get_data("chart.getTopTracks", limit=limit)
            tracks: list[dict[str, Any]] | dict[str, Any] = data.get("tracks", {}).get("track", [])

            # Normalize response
            if isinstance(tracks, dict):
                return [tracks]

            return tracks

        except (TimeoutError, ClientError, InvalidDataError, KeyError) as err:
            # Don't log full error to avoid exposing API key in URLs
            self.logger.warning("Failed to get top tracks chart: %s", type(err).__name__)
            return []
