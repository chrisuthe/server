"""Pandora music provider for Music Assistant."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import aiohttp
from music_assistant_models.config_entries import (
    ConfigActionResult,
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
    ImageType,
    MediaType,
    StreamType,
)
from music_assistant_models.errors import (
    InvalidDataError,
    LoginFailed,
    MediaNotFoundError,
    ProviderUnavailableError,
    ResourceTemporarilyUnavailable,
)
from music_assistant_models.media_items import (
    Album,
    Artist,
    AudioFormat,
    BrowseFolder,
    ItemMapping,
    MediaItemImage,
    MediaItemType,
    Playlist,
    ProviderMapping,
    SearchResults,
    Track,
)
from music_assistant_models.streamdetails import StreamDetails
from music_assistant_models.unique_list import UniqueList

from music_assistant.constants import (
    CONF_ENTRY_UNOFFICIAL_PROVIDER,
    CONF_PASSWORD,
    CONF_SOCKS_URL,
    CONF_USERNAME,
)
from music_assistant.helpers.aiohttp_client import create_clientsession, get_socks5_url
from music_assistant.helpers.util import parse_title_and_version
from music_assistant.models.music_provider import MusicProvider

from .constants import (
    ACCOUNT_FLAG_HIGH_QUALITY,
    ACCOUNT_FLAG_ON_DEMAND,
    CATALOG_ANNOTATE_ENDPOINT,
    CONF_QUALITY,
    CONF_TAKEOVER_ACTION,
    LOGIN_ENDPOINT,
    PLAYBACK_RESUMED_ENDPOINT,
    PLAYLIST_FRAGMENT_ENDPOINT,
    QUALITY_HIGH,
    QUALITY_STANDARD,
    RETRY_REASON_AUTH,
    RETRY_REASON_STREAM_VIOLATION,
    STATIONS_ENDPOINT,
)
from .fragments import PandoraFragment, PandoraStationSession, should_fetch_fragment
from .helpers import (
    create_auth_headers,
    get_csrf_token,
    handle_pandora_error,
    read_account_flags,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


class StreamViolationError(InvalidDataError):
    """Error raised when Pandora detects concurrent streaming on multiple devices."""


class PandoraProvider(MusicProvider):
    """Pandora Music Provider."""

    _auth_token: str | None = None
    _user_id: str | None = None
    _csrf_token: str | None = None
    _sessions: dict[str, PandoraStationSession]
    _socks_proxy: bool = False
    _high_quality_available: bool = False
    _on_demand_available: bool = False

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return Config entries to configure this provider."""
        return (
            CONF_ENTRY_UNOFFICIAL_PROVIDER,
            ConfigEntry(
                key=CONF_QUALITY,
                type=ConfigEntryType.STRING,
                required=True,
                default_value=QUALITY_STANDARD,
                options=[
                    ConfigValueOption(QUALITY_STANDARD),
                    ConfigValueOption(QUALITY_HIGH),
                ],
            ),
            ConfigEntry(
                key=CONF_SOCKS_URL,
                type=ConfigEntryType.STRING,
                required=False,
                default_value="",
                advanced=True,
            ),
            ConfigEntry(
                key=CONF_TAKEOVER_ACTION,
                type=ConfigEntryType.ACTION,
                action=CONF_TAKEOVER_ACTION,
                required=False,
            ),
        )

    async def handle_config_action(
        self, action: str
    ) -> tuple[ConfigEntry, ...] | ConfigActionResult | None:
        """Handle a one-shot config action button press."""
        if action == CONF_TAKEOVER_ACTION:
            await self.takeover_stream()
            return None
        return await super().handle_config_action(action)

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._sessions = {}

        # Authenticate with Pandora
        username = str(self.get_setup_value(CONF_USERNAME) or "")
        password = str(self.get_setup_value(CONF_PASSWORD) or "")
        if not username.strip() or not password.strip():
            raise LoginFailed("Username and password are required")
        socks_url = get_socks5_url(str(self.config.get_value(CONF_SOCKS_URL)))

        if socks_url:
            self.http_session = create_clientsession(
                self.mass, verify_ssl=True, socks_url=socks_url
            )
            self._socks_proxy = True
        else:
            self.http_session = self.mass.http_session
        await self._authenticate(username, password)

    async def unload(self, is_removed: bool = False) -> None:
        """Handle unload/close of the provider."""
        await self.close()
        await super().unload(is_removed)

    async def close(self) -> None:
        """Handle closing of http session if using socks."""
        if self._socks_proxy and self.http_session:
            await self.http_session.close()

    async def browse(self, path: str) -> Sequence[MediaItemType | ItemMapping | BrowseFolder]:
        """Browse the user's Pandora stations."""
        sub_path = path.split("://", 1)[1] if "://" in path else ""
        if sub_path:
            return await super().browse(path)
        return [station async for station in self._get_stations()]

    async def search(
        self,
        search_query: str,
        media_types: list[MediaType],
        limit: int = 25,
    ) -> SearchResults:
        """Search the user's stations by name."""
        # search is limited to the user's own stations: the API's catalogue search
        # requires the legacy endpoints this provider does not speak
        if MediaType.PLAYLIST not in media_types:
            return SearchResults()
        # substring rather than compare_strings: that helper answers "are these the same
        # entity", and its fuzzy mode rejects a length difference over four characters, so a
        # short query like "rock" could never reach a station called "Classic Rock Radio"
        query = search_query.lower()
        results: list[Playlist] = []
        async for station in self._get_stations():
            if query in station.name.lower():
                results.append(station)
                if len(results) >= limit:
                    break
        return SearchResults(playlists=results)

    async def get_library_playlists(self) -> AsyncGenerator[Playlist]:
        """Retrieve the user's stations as dynamic playlists."""
        async for station in self._get_stations():
            yield station

    async def get_playlist(self, prov_playlist_id: str) -> Playlist:
        """Get full station details by id."""
        async for station in self._get_stations():
            if station.item_id == prov_playlist_id:
                return station
        raise MediaNotFoundError(f"Station {prov_playlist_id} not found")

    async def get_playlist_tracks(self, prov_playlist_id: str, page: int = 0) -> list[Track]:
        """
        Get the currently playable tracks for the given station.

        :param prov_playlist_id: The Pandora station id.
        :param page: Paging index; a station serves a single batch, so anything beyond the
            first page terminates the caller's paging loop.
        """
        if page > 0:
            return []
        session = self._get_or_create_session(prov_playlist_id)
        fragment = session.current
        if fragment is None or should_fetch_fragment(fragment, time.time()):
            fragment = await self._fetch_fragment(session)
        # always serve the live fragment: an empty list would read as "this station has
        # ended" to the queue controller, which stops playback instead of continuing it.
        # Already-served tracks are withheld: the queue controller only de-duplicates refill
        # candidates against its unplayed tail, so a served track that scrolls out of that
        # tail would otherwise be re-added here and then fail once the fragment has moved on.
        return [self._parse_track(track, fragment.annotations) for track in fragment.pending]

    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        if (found := self._find_track_with_fragment(prov_track_id)) is None:
            raise MediaNotFoundError(f"Track {prov_track_id} not found")
        track, fragment = found
        return self._parse_track(track, fragment.annotations)

    async def get_album(self, prov_album_id: str) -> Album:
        """
        Get an album by id.

        A catalogue album is looked up directly. A station track's album has no catalogue
        identity, so it is addressed by the id of one of its tracks - see `_parse_album`.
        """
        if prov_album_id.startswith("AL:"):
            record = self._find_annotation(prov_album_id) or await self._annotate(prov_album_id)
            return self._parse_album_record(record, prov_album_id)
        if (found := self._find_track_with_fragment(prov_album_id)) and (
            album := self._parse_album(found[0], prov_album_id, {})
        ):
            return album
        raise MediaNotFoundError(f"Album {prov_album_id} not found")

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """
        Get an artist by id.

        A catalogue artist is looked up directly; a station artist is identified by name.
        """
        if prov_artist_id.startswith("AR:"):
            record = self._find_annotation(prov_artist_id) or await self._annotate(prov_artist_id)
            return self._parse_artist(str(record.get("name") or prov_artist_id), prov_artist_id)
        return self._parse_artist(prov_artist_id)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a station track."""
        if media_type != MediaType.TRACK:
            raise MediaNotFoundError(f"Unsupported media type: {media_type}")
        now = time.time()
        # only each session's live fragment: an older one's signed URL may already be expired
        # and there is no way to tell from here, so refuse rather than hand ffmpeg a link
        # that 403s mid-track
        holders = [
            (fragment, track)
            for session in self._sessions.values()
            if (fragment := session.current) is not None
            and (track := fragment.find(item_id)) is not None
        ]
        playable = [holder for holder in holders if not holder[0].urls_expired(now)]
        if not playable:
            if holders:
                # the signed URLs have outlived their TTL, which is what a long pause looks
                # like from here. Refusing keeps the failure named rather than an opaque
                # ffmpeg error. Note this asks a different question from is_stale: a fragment
                # can be idle long enough to be worth replacing while its URLs are still
                # perfectly playable, and refusing those would break resuming after a pause.
                raise MediaNotFoundError(f"Track {item_id} expired while playback was stopped")
            raise MediaNotFoundError(f"Track {item_id} is no longer available from Pandora")
        # stations overlap, so the same song can sit in several sessions at once. Serve the
        # freshest copy rather than whichever session was created first: an older station's
        # expired fragment must not fail a playable track, and the fragment that is marked
        # as having served the track has to be the one the audio URL came from.
        fragment, track = max(playable, key=lambda holder: holder[0].fetched_at)
        fragment.mark_resolved(item_id, now)
        duration = int(track.get("trackLength") or 0)
        can_seek = duration > 0
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=self._audio_format(),
            media_type=MediaType.TRACK,
            stream_type=StreamType.HTTP,
            path=track["audioURL"],
            duration=duration,
            can_seek=can_seek,
            allow_seek=can_seek,
        )

    async def takeover_stream(self) -> None:
        """
        Force Pandora to end any other active session and resume here.

        This sends "forceActive=true" to the playbackResumed endpoint, which instructs Pandora to
        terminate any conflicting stream on other devices. The user must manually restart playback
        in MA after clicking the config button that triggers this call.
        """
        self.logger.debug("Sending playbackResumed request to Pandora to attempt stream takeover.")
        await self._api_request(
            "POST",
            PLAYBACK_RESUMED_ENDPOINT,
            data={"forceActive": True},
            # This is called as part of handling a STREAM_VIOLATION 429, so mark that reason as
            # already exhausted to prevent _api_request from retrying on another 429.
            exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
        )

    async def _authenticate(self, username: str, password: str) -> None:
        """Authenticate with Pandora and get auth token."""
        try:
            self._csrf_token = await get_csrf_token(self.http_session)

            login_data = {
                "username": username,
                "password": password,
                "keepLoggedIn": True,
                "existingAuthToken": None,
            }

            headers = create_auth_headers(self._csrf_token)

            async with self.http_session.post(
                LOGIN_ENDPOINT,
                headers=headers,
                json=login_data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    await self.close()
                    raise LoginFailed(f"Login request failed with status {response.status}")

                response_data = await response.json()
                handle_pandora_error(response_data)

                self._auth_token = response_data.get("authToken")
                if not self._auth_token:
                    await self.close()
                    raise LoginFailed("No auth token received from Pandora")

                self._user_id = response_data.get("listenerId")

                # What this account is entitled to. Pandora sends config and flags as null
                # on some accounts, so read through them rather than guarding after the fact.
                flags = read_account_flags(response_data)
                self._high_quality_available = ACCOUNT_FLAG_HIGH_QUALITY in flags
                self._on_demand_available = ACCOUNT_FLAG_ON_DEMAND in flags

                self.logger.info(
                    "Successfully authenticated with Pandora "
                    "(high-quality streaming: %s, on-demand: %s)",
                    self._high_quality_available,
                    self._on_demand_available,
                )

        except aiohttp.ClientError as err:
            await self.close()
            self.logger.exception("Network error during authentication")
            raise ProviderUnavailableError(
                "Unable to connect to Pandora for authentication"
            ) from err

    async def _api_request(
        self,
        method: str,
        url: str,
        data: dict[str, Any] | None = None,
        exhausted_retry_reasons: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        """
        Make an API request to Pandora.

        :param method: HTTP method (GET, POST, etc.)
        :param url: API endpoint URL
        :param data: Optional JSON data to send
        :param exhausted_retry_reasons: Set of retry reasons already attempted for this request.
            Pass a pre-populated set to prevent specific retry strategies from being attempted.
        """
        if not self._csrf_token or not self._auth_token:
            await self.close()
            raise LoginFailed("Not authenticated with Pandora")

        headers = create_auth_headers(self._csrf_token, self._auth_token)

        try:
            async with self.http_session.request(
                method, url, json=data, headers=headers
            ) as response:
                # Check status BEFORE parsing JSON
                if response.status == 401:
                    if RETRY_REASON_AUTH not in exhausted_retry_reasons:
                        # Auth token expired, re-authenticate and retry once
                        username = str(self.get_setup_value(CONF_USERNAME) or "")
                        password = str(self.get_setup_value(CONF_PASSWORD) or "")
                        await self._authenticate(username, password)
                        return await self._api_request(
                            method,
                            url,
                            data,
                            exhausted_retry_reasons=exhausted_retry_reasons | {RETRY_REASON_AUTH},
                        )
                    await self.close()
                    raise LoginFailed("Pandora authentication failed after retry")
                if response.status == 404:
                    await self.close()
                    raise MediaNotFoundError("Resource not found")
                if response.status == 429:
                    # Another device may already be streaming on this account.
                    # Parse the body to confirm it is a STREAM_VIOLATION.
                    try:
                        error_body: dict[str, Any] = await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError) as err:
                        raise InvalidDataError(
                            "Unable to parse error 429 response body from Pandora"
                        ) from err
                    if error_body.get("errorString") == "STREAM_VIOLATION":
                        if RETRY_REASON_STREAM_VIOLATION not in exhausted_retry_reasons:
                            self.logger.warning(
                                "Pandora stream is already active on another device. "
                                "Automatically taking over the stream and retrying the request."
                            )
                            await self.takeover_stream()
                            return await self._api_request(
                                method,
                                url,
                                data,
                                exhausted_retry_reasons=exhausted_retry_reasons
                                | {RETRY_REASON_STREAM_VIOLATION},
                            )
                        raise StreamViolationError("STREAM_VIOLATION")
                    # This is some other, not concurrent streaming error kind of 429
                    raise ProviderUnavailableError(f"Pandora rate-limited (HTTP 429): {error_body}")
                if response.status >= 500:
                    await self.close()
                    raise ProviderUnavailableError("Pandora server error")
                if response.status >= 400:
                    await self.close()
                    raise InvalidDataError(f"Pandora API error: HTTP {response.status}")

                result: dict[str, Any] = await response.json()
                handle_pandora_error(result)
                return result

        except aiohttp.ClientError as err:
            await self.close()
            raise ProviderUnavailableError("Unable to connect to Pandora") from err
        except (ValueError, KeyError) as err:
            await self.close()
            raise InvalidDataError("Invalid response from Pandora") from err

    async def _fetch_fragment(self, session: PandoraStationSession) -> PandoraFragment:
        """Fetch the next fragment for a station and retain it as the live one."""
        is_station_start = not session.fragments
        try:
            result: dict[str, Any] = await self._api_request(
                "POST",
                PLAYLIST_FRAGMENT_ENDPOINT,
                data={
                    "stationId": session.station_id,
                    "isStationStart": is_station_start,
                    "fragmentRequestReason": "Normal",
                    "audioFormat": "mp3-hifi" if self._use_high_quality() else "aacplus",
                    "startingAtTrackId": None,
                    "onDemandArtistMessageArtistUidHex": None,
                    "onDemandArtistMessageIdHex": None,
                },
                # Mark stream violation retry as already exhausted for non-initial fragments
                # this prevents us from fighting with the concurrent streaming limit
                # if the user starts a stream on a different device while MA is already playing.
                exhausted_retry_reasons=frozenset()
                if is_station_start
                else frozenset({RETRY_REASON_STREAM_VIOLATION}),
            )
        except MediaNotFoundError:
            await self.close()
            raise
        except StreamViolationError:
            self.logger.warning(
                "Pandora stream is already active on another device. "
                "To manually take over the stream on this device, use the "
                "'Take over stream' button on the provider configuration page.",
            )
            raise
        except InvalidDataError as err:
            self.logger.error("Invalid fragment data for station %s: %s", session.station_id, err)
            await self.close()
            raise
        tracks = [
            track
            for track in result.get("tracks", [])
            if track.get("audioURL")
            and track.get("pandoraId")
            and "curator message" not in (track.get("songTitle") or "").lower()
        ]
        if not tracks:
            # retaining an empty fragment would make it the live one, and nothing can ever
            # spend it — the station would serve nothing until the staleness window elapsed
            raise MediaNotFoundError(
                f"Pandora returned no playable tracks for {session.station_id}"
            )
        return session.add_fragment(tracks, time.time(), await self._hydrate(tracks))

    async def _hydrate(self, tracks: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Return Pandora's catalogue records for the given tracks, keyed by pandoraId.

        Empty for an account without on-demand entitlement: these records exist to make
        albums and artists addressable, and a listener who cannot play a catalogue track
        must not be offered one.

        :param tracks: Retained fragment tracks, each carrying a `pandoraId`.
        """
        if not self._on_demand_available:
            return {}
        try:
            response = await self._api_request(
                "POST",
                CATALOG_ANNOTATE_ENDPOINT,
                data={
                    "pandoraIds": [track["pandoraId"] for track in tracks],
                    "annotateAlbumTracks": False,
                },
                # enrichment must never fight the concurrent-stream limit: taking the stream
                # over would stop playback on the user's other device as a side effect of
                # fetching metadata.
                exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
            )
        except (
            InvalidDataError,
            MediaNotFoundError,
            ProviderUnavailableError,
            ResourceTemporarilyUnavailable,
        ) as err:
            if self.http_session.closed:
                # several _api_request error paths close the transport before raising, and
                # for a socks user that is this provider's own session. Degrading past that
                # would report a healthy fragment fetch and leave the next call failing with
                # a bare "Session is closed" RuntimeError from somewhere unrelated, so only
                # degrade while the connection is still usable.
                raise
            # enrichment only: without it a track keeps its name, art and audio, so a
            # station must keep playing rather than fail on a metadata call. The cost is
            # that this fragment's songs fall back to track-scoped album and name-keyed
            # artist identities, so they can pick up a second album or artist row alongside
            # the catalogue ones - accepted, because serving no album at all is worse. Auth
            # errors are deliberately NOT caught here - _api_request already re-authenticates
            # once on a 401, so anything still raising past that must surface rather than
            # degrade into a silently unhydrated station.
            self.logger.warning("Could not annotate Pandora fragment tracks: %s", err)
            return {}
        return {key: value for key, value in response.items() if isinstance(value, dict)}

    async def _annotate(self, pandora_id: str) -> dict[str, Any]:
        """
        Return Pandora's catalogue record for one id.

        :raises MediaNotFoundError: If Pandora holds no record for the id.
        """
        response = await self._api_request(
            "POST",
            CATALOG_ANNOTATE_ENDPOINT,
            data={"pandoraIds": [pandora_id], "annotateAlbumTracks": False},
            # as in _hydrate: a metadata lookup must not take the stream over and stop
            # playback on another device.
            exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
        )
        record = response.get(pandora_id)
        if not isinstance(record, dict):
            raise MediaNotFoundError(f"Pandora has no record for {pandora_id}")
        return record

    async def _get_stations(self) -> AsyncGenerator[Playlist]:
        """Retrieve the user's stations from the provider."""
        response = await self._api_request("POST", STATIONS_ENDPOINT, data={"pageSize": 250})
        for station in response.get("stations", []):
            yield self._parse_station(station)

    def _get_or_create_session(self, station_id: str) -> PandoraStationSession:
        """Get or create a station session, with LRU eviction if needed."""
        # Simple LRU: limit to 10 active sessions
        if station_id not in self._sessions and len(self._sessions) >= 10:
            oldest = min(self._sessions.values(), key=lambda session: session.last_accessed)
            self.logger.debug("Evicting session for station %s", oldest.station_id)
            del self._sessions[oldest.station_id]
        if station_id not in self._sessions:
            self._sessions[station_id] = PandoraStationSession(station_id)
        session = self._sessions[station_id]
        session.last_accessed = time.time()
        return session

    def _find_track_with_fragment(
        self, prov_track_id: str
    ) -> tuple[dict[str, Any], PandoraFragment] | None:
        """
        Return raw track data and the freshest retained fragment holding it, or None.

        The id no longer names a station, so every retained session is searched. At most ten
        sessions hold at most four fragments of about four tracks, so this stays small.
        Stations overlap, so the freshest fragment decides: its annotations are the most
        recent answer Pandora gave for the track, and picking by dict order instead would
        let the same song resolve to a different album from one lookup to the next.
        """
        holders = [
            (track, fragment)
            for session in self._sessions.values()
            for fragment in session.fragments
            if (track := fragment.find(prov_track_id)) is not None
        ]
        return max(holders, key=lambda holder: holder[1].fetched_at, default=None)

    def _find_annotation(self, pandora_id: str) -> dict[str, Any] | None:
        """
        Return a catalogue record a retained fragment already holds for the given id, or None.

        Hydration annotates a whole fragment in one call, albums and artists included, so the
        record an album or artist lookup wants is usually in hand already. Music Assistant
        resolves those per item, so refetching them here would put the provider back to one
        network call per album and per artist in a listing.
        """
        for session in self._sessions.values():
            for fragment in session.fragments:
                if pandora_id in fragment.annotations:
                    record: dict[str, Any] = fragment.annotations[pandora_id]
                    return record
        return None

    def _parse_station(self, station: dict[str, Any]) -> Playlist:
        """Parse a station object into a dynamic playlist."""
        playlist = Playlist(
            item_id=station["stationId"],
            provider=self.instance_id,
            name=station["name"],
            is_dynamic=True,
            provider_mappings={
                ProviderMapping(
                    item_id=station["stationId"],
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )
        if art := station.get("art"):
            art_url = next(
                (item.get("url") for item in art if item.get("size") == 500), art[-1].get("url")
            )
            if art_url:
                playlist.metadata.add_image(
                    MediaItemImage(
                        type=ImageType.THUMB,
                        path=art_url,
                        provider=self.instance_id,
                        remotely_accessible=True,
                    )
                )
        return playlist

    def _parse_track(self, obj: dict[str, Any], annotations: dict[str, Any]) -> Track:
        """
        Parse a raw fragment track into a Track.

        :param obj: One raw track from a Pandora fragment.
        :param annotations: Catalogue records keyed by pandoraId, empty when the account is
            not entitled to on-demand playback.
        """
        name, version = parse_title_and_version(obj.get("songTitle") or "Unknown Song")
        track_id = obj["pandoraId"]
        record = annotations.get(track_id) or {}
        track = Track(
            item_id=track_id,
            provider=self.instance_id,
            name=name,
            version=version,
            duration=int(obj.get("trackLength") or 0),
            provider_mappings={
                ProviderMapping(
                    item_id=track_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    audio_format=self._audio_format(),
                    url=obj.get("songDetailURL"),
                )
            },
        )
        if album_art := obj.get("albumArt"):
            art_url = next(
                (art.get("url") for art in album_art if art.get("size") == 500),
                album_art[-1].get("url"),
            )
            if art_url:
                track.metadata.add_image(
                    MediaItemImage(
                        provider=self.instance_id,
                        type=ImageType.THUMB,
                        path=art_url,
                        remotely_accessible=True,
                    )
                )
        if artist_name := obj.get("artistName"):
            track.artists = UniqueList([self._parse_artist(artist_name, record.get("artistId"))])
        track.album = self._parse_album(obj, track_id, record)
        return track

    def _parse_album(
        self, obj: dict[str, Any], track_id: str, record: dict[str, Any]
    ) -> Album | None:
        """
        Parse the album a fragment track belongs to, if the API named one.

        A hydrated track names its album in Pandora's catalogue, which is the id that album
        carries everywhere else. A fragment on its own names no album at all, so the track's
        own id stands in - the two cannot be confused, since they carry different prefixes.
        """
        if not (url := obj.get("albumDetailURL")):
            return None
        album_id = str(record.get("albumId") or track_id)
        name, version = parse_title_and_version(obj.get("albumTitle") or "Unknown Album")
        return Album(
            item_id=album_id,
            provider=self.instance_id,
            name=name,
            version=version,
            provider_mappings={
                ProviderMapping(
                    item_id=album_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    url=url,
                )
            },
        )

    def _parse_artist(self, name: str, artist_id: str | None = None) -> Artist:
        """
        Parse an artist.

        Without a catalogue id, a Pandora fragment identifies its artist by name only.
        """
        item_id = artist_id or name
        return Artist(
            item_id=item_id,
            name=name,
            provider=self.instance_id,
            provider_mappings={
                ProviderMapping(
                    item_id=item_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )

    def _parse_album_record(self, record: dict[str, Any], album_id: str) -> Album:
        """
        Parse an album from a Pandora catalogue record.

        :param record: The catalogue record Pandora returned for the album.
        :param album_id: The id the album was requested by, which it keeps.
        """
        name, version = parse_title_and_version(str(record.get("name") or "Unknown Album"))
        return Album(
            item_id=album_id,
            provider=self.instance_id,
            name=name,
            version=version,
            provider_mappings={
                ProviderMapping(
                    item_id=album_id,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                )
            },
        )

    def _audio_format(self) -> AudioFormat:
        """Return the audio format the fragments are requested in."""
        return AudioFormat(
            content_type=ContentType.MP3 if self._use_high_quality() else ContentType.AAC
        )

    def _use_high_quality(self) -> bool:
        """
        Whether high quality audio should be requested from Pandora.

        This allows a graceful fallback to standard quality if the account is not eligible for
        high-quality streaming, while still respecting the user's preference if they are eligible.
        """
        return self._high_quality_available and self.config.get_value(CONF_QUALITY) == QUALITY_HIGH
