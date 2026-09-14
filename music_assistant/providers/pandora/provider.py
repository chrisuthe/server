"""Pandora music provider for Music Assistant."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiohttp
from music_assistant_models.config_entries import (
    ConfigActionResult,
    ConfigEntry,
    ConfigValueOption,
)
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
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
    MediaItemType,
    Radio,
    SearchResults,
    Track,
)
from music_assistant_models.streamdetails import StreamDetails

from music_assistant.constants import (
    CONF_ENTRY_UNOFFICIAL_PROVIDER,
    CONF_PASSWORD,
    CONF_SOCKS_URL,
    CONF_USERNAME,
)
from music_assistant.helpers.aiohttp_client import create_clientsession, get_socks5_url
from music_assistant.models.music_provider import MusicProvider

from .constants import (
    ACCOUNT_FLAG_HIGH_QUALITY,
    ACCOUNT_FLAG_ON_DEMAND,
    CATALOG_ANNOTATE_ENDPOINT,
    CATALOG_DETAILS_ENDPOINT,
    CONF_DEVICE_UUID,
    CONF_QUALITY,
    CONF_TAKEOVER_ACTION,
    LOGIN_ENDPOINT,
    NO_ON_DEMAND_MESSAGE,
    PLAYBACK_RESUMED_ENDPOINT,
    PLAYBACK_SOURCE_ENDPOINT,
    PLAYLIST_FRAGMENT_ENDPOINT,
    QUALITY_HIGH,
    QUALITY_STANDARD,
    RETRY_REASON_AUTH,
    RETRY_REASON_STREAM_VIOLATION,
    SOD_SEARCH_ENDPOINT,
    STATIONS_ENDPOINT,
)
from .fragments import (
    MAX_ACTIVE_SESSIONS,
    PandoraFragment,
    PandoraStationSession,
    should_fetch_fragment,
)
from .helpers import (
    create_auth_headers,
    get_csrf_token,
    handle_pandora_error,
    loudness_from_file_gain,
    raise_if_playback_refused,
    read_account_flags,
)
from .parsers import (
    parse_album,
    parse_album_record,
    parse_artist,
    parse_artist_record,
    parse_station,
    parse_track,
    parse_track_record,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence


class StreamViolationError(InvalidDataError):
    """Error raised when Pandora detects concurrent streaming on multiple devices."""


class PandoraProvider(MusicProvider):
    """Pandora Music Provider."""

    _auth_token: str | None = None
    _csrf_token: str | None = None
    _sessions: dict[str, PandoraStationSession]
    _socks_proxy: bool = False
    _high_quality_available: bool = False
    _on_demand_available: bool = False
    _device_uuid: str = ""

    @property
    def max_concurrent_streams(self) -> int:
        """Pandora enforces single-device streaming (stream violation on concurrent use)."""
        return 1

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

        if not (device_uuid := self.get_setup_value(CONF_DEVICE_UUID)):
            device_uuid = str(uuid4())
            self._update_setup_data(CONF_DEVICE_UUID, device_uuid)
        self._device_uuid = str(device_uuid)

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
        """Search the user's stations and, for an on-demand account, Pandora's catalogue."""
        query = search_query.strip()
        if not query:
            return SearchResults()
        stations = (
            await self._search_stations(query, limit) if MediaType.RADIO in media_types else []
        )
        types = [
            prefix
            for media_type, prefix in ((MediaType.TRACK, "TR"), (MediaType.ALBUM, "AL"))
            if media_type in media_types
        ]
        if not types or not self._on_demand_available:
            return SearchResults(radio=stations)
        tracks, albums = await self._search_catalogue(query, types, limit)
        return SearchResults(radio=stations, tracks=tracks, albums=albums)

    async def get_library_radios(self) -> AsyncGenerator[Radio]:
        """Retrieve the user's stations as dynamic radio stations."""
        async for station in self._get_stations():
            yield station

    async def get_radio(self, prov_radio_id: str) -> Radio:
        """Get full station details by id."""
        async for station in self._get_stations():
            if station.item_id == prov_radio_id:
                return station
        raise MediaNotFoundError(f"Station {prov_radio_id} not found")

    async def get_dynamic_radio_tracks(self, prov_radio_id: str) -> list[Track]:
        """
        Get the currently playable tracks for the given station.

        :param prov_radio_id: The Pandora station id.
        """
        session = self._get_or_create_session(prov_radio_id)
        fragment = session.current
        if fragment is None or should_fetch_fragment(fragment, time.time()):
            fragment = await self._fetch_fragment(session)
        # always serve the live fragment: an empty list would read as "this station has
        # ended" to the queue controller, which stops playback instead of continuing it.
        # Already-served tracks are withheld: the queue controller only de-duplicates refill
        # candidates against its unplayed tail, so a served track that scrolls out of that
        # tail would otherwise be re-added here and then fail once the fragment has moved on.
        return [parse_track(self, track, fragment.annotations) for track in fragment.pending]

    async def get_track(self, prov_track_id: str) -> Track:
        """Get full track details by id."""
        if (found := self._find_track_with_fragment(prov_track_id)) is not None:
            track, fragment = found
            return parse_track(self, track, fragment.annotations)
        records = self._find_annotations(prov_track_id) or await self._annotate_one(prov_track_id)
        return parse_track_record(self, records[prov_track_id], prov_track_id, records)

    async def get_album(self, prov_album_id: str) -> Album:
        """Get an album by its catalogue id, or by the id of a station track it holds."""
        if prov_album_id.startswith("AL:"):
            records = self._find_annotations(prov_album_id) or await self._annotate_one(
                prov_album_id
            )
            return parse_album_record(self, records[prov_album_id], prov_album_id, records)
        if (found := self._find_track_with_fragment(prov_album_id)) and (
            album := parse_album(self, found[0], prov_album_id)
        ):
            return album
        raise MediaNotFoundError(f"Album {prov_album_id} not found")

    async def get_album_tracks(self, prov_album_id: str) -> list[Track]:
        """
        Get an album's tracks, in the order Pandora lists them.

        :param prov_album_id: A catalogue album id, or the track id a station album is keyed by.
        :raises MediaNotFoundError: If the album is unknown, or the account may not play a
            catalogue album on demand.
        """
        if not prov_album_id.startswith("AL:"):
            if (found := self._find_track_with_fragment(prov_album_id)) is None:
                raise MediaNotFoundError(f"Album {prov_album_id} not found")
            track, fragment = found
            # a station album holds only the track it is keyed by
            return [
                parse_track(self, track, fragment.annotations, available=self._on_demand_available)
            ]
        if not self._on_demand_available:
            raise MediaNotFoundError(NO_ON_DEMAND_MESSAGE)
        response = await self._api_request(
            "POST",
            CATALOG_DETAILS_ENDPOINT,
            data={"pandoraId": prov_album_id},
            exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
        )
        annotations: dict[str, Any] = response.get("annotations") or {}
        if not isinstance(album := annotations.get(prov_album_id), dict):
            raise MediaNotFoundError(f"Pandora has no record for {prov_album_id}")
        track_ids = [str(track_id) for track_id in album.get("tracks") or []]
        missing = [
            track_id for track_id in track_ids if not isinstance(annotations.get(track_id), dict)
        ]
        if missing:
            annotations = {**annotations, **await self._annotate_ids(missing)}
        tracks: list[Track] = []
        for track_id in track_ids:
            record = annotations.get(track_id)
            if isinstance(record, dict) and (record.get("rightsInfo") or {}).get("hasInteractive"):
                tracks.append(parse_track_record(self, record, track_id, annotations))
        return tracks

    async def get_artist(self, prov_artist_id: str) -> Artist:
        """Get an artist by its catalogue id, or by name for a station artist."""
        if prov_artist_id.startswith("AR:"):
            records = self._find_annotations(prov_artist_id) or await self._annotate_one(
                prov_artist_id
            )
            return parse_artist_record(self, records[prov_artist_id], prov_artist_id)
        return parse_artist(self, prov_artist_id)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """Get streamdetails for a track, minting on demand when no live fragment holds it."""
        if media_type != MediaType.TRACK:
            raise MediaNotFoundError(f"Unsupported media type: {media_type}")
        now = time.time()
        # only each session's live fragment: an older one's signed URL may already be expired
        # and there is no way to tell from here, so an older one is never handed to ffmpeg to
        # 403 mid-track
        holders = [
            (fragment, track)
            for session in self._sessions.values()
            if (fragment := session.current) is not None
            and (track := fragment.find(item_id)) is not None
        ]
        if playable := [holder for holder in holders if not holder[0].urls_expired(now)]:
            # stations overlap, so the same song can sit in several sessions at once. Serve the
            # freshest copy, not the one from whichever session happens to be oldest: an older
            # station's expired fragment must not fail a playable track, and the fragment that is
            # marked as having served the track has to be the one the audio URL came from.
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
        if not self._on_demand_available:
            if holders:
                # the signed URLs have outlived their TTL, which is what a long pause looks
                # like from here. Refusing keeps the failure named rather than an opaque
                # ffmpeg error. Note this asks a different question from is_stale: a fragment
                # can be idle long enough to be worth replacing while its URLs are still
                # perfectly playable, and refusing those would break resuming after a pause.
                raise MediaNotFoundError(f"Track {item_id} expired while playback was stopped")
            raise MediaNotFoundError(NO_ON_DEMAND_MESSAGE)
        return await self._mint_stream_details(item_id)

    async def _mint_stream_details(self, source_id: str) -> StreamDetails:
        """
        Mint a signed URL for one playable source and describe the stream it names.

        :param source_id: The Pandora source id to play.
        :raises MediaNotFoundError: If Pandora will not play the source for this account.
        """
        response = await self._api_request(
            "POST",
            PLAYBACK_SOURCE_ENDPOINT,
            data={
                "sourceId": source_id,
                "includeItem": True,
                "includeSource": True,
                "deviceUuid": self._device_uuid,
            },
        )
        item = response.get("item") or {}
        if not (audio_url := item.get("audioUrl")):
            raise MediaNotFoundError(f"Pandora minted no audio URL for {source_id}")
        can_seek = "SEEK" in (item.get("interactions") or [])
        encoding = str(item.get("encoding") or "")
        return StreamDetails(
            provider=self.instance_id,
            item_id=source_id,
            audio_format=AudioFormat(
                content_type=ContentType.MP3 if encoding.startswith("mp3") else ContentType.AAC
            ),
            media_type=MediaType.TRACK,
            stream_type=StreamType.HTTP,
            path=str(audio_url),
            duration=int(item.get("duration") or 0),
            can_seek=can_seek,
            allow_seek=can_seek,
            loudness=loudness_from_file_gain(item.get("fileGain")),
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

                # What this account is entitled to. Pandora sends config and flags as null
                # on some accounts, so read through them rather than guarding after the fact.
                flags = read_account_flags(response_data)
                self._high_quality_available = ACCOUNT_FLAG_HIGH_QUALITY in flags
                self._on_demand_available = ACCOUNT_FLAG_ON_DEMAND in flags

                self.logger.info(
                    "Successfully authenticated with Pandora "
                    "(high-quality streaming available: %s, "
                    "on-demand playback available: %s)",
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
                if response.status == 400:
                    await raise_if_playback_refused(response)
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
        Return catalogue records for the given fragment tracks, keyed by pandoraId.

        Empty when the account is not entitled to on-demand playback or the lookup fails.
        """
        if not self._on_demand_available:
            return {}
        try:
            return await self._annotate_objects([track["pandoraId"] for track in tracks])
        except (
            InvalidDataError,
            MediaNotFoundError,
            ProviderUnavailableError,
            ResourceTemporarilyUnavailable,
        ) as err:
            # only degrade while the connection is still usable
            if self.http_session.closed:
                raise
            self.logger.warning("Could not annotate Pandora fragment tracks: %s", err)
            return {}

    async def _annotate_objects(self, pandora_ids: list[str]) -> dict[str, Any]:
        """
        Return catalogue records for the given ids in one request, keyed by pandoraId.

        Not gated on entitlement; `_annotate_ids` is.
        """
        response = await self._api_request(
            "POST",
            CATALOG_ANNOTATE_ENDPOINT,
            data={"pandoraIds": pandora_ids, "annotateAlbumTracks": False},
            # a metadata lookup never takes the stream over from another device
            exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
        )
        return {key: value for key, value in response.items() if isinstance(value, dict)}

    async def _annotate_ids(self, pandora_ids: list[str]) -> dict[str, Any]:
        """
        Return catalogue records for the given ids in one request, keyed by pandoraId.

        :raises MediaNotFoundError: If the account is not entitled to on-demand playback.
        """
        if not self._on_demand_available:
            raise MediaNotFoundError(NO_ON_DEMAND_MESSAGE)
        return await self._annotate_objects(pandora_ids)

    async def _annotate_one(self, pandora_id: str) -> dict[str, Any]:
        """
        Return the records Pandora answers one id with, siblings included, keyed by pandoraId.

        :raises MediaNotFoundError: If the account is not entitled to on-demand playback, or
            Pandora holds no record for the id.
        """
        records = await self._annotate_ids([pandora_id])
        if not isinstance(records.get(pandora_id), dict):
            raise MediaNotFoundError(f"Pandora has no record for {pandora_id}")
        return records

    async def _get_stations(self) -> AsyncGenerator[Radio]:
        """Retrieve the user's stations from the provider."""
        response = await self._api_request("POST", STATIONS_ENDPOINT, data={"pageSize": 250})
        for station in response.get("stations", []):
            yield parse_station(self, station)

    async def _search_stations(self, query: str, limit: int) -> list[Radio]:
        """Return the user's stations whose name contains the query."""
        # substring rather than compare_strings: that helper answers "are these the same
        # entity", and its fuzzy mode rejects a length difference over four characters, so a
        # short query like "rock" could never reach a station called "Classic Rock Radio"
        query = query.lower()
        results: list[Radio] = []
        async for station in self._get_stations():
            if query in station.name.lower():
                results.append(station)
                if len(results) >= limit:
                    break
        return results

    async def _search_catalogue(
        self, search_query: str, types: list[str], limit: int
    ) -> tuple[list[Track], list[Album]]:
        """
        Search Pandora's catalogue for the given type prefixes.

        :param types: Type prefixes to search for, as Pandora spells them - `["TR", "AL"]`.
        """
        response = await self._api_request(
            "POST",
            SOD_SEARCH_ENDPOINT,
            data={"query": search_query, "types": types, "count": limit, "annotate": True},
            exhausted_retry_reasons=frozenset({RETRY_REASON_STREAM_VIOLATION}),
        )
        annotations = response.get("annotations") or {}
        tracks: list[Track] = []
        albums: list[Album] = []
        for result_id in response.get("results") or []:
            if not isinstance(record := annotations.get(result_id), dict):
                continue
            rights = record.get("rightsInfo") or {}
            if "TR" in types and result_id.startswith("TR:") and rights.get("hasInteractive"):
                tracks.append(parse_track_record(self, record, result_id, annotations))
            elif "AL" in types and result_id.startswith("AL:"):
                albums.append(parse_album_record(self, record, result_id, annotations))
        return tracks, albums

    def _get_or_create_session(self, station_id: str) -> PandoraStationSession:
        """Get or create a station session, with LRU eviction if needed."""
        if station_id not in self._sessions and len(self._sessions) >= MAX_ACTIVE_SESSIONS:
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

        The id no longer names a station, so every retained session is searched. At most
        `MAX_ACTIVE_SESSIONS` sessions hold at most `MAX_RETAINED_FRAGMENTS` fragments of about
        four tracks each, so this stays small.
        Stations overlap, so the freshest fragment decides: it is the most recent answer
        Pandora gave for the track, and picking by dict order instead would let the same
        song resolve differently from one lookup to the next.
        """
        holders = [
            (track, fragment)
            for session in self._sessions.values()
            for fragment in session.fragments
            if (track := fragment.find(prov_track_id)) is not None
        ]
        return max(holders, key=lambda holder: holder[1].fetched_at, default=None)

    def _find_annotations(self, pandora_id: str) -> dict[str, Any] | None:
        """Return the annotations of the freshest retained fragment holding the id, or None."""
        holders = [
            fragment
            for session in self._sessions.values()
            for fragment in session.fragments
            if pandora_id in fragment.annotations
        ]
        freshest = max(holders, key=lambda fragment: fragment.fetched_at, default=None)
        return freshest.annotations if freshest is not None else None

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
