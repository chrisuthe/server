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
    MusicAssistantError,
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
    Playlist,
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
    ADD_SEED_ENDPOINT,
    CATALOG_ANNOTATE_ENDPOINT,
    CONF_ALLOW_STATION_DELETE,
    CONF_DEVICE_UUID,
    CONF_QUALITY,
    CONF_TAKEOVER_ACTION,
    CREATE_STATION_ENDPOINT,
    LOGIN_ENDPOINT,
    PLAYBACK_RESUMED_ENDPOINT,
    PLAYBACK_SOURCE_ENDPOINT,
    PLAYLIST_FRAGMENT_ENDPOINT,
    QUALITY_HIGH,
    QUALITY_STANDARD,
    REMOVE_STATION_ENDPOINT,
    RETRY_REASON_AUTH,
    RETRY_REASON_STREAM_VIOLATION,
    SEED_SEARCH_ENDPOINT,
    SEEDABLE_PREFIXES,
    SOD_SEARCH_ENDPOINT,
    STATIONS_ENDPOINT,
)
from .fragments import PandoraFragment, PandoraStationSession, should_fetch_fragment
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
    _user_id: str | None = None
    _csrf_token: str | None = None
    _sessions: dict[str, PandoraStationSession]
    _socks_proxy: bool = False
    _high_quality_available: bool = False
    _on_demand_available: bool = False
    _device_uuid: str

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
            ConfigEntry(
                key=CONF_ALLOW_STATION_DELETE,
                type=ConfigEntryType.BOOLEAN,
                required=False,
                default_value=False,
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

        # A stable per-install identity for playback/source, generated once and reused so
        # every restart does not look like a new device to a single-stream account.
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
        """
        Search the user's stations and, for an entitled account, Pandora's catalogue.

        Stations answer `MediaType.PLAYLIST`. Tracks and albums come from the catalogue and
        are offered only to an account that may play them on demand. Artists are not
        answered at all yet: a search result has to lead somewhere, and this provider has no
        artist page. Any other media type comes back empty.
        """
        stations = (
            await self._search_stations(search_query, limit)
            if MediaType.PLAYLIST in media_types
            else []
        )
        types = [
            prefix
            for media_type, prefix in ((MediaType.TRACK, "TR"), (MediaType.ALBUM, "AL"))
            if media_type in media_types
        ]
        if not types or not self._on_demand_available:
            return SearchResults(playlists=stations)
        tracks, albums = await self._search_catalogue(search_query, types, limit)
        return SearchResults(playlists=stations, tracks=tracks, albums=albums)

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
        return [parse_track(self, track, fragment.annotations) for track in fragment.pending]

    async def create_playlist(self, name: str, media_types: set[MediaType]) -> Playlist:
        """
        Create a Pandora station seeded from the given name.

        MA passes only a name, so the name doubles as the seed query - which is how
        Pandora's own UI creates stations. The station Pandora returns will be named
        after the seed it picked ("Radiohead Radio"), not after `name`.
        """
        seed_id = await self._resolve_seed(name)
        response = await self._api_request(
            "POST",
            CREATE_STATION_ENDPOINT,
            data={"pandoraId": seed_id, "stationName": name},
        )
        return parse_station(self, response)

    async def library_remove(self, prov_item_id: str, media_type: MediaType) -> bool:
        """
        Delete a station from the user's Pandora account, if the user opted in.

        Pandora has no follow/unfollow distinction - a station in getStations is the
        library - so removing one from the library deletes it outright, permanently.
        Measured: getStationDetails afterwards returns STATION_DOES_NOT_EXIST, with no
        tombstone and no way back. Library sync-back is on by default, so this stays
        behind an explicit opt-in rather than firing on a routine library edit.
        """
        if media_type != MediaType.PLAYLIST:
            return False
        if not self.config.get_value(CONF_ALLOW_STATION_DELETE):
            self.logger.info(
                "Leaving Pandora station %s in place: deleting it is permanent and "
                "'Allow deleting stations' is turned off for this provider.",
                prov_item_id,
            )
            return False
        await self._api_request("POST", REMOVE_STATION_ENDPOINT, data={"stationId": prov_item_id})
        return True

    async def library_add(self, item: MediaItemType) -> bool:
        """
        Accept a station that is already part of the account's library.

        Pandora exposes only the account's own stations, so anything Music Assistant can
        ask us to add is already there - there is no separate saved state to set.
        """
        return item.media_type == MediaType.PLAYLIST

    async def add_playlist_tracks(self, prov_playlist_id: str, prov_track_ids: list[str]) -> None:
        """
        Seed a station with the given tracks.

        A Pandora station holds seeds, not tracks, so adding a track adds a seed derived
        from it - Pandora's own "add variety". The track will not appear in the station's
        listing; it shifts what the station plays.
        """
        for prov_track_id in prov_track_ids:
            await self._api_request(
                "POST",
                ADD_SEED_ENDPOINT,
                data={"stationId": prov_playlist_id, "pandoraId": prov_track_id},
            )

    async def remove_playlist_tracks(
        self, prov_playlist_id: str, positions_to_remove: tuple[int, ...]
    ) -> None:
        """
        Refuse positional track removal, which a station cannot express.

        MA addresses removal by position in the playlist's track list. For a station that
        list is the live fragment, whose entries expire within minutes and have no
        correspondence to the station's seeds - so there is no position to delete.
        """
        raise MusicAssistantError(
            "Pandora stations hold seeds, not tracks - remove seeds in the Pandora app"
        )

    async def get_track(self, prov_track_id: str) -> Track:
        """
        Get full track details by id.

        A track a retained fragment still holds is described from that fragment, at no cost.
        Anything else is a catalogue track and is looked up directly - which is how a track
        found by search resolves at all.
        """
        if (found := self._find_track_with_fragment(prov_track_id)) is not None:
            track, fragment = found
            return parse_track(self, track, fragment.annotations)
        record = self._find_annotation(prov_track_id) or await self._annotate(prov_track_id)
        return parse_track_record(self, record, prov_track_id)

    async def get_album(self, prov_album_id: str) -> Album:
        """
        Get an album by id.

        A catalogue album is looked up directly. A station track's album has no catalogue
        identity, so it is addressed by the id of one of its tracks - see `parse_album`.
        """
        if prov_album_id.startswith("AL:"):
            record = self._find_annotation(prov_album_id) or await self._annotate(prov_album_id)
            return parse_album_record(self, record, prov_album_id)
        if (found := self._find_track_with_fragment(prov_album_id)) and (
            album := parse_album(self, found[0], prov_album_id)
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
            return parse_artist_record(self, record, prov_artist_id)
        return parse_artist(self, prov_artist_id)

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Get streamdetails for a Pandora track.

        A track a live fragment still holds streams from the URL that fragment already
        carries, at no cost. Anything else is minted per play, which needs the account's
        on-demand entitlement and spends an interactive play: a station track that has aged
        out of the retained fragments is playable again on an entitled account, where it
        used to raise.

        :raises MediaNotFoundError: If neither route can produce a URL for the track.
        """
        if media_type != MediaType.TRACK:
            raise MediaNotFoundError(f"Unsupported media type: {media_type}")
        now = time.time()
        # only each session's live fragment: an older one's signed URL may already be expired
        # and there is no way to tell from here, so it has to be re-minted rather than handed
        # to ffmpeg to 403 mid-track
        holders = [
            (fragment, track)
            for session in self._sessions.values()
            if (fragment := session.current) is not None
            and (track := fragment.find(item_id)) is not None
        ]
        if playable := [holder for holder in holders if not holder[0].urls_expired(now)]:
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
        if not self._on_demand_available:
            # answer here rather than let Pandora answer NO_ENTITLEMENTS: the refusal is
            # certain, and it costs neither a round trip nor an ad-free play the account does
            # not have. A retained holder means the signed URLs outlived their TTL, which is
            # what a long pause looks like from here - a different question from is_stale, as
            # a fragment can be idle long enough to be worth replacing while its URLs still
            # play perfectly well.
            if holders:
                raise MediaNotFoundError(f"Track {item_id} expired while playback was stopped")
            raise MediaNotFoundError("On-demand playback is not available on this Pandora account")
        return await self._mint_stream_details(item_id)

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
                    "(high-quality streaming available: %s, on-demand playback available: %s)",
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
                    # A free/non-Premium account gets a 400 for on-demand track requests;
                    # a per-track refusal must not tear down the session.
                    await raise_if_playback_refused(response)
                    await self.close()
                    raise InvalidDataError(f"Pandora API error: HTTP {response.status}")
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

        Gated on the account's on-demand entitlement: Music Assistant persists library rows,
        so an `AL:`/`AR:` id minted while an account was entitled can still be looked up after
        the subscription lapses. Without this gate that would fire a live catalogue call and
        return a navigable album or artist whose tracks cannot play - exactly what the
        entitlement check exists to prevent.

        :raises MediaNotFoundError: If the account is not entitled to on-demand playback, or
            Pandora holds no record for the id.
        """
        if not self._on_demand_available:
            raise MediaNotFoundError("On-demand playback is not available on this Pandora account")
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

    async def _mint_stream_details(self, source_id: str) -> StreamDetails:
        """
        Mint a signed URL for one playable source and describe the stream it names.

        Called once per playback and never to describe an item in a listing: each call spends
        an interactive play, and the URL it returns dies within minutes.

        :param source_id: The Pandora id to play. This provider only ever passes a `TR:` track
            id, but the endpoint plays other kinds of source too.
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
        # what the listener may do with this play, from Pandora itself, rather than the
        # fragment path's guess that anything with a duration can be seeked
        can_seek = "SEEK" in (item.get("interactions") or [])
        # the mint is not asked for a format and names the one it made, so the account's
        # quality preference does not describe what arrives here
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

    async def _get_stations(self) -> AsyncGenerator[Playlist]:
        """Retrieve the user's stations from the provider."""
        response = await self._api_request("POST", STATIONS_ENDPOINT, data={"pageSize": 250})
        for station in response.get("stations", []):
            yield parse_station(self, station)

    async def _search_stations(self, search_query: str, limit: int) -> list[Playlist]:
        """Return the user's stations whose name contains the query, up to the limit."""
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
        return results

    async def _search_catalogue(
        self, search_query: str, types: list[str], limit: int
    ) -> tuple[list[Track], list[Album]]:
        """
        Search Pandora's catalogue for the given type prefixes.

        One call answers every requested type, and the records it returns describe the
        results' albums and artists too, so nothing here needs a second lookup. A track the
        account may not play interactively is dropped rather than offered as a result that
        fails on click.

        :param types: Type prefixes to search for, as Pandora spells them - `["TR", "AL"]`.
        """
        response = await self._api_request(
            "POST",
            SOD_SEARCH_ENDPOINT,
            data={"query": search_query, "types": types, "count": limit, "annotate": True},
        )
        annotations = response.get("annotations") or {}
        tracks: list[Track] = []
        albums: list[Album] = []
        for result_id in response.get("results") or []:
            if not isinstance(record := annotations.get(result_id), dict):
                continue
            rights = record.get("rightsInfo") or {}
            if result_id.startswith("TR:") and rights.get("hasInteractive"):
                tracks.append(parse_track_record(self, record, result_id, annotations))
            elif result_id.startswith("AL:"):
                albums.append(parse_album_record(self, record, result_id))
        return tracks, albums

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
        Return the catalogue record the freshest retained fragment holds for the given id.

        Hydration annotates a whole fragment in one call, albums and artists included, so the
        record an album or artist lookup wants is usually in hand already. Music Assistant
        resolves those per item, so refetching them here would put the provider back to one
        network call per album and per artist in a listing. Fragments overlap the same way
        tracks do, so the freshest one holding the id decides, consistent with
        `_find_track_with_fragment` above.
        """
        holders = [
            fragment
            for session in self._sessions.values()
            for fragment in session.fragments
            if pandora_id in fragment.annotations
        ]
        freshest = max(holders, key=lambda fragment: fragment.fetched_at, default=None)
        return freshest.annotations[pandora_id] if freshest is not None else None

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

    async def _resolve_seed(self, query: str) -> str:
        """
        Return the highest-ranked Pandora id that createStation accepts as a seed.

        :param query: Free-text station name, used verbatim as the search query.
        :raises MediaNotFoundError: If nothing Pandora returned can seed a station.
        """
        response = await self._api_request(
            "POST", SEED_SEARCH_ENDPOINT, data={"query": query, "count": 5}
        )
        for item in response.get("items") or []:
            if not (pandora_id := item.get("pandoraId")):
                continue
            if pandora_id.split(":", 1)[0] in SEEDABLE_PREFIXES:
                return str(pandora_id)
        raise MediaNotFoundError(f"No Pandora seed matches '{query}'")
