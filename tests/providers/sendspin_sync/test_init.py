"""Tests for the Sendspin Sync plugin provider entry point."""

from __future__ import annotations

import pathlib
from contextlib import aclosing
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import Scope
from music_assistant_models.enums import MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import (
    ActionUnavailable,
    MediaNotFoundError,
    ResourceBusyError,
)
from music_assistant_models.provider import ProviderManifest

from music_assistant.providers import sendspin_sync
from music_assistant.providers.sendspin_sync import (
    AUDIO_SOURCE_ID,
    SESSION_TIMEOUT_SECONDS,
    SUPPORTED_FEATURES,
    SendspinSyncProvider,
    setup,
)
from music_assistant.providers.sendspin_sync.chirp import (
    BIT_DEPTH,
    CHANNELS,
    SAMPLE_RATE,
    build_chirp_period,
)
from music_assistant.providers.sendspin_sync.session import CalibrationSession

MANIFEST_PATH = pathlib.Path(sendspin_sync.__file__).parent / "manifest.json"


async def test_setup_returns_the_plugin_provider() -> None:
    """setup() hands Music Assistant a SendspinSyncProvider instance."""
    manifest = MagicMock()
    config = MagicMock()
    # both are load-bearing: Provider.__init__ names its logger after the domain and
    # passes the log level through str(), which a bare MagicMock turns into a repr
    # that setLevel rejects
    manifest.domain = "sendspin_sync"
    config.get_value = MagicMock(return_value="GLOBAL")
    provider = await setup(MagicMock(), manifest, config)
    assert isinstance(provider, SendspinSyncProvider)


async def test_manifest_declares_its_sendspin_dependency() -> None:
    """The shipped manifest couples the plugin to the sendspin provider."""
    manifest = await ProviderManifest.parse(str(MANIFEST_PATH))
    assert manifest.domain == "sendspin_sync"
    assert manifest.depends_on == "sendspin"


def test_provider_declares_the_audio_source_feature() -> None:
    """The calibration track is offered through the AudioSource feature."""
    assert {ProviderFeature.AUDIO_SOURCE} == SUPPORTED_FEATURES


async def test_the_calibration_source_can_be_started_by_music_assistant() -> None:
    """The exposed AudioSource is playable on demand and offers no transport controls."""
    provider = await _setup_provider()
    (source,) = await provider.get_audio_sources()
    assert source.item_id == AUDIO_SOURCE_ID
    assert source.can_initiate
    assert source.exclusive
    assert not source.can_play_pause
    assert not source.can_seek
    assert not source.can_next_previous
    assert not source.allow_external_trigger


async def test_stream_details_describe_the_generated_pcm() -> None:
    """The declared format matches the bytes the generator actually emits."""
    provider = await _setup_provider()
    streamdetails = await provider.get_stream_details(AUDIO_SOURCE_ID, MediaType.AUDIO_SOURCE)
    assert streamdetails.stream_type == StreamType.CUSTOM
    assert streamdetails.audio_format.sample_rate == SAMPLE_RATE
    assert streamdetails.audio_format.bit_depth == BIT_DEPTH
    assert streamdetails.audio_format.channels == CHANNELS


async def test_stream_details_reject_an_unknown_source() -> None:
    """A request for an item this plugin does not own is a MediaNotFoundError."""
    provider = await _setup_provider()
    with pytest.raises(MediaNotFoundError):
        await provider.get_stream_details("nope", MediaType.AUDIO_SOURCE)


async def test_stream_details_claim_nothing() -> None:
    """Fetching stream details leaves ownership untouched, so queue preload is safe."""
    provider = await _setup_provider()
    await provider.get_stream_details(AUDIO_SOURCE_ID, MediaType.AUDIO_SOURCE)
    assert provider._in_use_by_queue is None
    assert provider._active_session_id is None


async def test_audio_stream_repeats_the_period_seamlessly() -> None:
    """Consecutive chunks concatenate back into whole periods, with no gap at the loop."""
    provider = await _setup_provider()
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "session")
    period = build_chirp_period()
    streamed = await _take_bytes(provider, len(period) * 2)
    assert streamed == period * 2


async def test_audio_stream_stops_when_another_session_supersedes_it() -> None:
    """A fresh claim on the same queue ends the previous session's generator."""
    provider = await _setup_provider()
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "session")
    async with aclosing(provider.get_audio_stream(MagicMock())) as stream:
        assert await anext(stream)
        await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "newer-session")
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
    # the superseded generator must not release the newer session's claim
    assert provider._in_use_by_queue == "queue"
    assert provider._active_session_id == "newer-session"


async def test_teardown_releases_the_source() -> None:
    """The matching unselect frees the source for another queue."""
    provider = await _setup_provider()
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "session")
    await provider.on_source_unselected(AUDIO_SOURCE_ID, "queue", "session")
    assert provider._in_use_by_queue is None
    assert provider._active_session_id is None


async def test_a_stale_teardown_leaves_the_live_claim_alone() -> None:
    """An unselect from a superseded same-queue request is ignored."""
    provider = await _setup_provider()
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "newer-session")
    await provider.on_source_unselected(AUDIO_SOURCE_ID, "queue", "older-session")
    assert provider._in_use_by_queue == "queue"
    assert provider._active_session_id == "newer-session"


async def test_source_hooks_ignore_another_plugins_source() -> None:
    """Lifecycle callbacks for a source this plugin does not own change nothing."""
    provider = await _setup_provider()
    await provider.on_source_selected("elsewhere", "player", "queue", "session")
    assert provider._in_use_by_queue is None

    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "queue", "session")
    await provider.on_source_unselected("elsewhere", "queue", "session")
    assert provider._in_use_by_queue == "queue"


async def test_api_commands_are_registered_and_released_on_unload() -> None:
    """Every session command is registered under the plugin namespace and given back."""
    provider = await _setup_provider()
    unregister_calls: list[str] = []

    def _register(command: str, _handler: object, **_kwargs: object) -> MagicMock:
        return MagicMock(side_effect=lambda: unregister_calls.append(command))

    mass = _mock_mass(provider)
    mass.register_api_command = MagicMock(side_effect=_register)
    await provider.loaded_in_mass()

    registered = [call.args[0] for call in mass.register_api_command.call_args_list]
    assert registered == [
        "sendspin_sync/eligible_players",
        "sendspin_sync/session",
        "sendspin_sync/start_session",
        "sendspin_sync/solo_player",
        "sendspin_sync/stop_session",
    ]
    scopes = [call.kwargs["required_scope"] for call in mass.register_api_command.call_args_list]
    assert scopes == [
        Scope.PLAYERS_READ,
        Scope.PLAYERS_READ,
        Scope.PLAYERS_CONTROL,
        Scope.PLAYERS_CONTROL,
        Scope.PLAYERS_CONTROL,
    ]

    await provider.unload()
    assert unregister_calls == registered


async def test_unload_stops_a_running_session() -> None:
    """Unloading the plugin does not leave a user's speakers muted and regrouped."""
    provider = await _setup_provider()
    session = MagicMock()
    session.stop = AsyncMock()
    provider._session = session

    await provider.unload()

    session.stop.assert_awaited_once()
    assert await provider.get_session() is None
    _mock_mass(provider).cancel_timer.assert_called_once()


async def test_no_session_reports_no_state() -> None:
    """The session command answers with None while nothing is running."""
    provider = await _setup_provider()
    assert await provider.get_session() is None


async def test_a_second_session_is_refused() -> None:
    """Only one calibration session can hold the speakers at a time."""
    provider = await _setup_provider()
    provider._session = MagicMock()

    with pytest.raises(ResourceBusyError):
        await provider.start_session(["player"])


async def test_solo_without_a_session_is_refused() -> None:
    """Soloing a speaker needs a session to solo it within."""
    provider = await _setup_provider()

    with pytest.raises(ActionUnavailable):
        await provider.solo_player("player")


async def test_stopping_without_a_session_is_a_no_op() -> None:
    """Stopping when nothing is running is not an error."""
    provider = await _setup_provider()
    await provider.stop_session()
    assert provider._session is None


async def test_the_stream_ending_externally_stops_the_session() -> None:
    """A session can not outlive the stream that carries its phase reference."""
    provider = await _setup_provider()
    session = MagicMock()
    session.queue_id = "anchor"
    session.stopped = False
    provider._session = session
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "anchor", "session")

    await provider.on_source_unselected(AUDIO_SOURCE_ID, "anchor", "session")

    create_task = _mock_mass(provider).create_task
    create_task.assert_called_once()
    # the stubbed create_task never runs the teardown it was handed
    create_task.call_args.args[0].close()


async def test_another_queues_teardown_leaves_the_session_alone() -> None:
    """An unselect from a queue that is not the session's anchor changes nothing."""
    provider = await _setup_provider()
    session = MagicMock()
    session.queue_id = "anchor"
    session.stopped = False
    provider._session = session
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "elsewhere", "session")

    await provider.on_source_unselected(AUDIO_SOURCE_ID, "elsewhere", "session")

    _mock_mass(provider).create_task.assert_not_called()


async def test_a_stopped_session_is_not_torn_down_again() -> None:
    """The unselect fired by the session's own stop does not re-enter that stop."""
    provider = await _setup_provider()
    session = MagicMock()
    session.queue_id = "anchor"
    session.stopped = True
    provider._session = session
    await provider.on_source_selected(AUDIO_SOURCE_ID, "player", "anchor", "session")

    await provider.on_source_unselected(AUDIO_SOURCE_ID, "anchor", "session")

    _mock_mass(provider).create_task.assert_not_called()


async def test_starting_a_session_arms_the_inactivity_timeout() -> None:
    """A phone that walks away can not leave a house muted, so the session is on a timer."""
    provider = await _setup_provider()
    mass = _mock_mass(provider)
    session = _stub_session()

    with patch.object(CalibrationSession, "create", AsyncMock(return_value=session)):
        await provider.start_session(["player"])

    mass.call_later.assert_called_once()
    assert mass.call_later.call_args.args[0] == SESSION_TIMEOUT_SECONDS
    assert mass.call_later.call_args.kwargs["task_id"] == provider._session_timeout_id
    session.begin.assert_awaited_once()


async def test_the_session_is_visible_before_its_stream_starts() -> None:
    """
    The teardown hook can only save a session it can already see.

    Starting the stream fires the source lifecycle hooks, so a session registered
    only after begin() returns would miss a stream that died on the way up and sit
    holding the speakers until the timeout.
    """
    provider = await _setup_provider()
    session = _stub_session()
    visible_during_begin: list[bool] = []
    session.begin = AsyncMock(
        side_effect=lambda _source: visible_during_begin.append(provider._session is session)
    )

    with patch.object(CalibrationSession, "create", AsyncMock(return_value=session)):
        await provider.start_session(["player"])

    assert visible_during_begin == [True]


async def test_each_solo_rearms_the_inactivity_timeout() -> None:
    """Driving the session keeps it alive; the timeout only fires once it is forgotten."""
    provider = await _setup_provider()
    mass = _mock_mass(provider)
    session = _stub_session()

    with patch.object(CalibrationSession, "create", AsyncMock(return_value=session)):
        await provider.start_session(["player"])
    await provider.solo_player("player")

    armed_with = [call.kwargs["task_id"] for call in mass.call_later.call_args_list]
    assert armed_with == [provider._session_timeout_id, provider._session_timeout_id]
    assert mass.cancel_timer.call_args_list == []


async def test_the_timeout_cancels_the_session_it_was_armed_for() -> None:
    """When the timeout fires it restores the speakers and clears the session."""
    provider = await _setup_provider()
    session = _stub_session()

    with patch.object(CalibrationSession, "create", AsyncMock(return_value=session)):
        await provider.start_session(["player"])
    await provider._handle_session_timeout()

    session.stop.assert_awaited_once()
    assert await provider.get_session() is None


async def test_a_stream_that_never_starts_releases_the_session() -> None:
    """A failed start leaves nothing behind for the timeout to trip over."""
    provider = await _setup_provider()
    mass = _mock_mass(provider)
    session = _stub_session()
    session.begin = AsyncMock(side_effect=RuntimeError("no stream"))

    with (
        patch.object(CalibrationSession, "create", AsyncMock(return_value=session)),
        pytest.raises(RuntimeError),
    ):
        await provider.start_session(["player"])

    session.stop.assert_awaited_once()
    assert await provider.get_session() is None
    mass.cancel_timer.assert_called_with(provider._session_timeout_id)


def _stub_session() -> MagicMock:
    """Return a stub calibration session that reports itself live."""
    session = MagicMock()
    session.queue_id = "anchor"
    session.stopped = False
    session.begin = AsyncMock()
    session.solo = AsyncMock()
    session.stop = AsyncMock()
    return session


async def _setup_provider() -> SendspinSyncProvider:
    """Return a fully initialized provider backed by mocked MA plumbing."""
    manifest = MagicMock()
    manifest.domain = "sendspin_sync"
    config = MagicMock()
    config.get_value = MagicMock(return_value="GLOBAL")
    config.name = "Sendspin Sync"
    config.instance_id = "sendspin_sync--test"
    provider = await setup(MagicMock(), manifest, config)
    assert isinstance(provider, SendspinSyncProvider)
    await provider.handle_async_init()
    return provider


def _mock_mass(provider: SendspinSyncProvider) -> MagicMock:
    """Return the stub MusicAssistant the given provider was set up with."""
    return cast("MagicMock", provider.mass)


async def _take_bytes(provider: SendspinSyncProvider, count: int) -> bytes:
    """Return the first ``count`` bytes the provider's audio stream yields."""
    collected = bytearray()
    async with aclosing(provider.get_audio_stream(MagicMock())) as stream:
        async for chunk in stream:
            collected += chunk
            if len(collected) >= count:
                break
    return bytes(collected[:count])
