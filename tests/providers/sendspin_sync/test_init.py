"""Tests for the Sendspin Sync plugin provider entry point."""

from __future__ import annotations

import pathlib
from contextlib import aclosing
from unittest.mock import MagicMock

import pytest
from music_assistant_models.enums import MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.provider import ProviderManifest

from music_assistant.providers import sendspin_sync
from music_assistant.providers.sendspin_sync import (
    AUDIO_SOURCE_ID,
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


async def _take_bytes(provider: SendspinSyncProvider, count: int) -> bytes:
    """Return the first ``count`` bytes the provider's audio stream yields."""
    collected = bytearray()
    async with aclosing(provider.get_audio_stream(MagicMock())) as stream:
        async for chunk in stream:
            collected += chunk
            if len(collected) >= count:
                break
    return bytes(collected[:count])
