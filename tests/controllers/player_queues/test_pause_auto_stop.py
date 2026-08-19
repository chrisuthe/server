"""Tests that the paused-player watchdog waits longer for long-form media."""

from __future__ import annotations

from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock

from music_assistant_models.enums import PlaybackState
from music_assistant_models.media_items import Audiobook, Podcast, PodcastEpisode
from music_assistant_models.media_items.provider_mapping import ProviderMapping
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.constants import (
    PAUSE_AUTO_STOP_TIMEOUT,
    PAUSE_AUTO_STOP_TIMEOUT_LONG_FORM,
)
from music_assistant.controllers.player_queues.state import PlayerQueueData

if TYPE_CHECKING:
    import pytest

QUEUE_ID = "q1"
_PROVIDER_MAPPINGS = {
    ProviderMapping(item_id="x", provider_domain="test", provider_instance="test")
}


def _episode() -> PodcastEpisode:
    return PodcastEpisode(
        item_id="ep1",
        provider="test",
        name="Ep",
        provider_mappings=_PROVIDER_MAPPINGS,
        position=1,
        podcast=Podcast(
            item_id="pod1", provider="test", name="Pod", provider_mappings=_PROVIDER_MAPPINGS
        ),
    )


def _audiobook() -> Audiobook:
    return Audiobook(
        item_id="ab1", provider="test", name="Book", provider_mappings=_PROVIDER_MAPPINGS
    )


async def _seconds_waited_before_stop(
    media_item: Audiobook | PodcastEpisode | None, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, AsyncMock]:
    """
    Pause a queue holding media_item and run its watchdog with the sleeps stubbed out.

    Returns how many one-second waits the watchdog sat through, and the stop mock so the
    caller can check it was ultimately called.
    """
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    queue = PlayerQueue(queue_id=QUEUE_ID, active=True, display_name="Q1", available=True, items=1)
    queue.state = PlaybackState.PLAYING
    queue.current_item = QueueItem(
        queue_id=QUEUE_ID,
        queue_item_id="item1",
        name="item1",
        duration=9028,
        media_item=media_item,
    )
    ctrl._queue_data = {QUEUE_ID: PlayerQueueData(queue=queue)}
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    ctrl._set_transitioning = Mock()  # type: ignore[method-assign]
    stop = AsyncMock()
    ctrl.stop = stop  # type: ignore[method-assign]
    ctrl.mass = MagicMock()
    ctrl.mass.players._handle_cmd_pause = AsyncMock()

    # a player that stays paused forever, so the watchdog runs its full grace period
    player = MagicMock()
    player.state.playback_state = PlaybackState.PAUSED
    player.extra_data = {}
    ctrl.mass.players.get_player = Mock(return_value=player)

    # pause() hands the watchdog to create_task; capture it so we can drive it here
    watchdogs: list[Coroutine[Any, Any, None]] = []
    ctrl.mass.create_task = Mock(side_effect=lambda coro, **_kwargs: watchdogs.append(coro))

    waits = 0

    async def _counting_sleep(_delay: float) -> None:
        nonlocal waits
        waits += 1

    monkeypatch.setattr(
        "music_assistant.controllers.player_queues.controller.asyncio.sleep", _counting_sleep
    )

    await ctrl.pause(QUEUE_ID)
    assert len(watchdogs) == 1
    await watchdogs[0]
    return waits, stop


async def test_paused_track_is_stopped_at_the_default_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A track's stream is cheap to rebuild, so the short grace period stands."""
    waits, stop = await _seconds_waited_before_stop(None, monkeypatch)
    assert waits == PAUSE_AUTO_STOP_TIMEOUT
    stop.assert_awaited_once_with(QUEUE_ID)


async def test_paused_podcast_episode_gets_the_extended_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A paused episode is left alone far longer, so resuming needs no re-open and seek."""
    waits, stop = await _seconds_waited_before_stop(_episode(), monkeypatch)
    assert waits == PAUSE_AUTO_STOP_TIMEOUT_LONG_FORM
    stop.assert_awaited_once_with(QUEUE_ID)


async def test_paused_audiobook_gets_the_extended_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audiobooks get the same grace period as podcast episodes."""
    waits, _stop = await _seconds_waited_before_stop(_audiobook(), monkeypatch)
    assert waits == PAUSE_AUTO_STOP_TIMEOUT_LONG_FORM
