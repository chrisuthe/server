"""Tests that a queue's resume anchor outlives a start that never got off the ground."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.player_queue import PlayerQueue
from music_assistant_models.queue_item import QueueItem

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData

QUEUE_ID = "q1"
# a podcast episode paused 75 minutes in - the position a resume has to get back to
PAUSE_POSITION = 4500


def _controller_with_paused_queue() -> tuple[PlayerQueuesController, PlayerQueue]:
    """Build a bare controller holding an item paused at PAUSE_POSITION."""
    ctrl = PlayerQueuesController.__new__(PlayerQueuesController)
    queue = PlayerQueue(queue_id=QUEUE_ID, active=True, display_name="Q1", available=True, items=2)
    items = [
        QueueItem(queue_id=QUEUE_ID, queue_item_id="ep1", name="ep1", duration=9028),
        QueueItem(queue_id=QUEUE_ID, queue_item_id="ep2", name="ep2", duration=6787),
    ]
    queue.current_index = 0
    queue.current_item = items[0]
    queue.resume_pos = PAUSE_POSITION
    queue.elapsed_time = PAUSE_POSITION
    queue.elapsed_time_last_updated = time.time()
    queue_data = PlayerQueueData(queue=queue)
    queue_data.items = items
    ctrl._queue_data = {QUEUE_ID: queue_data}

    ctrl.signal_update = Mock()  # type: ignore[method-assign]
    ctrl._check_player_permission = Mock()  # type: ignore[method-assign]
    ctrl._set_transitioning = Mock()  # type: ignore[method-assign]
    ctrl._load_item = AsyncMock()  # type: ignore[method-assign]
    ctrl._get_next_index = Mock(return_value=None)  # type: ignore[method-assign]
    ctrl.player_media_from_queue_item = AsyncMock()  # type: ignore[method-assign]
    ctrl.stop = AsyncMock()  # type: ignore[method-assign]
    ctrl.mass = MagicMock()
    ctrl.mass.players.play_media = AsyncMock()
    ctrl.logger = MagicMock()  # the failure paths log
    return ctrl, queue


async def test_failed_start_keeps_the_resume_anchor() -> None:
    """
    A start that never produced audio must leave the pause position intact.

    Losing it here silently restarts the item from the beginning on the next play,
    and a few seconds of that overwrites the stored progress for good.
    """
    ctrl, queue = _controller_with_paused_queue()
    ctrl._load_item = AsyncMock(side_effect=MediaNotFoundError("source too slow"))  # type: ignore[method-assign]

    with pytest.raises(MediaNotFoundError):
        await ctrl.play_index(QUEUE_ID, 0, seek_position=PAUSE_POSITION)

    assert queue.resume_pos == PAUSE_POSITION


async def test_successful_start_clears_the_resume_anchor() -> None:
    """Once the item is actually under way the anchor has been consumed."""
    ctrl, queue = _controller_with_paused_queue()

    await ctrl.play_index(QUEUE_ID, 0, seek_position=PAUSE_POSITION)

    assert queue.resume_pos == 0


async def test_next_clears_the_resume_anchor() -> None:
    """
    Skipping to another item abandons the anchor.

    next() switches current_item up front, so a surviving anchor would belong to the
    item we just left and would seek the new one to that position.
    """
    ctrl, queue = _controller_with_paused_queue()
    ctrl._get_next_index = Mock(return_value=1)  # type: ignore[method-assign]

    await ctrl.next(QUEUE_ID)

    assert queue.resume_pos == 0


async def test_previous_clears_the_resume_anchor() -> None:
    """Going back abandons the anchor for the same reason next() does."""
    ctrl, queue = _controller_with_paused_queue()
    queue.current_index = 1
    queue.current_item = ctrl._queue_data[QUEUE_ID].items[1]

    await ctrl.previous(QUEUE_ID)

    assert queue.resume_pos == 0
