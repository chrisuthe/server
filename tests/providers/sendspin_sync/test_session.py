"""Tests for the Sendspin Sync calibration session orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.constants import (
    PLAYER_CONTROL_NATIVE,
    PLAYER_CONTROL_NONE,
)
from music_assistant_models.enums import MediaType, PlaybackState, PlayerFeature, PlayerType
from music_assistant_models.errors import (
    ActionUnavailable,
    InvalidDataError,
    PlayerUnavailableError,
    UnsupportedFeaturedException,
)

from music_assistant.providers.sendspin_sync.session import (
    CalibrationSession,
    CalibrationSessionState,
    is_eligible,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

ANCHOR_ID = "virtual_calibration"


async def test_start_groups_every_target_and_starts_the_track_once() -> None:
    """A session attaches all targets to the anchor and plays the track a single time."""
    mass = _make_mass(_make_player("a"), _make_player("b"), _make_player("c"))
    session = await _start(mass, ["a", "b", "c"])

    shared = _shared_session(session)
    assert [call.args[0] for call in shared.add_guest_listener.await_args_list] == ["a", "b", "c"]
    mass.player_queues.play_media.assert_awaited_once()
    assert mass.player_queues.play_media.await_args.args[0] == ANCHOR_ID


async def test_the_session_is_silent_until_its_stream_begins() -> None:
    """Commandeering the speakers and starting the track are separate steps."""
    mass = _make_mass(_make_player("a"))

    with patch(
        "music_assistant.providers.sendspin_sync.session.SharedPlaybackSession"
    ) as shared_cls:
        shared_cls.create_remote = AsyncMock(return_value=_make_shared_session())
        session = await CalibrationSession.create(
            mass, MagicMock(), "provider.sendspin_sync", "sendspin_sync--test", ["a"]
        )

    mass.player_queues.play_media.assert_not_awaited()
    await session.begin(MagicMock())
    mass.player_queues.play_media.assert_awaited_once()


async def test_soloing_never_restarts_the_stream() -> None:
    """
    Walking every speaker leaves the single calibration stream untouched.

    This is the property the whole measurement depends on: the chirp train is a
    metronome on the server clock and a restart resets its timeline.
    """
    mass = _make_mass(_make_player("a"), _make_player("b"), _make_player("c"))
    session = await _start(mass, ["a", "b", "c"])
    mass.player_queues.play_media.reset_mock()

    for player_id in ("a", "b", "c", "a"):
        await session.solo(player_id)

    mass.player_queues.play_media.assert_not_awaited()
    mass.player_queues.stop.assert_not_awaited()


async def test_solo_mutes_the_others_and_never_mutes_the_target() -> None:
    """Isolating a speaker silences its peers and issues no mute command against it."""
    mass = _make_mass(_make_player("a"), _make_player("b"), _make_player("c"))
    session = await _start(mass, ["a", "b", "c"])

    await session.solo("b")

    assert _silenced(mass) == {"a", "c"}
    assert "b" not in _mute_command_targets(mass)
    assert session.soloed_player_id == "b"


async def test_solo_hands_the_next_target_over_unmuted() -> None:
    """The speaker whose turn it is comes out of a previous solo's mute before its peers go in."""
    mass = _make_mass(_make_player("a"), _make_player("b"))
    session = await _start(mass, ["a", "b"])

    await session.solo("a")
    assert _silenced(mass) == {"b"}
    await session.solo("b")

    assert _silenced(mass) == {"a"}


async def test_start_rejects_a_muted_player() -> None:
    """
    A speaker the user had already muted can not be calibrated.

    The session refuses to claim a mute it did not make, so such a speaker could be
    soloed but never actually heard - better to say so up front than to hand back a
    session with a member that yields no measurement.
    """
    mass = _make_mass(_make_player("a"), _make_player("b", volume_muted=True))

    with pytest.raises(ActionUnavailable, match="muted"):
        await _start(mass, ["a", "b"])


async def test_start_rejects_a_player_turned_all_the_way_down() -> None:
    """A speaker at volume zero is as inaudible as a muted one, force flag or not."""
    mass = _make_mass(_make_player("a", volume_level=0))

    with pytest.raises(ActionUnavailable):
        await _start(mass, ["a"], force=True)


async def test_solo_silences_a_player_without_a_mute_control_by_volume() -> None:
    """A Sendspin client that does not implement mute is taken out of the mix by volume."""
    mass = _make_mass(
        _make_player("a"),
        _make_player("b", mute_control=PLAYER_CONTROL_NONE, volume_level=55),
    )
    session = await _start(mass, ["a", "b"])

    await session.solo("a")

    assert ("b", 0) in _volume_calls(mass)
    assert "b" not in _mute_command_targets(mass)

    await session.stop()
    assert ("b", 55) in _volume_calls(mass)


async def test_a_volume_silenced_player_stays_tracked_when_its_state_lags() -> None:
    """
    A speaker whose volume command has not landed yet is still re-silenced.

    Sendspin's volume_set only reports back once the client acknowledges it, so a
    live read of volume_level can still show the previous solo's zero. Deciding
    ownership from that read would drop the speaker from the session's bookkeeping
    and let it rise back to full volume mid-measurement.
    """
    mass = _make_mass(
        _make_player("a", mute_control=PLAYER_CONTROL_NONE),
        _make_player("b", mute_control=PLAYER_CONTROL_NONE, volume_level=55),
        optimistic_volume=False,
    )
    session = await _start(mass, ["a", "b"])

    await session.solo("a")
    await session.solo("b")
    mass.players.cmd_volume_set.reset_mock()
    await session.solo("a")

    assert ("b", 0) in _volume_calls(mass)


async def test_a_player_that_was_only_muted_keeps_a_volume_change() -> None:
    """The restore undoes the control it drove, so a volume the user moved is left alone."""
    player = _make_player("b")
    mass = _make_mass(_make_player("a"), player)
    session = await _start(mass, ["a", "b"])
    await session.solo("a")
    # the user turns 'b' down while it is muted for the measurement
    player.state.volume_level = 15

    await session.stop()

    assert "b" not in {player_id for player_id, _ in _volume_calls(mass)}
    assert ("b", False) in _mute_calls(mass)


async def test_solo_rejects_a_player_outside_the_session() -> None:
    """Only members of the running session can be isolated."""
    mass = _make_mass(_make_player("a"), _make_player("b"))
    session = await _start(mass, ["a"])

    with pytest.raises(InvalidDataError):
        await session.solo("b")


async def test_solo_rejects_a_stopped_session() -> None:
    """A session that has been torn down can not be driven any further."""
    mass = _make_mass(_make_player("a"))
    session = await _start(mass, ["a"])
    await session.stop()

    with pytest.raises(ActionUnavailable):
        await session.solo("a")


async def test_the_state_a_client_renders() -> None:
    """The payload the API hands a UI describes the anchor, the members and the stream."""
    mass = _make_mass(_make_player("a"), _make_player("b"))
    session = await _start(mass, ["a", "b"])

    await session.solo("b")

    assert session.state == CalibrationSessionState(
        anchor_player_id=ANCHOR_ID,
        queue_id=ANCHOR_ID,
        player_ids=["a", "b"],
        soloed_player_id="b",
        streaming=True,
    )


async def test_a_stream_that_is_no_longer_the_calibration_track_is_not_streaming() -> None:
    """The reported stream state follows the anchor's queue, not the session's intent."""
    mass = _make_mass(_make_player("a"))
    session = await _start(mass, ["a"])
    assert session.streaming

    mass.player_queues.get.return_value.state = PlaybackState.IDLE
    assert not session.streaming


async def test_stop_restores_volume_mute_grouping_and_playback() -> None:
    """Every target gets its volume, mute, group membership and playback put back."""
    mass = _make_mass(
        _make_player("a", volume_level=30, synced_to="leader"),
        _make_player(
            "b", volume_level=70, active_group="group", playback_state=PlaybackState.PLAYING
        ),
        _make_player(
            "c",
            volume_level=55,
            playback_state=PlaybackState.PLAYING,
            active_source="c_queue",
        ),
    )
    session = await _start(mass, ["a", "b", "c"], force=True)
    await session.solo("a")

    await session.stop()

    shared = _shared_session(session)
    mass.player_queues.stop.assert_awaited_once_with(ANCHOR_ID)
    assert {call.args[0] for call in shared.remove_guest_listener.await_args_list} == {
        "a",
        "b",
        "c",
    }
    assert ("b", False) in _mute_calls(mass)
    assert ("c", False) in _mute_calls(mass)
    assert mass.players.cmd_set_members.await_args_list[0].args[0] == "leader"
    assert mass.players.cmd_set_members.await_args_list[1].args[0] == "group"
    # 'c' was playing its own content standalone, so it is resumed on the source it
    # was on; 'b' is left to the group it was rejoined to, which restarts it itself
    mass.players.cmd_resume.assert_awaited_once()
    assert mass.players.cmd_resume.await_args.args[0] == "c"
    assert mass.players.cmd_resume.await_args.args[1] == "c_queue"
    shared.close.assert_awaited_once()


async def test_stop_restarts_a_group_that_cannot_take_members_back() -> None:
    """A group player without SET_MEMBERS is restarted instead of re-joined member by member."""
    leader = _make_player("group", supported_features=set(), type=PlayerType.GROUP)
    mass = _make_mass(
        _make_player("a", active_group="group", playback_state=PlaybackState.PLAYING), leader
    )
    session = await _start(mass, ["a"], force=True)

    await session.stop()

    mass.players.cmd_set_members.assert_not_awaited()
    mass.players.cmd_play.assert_awaited_once_with("group")


async def test_stop_powers_a_player_back_off() -> None:
    """Grouping powers a speaker on, so one that was off is switched off again."""
    mass = _make_mass(_make_player("a", powered=False, power_control=PLAYER_CONTROL_NATIVE))
    session = await _start(mass, ["a"])

    await session.stop()

    mass.players.cmd_power.assert_awaited_once_with("a", False)
    mass.players.cmd_resume.assert_not_awaited()


async def test_stop_leaves_a_player_without_power_control_alone() -> None:
    """A speaker with no power control has no power state to restore."""
    mass = _make_mass(_make_player("a", powered=None))
    session = await _start(mass, ["a"])

    await session.stop()

    mass.players.cmd_power.assert_not_awaited()


async def test_stop_leaves_an_idle_player_stopped() -> None:
    """A speaker that was not playing is not started up by the restore."""
    mass = _make_mass(_make_player("a"))
    session = await _start(mass, ["a"])

    await session.stop()

    mass.players.cmd_resume.assert_not_awaited()


async def test_stop_is_idempotent() -> None:
    """Stopping an already stopped session restores nothing a second time."""
    mass = _make_mass(_make_player("a"))
    session = await _start(mass, ["a"])
    await session.stop()
    mass.player_queues.stop.reset_mock()

    await session.stop()

    assert session.stopped
    mass.player_queues.stop.assert_not_awaited()


async def test_stop_restores_the_players_when_the_anchor_is_already_gone() -> None:
    """
    A vanished anchor queue does not skip the restore.

    The anchor lives in the Sendspin provider's memory, so a Sendspin reload takes
    it (and its queue) with it while the session still holds the speakers.
    """
    mass = _make_mass(_make_player("a", volume_level=30))
    session = await _start(mass, ["a"])
    await session.solo("a")
    mass.player_queues.get.return_value = None

    await session.stop()

    mass.player_queues.stop.assert_not_awaited()
    _shared_session(session).remove_guest_listener.assert_awaited_once_with("a")


async def test_stop_restores_the_players_when_stopping_the_stream_fails() -> None:
    """A stream that refuses to stop may not cost the user their speaker state."""
    mass = _make_mass(_make_player("a", volume_level=30), _make_player("b", volume_level=70))
    session = await _start(mass, ["a", "b"])
    await session.solo("a")
    mass.player_queues.stop.side_effect = RuntimeError("queue is gone")

    await session.stop()

    assert ("b", False) in _mute_calls(mass)
    _shared_session(session).close.assert_awaited_once()


async def test_stop_restores_the_remaining_players_when_one_fails() -> None:
    """A player that has gone away mid-session can not strand its peers muted."""
    peer = _make_player("b")
    mass = _make_mass(_make_player("a"), peer, _make_player("c"))
    session = await _start(mass, ["a", "b", "c"])
    await session.solo("c")
    assert peer.state.volume_muted
    _fail_for(mass.players.cmd_volume_mute, "a", PlayerUnavailableError("gone"))

    await session.stop()

    assert not peer.state.volume_muted


async def test_stop_restores_the_remaining_players_when_one_cannot_be_ungrouped() -> None:
    """An ungroup that fails outside the MA error hierarchy still leaves the peers restored."""
    peer = _make_player("b")
    mass = _make_mass(_make_player("a"), peer, _make_player("c"))
    session = await _start(mass, ["a", "b", "c"])
    await session.solo("c")
    shared = _shared_session(session)
    _fail_for(shared.remove_guest_listener, "a", RuntimeError("client exploded"))

    await session.stop()

    assert not peer.state.volume_muted
    shared.close.assert_awaited_once()


async def test_a_player_that_cannot_be_ungrouped_is_still_unmuted() -> None:
    """Each restore step is attempted on its own, so a stuck ungroup leaves nothing silent."""
    stuck = _make_player("a")
    mass = _make_mass(stuck, _make_player("b"))
    session = await _start(mass, ["a", "b"])
    await session.solo("b")
    assert stuck.state.volume_muted
    _fail_for(_shared_session(session).remove_guest_listener, "a", RuntimeError("stuck"))

    await session.stop()

    assert not stuck.state.volume_muted


async def test_a_solo_that_cannot_make_its_target_audible_fails() -> None:
    """
    A target that can not be unmuted is an error, not a session reporting it isolated.

    An inaudible target yields no measurement at all, so the caller has to know.
    """
    mass = _make_mass(_make_player("a"), _make_player("b"))
    session = await _start(mass, ["a", "b"])
    await session.solo("a")
    _fail_for(mass.players.cmd_volume_mute, "b", PlayerUnavailableError("gone"))

    with pytest.raises(PlayerUnavailableError):
        await session.solo("b")

    assert session.soloed_player_id == "a"


async def test_a_failure_while_grouping_releases_only_what_was_taken() -> None:
    """An exception mid-takeover releases the players it grouped and leaves the rest alone."""
    mass = _make_mass(
        _make_player("a"),
        _make_player("b"),
        _make_player("c", playback_state=PlaybackState.PLAYING),
    )
    shared = _make_shared_session()
    _fail_for(shared.add_guest_listener, "b", PlayerUnavailableError("gone"))

    with pytest.raises(PlayerUnavailableError):
        await _start(mass, ["a", "b", "c"], force=True, shared=shared)

    # 'a' was grouped and is released; 'b' failed to group and 'c' was never reached,
    # so neither is restored - in particular 'c' keeps playing untouched
    assert {call.args[0] for call in shared.remove_guest_listener.await_args_list} == {"a"}
    mass.players.cmd_resume.assert_not_awaited()
    shared.close.assert_awaited_once()
    mass.player_queues.play_media.assert_not_awaited()


async def test_start_refuses_a_playing_player() -> None:
    """A session may not silently hijack music the user is listening to."""
    mass = _make_mass(_make_player("a"), _make_player("b", playback_state=PlaybackState.PLAYING))

    with pytest.raises(ActionUnavailable, match="currently playing"):
        await _start(mass, ["a", "b"])

    mass.player_queues.play_media.assert_not_awaited()


async def test_start_refuses_a_paused_player() -> None:
    """A paused speaker is holding the user's content just as much as a playing one."""
    mass = _make_mass(_make_player("a", playback_state=PlaybackState.PAUSED))

    with pytest.raises(ActionUnavailable):
        await _start(mass, ["a"])


async def test_start_takes_a_playing_player_over_when_forced() -> None:
    """An explicit force flag is what it takes to interrupt playback."""
    mass = _make_mass(_make_player("a", playback_state=PlaybackState.PLAYING))

    session = await _start(mass, ["a"], force=True)

    assert session.player_ids == ["a"]
    mass.player_queues.play_media.assert_awaited_once()


async def test_start_rejects_a_non_sendspin_player() -> None:
    """Only Sendspin speakers can be calibrated, enforced rather than documented."""
    mass = _make_mass(_make_player("a"), _make_player("b", domain="sonos"))

    with pytest.raises(UnsupportedFeaturedException):
        await _start(mass, ["a", "b"])


async def test_start_rejects_an_unknown_player() -> None:
    """A player id that resolves to nothing is an error, not a silently skipped target."""
    mass = _make_mass(_make_player("a"))

    with pytest.raises(PlayerUnavailableError):
        await _start(mass, ["a", "nope"])


async def test_start_rejects_an_empty_target_list() -> None:
    """A session needs at least one speaker to calibrate."""
    mass = _make_mass(_make_player("a"))

    with pytest.raises(InvalidDataError):
        await _start(mass, [])


async def test_start_rejects_a_duplicated_target() -> None:
    """A speaker may only appear once in a session."""
    mass = _make_mass(_make_player("a"))

    with pytest.raises(InvalidDataError):
        await _start(mass, ["a", "a"])


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"type": PlayerType.STEREO_PAIR}, True),
        ({"domain": "sonos"}, False),
        ({"available": False}, False),
        ({"type": PlayerType.GROUP}, False),
        ({"type": PlayerType.PROTOCOL}, False),
        ({"type": PlayerType.DISPLAY}, False),
        ({"type": PlayerType.VISUALIZER}, False),
        ({"type": PlayerType.LIGHT}, False),
        ({"mute_control": PLAYER_CONTROL_NONE}, True),
        ({"mute_control": PLAYER_CONTROL_NONE, "volume_control": PLAYER_CONTROL_NONE}, False),
    ],
)
async def test_eligibility(overrides: dict[str, Any], expected: bool) -> None:
    """Only available, controllable, audio-rendering Sendspin speakers are eligible."""
    player = _make_player("a", **overrides)
    assert is_eligible(_make_mass(player), player) is expected


async def test_the_session_anchor_is_not_itself_eligible() -> None:
    """The hidden player that leads a calibration group is not a speaker to calibrate."""
    player = _make_player(ANCHOR_ID)
    mass = _make_mass(player)
    mass.get_provider.return_value.is_virtual_player = MagicMock(return_value=True)
    assert is_eligible(mass, player) is False


async def test_a_speaker_that_cannot_take_a_static_delay_is_still_eligible() -> None:
    """
    A speaker whose client carries no static delay is measured like any other.

    Offsets are relative, so leaving it out would take it out of the picture the rest
    are normalised against rather than merely deny it a correction.
    """
    player = _make_player("a")
    mass = _make_mass(player)
    mass.get_provider.return_value.supports_player_static_delay = MagicMock(return_value=False)

    assert is_eligible(mass, player) is True


async def test_start_accepts_a_speaker_that_cannot_take_a_static_delay() -> None:
    """A session runs over such a speaker; the result comes back for the user to apply."""
    mass = _make_mass(_make_player("a"))
    mass.get_provider.return_value.supports_player_static_delay = MagicMock(return_value=False)

    session = await _start(mass, ["a"])

    assert session.player_ids == ["a"]


async def _start(
    mass: MagicMock,
    player_ids: list[str],
    force: bool = False,
    shared: MagicMock | None = None,
) -> CalibrationSession:
    """
    Return a started, streaming session anchored on a stubbed shared playback session.

    :param mass: The stub MusicAssistant the session runs against.
    :param player_ids: The players to calibrate.
    :param force: Take over players that are busy playing.
    :param shared: Anchor stub to use, so a caller can inspect it after a failed start.
    """
    with patch(
        "music_assistant.providers.sendspin_sync.session.SharedPlaybackSession"
    ) as shared_cls:
        shared_cls.create_remote = AsyncMock(return_value=shared or _make_shared_session())
        session = await CalibrationSession.create(
            mass,
            MagicMock(),
            "provider.sendspin_sync",
            "sendspin_sync--test",
            player_ids,
            force=force,
        )
    await session.begin(MagicMock())
    return session


def _fail_for(mock: AsyncMock, player_id: str, error: Exception) -> None:
    """
    Make a player command raise for one player id, keeping its behaviour for the rest.

    :param mock: The stubbed command to make fail.
    :param player_id: The player the command should raise for.
    :param error: The error to raise for that player.
    """
    original = mock.side_effect

    async def _side_effect(target_id: str, *args: object) -> None:
        if target_id == player_id:
            raise error
        if original is not None:
            await original(target_id, *args)

    mock.side_effect = _side_effect


def _make_shared_session() -> MagicMock:
    """Return a stub of the shared playback session that anchors a calibration group."""
    shared = MagicMock()
    shared.player_id = ANCHOR_ID
    shared.queue_id = ANCHOR_ID
    shared.add_guest_listener = AsyncMock()
    shared.remove_guest_listener = AsyncMock()
    shared.close = AsyncMock()
    return shared


def _make_player(player_id: str, *, domain: str = "sendspin", **overrides: object) -> MagicMock:
    """
    Return a stub player carrying just the state a calibration session reads.

    :param player_id: Id and display name of the stub player.
    :param domain: Domain of the player provider the stub belongs to.
    :param overrides: PlayerState attributes to set on top of an idle, unmuted speaker.
    """
    player = MagicMock()
    player.player_id = player_id
    player.provider.domain = domain
    player.extra_data = {}
    state: dict[str, object] = {
        "name": player_id,
        "available": True,
        "type": PlayerType.PLAYER,
        "playback_state": PlaybackState.IDLE,
        "powered": True,
        "power_control": PLAYER_CONTROL_NATIVE,
        "active_source": None,
        "current_media": None,
        "volume_muted": False,
        "volume_level": 42,
        "synced_to": None,
        "active_group": None,
        "mute_control": PLAYER_CONTROL_NATIVE,
        "volume_control": PLAYER_CONTROL_NATIVE,
        "supported_features": {PlayerFeature.SET_MEMBERS},
    }
    if "powered" in overrides and overrides["powered"] is None:
        # a player without power control reports no power state either
        state["power_control"] = PLAYER_CONTROL_NONE
    for key, value in (state | overrides).items():
        setattr(player.state, key, value)
    return player


def _make_mass(*players: MagicMock, optimistic_volume: bool = True) -> MagicMock:
    """
    Return a stub MusicAssistant whose mute and volume commands move player state.

    :param players: The stub players to register.
    :param optimistic_volume: Whether a volume command lands in player state immediately.
        Sendspin's does not - it waits for the client to acknowledge it.
    """
    registry = {player.player_id: player for player in players}
    mass = MagicMock()

    async def _mute(player_id: str, muted: bool) -> None:
        registry[player_id].state.volume_muted = muted

    async def _set_volume(player_id: str, volume_level: int) -> None:
        if not optimistic_volume:
            return
        registry[player_id].state.volume_level = volume_level
        registry[player_id].state.volume_muted = False

    mass.players.get_player = MagicMock(side_effect=lambda player_id, *_: registry.get(player_id))
    mass.players.all_players = MagicMock(return_value=list(players))
    mass.players.cmd_volume_mute = AsyncMock(side_effect=_mute)
    mass.players.cmd_volume_set = AsyncMock(side_effect=_set_volume)
    mass.players.cmd_set_members = AsyncMock()
    mass.players.cmd_power = AsyncMock()
    mass.players.cmd_play = AsyncMock()
    mass.players.cmd_resume = AsyncMock()
    mass.player_queues.play_media = AsyncMock()
    mass.player_queues.stop = AsyncMock()
    mass.player_queues.get = MagicMock(return_value=_make_anchor_queue())
    sendspin = MagicMock()
    sendspin.is_virtual_player = MagicMock(return_value=False)
    sendspin.supports_player_static_delay = MagicMock(return_value=True)
    mass.get_provider = MagicMock(return_value=sendspin)
    return mass


def _make_anchor_queue() -> MagicMock:
    """Return a stub anchor queue that is playing the calibration track."""
    queue = MagicMock()
    queue.state = PlaybackState.PLAYING
    queue.current_item.media_type = MediaType.AUDIO_SOURCE
    return queue


def _shared_session(session: CalibrationSession) -> MagicMock:
    """Return the stubbed shared playback session backing the given session."""
    return cast("MagicMock", session._shared_session)


def _mute_calls(mass: MagicMock) -> list[tuple[str, bool]]:
    """Return every (player_id, muted) pair passed to cmd_volume_mute."""
    return [(call.args[0], call.args[1]) for call in mass.players.cmd_volume_mute.await_args_list]


def _volume_calls(mass: MagicMock) -> list[tuple[str, int]]:
    """Return every (player_id, volume_level) pair passed to cmd_volume_set."""
    return [(call.args[0], call.args[1]) for call in mass.players.cmd_volume_set.await_args_list]


def _mute_command_targets(mass: MagicMock) -> set[str]:
    """Return the ids of every player a mute command was issued against."""
    return {player_id for player_id, _ in _mute_calls(mass)}


def _silenced(mass: MagicMock) -> set[str]:
    """Return the ids of every player that is currently inaudible."""
    return {
        player.player_id
        for player in _registered_players(mass)
        if player.state.volume_muted or player.state.volume_level == 0
    }


def _registered_players(mass: MagicMock) -> Iterable[MagicMock]:
    """Return the stub players registered on the given stub MusicAssistant."""
    return cast("list[MagicMock]", mass.players.all_players.return_value)
