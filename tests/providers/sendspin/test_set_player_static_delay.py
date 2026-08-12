"""Tests for the published static delay applier on the Sendspin provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from aiosendspin.models.types import PlayerCommand
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, PlayerType
from music_assistant_models.errors import (
    InvalidDataError,
    PlayerUnavailableError,
    UnsupportedFeaturedException,
)

from music_assistant.models.player import Player
from music_assistant.providers.sendspin.constants import (
    CONF_SENDSPIN_STATIC_DELAY,
    MAX_SENDSPIN_STATIC_DELAY,
    MIN_SENDSPIN_STATIC_DELAY,
)
from music_assistant.providers.sendspin.player import SendspinPlayer
from music_assistant.providers.sendspin.provider import SendspinProvider

if TYPE_CHECKING:
    from aiosendspin.server.client import SendspinClient

    from music_assistant.mass import MusicAssistant

PLAYER_ID = "calibration_target"


def _make_player_role(*, supported: bool = True) -> Mock:
    """Build a player role stand-in whose recorded calls expose what was pushed to it."""
    role = Mock(spec=PlayerRoleProtocol)
    role.state_supported_commands = [PlayerCommand.SET_STATIC_DELAY] if supported else []
    return role


class _StaticDelayPlayer(SendspinPlayer):
    """
    A SendspinPlayer with a stubbed player role instead of a live Sendspin client.

    Only the static-delay surface is stubbed: config handling, ``on_config_updated``
    and ``_apply_static_delay`` all run the real provider code, so a test can follow a
    save all the way to the role command.
    """

    def __init__(self, provider: SendspinProvider, player_id: str, role: Mock) -> None:
        """
        Initialize the player without touching the Sendspin server.

        Deliberately runs ``Player.__init__`` rather than the SendspinPlayer one: it is
        what creates the default player config, without which the config controller
        cannot read this player back.
        """
        Player.__init__(self, provider, player_id)
        self._role = role
        # a paired client with its roles activated, so state calculation reads it as ready
        self.api = cast(
            "SendspinClient",
            SimpleNamespace(connection_security=object(), active_roles=("player@v1",)),
        )
        self._attr_name = "Calibration Target"
        self._attr_type = PlayerType.PROTOCOL
        self._cache.clear()
        self.update_state(signal_event=False)

    @property
    def _player_role(self) -> PlayerRoleProtocol | None:
        """Return the stubbed player role."""
        return cast("PlayerRoleProtocol", self._role)

    async def stop(self) -> None:
        """Stop playback (required abstract method)."""

    async def get_config_entries(self) -> list[ConfigEntry]:
        """
        Return only the static delay entry.

        Mirrors the entry in ``SendspinPlayer.get_config_entries``, kept in step with it
        through the shared ``supports_static_delay`` gate and MIN/MAX constants.
        """
        if not self.supports_static_delay:
            return []
        return [
            ConfigEntry(
                key=CONF_SENDSPIN_STATIC_DELAY,
                type=ConfigEntryType.INTEGER,
                required=False,
                default_value=self.static_delay_default_ms,
                range=(MIN_SENDSPIN_STATIC_DELAY, MAX_SENDSPIN_STATIC_DELAY),
                immediate_apply=True,
            )
        ]


def _get_sendspin_provider(mass: MusicAssistant) -> SendspinProvider:
    """Return the loaded Sendspin provider instance."""
    provider = mass.get_provider("sendspin")
    assert provider is not None
    return cast("SendspinProvider", provider)


def _register_player(
    mass: MusicAssistant, *, supported: bool = True
) -> tuple[SendspinProvider, Mock]:
    """Register a Sendspin player stand-in and return its provider and player role."""
    sendspin = _get_sendspin_provider(mass)
    role = _make_player_role(supported=supported)
    player = _StaticDelayPlayer(sendspin, PLAYER_ID, role)
    mass.players._players[PLAYER_ID] = player
    return sendspin, role


def _pushed_delays(role: Mock) -> list[int]:
    """Return every static delay pushed to the player role, in order."""
    return [call.args[0] for call in role.set_static_delay.call_args_list]


async def test_static_delay_reaches_the_player_role(mass: MusicAssistant) -> None:
    """A saved static delay travels through the config change dispatch to the device."""
    sendspin, role = _register_player(mass)

    await sendspin.set_player_static_delay(PLAYER_ID, 320)

    assert _pushed_delays(role) == [320]
    assert mass.config.get_raw_player_config_value(PLAYER_ID, CONF_SENDSPIN_STATIC_DELAY) == 320


async def test_reapplying_the_same_delay_does_not_push_again(mass: MusicAssistant) -> None:
    """The save yields no changed keys, so an unchanged value never reaches the role again."""
    sendspin, role = _register_player(mass)
    await sendspin.set_player_static_delay(PLAYER_ID, 320)
    role.reset_mock()

    await sendspin.set_player_static_delay(PLAYER_ID, 320)

    assert _pushed_delays(role) == []


@pytest.mark.parametrize("delay_ms", [MIN_SENDSPIN_STATIC_DELAY, MAX_SENDSPIN_STATIC_DELAY])
async def test_range_boundaries_are_accepted(mass: MusicAssistant, delay_ms: int) -> None:
    """Both ends of the supported range are applied rather than rejected."""
    sendspin, role = _register_player(mass)
    # 0 is the player's default, which is not persisted as an override; the push to the
    # role is the only observable effect, so seed a non-default value to change away from
    await sendspin.set_player_static_delay(PLAYER_ID, 100)
    role.reset_mock()

    await sendspin.set_player_static_delay(PLAYER_ID, delay_ms)

    assert _pushed_delays(role) == [delay_ms]


@pytest.mark.parametrize(
    "delay_ms",
    [MIN_SENDSPIN_STATIC_DELAY - 1, MAX_SENDSPIN_STATIC_DELAY + 1, -500, 10_000],
)
async def test_out_of_range_delay_raises(mass: MusicAssistant, delay_ms: int) -> None:
    """An out-of-range delay is rejected outright, never clamped into range."""
    sendspin, role = _register_player(mass)

    with pytest.raises(InvalidDataError):
        await sendspin.set_player_static_delay(PLAYER_ID, delay_ms)

    assert _pushed_delays(role) == []


async def test_unknown_player_raises(mass: MusicAssistant) -> None:
    """An unknown player id raises instead of writing config for a player that isn't there."""
    sendspin = _get_sendspin_provider(mass)

    with pytest.raises(PlayerUnavailableError):
        await sendspin.set_player_static_delay("no_such_player", 100)


async def test_non_sendspin_player_raises(mass: MusicAssistant) -> None:
    """A player belonging to another provider is refused."""
    sendspin = _get_sendspin_provider(mass)
    foreign_player = Mock(spec=Player)
    mass.players._players["foreign_player"] = foreign_player

    with pytest.raises(UnsupportedFeaturedException):
        await sendspin.set_player_static_delay("foreign_player", 100)


async def test_player_without_static_delay_support_raises(mass: MusicAssistant) -> None:
    """
    A player whose role won't take a static delay is refused, not silently accepted.

    Without the support check the save produces no changed keys, so the caller would
    get a clean return while the device never hears about the new delay.
    """
    sendspin, role = _register_player(mass, supported=False)

    with pytest.raises(UnsupportedFeaturedException):
        await sendspin.set_player_static_delay(PLAYER_ID, 100)

    assert _pushed_delays(role) == []
