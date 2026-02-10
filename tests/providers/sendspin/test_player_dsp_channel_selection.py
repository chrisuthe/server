"""Tests for Sendspin DSP channel selection helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from aiosendspin.server.channels import MAIN_CHANNEL

from music_assistant.providers.sendspin.player import _compute_unique_dsp_channels, _get_player_channel


def _make_dsp_config(*, enabled: bool, filter_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=enabled,
        input_gain=0,
        output_gain=0,
        filters=[SimpleNamespace(enabled=filter_enabled)],
    )


def _make_mass(
    dsp_by_player: dict[str, SimpleNamespace],
    output_channels_by_player: dict[str, str] | None = None,
) -> Any:
    channels = output_channels_by_player or {}

    class _Config:
        def get_player_dsp_config(self, player_id: str) -> SimpleNamespace:
            src = dsp_by_player[player_id]
            # Return a fresh mutable object because helper code trims disabled filters in place.
            return SimpleNamespace(
                enabled=src.enabled,
                input_gain=src.input_gain,
                output_gain=src.output_gain,
                filters=[SimpleNamespace(enabled=f.enabled) for f in src.filters],
            )

        def get_raw_player_config_value(self, player_id: str, key: str, default: str) -> str:
            assert key == "output_channels"
            return channels.get(player_id, default)

    return SimpleNamespace(config=_Config())


def test_get_player_channel_uses_main_channel_for_disabled_stereo() -> None:
    """Stereo output with DSP disabled uses MAIN_CHANNEL."""
    mass = _make_mass({"p1": _make_dsp_config(enabled=False)})

    assert _get_player_channel(mass, "p1") == MAIN_CHANNEL


def test_get_player_channel_uses_custom_channel_when_dsp_enabled() -> None:
    """DSP enabled always uses a custom per-player channel."""
    mass = _make_mass({"p1": _make_dsp_config(enabled=True)})

    channel_1 = _get_player_channel(mass, "p1")
    channel_2 = _get_player_channel(mass, "p1")

    assert channel_1 != MAIN_CHANNEL
    assert channel_2 == channel_1


def test_get_player_channel_uses_custom_channel_for_non_stereo_output() -> None:
    """Non-stereo output mode gets a custom channel even with DSP disabled."""
    mass = _make_mass(
        {"p1": _make_dsp_config(enabled=False)},
        output_channels_by_player={"p1": "left"},
    )

    assert _get_player_channel(mass, "p1") != MAIN_CHANNEL


def test_compute_unique_dsp_channels_does_not_dedupe_players() -> None:
    """Each player gets its own channel even with identical DSP configs."""
    shared_dsp = _make_dsp_config(enabled=True, filter_enabled=True)
    mass = _make_mass({"p1": shared_dsp, "p2": shared_dsp})

    channels = _compute_unique_dsp_channels(mass, ["p1", "p2"])

    assert len(channels) == 2
    all_members = sorted(member for _cfg, members in channels.values() for member in members)
    assert all_members == ["p1", "p2"]


def test_compute_unique_dsp_channels_strips_disabled_filters() -> None:
    """Disabled DSP filters should be removed from returned configs."""
    mass = _make_mass({"p1": _make_dsp_config(enabled=True, filter_enabled=False)})

    channels = _compute_unique_dsp_channels(mass, ["p1"])
    assert len(channels) == 1
    config, members = next(iter(channels.values()))

    assert members == ["p1"]
    assert config.filters == []
