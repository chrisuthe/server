"""Sendspin Player implementation."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from aiosendspin.models import AudioCodec, MediaCommand
from aiosendspin.models.types import PlaybackStateType
from aiosendspin.models.types import RepeatMode as SendspinRepeatMode
from aiosendspin.server import (
    ClientEvent,
    GroupEvent,
    SendspinGroup,
    VolumeChangedEvent,
)
from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import DisconnectBehaviour
from aiosendspin.server.events import (
    ClientGroupChangedEvent,
    GroupDeletedEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
    GroupStateChangedEvent,
)
from aiosendspin.server.roles import (
    ArtworkGroupRole,
    ControllerEvent,
    ControllerGroupRole,
    ControllerNextEvent,
    ControllerPauseEvent,
    ControllerPlayEvent,
    ControllerPreviousEvent,
    ControllerRepeatEvent,
    ControllerShuffleEvent,
    ControllerStopEvent,
    MetadataGroupRole,
)
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.constants import PLAYER_CONTROL_NONE
from music_assistant_models.enums import (
    ConfigEntryType,
    ImageType,
    PlaybackState,
    PlayerFeature,
    PlayerType,
    RepeatMode,
)
from music_assistant_models.media_items import AudioFormat
from music_assistant_models.player import DeviceInfo
from PIL import Image

from music_assistant.constants import (
    CONF_ENTRY_FLOW_MODE,
    CONF_ENTRY_HTTP_PROFILE_HIDDEN,
    CONF_ENTRY_OUTPUT_CODEC_HIDDEN,
    CONF_ENTRY_SAMPLE_RATES,
    CONF_OUTPUT_CHANNELS,
)
from music_assistant.helpers.audio import get_player_filter_params
from music_assistant.helpers.ffmpeg import FFMpeg
from music_assistant.models.player import Player, PlayerMedia
from music_assistant.providers.sendspin.playback import (
    _HistoricalInjection,
    _MainPCMCacheChunk,
    pcm_duration_us,
    prepare_dsp_for_join,
    run_playback,
)

# Supported group commands for Sendspin players
SUPPORTED_GROUP_COMMANDS = [
    MediaCommand.PLAY,
    MediaCommand.PAUSE,
    MediaCommand.STOP,
    MediaCommand.NEXT,
    MediaCommand.PREVIOUS,
    MediaCommand.REPEAT_OFF,
    MediaCommand.REPEAT_ONE,
    MediaCommand.REPEAT_ALL,
    MediaCommand.SHUFFLE,
    MediaCommand.UNSHUFFLE,
]

# Config constants for Sendspin audio format
CONF_PREFERRED_SENDSPIN_FORMAT = "preferred_sendspin_format"
SENDSPIN_FORMAT_AUTOMATIC = "automatic"


def format_to_option_value(fmt: SupportedAudioFormat) -> str:
    """Convert SupportedAudioFormat to "codec:sample_rate:bit_depth:channels"."""
    return f"{fmt.codec.value}:{fmt.sample_rate}:{fmt.bit_depth}:{fmt.channels}"


def option_value_to_format(value: str) -> tuple[AudioCodec, SendspinAudioFormat] | None:
    """Parse option value back to (AudioCodec, SendspinAudioFormat).

    :param value: Option value in format "codec:sample_rate:bit_depth:channels".
    :return: Tuple of (AudioCodec, SendspinAudioFormat) or None if parsing fails.
    """
    try:
        codec_str, sample_rate_str, bit_depth_str, channels_str = value.split(":")
        codec = AudioCodec(codec_str)
        audio_format = SendspinAudioFormat(
            sample_rate=int(sample_rate_str),
            bit_depth=int(bit_depth_str),
            channels=int(channels_str),
        )
        return (codec, audio_format)
    except (ValueError, KeyError):
        return None


def format_to_display_string(fmt: SupportedAudioFormat) -> str:
    """Convert to display string like "FLAC 48kHz/24bit stereo"."""
    codec_name = fmt.codec.name
    sample_rate_khz = fmt.sample_rate / 1000
    # Format sample rate: show as integer if whole number, otherwise one decimal
    if sample_rate_khz == int(sample_rate_khz):
        sample_rate_str = f"{int(sample_rate_khz)}kHz"
    else:
        sample_rate_str = f"{sample_rate_khz:.1f}kHz"
    if fmt.channels == 2:
        channels_str = "stereo"
    elif fmt.channels == 1:
        channels_str = "mono"
    else:
        channels_str = f"{fmt.channels}ch"
    return f"{codec_name} {sample_rate_str}/{fmt.bit_depth}bit {channels_str}"


if TYPE_CHECKING:
    from aiosendspin.models.player import SupportedAudioFormat
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.push_stream import PushStream
    from music_assistant_models.config_entries import ConfigValueType
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

    from .provider import SendspinProvider


@dataclass
class _DSPChannel:
    """A DSP processing channel backed by an ffmpeg process."""

    channel_id: UUID
    filter_params: list[str]
    ffmpeg: FFMpeg
    output_channels: int  # 1 for mono (left/right mode), 2 for stereo
    pending: bytearray = field(default_factory=bytearray)
    chunks_processed: int = 0
    chunks_failed: int = 0
    input_bytes_total: int = 0
    input_us_total: int = 0
    raw_output_bytes_total: int = 0
    raw_output_us_total: int = 0
    delivered_output_bytes_total: int = 0
    delivered_output_us_total: int = 0
    pending_peak_bytes: int = 0
    pending_condition: asyncio.Condition = field(default_factory=asyncio.Condition)
    stdout_reader_task: asyncio.Task[None] | None = None


def _get_dsp_filter_params(
    mass: MusicAssistant, player_id: str, pcm_format: AudioFormat
) -> list[str] | None:
    """Return ffmpeg filter params if player needs DSP, else None."""
    dsp_enabled = mass.config.get_player_dsp_config(player_id).enabled
    output_channels = mass.config.get_raw_player_config_value(
        player_id, CONF_OUTPUT_CHANNELS, "stereo"
    )
    if not dsp_enabled and output_channels == "stereo":
        return None
    filter_params = get_player_filter_params(mass, player_id, pcm_format, pcm_format)
    return filter_params if filter_params else None


class SendspinPlayer(Player):
    """A sendspin audio player in Music Assistant."""

    api: SendspinClient
    unsub_event_cb: Callable[[], None]
    unsub_group_event_cb: Callable[[], None]
    last_sent_artwork_url: str | None = None
    last_sent_artist_artwork_url: str | None = None
    _playback_task: asyncio.Task[None] | None = None
    _push_stream: PushStream | None = None
    _dsp_channels: dict[str, _DSPChannel]
    _player_channel_map: dict[str, UUID]
    _pending_join_members: set[str]
    _pending_historical_injections: dict[str, _HistoricalInjection]
    _main_pcm_cache: deque[_MainPCMCacheChunk]
    _pcm_format: AudioFormat | None = None
    is_web_player: bool = False

    @property
    def requires_flow_mode(self) -> bool:
        """Return if the player requires flow mode."""
        return True

    def __init__(self, provider: SendspinProvider, player_id: str) -> None:
        """Initialize the Player."""
        super().__init__(provider, player_id)
        sendspin_client = provider.server_api.get_client(player_id)
        assert sendspin_client is not None
        self.api = sendspin_client
        self.api.disconnect_behaviour = DisconnectBehaviour.STOP
        self.unsub_event_cb = sendspin_client.add_event_listener(self.event_cb)
        self.unsub_group_event_cb = sendspin_client.group.add_event_listener(self.group_event_cb)
        if controller_role := self._controller_role:
            controller_role.set_supported_commands(SUPPORTED_GROUP_COMMANDS)

        self._dsp_channels = {}
        self._player_channel_map = {}
        self._pending_join_members = set()
        self._pending_historical_injections = {}
        self._main_pcm_cache = deque()

        self.logger = self.provider.logger.getChild(player_id)
        # init some static variables
        self._attr_name = sendspin_client.name
        self._attr_type = PlayerType.PLAYER
        self._attr_supported_features = {
            PlayerFeature.SET_MEMBERS,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.MULTI_DEVICE_DSP,
        }
        self._attr_can_group_with = {provider.instance_id}
        self._attr_power_control = PLAYER_CONTROL_NONE
        if device_info := sendspin_client.info.device_info:
            self._attr_device_info = DeviceInfo(
                model=device_info.product_name or "Unknown model",
                manufacturer=device_info.manufacturer or "Unknown Manufacturer",
                software_version=device_info.software_version,
            )
        else:
            self._attr_device_info = DeviceInfo()
        if sendspin_client.info.player_support:
            for role in sendspin_client.roles_by_family("player"):
                volume = role.get_player_volume()
                muted = role.get_player_muted()
                if volume is not None:
                    self._attr_volume_level = volume
                if muted is not None:
                    self._attr_volume_muted = muted
                if volume is not None or muted is not None:
                    break
        self._attr_available = True
        self.is_web_player = sendspin_client.name.startswith(
            "Web ("  # The regular Web Interface
        ) or sendspin_client.name.startswith(
            "PWA ("  # The PWA App
        )
        self._attr_expose_to_ha_by_default = not self.is_web_player
        self._attr_hidden_by_default = self.is_web_player

    @property
    def _artwork_role(self) -> ArtworkGroupRole | None:
        """Get the ArtworkGroupRole for this player's group."""
        role = self.api.group.group_role("artwork")
        if isinstance(role, ArtworkGroupRole):
            return role
        return None

    @property
    def _metadata_role(self) -> MetadataGroupRole | None:
        """Get the MetadataGroupRole for this player's group."""
        role = self.api.group.group_role("metadata")
        if isinstance(role, MetadataGroupRole):
            return role
        return None

    @property
    def _controller_role(self) -> ControllerGroupRole | None:
        """Get the ControllerGroupRole for this player's group."""
        role = self.api.group.group_role("controller")
        if isinstance(role, ControllerGroupRole):
            return role
        return None

    @property
    def _player_role(self) -> PlayerRoleProtocol | None:
        """Get the player role for this client (not group role)."""
        for role in self.api.roles_by_family("player"):
            if isinstance(role, PlayerRoleProtocol):
                return role
        return None

    async def _handle_controller_event(self, event: ControllerEvent) -> None:
        """Handle a controller event from the ControllerGroupRole."""
        queue = self.mass.player_queues.get_active_queue(self.player_id)
        match event:
            case ControllerPlayEvent():
                await self.mass.players.cmd_play(self.player_id)
            case ControllerPauseEvent():
                await self.mass.players.cmd_pause(self.player_id)
            case ControllerStopEvent():
                await self.mass.players.cmd_stop(self.player_id)
            case ControllerNextEvent():
                await self.mass.players.cmd_next_track(self.player_id)
            case ControllerPreviousEvent():
                await self.mass.players.cmd_previous_track(self.player_id)
            case ControllerRepeatEvent(mode=mode) if queue:
                match mode:
                    case SendspinRepeatMode.OFF:
                        self.mass.player_queues.set_repeat(queue.queue_id, RepeatMode.OFF)
                    case SendspinRepeatMode.ONE:
                        self.mass.player_queues.set_repeat(queue.queue_id, RepeatMode.ONE)
                    case SendspinRepeatMode.ALL:
                        self.mass.player_queues.set_repeat(queue.queue_id, RepeatMode.ALL)
            case ControllerShuffleEvent(shuffle=shuffle) if queue:
                await self.mass.player_queues.set_shuffle(queue.queue_id, shuffle_enabled=shuffle)

    def event_cb(self, client: SendspinClient, event: ClientEvent) -> None:
        """Event callback registered to the sendspin client."""
        self.logger.debug("Received PlayerEvent: %s", event)
        match event:
            case VolumeChangedEvent(volume=volume, muted=muted):
                self._attr_volume_level = volume
                self._attr_volume_muted = muted
                self.update_state()
            case ClientGroupChangedEvent(new_group=new_group):
                self.unsub_group_event_cb()
                self.unsub_group_event_cb = new_group.add_event_listener(self.group_event_cb)
                if controller_role := self._controller_role:
                    controller_role.set_supported_commands(SUPPORTED_GROUP_COMMANDS)
                # Sync playback state from the new group
                match new_group.state:
                    case PlaybackStateType.PLAYING:
                        self._attr_playback_state = PlaybackState.PLAYING
                    case PlaybackStateType.PAUSED:
                        self._attr_playback_state = PlaybackState.PAUSED
                    case PlaybackStateType.STOPPED:
                        self._attr_playback_state = PlaybackState.IDLE
                        self._attr_elapsed_time = 0
                        self._attr_elapsed_time_last_updated = time.time()
                        if self._playback_task and not self._playback_task.done():
                            self._playback_task.cancel()
                # Update in case this is a newly created group
                # GroupMemberAddedEvent or GroupMemberRemovedEvent will be fired before this
                # so group members are already up to date at this point
                if self.synced_to is None:
                    # We are the leader, stop on disconnect
                    self.api.disconnect_behaviour = DisconnectBehaviour.STOP
                else:
                    self.api.disconnect_behaviour = DisconnectBehaviour.UNGROUP
                self.update_state()

    def group_event_cb(self, group: SendspinGroup, event: GroupEvent) -> None:
        """Event callback registered to the sendspin group this player belongs to."""
        if self.synced_to is not None:
            # Only handle group events as the leader, except for:
            # - GroupMemberRemovedEvent: to handle being removed from a group
            # - GroupStateChangedEvent: to update playback state when leader stops/disconnects
            if not isinstance(event, (GroupMemberRemovedEvent, GroupStateChangedEvent)):
                return
        self.logger.debug("Received GroupEvent: %s", event)

        match event:
            case GroupStateChangedEvent(state=state):
                self.logger.debug("Group state changed to: %s", state)
                match state:
                    case PlaybackStateType.PLAYING:
                        self._attr_playback_state = PlaybackState.PLAYING
                    case PlaybackStateType.PAUSED:
                        self._attr_playback_state = PlaybackState.PAUSED
                    case PlaybackStateType.STOPPED:
                        self._attr_playback_state = PlaybackState.IDLE
                        self._attr_elapsed_time = 0
                        self._attr_elapsed_time_last_updated = time.time()
                        if self._playback_task and not self._playback_task.done():
                            self._playback_task.cancel()
                self.update_state()
            case GroupMemberAddedEvent(client_id=client_id):
                self.logger.debug("Group member added: %s", client_id)
                if client_id not in self._attr_group_members:
                    self._attr_group_members.append(client_id)
                    self.update_state()
            case GroupMemberRemovedEvent(client_id=client_id):
                self.logger.debug("Group member removed: %s", client_id)
                self.mass.create_task(self._handle_group_member_removed(group, client_id))
            case GroupDeletedEvent():
                pass
            case ControllerEvent() as controller_event:
                if self.synced_to is None:
                    self.mass.create_task(self._handle_controller_event(controller_event))

    async def _handle_group_member_removed(self, group: SendspinGroup, client_id: str) -> None:
        """Handle a group member being removed asynchronously."""
        if client_id == self.player_id:
            if len(self._attr_group_members) > 0:
                # We were just removed as a leader:
                # 1. stop playback on the old group
                await group.stop()
                # 2. clear our members (since we are now alone)
                group_members = [
                    member for member in self._attr_group_members if member != client_id
                ]
                self._attr_group_members = []
                # 3. assign new leader if there are members left
                if len(group_members) > 0 and (
                    new_leader := self.mass.players.get(group_members[0])
                ):
                    new_leader = cast("SendspinPlayer", new_leader)
                    new_leader._attr_group_members = group_members[1:]
                    new_leader.api.disconnect_behaviour = DisconnectBehaviour.STOP
                    new_leader.update_state()
            self.update_state()
        elif client_id in self._attr_group_members:
            # Someone else left our group
            self._attr_group_members.remove(client_id)
            self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """Handle VOLUME_SET command on the player."""
        roles = self.api.roles_by_family("player")
        for role in roles:
            role.set_player_volume(volume_level)

    async def volume_mute(self, muted: bool) -> None:
        """Handle VOLUME MUTE command on the player."""
        roles = self.api.roles_by_family("player")
        for role in roles:
            role.set_player_mute(muted)

    async def stop(self) -> None:
        """Stop command."""
        self.logger.debug("Received STOP command on player %s", self.display_name)
        self.mark_stop_called()
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_elapsed_time = 0
        self._attr_elapsed_time_last_updated = time.time()
        self.update_state()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._playback_task
        # We don't care if we stopped the stream or it was already stopped
        await self.api.group.stop()
        # Clear the playback task reference (group.stop() handles stopping the stream)
        self._playback_task = None
        self._attr_current_media = None
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_elapsed_time = 0
        self._attr_elapsed_time_last_updated = time.time()
        self.update_state()

    async def play_media(self, media: PlayerMedia) -> None:
        """Play media command."""
        self.logger.debug(
            "Received PLAY_MEDIA command on player %s with uri %s", self.display_name, media.uri
        )

        # Update player state optimistically
        self._attr_current_media = media
        self._attr_elapsed_time = 0
        self._attr_elapsed_time_last_updated = time.time()
        # playback_state will be set by the group state change event

        # Stop previous stream in case we were already playing something
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._playback_task
        await self.api.group.stop()
        # Run playback in background task to immediately return
        self._playback_task = asyncio.create_task(self._run_playback(media))
        self.update_state()

    async def on_config_updated(self) -> None:
        """Apply preferred format when config changes."""
        await self._apply_preferred_format()

    async def _apply_preferred_format(self) -> None:
        """Read config and call set_preferred_format() if not automatic."""
        player_role = self._player_role
        if player_role is None:
            return

        config_value = cast(
            "str",
            self.config.get_value(CONF_PREFERRED_SENDSPIN_FORMAT, SENDSPIN_FORMAT_AUTOMATIC),
        )
        if config_value == SENDSPIN_FORMAT_AUTOMATIC:
            # Automatic mode: don't set a preferred format, let client decide
            self.logger.debug("Audio format set to automatic for player %s", self.display_name)
            return

        parsed = option_value_to_format(config_value)
        if parsed is None:
            self.logger.warning(
                "Invalid audio format config value '%s' for player %s",
                config_value,
                self.display_name,
            )
            return

        codec, audio_format = parsed
        if not player_role.set_preferred_format(audio_format, codec):
            self.logger.warning(
                "Failed to set preferred audio format %s %s for player %s",
                codec.name,
                audio_format,
                self.display_name,
            )

    def _resolve_channel_for_player(self, player_id: str, pcm_format: AudioFormat) -> UUID:
        """Resolve channel ID for a player and update local maps."""
        filter_params = _get_dsp_filter_params(self.mass, player_id, pcm_format)
        if filter_params is None:
            self._player_channel_map[player_id] = MAIN_CHANNEL
            return MAIN_CHANNEL
        channel_id = uuid4()
        self._player_channel_map[player_id] = channel_id
        return channel_id

    async def _create_dsp_channel(
        self,
        player_id: str,
        pcm_format: AudioFormat,
    ) -> _DSPChannel:
        """Create a new DSP channel with an ffmpeg process.

        :param player_id: Player ID this DSP channel belongs to.
        :param pcm_format: Input PCM audio format.
        """
        filter_params = _get_dsp_filter_params(self.mass, player_id, pcm_format) or []
        output_channels = 1 if any("pan=mono" in p for p in filter_params) else 2
        output_format = AudioFormat(
            content_type=pcm_format.content_type,
            sample_rate=pcm_format.sample_rate,
            bit_depth=pcm_format.bit_depth,
            channels=output_channels,
        )
        channel_id = self._player_channel_map[player_id]
        ffmpeg = FFMpeg(
            audio_input="-",
            input_format=pcm_format,
            output_format=output_format,
            filter_params=filter_params,
        )
        await ffmpeg.start()
        dsp = _DSPChannel(
            channel_id=channel_id,
            filter_params=filter_params,
            ffmpeg=ffmpeg,
            output_channels=output_channels,
        )
        dsp.stdout_reader_task = asyncio.create_task(
            self._read_dsp_stdout(dsp),
            name=f"sendspin-dsp-stdout-{dsp.channel_id}",
        )
        return dsp

    async def _close_dsp_channel(self, dsp: _DSPChannel) -> None:
        """Close a DSP channel and release ffmpeg resources."""
        if dsp.stdout_reader_task is not None:
            dsp.stdout_reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await dsp.stdout_reader_task
        with suppress(Exception):
            await dsp.ffmpeg.close()

    async def _read_dsp_stdout(self, dsp: _DSPChannel) -> None:
        """Continuously read ffmpeg stdout into dsp.pending.

        We need to drain stdout continuously; otherwise ffmpeg can block on its stdout
        pipe and/or our per-chunk reads can artificially lag behind real output.
        """
        try:
            while True:
                data = await dsp.ffmpeg.read(65536)
                if not data:
                    return
                dsp.pending.extend(data)
                dsp.pending_peak_bytes = max(dsp.pending_peak_bytes, len(dsp.pending))
                async with dsp.pending_condition:
                    dsp.pending_condition.notify_all()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger.debug(
                "DSP stdout reader failed for channel %s", dsp.channel_id, exc_info=True
            )
        finally:
            async with dsp.pending_condition:
                dsp.pending_condition.notify_all()

    async def _process_dsp_chunk(
        self, dsp: _DSPChannel, chunk: bytes, pcm_format: AudioFormat
    ) -> bytes:
        """Process a PCM chunk through the DSP channel's ffmpeg.

        :param dsp: DSP channel to process through.
        :param chunk: Raw PCM audio data.
        :param pcm_format: Input PCM format (used to calculate expected output size).
        """
        expected = len(chunk) // 2 if dsp.output_channels == 1 else len(chunk)
        input_duration_us = pcm_duration_us(
            len(chunk),
            pcm_format.sample_rate,
            pcm_format.bit_depth,
            pcm_format.channels,
        )
        output_duration_us = pcm_duration_us(
            expected,
            pcm_format.sample_rate,
            pcm_format.bit_depth,
            dsp.output_channels,
        )
        dsp.chunks_processed += 1
        dsp.input_bytes_total += len(chunk)
        dsp.input_us_total += input_duration_us
        try:
            await asyncio.wait_for(dsp.ffmpeg.write(chunk), timeout=5.0)
        except Exception:
            dsp.chunks_failed += 1
            dsp.delivered_output_bytes_total += expected
            dsp.delivered_output_us_total += output_duration_us
            self.logger.warning("DSP failed for channel %s", dsp.channel_id, exc_info=True)
            # Keep timeline continuity by filling this chunk with silence.
            return b"\x00" * expected

        deadline = time.monotonic() + 2.0
        while len(dsp.pending) < expected:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            async with dsp.pending_condition:
                if len(dsp.pending) >= expected:
                    break
                if dsp.stdout_reader_task is not None and dsp.stdout_reader_task.done():
                    break
                with suppress(TimeoutError):
                    await asyncio.wait_for(dsp.pending_condition.wait(), timeout=remaining)

        raw_consumed_bytes = min(expected, len(dsp.pending))
        raw_consumed_us = 0
        if raw_consumed_bytes:
            raw_consumed_us = pcm_duration_us(
                raw_consumed_bytes,
                pcm_format.sample_rate,
                pcm_format.bit_depth,
                dsp.output_channels,
            )
            dsp.raw_output_bytes_total += raw_consumed_bytes
            dsp.raw_output_us_total += raw_consumed_us
        dsp.delivered_output_bytes_total += expected
        dsp.delivered_output_us_total += output_duration_us
        if raw_consumed_bytes >= expected:
            result = bytes(dsp.pending[:expected])
            del dsp.pending[:expected]
            return result

        # Still short: output silence for the missing tail to keep stream moving,
        # but record the fact that ffmpeg is lagging behind.
        result = bytes(dsp.pending) + b"\x00" * (expected - len(dsp.pending))
        dsp.pending.clear()
        self.logger.debug(
            "DSP short read channel=%s expected=%s got=%s lag_us=%s",
            dsp.channel_id,
            expected,
            raw_consumed_bytes,
            max(0, dsp.input_us_total - dsp.raw_output_us_total),
        )
        return result

    async def _release_player_channel(self, player_id: str) -> None:
        """Release channel bookkeeping for a player and close its DSP channel."""
        self._player_channel_map.pop(player_id, None)
        self._pending_historical_injections.pop(player_id, None)
        dsp = self._dsp_channels.pop(player_id, None)
        if dsp is not None:
            await self._close_dsp_channel(dsp)

    async def _run_playback(self, media: PlayerMedia) -> None:
        """Run the actual playback in a background task."""
        await run_playback(self, media)

    async def set_members(
        self,
        player_ids_to_add: list[str] | None = None,
        player_ids_to_remove: list[str] | None = None,
    ) -> None:
        """Handle SET_MEMBERS command on the player."""
        self.logger.debug(
            "set_members called: adding %s, removing %s", player_ids_to_add, player_ids_to_remove
        )
        for player_id in player_ids_to_remove or []:
            player = self.mass.players.get(player_id, True)
            player = cast("SendspinPlayer", player)  # For type checking
            await self.api.group.remove_client(player.api)
            await self._release_player_channel(player_id)
        for player_id in player_ids_to_add or []:
            self._pending_join_members.add(player_id)
            join_started = time.perf_counter()
            try:
                if self._pcm_format is not None and self._push_stream is not None:
                    await prepare_dsp_for_join(self, player_id)
                player = self.mass.players.get(player_id, True)
                player = cast("SendspinPlayer", player)  # For type checking
                await self.api.group.add_client(player.api)
                join_elapsed_ms = round((time.perf_counter() - join_started) * 1000, 1)
                self.logger.debug("Joined player %s in %sms", player_id, join_elapsed_ms)
            except Exception:
                await self._release_player_channel(player_id)
                raise
            finally:
                self._pending_join_members.discard(player_id)
        # self.group_members will be updated by the group event callback

    async def _send_album_artwork(self, current_item: QueueItem) -> str | None:
        """
        Send album artwork to the sendspin group.

        Args:
            current_item: The current queue item.
        """
        artwork_url = None
        if current_item.image is not None:
            artwork_url = self.mass.metadata.get_image_url(current_item.image)

        if artwork_url != self.last_sent_artwork_url:
            # Image changed, resend the artwork
            self.last_sent_artwork_url = artwork_url
            if artwork_url is not None and current_item.media_item is not None:
                image_data = await self.mass.metadata.get_image_data_for_item(
                    current_item.media_item
                )
                if image_data is not None:
                    image = await asyncio.to_thread(Image.open, BytesIO(image_data))
                    if (artwork_role := self._artwork_role) is not None:
                        await artwork_role.set_album_artwork(image)
            # Clear artwork if none available
            elif (artwork_role := self._artwork_role) is not None:
                await artwork_role.set_album_artwork(None)

        return artwork_url

    async def _send_artist_artwork(self, current_item: QueueItem) -> None:
        """
        Send artist artwork to the sendspin group.

        Args:
            current_item: The current queue item.
        """
        # Extract primary artist if available
        artist_artwork_url = None
        if current_item.media_item is not None and hasattr(current_item.media_item, "artists"):
            artists = getattr(current_item.media_item, "artists", None)
            if artists and len(artists) > 0:
                primary_artist = artists[0]
                if hasattr(primary_artist, "image"):
                    artist_image = getattr(primary_artist, "image", None)
                    if artist_image is not None:
                        artist_artwork_url = self.mass.metadata.get_image_url(artist_image)

        if artist_artwork_url != self.last_sent_artist_artwork_url:
            # Artist image changed, resend the artwork
            self.last_sent_artist_artwork_url = artist_artwork_url
            if artist_artwork_url is not None:
                artist_image_data = await self.mass.metadata.get_image_data_for_item(
                    primary_artist, img_type=ImageType.THUMB
                )
                if artist_image_data is not None:
                    artist_image = await asyncio.to_thread(Image.open, BytesIO(artist_image_data))
                    if (artwork_role := self._artwork_role) is not None:
                        await artwork_role.set_artist_artwork(artist_image)
            # Clear artist artwork if none available
            elif (artwork_role := self._artwork_role) is not None:
                await artwork_role.set_artist_artwork(None)

    def _on_player_media_updated(self) -> None:
        """Handle callback when the current media of the player is updated."""
        if self.synced_to is not None:
            # Only leader sends metadata
            return

        if self.current_media is None:
            # Clear metadata when no media loaded
            if (metadata_role := self._metadata_role) is not None:
                metadata_role.set_metadata(Metadata())
            return
        self.mass.create_task(self.send_current_media_metadata())

    async def send_current_media_metadata(self) -> None:
        """Send the current media metadata to the sendspin group."""
        if not self.available:
            return
        current_media = self.current_media
        if current_media is None:
            return
        # check if we are playing a MA queue item
        queue_item: QueueItem | None = None
        queue: PlayerQueue | None = None
        if current_media.source_id and current_media.queue_item_id:
            queue = self.mass.player_queues.get(current_media.source_id)
            queue_item = self.mass.player_queues.get_item(
                current_media.source_id, current_media.queue_item_id
            )

        # Send album and artist artwork
        if queue_item:
            await self._send_album_artwork(queue_item)
            await self._send_artist_artwork(queue_item)

        track_duration = current_media.duration or 0
        repeat = SendspinRepeatMode.OFF
        if queue and queue.repeat_mode == RepeatMode.ALL:
            repeat = SendspinRepeatMode.ALL
        elif queue and queue.repeat_mode == RepeatMode.ONE:
            repeat = SendspinRepeatMode.ONE

        shuffle = queue.shuffle_enabled if queue else False

        metadata = Metadata(
            title=current_media.title,
            artist=current_media.artist,
            album_artist=None,
            album=current_media.album,
            artwork_url=current_media.image_url,
            year=None,
            track=None,
            track_duration=track_duration * 1000 if track_duration is not None else None,
            track_progress=int(current_media.corrected_elapsed_time * 1000)
            if current_media.corrected_elapsed_time
            else 0,
            playback_speed=1000,
            repeat=repeat,
            shuffle=shuffle,
        )

        # Send metadata to the group
        if (metadata_role := self._metadata_role) is not None:
            metadata_role.set_metadata(metadata)

    async def get_config_entries(
        self,
        action: str | None = None,
        values: dict[str, ConfigValueType] | None = None,
    ) -> list[ConfigEntry]:
        """Return all (provider/player specific) Config Entries for the player."""
        default_entries = await super().get_config_entries(action=action, values=values)
        entries = [
            *default_entries,
            ConfigEntry.from_dict(
                {**CONF_ENTRY_FLOW_MODE.to_dict(), "default_value": True, "hidden": True}
            ),
            CONF_ENTRY_OUTPUT_CODEC_HIDDEN,
            CONF_ENTRY_HTTP_PROFILE_HIDDEN,
            ConfigEntry.from_dict({**CONF_ENTRY_SAMPLE_RATES.to_dict(), "hidden": True}),
        ]

        # Build dynamic format options from player's supported formats
        player_role = self._player_role
        if player_role is not None:
            supported_formats = player_role.get_supported_formats()
            if supported_formats:
                format_options = [
                    ConfigValueOption(
                        title="Automatic (let client decide)",
                        value=SENDSPIN_FORMAT_AUTOMATIC,
                    ),
                ]
                for fmt in supported_formats:
                    format_options.append(
                        ConfigValueOption(
                            title=format_to_display_string(fmt),
                            value=format_to_option_value(fmt),
                        )
                    )
                entries.append(
                    ConfigEntry(
                        key=CONF_PREFERRED_SENDSPIN_FORMAT,
                        type=ConfigEntryType.STRING,
                        label="Preferred audio format",
                        description="Select the audio format to use for playback on this player.",
                        category="audio",
                        default_value=SENDSPIN_FORMAT_AUTOMATIC,
                        options=format_options,
                    )
                )

        return entries

    async def on_unload(self) -> None:
        """Handle logic when the player is unloaded from the Player controller."""
        await super().on_unload()
        self.unsub_event_cb()
        self.unsub_group_event_cb()
