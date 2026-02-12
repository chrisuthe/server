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
from uuid import UUID, uuid5

from aiosendspin.models import AudioCodec, MediaCommand
from aiosendspin.models.types import PlaybackStateType
from aiosendspin.models.types import RepeatMode as SendspinRepeatMode
from aiosendspin.server import (
    ClientEvent,
    GroupEvent,
    GroupStateChangedEvent,
    SendspinGroup,
    VolumeChangedEvent,
)
from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from aiosendspin.server.client import DisconnectBehaviour
from aiosendspin.server.events import ClientGroupChangedEvent
from aiosendspin.server.group import (
    GroupDeletedEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
)
from aiosendspin.server.push_stream import CachedPCMChunk
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
    PlayerGroupRole,
)
from aiosendspin.server.roles.metadata.state import Metadata
from aiosendspin.server.roles.player.types import PlayerRoleProtocol
from music_assistant_models.config_entries import ConfigEntry, ConfigValueOption
from music_assistant_models.constants import PLAYER_CONTROL_NONE
from music_assistant_models.enums import (
    ConfigEntryType,
    ContentType,
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
)
from music_assistant.helpers.audio import get_player_filter_params
from music_assistant.helpers.ffmpeg import FFMpeg
from music_assistant.models.player import Player, PlayerMedia

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
    from collections.abc import AsyncGenerator

    from aiosendspin.models.player import SupportedAudioFormat
    from aiosendspin.server.client import SendspinClient
    from aiosendspin.server.push_stream import PushStream
    from music_assistant_models.config_entries import ConfigValueType
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

    from .provider import SendspinProvider

# Namespace for generating deterministic DSP channel UUIDs from filter params
_DSP_CHANNEL_NAMESPACE = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


@dataclass
class _DSPChannel:
    """A DSP processing channel backed by an ffmpeg process."""

    channel_id: UUID
    filter_params: list[str]
    ffmpeg: FFMpeg
    output_channels: int  # 1 for mono (left/right mode), 2 for stereo
    pending: bytearray = field(default_factory=bytearray)


def _needs_dsp_channel(
    mass: MusicAssistant, player_id: str, pcm_format: AudioFormat
) -> tuple[str, ...] | None:
    """Return filter_key if player needs DSP, else None."""
    filter_params = get_player_filter_params(mass, player_id, pcm_format, pcm_format)
    return tuple(filter_params) if filter_params else None


class SendspinPlayer(Player):
    """A sendspin audio player in Music Assistant."""

    api: SendspinClient
    unsub_event_cb: Callable[[], None]
    unsub_group_event_cb: Callable[[], None]
    unsub_controller_event_cb: Callable[[], None] | None = None
    last_sent_artwork_url: str | None = None
    last_sent_artist_artwork_url: str | None = None
    _playback_task: asyncio.Task[None] | None = None
    _push_stream: PushStream | None = None
    _dsp_channels: dict[tuple[str, ...], _DSPChannel]
    _prepared_dsp_channels: dict[tuple[str, ...], _DSPChannel]
    _player_channel_map: dict[str, UUID]
    _pending_join_members: set[str]
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
        self._subscribe_to_controller_events(sendspin_client.group)
        sendspin_client.group.set_supported_commands(SUPPORTED_GROUP_COMMANDS)

        self._dsp_channels = {}
        self._prepared_dsp_channels = {}
        self._player_channel_map = {}
        self._pending_join_members = set()

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
    def _player_group_role(self) -> PlayerGroupRole | None:
        """Get the PlayerGroupRole for this player's group."""
        role = self.api.group.group_role("player")
        if isinstance(role, PlayerGroupRole):
            return role
        return None

    @property
    def _controller_role(self) -> ControllerGroupRole | None:
        """Get the ControllerGroupRole for this player's group."""
        role = self.api.group.group_role("controller")
        if isinstance(role, ControllerGroupRole):
            return role
        return None

    def _subscribe_to_controller_events(self, group: SendspinGroup) -> None:
        """Subscribe to controller events from the group's ControllerGroupRole."""
        if self.unsub_controller_event_cb is not None:
            self.unsub_controller_event_cb()
            self.unsub_controller_event_cb = None
        controller_role = group.group_role("controller")
        self.logger.debug(
            "Subscribing to controller events: group=%s, controller_role=%s",
            group.group_id,
            controller_role,
        )
        if isinstance(controller_role, ControllerGroupRole):
            self.unsub_controller_event_cb = controller_role.add_event_listener(
                self.controller_event_cb
            )

    def controller_event_cb(self, event: ControllerEvent) -> None:
        """Event callback registered to the ControllerGroupRole."""
        self.logger.debug(
            "Received ControllerEvent: %s, synced_to=%s, player_id=%s",
            event,
            self.synced_to,
            self.player_id,
        )
        if self.synced_to is not None:
            # Only leader handles controller events
            return
        self.mass.create_task(self._handle_controller_event(event))

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
                # Re-subscribe to controller events for the new group
                self._subscribe_to_controller_events(new_group)
                # Sync playback state from the new group
                match new_group.state:
                    case PlaybackStateType.PLAYING:
                        self._attr_playback_state = PlaybackState.PLAYING
                    case PlaybackStateType.PAUSED:
                        self._attr_playback_state = PlaybackState.PAUSED
                    case PlaybackStateType.STOPPED:
                        self._attr_playback_state = PlaybackState.IDLE
                # Update in case this is a newly created group
                new_group.set_supported_commands(SUPPORTED_GROUP_COMMANDS)
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
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._playback_task
        # We don't care if we stopped the stream or it was already stopped
        await self.api.group.stop()
        # Clear the playback task reference (group.stop() handles stopping the stream)
        self._playback_task = None
        self._attr_current_media = None
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

    def _get_player_role(self) -> PlayerRoleProtocol | None:
        """Get the player role for this client (not group role)."""
        for role in self.api.roles_by_family("player"):
            if isinstance(role, PlayerRoleProtocol):
                return role
        return None

    async def on_config_updated(self) -> None:
        """Apply preferred format when config changes."""
        await self._apply_preferred_format()

    async def _apply_preferred_format(self) -> None:
        """Read config and call set_preferred_format() if not automatic."""
        player_role = self._get_player_role()
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
        success = player_role.set_preferred_format(audio_format, codec)
        if success:
            self.logger.debug(
                "Set preferred audio format to %s %s for player %s",
                codec.name,
                audio_format,
                self.display_name,
            )
        else:
            self.logger.warning(
                "Failed to set preferred audio format %s %s for player %s",
                codec.name,
                audio_format,
                self.display_name,
            )

    async def _throttle_playback(
        self,
        chunk_size: int,
        pcm_format: AudioFormat,
        play_start_us: int,
    ) -> int:
        """Throttle audio production to stay at most ~6s ahead of realtime.

        :param chunk_size: Size of the current audio chunk in bytes.
        :param pcm_format: The PCM audio format.
        :param play_start_us: The play start timestamp from commit_audio().
        :return: The playback end timestamp in microseconds.
        """
        bytes_per_second = (
            pcm_format.sample_rate * pcm_format.channels * (pcm_format.bit_depth // 8)
        )
        chunk_duration_us = int(chunk_size * 1_000_000 / bytes_per_second)
        playback_end_us = play_start_us + max(chunk_duration_us, 0)
        target_ahead_us = 6_000_000
        while True:
            now_us = int(self.mass.loop.time() * 1_000_000)
            ahead_us = playback_end_us - now_us
            if ahead_us <= target_ahead_us:
                break
            await asyncio.sleep(min((ahead_us - target_ahead_us) / 1_000_000, 1.0))
        return playback_end_us

    async def _create_dsp_channel(
        self,
        filter_key: tuple[str, ...],
        pcm_format: AudioFormat,
    ) -> _DSPChannel:
        """Create a new DSP channel with an ffmpeg process.

        :param filter_key: Tuple of ffmpeg filter parameters.
        :param pcm_format: Input PCM audio format.
        """
        filter_params = list(filter_key)
        output_channels = 1 if any("pan=mono" in p for p in filter_params) else 2
        output_format = AudioFormat(
            content_type=pcm_format.content_type,
            sample_rate=pcm_format.sample_rate,
            bit_depth=pcm_format.bit_depth,
            channels=output_channels,
        )
        channel_id = uuid5(_DSP_CHANNEL_NAMESPACE, str(filter_key))
        ffmpeg = FFMpeg(
            audio_input="-",
            input_format=pcm_format,
            output_format=output_format,
            filter_params=filter_params,
        )
        await ffmpeg.start()
        return _DSPChannel(
            channel_id=channel_id,
            filter_params=filter_params,
            ffmpeg=ffmpeg,
            output_channels=output_channels,
        )

    async def _setup_dsp_channels(self, pcm_format: AudioFormat) -> None:
        """Set up DSP channels for all current group members at playback start."""
        member_ids = [self.player_id, *self._attr_group_members]
        for member_id in member_ids:
            filter_key = _needs_dsp_channel(self.mass, member_id, pcm_format)
            if filter_key is None:
                self._player_channel_map[member_id] = MAIN_CHANNEL
                continue
            if filter_key not in self._dsp_channels:
                self._dsp_channels[filter_key] = await self._create_dsp_channel(
                    filter_key, pcm_format
                )
            self._player_channel_map[member_id] = self._dsp_channels[filter_key].channel_id

    def _get_channel_cache_tail_us(self, channel_id: UUID) -> int | None:
        """Return cached channel tail timestamp (end of newest chunk), if available."""
        if self._push_stream is None:
            return None
        cached_chunks = self._push_stream.get_cached_pcm_chunks(channel_id)
        if not cached_chunks:
            return None
        last_chunk = cached_chunks[-1]
        return last_chunk.timestamp_us + last_chunk.duration_us

    async def _close_dsp_channel(self, dsp: _DSPChannel) -> None:
        """Close a DSP channel and clear all related stream cache state."""
        if self._push_stream is not None:
            self._push_stream.disable_pcm_cache_for_channel(dsp.channel_id)
            self._push_stream._channel_timing.pop(dsp.channel_id, None)
        with suppress(Exception):
            await dsp.ffmpeg.close()

    def _shift_channel_pcm_cache(self, channel_id: UUID, shift_us: int) -> None:
        """Shift all cached PCM timestamps for a channel by a positive offset."""
        if self._push_stream is None or shift_us <= 0:
            return
        channel_int = channel_id.int
        cached = self._push_stream._pcm_chunk_cache.get(channel_int)
        if not cached:
            return
        shifted_cache: deque[CachedPCMChunk] = deque()
        for chunk in cached:
            shifted_cache.append(
                CachedPCMChunk(
                    timestamp_us=chunk.timestamp_us + shift_us,
                    duration_us=chunk.duration_us,
                    pcm_data=chunk.pcm_data,
                    sample_rate=chunk.sample_rate,
                    bit_depth=chunk.bit_depth,
                    channels=chunk.channels,
                )
            )
        self._push_stream._pcm_chunk_cache[channel_int] = shifted_cache

    def _activate_prepared_dsp_channel(self, filter_key: tuple[str, ...]) -> None:
        """Move a prepared DSP channel to active playback and align its timing."""
        if self._push_stream is None:
            return
        dsp = self._prepared_dsp_channels.pop(filter_key, None)
        if dsp is None:
            return

        self._dsp_channels[filter_key] = dsp
        cache_tail_us = self._get_channel_cache_tail_us(dsp.channel_id)
        main_timing_us = self._push_stream._channel_timing.get(MAIN_CHANNEL)
        if (
            main_timing_us is not None
            and cache_tail_us is not None
            and cache_tail_us < main_timing_us
        ):
            self._shift_channel_pcm_cache(dsp.channel_id, main_timing_us - cache_tail_us)
            cache_tail_us = main_timing_us
        if main_timing_us is not None and cache_tail_us is not None:
            self._push_stream._channel_timing[dsp.channel_id] = max(main_timing_us, cache_tail_us)
        elif cache_tail_us is not None:
            self._push_stream._channel_timing[dsp.channel_id] = cache_tail_us
        elif main_timing_us is not None:
            self._push_stream._channel_timing[dsp.channel_id] = main_timing_us

    async def _drain_stdout(self, dsp: _DSPChannel, target: int) -> None:
        """Read ffmpeg stdout into the pending buffer until target bytes or timeout.

        :param dsp: DSP channel to drain.
        :param target: Target number of bytes to accumulate.
        """
        while len(dsp.pending) < target:
            try:
                data = await asyncio.wait_for(dsp.ffmpeg.read(target), timeout=0.01)
                if not data:
                    break
                dsp.pending.extend(data)
            except TimeoutError:
                break

    async def _process_dsp_chunk(
        self, dsp: _DSPChannel, chunk: bytes, pcm_format: AudioFormat
    ) -> bytes:
        """Process a PCM chunk through the DSP channel's ffmpeg.

        :param dsp: DSP channel to process through.
        :param chunk: Raw PCM audio data.
        :param pcm_format: Input PCM format (used to calculate expected output size).
        """
        expected = len(chunk) // 2 if dsp.output_channels == 1 else len(chunk)
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    dsp.ffmpeg.write(chunk),
                    self._drain_stdout(dsp, expected),
                ),
                timeout=5.0,
            )
        except Exception:
            self.logger.warning("DSP failed for channel %s", dsp.channel_id, exc_info=True)
            # Keep timeline continuity by filling this chunk with silence.
            return b"\x00" * expected
        if len(dsp.pending) >= expected:
            result = bytes(dsp.pending[:expected])
            del dsp.pending[:expected]
            return result
        # First chunk: limiter withheld ~960 bytes (5ms lookahead). Pad with silence.
        result = bytes(dsp.pending) + b"\x00" * (expected - len(dsp.pending))
        dsp.pending.clear()
        return result

    async def _sync_dsp_channels(self) -> None:
        """Sync DSP channels with current group membership.

        Handles departed members only. New members are pre-configured in
        set_members (before add_client) so the channel resolver is correct
        when aiosendspin performs late-join catch-up.
        """
        assert self._push_stream is not None
        assert self._pcm_format is not None
        current_members = {self.player_id, *self._attr_group_members}
        known_members = set(self._player_channel_map.keys())

        # Handle new members not yet configured (e.g., initial setup race)
        for member_id in current_members - known_members:
            filter_key = _needs_dsp_channel(self.mass, member_id, self._pcm_format)
            if filter_key is None:
                self._player_channel_map[member_id] = MAIN_CHANNEL
                continue
            if filter_key in self._prepared_dsp_channels:
                self._player_channel_map[member_id] = self._prepared_dsp_channels[
                    filter_key
                ].channel_id
                continue
            if filter_key not in self._dsp_channels:
                dsp = await self._create_dsp_channel(filter_key, self._pcm_format)
                self._dsp_channels[filter_key] = dsp
                self._push_stream.enable_pcm_cache_for_channel(dsp.channel_id)
            self._player_channel_map[member_id] = self._dsp_channels[filter_key].channel_id

        # Handle departed members
        departed_members = known_members - current_members - self._pending_join_members
        for member_id in departed_members:
            channel_id = self._player_channel_map.pop(member_id)
            if channel_id == MAIN_CHANNEL:
                continue
            if any(v == channel_id for v in self._player_channel_map.values()):
                continue
            for key, dsp in list(self._prepared_dsp_channels.items()):
                if dsp.channel_id == channel_id:
                    await self._close_dsp_channel(dsp)
                    del self._prepared_dsp_channels[key]
                    break
            for key, dsp in list(self._dsp_channels.items()):
                if dsp.channel_id == channel_id:
                    await self._close_dsp_channel(dsp)
                    del self._dsp_channels[key]
                    break

    async def _prepare_dsp_for_join(self, player_id: str) -> None:  # noqa: PLR0915
        """Pre-configure DSP channel for a player about to join the group.

        Called from set_members BEFORE add_client so the channel resolver
        returns the correct channel when aiosendspin performs catch-up.
        Also populates the DSP channel's PCM cache from MAIN_CHANNEL cache
        so catch-up uses DSP-processed audio.

        :param player_id: ID of the player about to join.
        """
        if self._push_stream is None or self._pcm_format is None:
            return

        pcm_format = self._pcm_format
        filter_key = _needs_dsp_channel(self.mass, player_id, pcm_format)
        if filter_key is None:
            self._player_channel_map[player_id] = MAIN_CHANNEL
            return

        if filter_key in self._dsp_channels:
            # Reuse existing channel (shared DSP config)
            self._player_channel_map[player_id] = self._dsp_channels[filter_key].channel_id
            return
        if filter_key in self._prepared_dsp_channels:
            self._player_channel_map[player_id] = self._prepared_dsp_channels[filter_key].channel_id
            return

        # Create new DSP channel and bootstrap its PCM cache
        prep_started = time.perf_counter()
        dsp = await self._create_dsp_channel(filter_key, pcm_format)
        self._push_stream.enable_pcm_cache_for_channel(dsp.channel_id)

        # Process cached MAIN_CHANNEL PCM through the new DSP ffmpeg.
        # Collect all results before exposing this channel to the live loop.
        target_us = self._push_stream.get_late_join_target_timestamp_us()
        dsp_fmt = SendspinAudioFormat(
            pcm_format.sample_rate, pcm_format.bit_depth, dsp.output_channels
        )

        processed_chunks: list[tuple[bytes, int, int]] = []  # (pcm, timestamp_us, duration_us)
        source_tail_us: int | None = None
        max_catchup_loops = 8
        for _ in range(max_catchup_loops):
            cached_chunks = self._push_stream.get_cached_pcm_chunks(MAIN_CHANNEL)
            if source_tail_us is None:
                eligible = [c for c in cached_chunks if c.timestamp_us + c.duration_us > target_us]
            else:
                eligible = [c for c in cached_chunks if c.timestamp_us >= source_tail_us]
            if not eligible:
                break

            for cached in eligible:
                processed = await self._process_dsp_chunk(dsp, cached.pcm_data, pcm_format)
                bytes_per_sample = pcm_format.bit_depth // 8
                frame_stride = bytes_per_sample * dsp.output_channels
                sample_count = len(processed) // frame_stride
                duration_us = int(sample_count * 1_000_000 / pcm_format.sample_rate)
                processed_chunks.append((processed, cached.timestamp_us, duration_us))
                source_tail_us = cached.timestamp_us + cached.duration_us

            main_timing_us = self._push_stream._channel_timing.get(MAIN_CHANNEL)
            if main_timing_us is None or source_tail_us is None or source_tail_us >= main_timing_us:
                break
            await asyncio.sleep(0)

        # Fallback: if we could not process up to main tail, shift to keep timeline continuity.
        if processed_chunks and MAIN_CHANNEL in self._push_stream._channel_timing:
            main_timing_us = self._push_stream._channel_timing[MAIN_CHANNEL]
            processed_end_us = processed_chunks[-1][1] + processed_chunks[-1][2]
            if processed_end_us < main_timing_us:
                shift_us = main_timing_us - processed_end_us
                self.logger.warning(
                    "DSP prep lagged live tail by %sms for %s; shifting catch-up cache forward",
                    round(shift_us / 1000, 1),
                    player_id,
                )
                processed_chunks = [
                    (pcm_data, timestamp_us + shift_us, duration_us)
                    for pcm_data, timestamp_us, duration_us in processed_chunks
                ]

        prep_elapsed_ms = round((time.perf_counter() - prep_started) * 1000, 1)
        processed_audio_us = sum(
            duration_us for _pcm_data, _timestamp_us, duration_us in processed_chunks
        )
        main_tail_us = self._push_stream._channel_timing.get(MAIN_CHANNEL)
        processed_tail_us = (
            processed_chunks[-1][1] + processed_chunks[-1][2] if processed_chunks else None
        )
        tail_gap_ms = (
            round((main_tail_us - processed_tail_us) / 1000, 1)
            if main_tail_us is not None and processed_tail_us is not None
            else None
        )
        self.logger.debug(
            "Prepared DSP channel %s for %s in %sms (chunks=%s audio_ms=%s tail_gap_ms=%s)",
            dsp.channel_id,
            player_id,
            prep_elapsed_ms,
            len(processed_chunks),
            round(processed_audio_us / 1000, 1),
            tail_gap_ms,
        )

        # Populate DSP PCM cache directly so catch-up encoding can use it
        if processed_chunks:
            cache_deque: deque[CachedPCMChunk] = deque()
            for pcm_data, timestamp_us, duration_us in processed_chunks:
                cache_deque.append(
                    CachedPCMChunk(
                        timestamp_us=timestamp_us,
                        duration_us=duration_us,
                        pcm_data=pcm_data,
                        sample_rate=dsp_fmt.sample_rate,
                        bit_depth=dsp_fmt.bit_depth,
                        channels=dsp_fmt.channels,
                    )
                )
            self._push_stream._pcm_chunk_cache[dsp.channel_id.int] = cache_deque

        # Keep this channel prepared-but-inactive until add_client completes.
        self._prepared_dsp_channels[filter_key] = dsp
        self._player_channel_map[player_id] = dsp.channel_id

    async def _run_playback(self, media: PlayerMedia) -> None:  # noqa: PLR0915
        """Run the actual playback in a background task."""
        audio_source: AsyncGenerator[bytes, None] | None = None
        playback_end_us: int | None = None
        cancelled = False
        errored = False
        try:
            pcm_format = AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            )
            self._pcm_format = pcm_format
            sendspin_fmt = SendspinAudioFormat(
                sample_rate=pcm_format.sample_rate,
                bit_depth=pcm_format.bit_depth,
                channels=pcm_format.channels,
            )

            # Set up per-player DSP channels
            await self._setup_dsp_channels(pcm_format)

            def channel_resolver(player_id: str) -> UUID:
                return self._player_channel_map.get(player_id, MAIN_CHANNEL)

            self._push_stream = self.api.group.start_stream(channel_resolver=channel_resolver)
            for dsp in self._dsp_channels.values():
                self._push_stream.enable_pcm_cache_for_channel(dsp.channel_id)

            audio_source = self.mass.streams.get_stream(media, pcm_format)

            async for chunk in audio_source:
                if self._push_stream is None or self._push_stream.is_stopped:
                    break

                await self._sync_dsp_channels()

                # MAIN_CHANNEL: raw PCM (always present)
                self._push_stream.prepare_audio(chunk, sendspin_fmt)

                # DSP channels: processed PCM
                for dsp in self._dsp_channels.values():
                    processed = await self._process_dsp_chunk(dsp, chunk, pcm_format)
                    dsp_fmt = SendspinAudioFormat(
                        pcm_format.sample_rate,
                        pcm_format.bit_depth,
                        dsp.output_channels,
                    )
                    self._push_stream.prepare_audio(processed, dsp_fmt, channel_id=dsp.channel_id)

                play_start_us = await self._push_stream.commit_audio()
                playback_end_us = await self._throttle_playback(
                    len(chunk), pcm_format, play_start_us
                )

        except asyncio.CancelledError:
            cancelled = True
            self.logger.debug("Playback cancelled for player %s", self.display_name)
            raise
        except Exception:
            errored = True
            self.logger.exception("Error during playback for player %s", self.display_name)
            raise
        finally:
            # Stop the stream FIRST so clients stop immediately
            if (
                not cancelled
                and not errored
                and playback_end_us is not None
                and self._push_stream is not None
                and not self._push_stream.is_stopped
            ):
                with suppress(Exception):
                    await self.api.group.stop(stop_time_us=playback_end_us)
            else:
                with suppress(Exception):
                    self.api.group.stop_stream()
            self._push_stream = None
            self._pcm_format = None
            # Then clean up audio source and DSP processes
            if audio_source is not None:
                with suppress(Exception):
                    await audio_source.aclose()
            for dsp in self._dsp_channels.values():
                with suppress(Exception):
                    await dsp.ffmpeg.close()
            for dsp in self._prepared_dsp_channels.values():
                with suppress(Exception):
                    await dsp.ffmpeg.close()
            self._dsp_channels.clear()
            self._prepared_dsp_channels.clear()
            self._player_channel_map.clear()
            self._pending_join_members.clear()

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
        for player_id in player_ids_to_add or []:
            self._pending_join_members.add(player_id)
            filter_key: tuple[str, ...] | None = None
            join_started = time.perf_counter()
            if self._pcm_format is not None:
                filter_key = _needs_dsp_channel(self.mass, player_id, self._pcm_format)
            try:
                # Pre-configure DSP channel before add_client so the channel resolver
                # is correct when aiosendspin performs late-join catch-up.
                await self._prepare_dsp_for_join(player_id)
                player = self.mass.players.get(player_id, True)
                player = cast("SendspinPlayer", player)  # For type checking
                await self.api.group.add_client(player.api)
                if filter_key is not None:
                    self._activate_prepared_dsp_channel(filter_key)
                join_elapsed_ms = round((time.perf_counter() - join_started) * 1000, 1)
                self.logger.debug("Joined player %s in %sms", player_id, join_elapsed_ms)
            except Exception:
                self._player_channel_map.pop(player_id, None)
                if filter_key is not None and filter_key in self._prepared_dsp_channels:
                    dsp = self._prepared_dsp_channels.pop(filter_key)
                    await self._close_dsp_channel(dsp)
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
            album_artist=None,  # TODO: extract from optional queue item
            album=current_media.album,
            artwork_url=current_media.image_url,
            year=None,  # TODO: extract from optional queue item
            track=None,  # TODO: extract from optional queue item
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
        player_role = self._get_player_role()
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
        if self.unsub_controller_event_cb is not None:
            self.unsub_controller_event_cb()
