"""Sendspin Player implementation."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable
from contextlib import suppress
from io import BytesIO
from typing import TYPE_CHECKING, cast
from uuid import UUID

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
from aiosendspin.server.channels import MAIN_CHANNEL, ChannelResolver
from aiosendspin.server.client import DisconnectBehaviour
from aiosendspin.server.events import ClientGroupChangedEvent
from aiosendspin.server.group import (
    GroupDeletedEvent,
    GroupMemberAddedEvent,
    GroupMemberRemovedEvent,
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
from music_assistant.helpers.ffmpeg import get_ffmpeg_stream
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
    from aiosendspin.server.push_stream import CachedPCMChunk, PushStream
    from music_assistant_models.config_entries import ConfigValueType
    from music_assistant_models.dsp import DSPConfig
    from music_assistant_models.player_queue import PlayerQueue
    from music_assistant_models.queue_item import QueueItem

    from music_assistant.mass import MusicAssistant

    from .provider import SendspinProvider


def _hash_dsp_config(dsp_config: DSPConfig) -> str:
    """Create a deterministic hash from a DSP configuration.

    Players with identical DSP settings produce the same hash, allowing
    them to share a single DSP processing channel.

    :param dsp_config: The DSP configuration to hash.
    :return: Hex digest string.
    """
    h = hashlib.md5(usedforsecurity=False)
    h.update(f"input_gain={dsp_config.input_gain}".encode())
    h.update(f"output_gain={dsp_config.output_gain}".encode())
    for f in dsp_config.filters:
        if not f.enabled:
            continue
        # Use the dict representation for a stable, complete hash of filter params
        h.update(str(f.to_dict()).encode())
    return h.hexdigest()


def _build_channel_map(
    mass: MusicAssistant,
    leader_player_id: str,
    group_member_ids: list[str],
    player_to_channel: dict[str, UUID] | None = None,
) -> tuple[ChannelResolver, dict[UUID, tuple[DSPConfig, list[str]]], dict[str, UUID]]:
    """Build a channel map for grouped playback with per-player DSP.

    :param mass: MusicAssistant instance.
    :param leader_player_id: Player ID of the group leader.
    :param group_member_ids: Player IDs of group members (excluding leader).
    :param player_to_channel: Optional mutable dict to populate/update. If provided,
        the resolver closure captures this dict, allowing dynamic updates.
    :return: Tuple of (channel_resolver, unique_channels, player_to_channel) where
        unique_channels maps channel_id -> (dsp_config, player_ids) for channels
        that need DSP processing.
    """
    all_player_ids = [leader_player_id, *group_member_ids]

    # Map each unique DSP hash to a channel UUID and config
    hash_to_channel: dict[str, UUID] = {}
    hash_to_config: dict[str, DSPConfig] = {}
    if player_to_channel is None:
        player_to_channel = {}

    for player_id in all_player_ids:
        dsp_config = mass.config.get_player_dsp_config(player_id)

        # Check if DSP is enabled and has any active filters/gains
        has_active_dsp = dsp_config.enabled and (
            dsp_config.input_gain != 0
            or dsp_config.output_gain != 0
            or any(f.enabled for f in dsp_config.filters)
        )

        if not has_active_dsp:
            player_to_channel[player_id] = MAIN_CHANNEL
            continue

        # Remove disabled filters for consistent hashing
        dsp_config.filters = [f for f in dsp_config.filters if f.enabled]
        config_hash = _hash_dsp_config(dsp_config)

        if config_hash not in hash_to_channel:
            # Generate a deterministic UUID from the hash
            channel_uuid = UUID(
                hashlib.md5(config_hash.encode(), usedforsecurity=False).hexdigest()
            )
            hash_to_channel[config_hash] = channel_uuid
            hash_to_config[config_hash] = dsp_config

        player_to_channel[player_id] = hash_to_channel[config_hash]

    # Build unique_channels: channel_id -> (dsp_config, [player_ids])
    unique_channels: dict[UUID, tuple[DSPConfig, list[str]]] = {}
    for config_hash, channel_id in hash_to_channel.items():
        players_for_channel = [pid for pid, cid in player_to_channel.items() if cid == channel_id]
        unique_channels[channel_id] = (hash_to_config[config_hash], players_for_channel)

    def channel_resolver(player_id: str) -> UUID:
        return player_to_channel.get(player_id, MAIN_CHANNEL)

    return channel_resolver, unique_channels, player_to_channel


def _dsp_config_to_filter_params(
    mass: MusicAssistant,
    player_id: str,
    pcm_format: AudioFormat,
) -> list[str]:
    """Get ffmpeg filter parameters for a player's DSP configuration.

    Reuses the existing ``get_player_filter_params`` helper which handles
    input/output gain, EQ filters, channel mixing, and the output limiter.

    :param mass: MusicAssistant instance.
    :param player_id: The player to get filter params for.
    :param pcm_format: The PCM audio format (used as both input and output format).
    :return: List of ffmpeg filter parameter strings.
    """
    return get_player_filter_params(mass, player_id, pcm_format, pcm_format)


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

        self.logger = self.provider.logger.getChild(player_id)
        # init some static variables
        self._attr_name = sendspin_client.name
        self._attr_type = PlayerType.PLAYER
        self._attr_supported_features = {
            PlayerFeature.SET_MEMBERS,
            PlayerFeature.MULTI_DEVICE_DSP,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
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

    def _setup_dsp_channels(
        self,
        push_stream: PushStream,
        pcm_format: AudioFormat,
        sendspin_source_format: SendspinAudioFormat,
    ) -> tuple[dict[UUID, asyncio.Queue[bytes | None]], list[asyncio.Task[None]]]:
        """Set up per-player DSP processing channels.

        :param push_stream: The active PushStream.
        :param pcm_format: The PCM audio format.
        :param sendspin_source_format: The sendspin source format.
        :return: Tuple of (dsp_queues, dsp_tasks).
        """
        _, unique_channels, _ = _build_channel_map(
            self.mass, self.player_id, self._attr_group_members
        )

        dsp_queues: dict[UUID, asyncio.Queue[bytes | None]] = {}
        dsp_tasks: list[asyncio.Task[None]] = []

        if unique_channels:
            self.logger.debug(
                "Multi-device DSP: %d unique DSP channel(s) for %d players",
                len(unique_channels),
                1 + len(self._attr_group_members),
            )

        for channel_id, (_dsp_config, player_ids) in unique_channels.items():
            queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=10)
            dsp_queues[channel_id] = queue
            task = asyncio.create_task(
                self._channel_dsp_processor(
                    channel_id=channel_id,
                    player_id=player_ids[0],
                    raw_queue=queue,
                    push_stream=push_stream,
                    pcm_format=pcm_format,
                    sendspin_format=sendspin_source_format,
                )
            )
            dsp_tasks.append(task)
            push_stream.enable_pcm_cache_for_channel(channel_id)

        return dsp_queues, dsp_tasks

    async def _process_cached_audio_through_dsp(
        self,
        cached_chunks: list[CachedPCMChunk],
        player_id: str,
        pcm_format: AudioFormat,
    ) -> list[bytes]:
        """Process cached PCM chunks through a player's DSP filter chain.

        :param cached_chunks: Cached PCM chunks to process.
        :param player_id: Player ID for DSP config lookup.
        :param pcm_format: PCM audio format.
        :return: List of processed PCM chunks in chronological order.
        """
        filter_params = _dsp_config_to_filter_params(self.mass, player_id, pcm_format)

        if not filter_params:
            return [chunk.pcm_data for chunk in cached_chunks]

        async def chunk_generator() -> AsyncGenerator[bytes, None]:
            for chunk in cached_chunks:
                yield chunk.pcm_data

        processed_chunks: list[bytes] = []
        async for processed_chunk in get_ffmpeg_stream(
            audio_input=chunk_generator(),
            input_format=pcm_format,
            output_format=pcm_format,
            filter_params=filter_params,
        ):
            processed_chunks.append(processed_chunk)

        return processed_chunks

    async def _bootstrap_new_dsp_channel(
        self,
        channel_id: UUID,
        player_id: str,
        push_stream: PushStream,
        pcm_format: AudioFormat,
        sendspin_format: SendspinAudioFormat,
    ) -> tuple[asyncio.Queue[bytes | None], asyncio.Task[None]]:
        """Bootstrap a new DSP channel for a late-joining player with unique DSP config.

        Retrieves MAIN_CHANNEL PCM cache, processes it through the player's DSP,
        and injects the result as historical audio before starting the live DSP
        processor task.

        :param channel_id: UUID of the new channel to create.
        :param player_id: Player ID for DSP config lookup.
        :param push_stream: Active PushStream instance.
        :param pcm_format: PCM format for audio processing.
        :param sendspin_format: Sendspin audio format.
        :return: Tuple of (raw_pcm_queue, dsp_processor_task).
        """
        # Retrieve cached MAIN_CHANNEL PCM and process through DSP
        cached_chunks = push_stream.get_cached_pcm_chunks(MAIN_CHANNEL)
        if cached_chunks:
            try:
                processed = await self._process_cached_audio_through_dsp(
                    cached_chunks, player_id, pcm_format
                )
                sendspin_fmt = SendspinAudioFormat(
                    sample_rate=pcm_format.sample_rate,
                    bit_depth=pcm_format.bit_depth,
                    channels=pcm_format.channels,
                )
                for chunk_pcm in processed:
                    push_stream.prepare_historical_audio(
                        chunk_pcm, sendspin_fmt, channel_id=channel_id
                    )
            except Exception:
                self.logger.exception(
                    "Failed to process historical audio for channel %s (player %s), "
                    "starting without catch-up",
                    channel_id,
                    player_id,
                )

        # Enable PCM caching for the new channel
        push_stream.enable_pcm_cache_for_channel(channel_id)

        # Create queue and DSP processor task for ongoing live audio
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=10)
        task = asyncio.create_task(
            self._channel_dsp_processor(
                channel_id=channel_id,
                player_id=player_id,
                raw_queue=queue,
                push_stream=push_stream,
                pcm_format=pcm_format,
                sendspin_format=sendspin_format,
            )
        )

        return queue, task

    async def _run_playback(self, media: PlayerMedia) -> None:
        """Run the actual playback in a background task."""
        audio_source: AsyncGenerator[bytes, None] | None = None
        playback_end_us: int | None = None
        cancelled = False
        errored = False
        dsp_tasks: list[asyncio.Task[None]] = []
        dsp_queues: dict[UUID, asyncio.Queue[bytes | None]] = {}
        try:
            # Define source PCM format for streaming (what MA yields).
            # aiosendspin only supports 16-bit PCM for now.
            pcm_format = AudioFormat(
                content_type=ContentType.PCM_S16LE,
                sample_rate=48000,
                bit_depth=16,
                channels=2,
            )

            # PushStream expects raw PCM with a PCM source format; encoding happens per-player.
            sendspin_source_format = SendspinAudioFormat(
                sample_rate=pcm_format.sample_rate,
                bit_depth=pcm_format.bit_depth,
                channels=pcm_format.channels,
            )

            # Build channel resolver for per-player DSP routing.
            # player_to_channel is mutable: late joiners can be added dynamically.
            player_to_channel: dict[str, UUID] = {}
            channel_resolver, _, player_to_channel = _build_channel_map(
                self.mass,
                self.player_id,
                self._attr_group_members,
                player_to_channel,
            )

            # Always pass the channel resolver so late joiners with unique DSP
            # can be routed to new channels dynamically.
            self._push_stream = self.api.group.start_stream(
                channel_resolver=channel_resolver,
            )

            # Set up per-player DSP processing channels
            dsp_queues, dsp_tasks = self._setup_dsp_channels(
                self._push_stream, pcm_format, sendspin_source_format
            )

            # Track known group members for late-joiner detection
            known_members = set(self._attr_group_members)
            known_members.add(self.player_id)

            bytes_per_second = (
                pcm_format.sample_rate * pcm_format.channels * (pcm_format.bit_depth // 8)
            )

            # Get raw audio source from Music Assistant (no DSP applied at source level)
            audio_source = self.mass.streams.get_stream(media, pcm_format)

            # Push audio chunks to connected players
            async for chunk in audio_source:
                if self._push_stream is None or self._push_stream.is_stopped:
                    break

                # Send raw PCM to MAIN_CHANNEL (players without DSP)
                self._push_stream.prepare_audio(chunk, sendspin_source_format)

                # Multicast raw PCM to all DSP processor queues
                if dsp_queues:
                    await asyncio.gather(*(q.put(chunk) for q in dsp_queues.values()))

                play_start_us = await self._push_stream.commit_audio()

                # Check for new group members with unique DSP configs
                await self._check_for_new_dsp_channels(
                    known_members,
                    player_to_channel,
                    dsp_queues,
                    dsp_tasks,
                    self._push_stream,
                    pcm_format,
                    sendspin_source_format,
                )

                # Track playback end for graceful stop at end-of-stream
                playback_end_us = play_start_us + int(len(chunk) * 1_000_000 / bytes_per_second)

                await self._push_stream.sleep_to_limit_buffer(30_000_000)

        except asyncio.CancelledError:
            cancelled = True
            self.logger.debug("Playback cancelled for player %s", self.display_name)
            raise
        except Exception:
            errored = True
            self.logger.exception("Error during playback for player %s", self.display_name)
            raise
        finally:
            await self._cleanup_playback(
                dsp_queues,
                dsp_tasks,
                audio_source,
                cancelled=cancelled,
                errored=errored,
                playback_end_us=playback_end_us,
            )

    async def _cleanup_playback(
        self,
        dsp_queues: dict[UUID, asyncio.Queue[bytes | None]],
        dsp_tasks: list[asyncio.Task[None]],
        audio_source: AsyncGenerator[bytes, None] | None,
        *,
        cancelled: bool,
        errored: bool,
        playback_end_us: int | None,
    ) -> None:
        """Clean up playback resources after stream ends.

        :param dsp_queues: Active DSP queues to drain.
        :param dsp_tasks: Active DSP tasks to await.
        :param audio_source: Audio source generator to close.
        :param cancelled: Whether playback was cancelled.
        :param errored: Whether playback errored.
        :param playback_end_us: Scheduled end timestamp for graceful stop.
        """
        # Signal DSP tasks to stop by sending sentinel values
        for queue in dsp_queues.values():
            with suppress(Exception):
                await queue.put(None)
        if dsp_tasks:
            await asyncio.gather(*dsp_tasks, return_exceptions=True)

        # Ensure the source generator is closed to avoid leaking ffmpeg processes.
        if audio_source is not None:
            with suppress(Exception):
                await audio_source.aclose()

        # If we reached end-of-stream normally, schedule stop at the actual end time
        # so clients can play the queued audio. On cancel/error: stop immediately.
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

    async def _check_for_new_dsp_channels(
        self,
        known_members: set[str],
        player_to_channel: dict[str, UUID],
        dsp_queues: dict[UUID, asyncio.Queue[bytes | None]],
        dsp_tasks: list[asyncio.Task[None]],
        push_stream: PushStream,
        pcm_format: AudioFormat,
        sendspin_format: SendspinAudioFormat,
    ) -> None:
        """Detect new group members and bootstrap DSP channels if needed.

        :param known_members: Set of player IDs already known.
        :param player_to_channel: Mutable resolver dict (updated in-place).
        :param dsp_queues: Active DSP queues (updated in-place).
        :param dsp_tasks: Active DSP tasks (updated in-place).
        :param push_stream: Active PushStream instance.
        :param pcm_format: PCM audio format.
        :param sendspin_format: Sendspin audio format.
        """
        current_members = {self.player_id, *self._attr_group_members}
        new_members = current_members - known_members
        if not new_members:
            return

        for new_player_id in new_members:
            known_members.add(new_player_id)

            # Determine which channel this player should use
            dsp_config = self.mass.config.get_player_dsp_config(new_player_id)
            has_active_dsp = dsp_config.enabled and (
                dsp_config.input_gain != 0
                or dsp_config.output_gain != 0
                or any(f.enabled for f in dsp_config.filters)
            )

            if not has_active_dsp:
                player_to_channel[new_player_id] = MAIN_CHANNEL
                continue

            # Generate channel UUID from DSP config hash
            dsp_config.filters = [f for f in dsp_config.filters if f.enabled]
            config_hash = _hash_dsp_config(dsp_config)
            channel_id = UUID(hashlib.md5(config_hash.encode(), usedforsecurity=False).hexdigest())
            player_to_channel[new_player_id] = channel_id

            if channel_id in dsp_queues:
                # Channel already exists, player shares it
                continue

            # New unique DSP config: bootstrap the channel
            self.logger.info(
                "Bootstrapping new DSP channel %s for late-joining player %s",
                channel_id,
                new_player_id,
            )
            try:
                queue, task = await self._bootstrap_new_dsp_channel(
                    channel_id,
                    new_player_id,
                    push_stream,
                    pcm_format,
                    sendspin_format,
                )
                dsp_queues[channel_id] = queue
                dsp_tasks.append(task)
            except Exception:
                self.logger.exception(
                    "Failed to bootstrap DSP channel for player %s",
                    new_player_id,
                )

    async def _channel_dsp_processor(
        self,
        channel_id: UUID,
        player_id: str,
        raw_queue: asyncio.Queue[bytes | None],
        push_stream: PushStream,
        pcm_format: AudioFormat,
        sendspin_format: SendspinAudioFormat,
    ) -> None:
        """Process a single DSP channel by piping raw PCM through ffmpeg.

        Reads raw PCM chunks from the queue, pipes them through an ffmpeg process
        with the player's DSP filter chain, and feeds the processed output to
        the push stream on the assigned channel.

        :param channel_id: The channel UUID to push processed audio to.
        :param player_id: Player ID used to derive DSP filter parameters.
        :param raw_queue: Queue receiving raw PCM chunks (None = end of stream).
        :param push_stream: The PushStream to push processed audio to.
        :param pcm_format: The PCM audio format for input and output.
        :param sendspin_format: The sendspin audio format for push_stream.
        """
        filter_params = _dsp_config_to_filter_params(self.mass, player_id, pcm_format)

        if not filter_params:
            # No active filters; just pass through to channel
            while True:
                chunk = await raw_queue.get()
                if chunk is None or push_stream.is_stopped:
                    break
                push_stream.prepare_audio(chunk, sendspin_format, channel_id=channel_id)
            return

        async def _queue_to_generator() -> AsyncGenerator[bytes, None]:
            """Convert the raw queue into an async generator for ffmpeg input."""
            while True:
                chunk = await raw_queue.get()
                if chunk is None:
                    return
                yield chunk

        try:
            async for processed_chunk in get_ffmpeg_stream(
                audio_input=_queue_to_generator(),
                input_format=pcm_format,
                output_format=pcm_format,
                filter_params=filter_params,
            ):
                if push_stream.is_stopped:
                    break
                push_stream.prepare_audio(processed_chunk, sendspin_format, channel_id=channel_id)
        except Exception:
            self.logger.exception(
                "DSP processor error for channel %s (player %s)", channel_id, player_id
            )

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
            player = self.mass.players.get(player_id, True)
            player = cast("SendspinPlayer", player)  # For type checking
            await self.api.group.add_client(player.api)
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
