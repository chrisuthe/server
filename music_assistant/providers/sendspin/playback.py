"""Playback pipeline helpers for Sendspin players."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from aiosendspin.server.audio import AudioFormat as SendspinAudioFormat
from aiosendspin.server.channels import MAIN_CHANNEL
from music_assistant_models.enums import ContentType, PlaybackState
from music_assistant_models.media_items import AudioFormat

from music_assistant.models.player import PlayerMedia

if TYPE_CHECKING:
    from .player import SendspinPlayer

# Use a frame size that matches ffmpeg's typical internal audio frame granularity
# (4096 samples at 48kHz => 16384 bytes for stereo s16le). This avoids startup
# "short read" behavior that can otherwise cause a permanent offset between the
# main and DSP channels when a DSP channel is (re)created mid-stream.
_FRAME_SAMPLES = 4096
_MAX_BUFFER_AHEAD_US = 5_000_000
_MAIN_PCM_CACHE_RETENTION_US = 10_000_000
_ENABLE_HISTORICAL_DSP_INJECTION = True


@dataclass
class _PreparedCommitFrame:
    """Prepared audio frame for a single commit."""

    seq: int
    main_pcm: bytes
    main_duration_us: int
    dsp_pcm: dict[UUID, tuple[bytes, SendspinAudioFormat]]


@dataclass
class _MainPCMCacheChunk:
    """Main channel chunk kept for late-join DSP preprocessing."""

    timestamp_us: int
    duration_us: int
    pcm_data: bytes


@dataclass
class _HistoricalInjection:
    """Historical DSP audio to inject on next commit."""

    channel_id: UUID
    audio_format: SendspinAudioFormat
    chunks: list[bytes]


def pcm_duration_us(byte_count: int, sample_rate: int, bit_depth: int, channels: int) -> int:
    """Calculate PCM duration from byte count and format."""
    frame_stride = (bit_depth // 8) * channels
    sample_count = byte_count // frame_stride if frame_stride else 0
    return int(sample_count * 1_000_000 / sample_rate) if sample_rate else 0


def _expected_dsp_bytes(
    input_pcm: bytes, bit_depth: int, input_channels: int, output_channels: int
) -> int:
    """Calculate expected DSP PCM byte length for preserved sample count."""
    bytes_per_sample = bit_depth // 8
    input_frame_stride = bytes_per_sample * input_channels
    sample_count = len(input_pcm) // input_frame_stride if input_frame_stride else 0
    return sample_count * bytes_per_sample * output_channels


def _normalize_pcm_size(pcm_data: bytes, expected_size: int) -> bytes:
    """Trim/pad PCM data to expected size."""
    if len(pcm_data) == expected_size:
        return pcm_data
    if len(pcm_data) > expected_size:
        return pcm_data[:expected_size]
    return pcm_data + (b"\x00" * (expected_size - len(pcm_data)))


async def _sync_live_dsp_channels(player: SendspinPlayer, pcm_format: AudioFormat) -> None:
    """Sync active DSP channels with current group members."""
    # Include pending joiners so we don't "drop" their DSP channel between the
    # set_members() call and the moment group membership state is updated.
    current_members = {
        player.player_id,
        *player._attr_group_members,
        *player._pending_join_members,
    }
    for member_id in current_members:
        if (
            member_id not in player._player_channel_map
            or member_id not in player._player_filter_key_map
        ):
            player._resolve_channel_for_player(member_id, pcm_format)

    departed_members = (
        set(player._player_channel_map) - current_members - player._pending_join_members
    )
    for member_id in departed_members:
        await player._release_player_channel(member_id)

    required_filter_keys = {
        filter_key
        for member_id in current_members
        if (filter_key := player._player_filter_key_map.get(member_id)) is not None
    }
    for filter_key in required_filter_keys:
        if filter_key not in player._dsp_channels:
            player._dsp_channels[filter_key] = await player._create_dsp_channel(
                filter_key, pcm_format
            )
    player._active_dsp_filter_keys = required_filter_keys

    for filter_key in list(player._dsp_channels):
        if filter_key in required_filter_keys:
            continue
        if filter_key in player._pending_historical_injections:
            continue
        if any(key == filter_key for key in player._player_filter_key_map.values()):
            continue
        dsp = player._dsp_channels.pop(filter_key)
        await player._close_dsp_channel(dsp)


async def _iter_pcm_frames(
    audio_source: AsyncGenerator[bytes, None], pcm_format: AudioFormat
) -> AsyncGenerator[bytes, None]:
    """Split source PCM into fixed-size sample frames."""
    bytes_per_sample = pcm_format.bit_depth // 8
    frame_stride = bytes_per_sample * pcm_format.channels
    if frame_stride <= 0:
        async for chunk in audio_source:
            if chunk:
                yield chunk
        return

    frame_size = _FRAME_SAMPLES * frame_stride
    pending = bytearray()

    async for chunk in audio_source:
        if not chunk:
            continue
        pending.extend(chunk)
        while len(pending) >= frame_size:
            frame = bytes(pending[:frame_size])
            del pending[:frame_size]
            yield frame
    if pending:
        yield bytes(pending)


async def _prepare_commit_frames(
    player: SendspinPlayer,
    audio_source: AsyncGenerator[bytes, None],
    pcm_format: AudioFormat,
    prepared_queue: asyncio.Queue[_PreparedCommitFrame | None],
) -> None:
    """Read source PCM, process DSP channels, and queue commit-ready frames."""
    seq = 0
    try:
        async for frame in _iter_pcm_frames(audio_source, pcm_format):
            if player._push_stream is None or player._push_stream.is_stopped:
                break
            await _sync_live_dsp_channels(player, pcm_format)

            active_dsps = [
                player._dsp_channels[key]
                for key in player._active_dsp_filter_keys
                if key in player._dsp_channels
            ]
            processed_by_channel: dict[UUID, tuple[bytes, SendspinAudioFormat]] = {}
            if active_dsps:
                processed_frames = await asyncio.gather(
                    *(player._process_dsp_chunk(dsp, frame, pcm_format) for dsp in active_dsps)
                )
                for dsp, processed in zip(active_dsps, processed_frames, strict=True):
                    expected_size = _expected_dsp_bytes(
                        frame, pcm_format.bit_depth, pcm_format.channels, dsp.output_channels
                    )
                    normalized = _normalize_pcm_size(processed, expected_size)
                    dsp_fmt = SendspinAudioFormat(
                        sample_rate=pcm_format.sample_rate,
                        bit_depth=pcm_format.bit_depth,
                        channels=dsp.output_channels,
                    )
                    processed_by_channel[dsp.channel_id] = (normalized, dsp_fmt)

            main_duration_us = pcm_duration_us(
                len(frame),
                pcm_format.sample_rate,
                pcm_format.bit_depth,
                pcm_format.channels,
            )
            await prepared_queue.put(
                _PreparedCommitFrame(
                    seq=seq,
                    main_pcm=frame,
                    main_duration_us=main_duration_us,
                    dsp_pcm=processed_by_channel,
                )
            )
            seq += 1
    finally:
        await prepared_queue.put(None)


def _inject_pending_historical_audio(player: SendspinPlayer) -> bool:
    """Queue pending historical DSP audio into PushStream before live prepare_audio."""
    if player._push_stream is None:
        return False
    injected = False
    for filter_key, injection in list(player._pending_historical_injections.items()):
        if not injection.chunks:
            del player._pending_historical_injections[filter_key]
            continue
        try:
            if not _ENABLE_HISTORICAL_DSP_INJECTION:
                discarded_chunks = len(injection.chunks)
                discarded_bytes = sum(len(chunk) for chunk in injection.chunks)
                player.logger.debug(
                    "Discarded historical DSP audio for channel %s: chunks=%s bytes=%s",
                    injection.channel_id,
                    discarded_chunks,
                    discarded_bytes,
                )
                continue

            for chunk in injection.chunks:
                player._push_stream.prepare_historical_audio(
                    chunk,
                    injection.audio_format,
                    channel_id=injection.channel_id,
                )
            injected = True
            player.logger.debug(
                "Injected historical DSP audio for channel %s: chunks=%s bytes=%s",
                injection.channel_id,
                len(injection.chunks),
                sum(len(chunk) for chunk in injection.chunks),
            )
        except Exception:
            player.logger.warning(
                "Failed to inject historical DSP audio for channel %s",
                injection.channel_id,
                exc_info=True,
            )
        finally:
            del player._pending_historical_injections[filter_key]
    return injected


def _append_main_pcm_cache(
    player: SendspinPlayer, timestamp_us: int, duration_us: int, pcm_data: bytes
) -> None:
    """Store committed main-channel PCM for late-join DSP preprocessing."""
    player._main_pcm_cache.append(
        _MainPCMCacheChunk(timestamp_us=timestamp_us, duration_us=duration_us, pcm_data=pcm_data)
    )


def _prune_main_pcm_cache(player: SendspinPlayer) -> None:
    """Prune played/old main-channel PCM cache entries."""
    if player._push_stream is None:
        player._main_pcm_cache.clear()
        return
    while len(player._main_pcm_cache) > 1:
        oldest = player._main_pcm_cache[0]
        newest = player._main_pcm_cache[-1]
        window_us = (newest.timestamp_us + newest.duration_us) - oldest.timestamp_us
        if window_us <= _MAIN_PCM_CACHE_RETENTION_US:
            break
        player._main_pcm_cache.popleft()


async def prepare_dsp_for_join(player: SendspinPlayer, player_id: str) -> tuple[str, ...] | None:
    """Preprocess historical PCM for a joining player's new DSP channel."""
    if player._push_stream is None or player._pcm_format is None:
        return None
    pcm_format = player._pcm_format
    channel_id = player._resolve_channel_for_player(player_id, pcm_format)
    filter_key = player._player_filter_key_map.get(player_id)
    if filter_key is None:
        return None

    if filter_key in player._dsp_channels:
        player.logger.debug("DSP channel already active for %s: channel=%s", player_id, channel_id)
        return filter_key

    dsp = await player._create_dsp_channel(filter_key, pcm_format)
    player._dsp_channels[filter_key] = dsp

    if not _ENABLE_HISTORICAL_DSP_INJECTION:
        return filter_key

    target_us = player._push_stream.get_late_join_target_timestamp_us()
    historical_source = [
        chunk
        for chunk in player._main_pcm_cache
        if chunk.timestamp_us + chunk.duration_us > target_us
    ]
    if not historical_source:
        player.logger.debug(
            "No main PCM cache to backfill for %s (target=%sus)", player_id, target_us
        )
        return filter_key

    first_chunk = historical_source[0]
    if first_chunk.timestamp_us > target_us + 200_000:
        player.logger.debug(
            "Skipping historical DSP backfill for %s: first_chunk=%sus target=%sus",
            player_id,
            first_chunk.timestamp_us,
            target_us,
        )
        return filter_key

    dsp_fmt = SendspinAudioFormat(
        sample_rate=pcm_format.sample_rate,
        bit_depth=pcm_format.bit_depth,
        channels=dsp.output_channels,
    )
    processed_chunks: list[bytes] = []
    for chunk in historical_source:
        processed = await player._process_dsp_chunk(dsp, chunk.pcm_data, pcm_format)
        expected_size = _expected_dsp_bytes(
            chunk.pcm_data,
            pcm_format.bit_depth,
            pcm_format.channels,
            dsp.output_channels,
        )
        normalized = _normalize_pcm_size(processed, expected_size)
        processed_chunks.append(normalized)

    if processed_chunks:
        player._pending_historical_injections[filter_key] = _HistoricalInjection(
            channel_id=dsp.channel_id,
            audio_format=dsp_fmt,
            chunks=processed_chunks,
        )
        player.logger.debug(
            "Prepared historical DSP audio for %s: channel=%s chunks=%s",
            player_id,
            dsp.channel_id,
            len(processed_chunks),
        )
    return filter_key


async def run_playback(player: SendspinPlayer, media: PlayerMedia) -> None:  # noqa: PLR0915
    """Run Sendspin playback in a background task."""
    audio_source: AsyncGenerator[bytes, None] | None = None
    prepared_queue: asyncio.Queue[_PreparedCommitFrame | None] | None = None
    prepare_task: asyncio.Task[None] | None = None
    commit_count = 0
    stream_position_us = 0
    first_main_start_us: int | None = None
    try:
        pcm_format = AudioFormat(
            content_type=ContentType.PCM_S16LE,
            sample_rate=48000,
            bit_depth=16,
            channels=2,
        )
        player._pcm_format = pcm_format
        sendspin_fmt = SendspinAudioFormat(
            sample_rate=pcm_format.sample_rate,
            bit_depth=pcm_format.bit_depth,
            channels=pcm_format.channels,
        )

        player._player_channel_map.clear()
        player._player_filter_key_map.clear()
        player._active_dsp_filter_keys.clear()
        player._pending_historical_injections.clear()
        player._main_pcm_cache.clear()

        def channel_resolver(player_id: str) -> UUID:
            if player._pcm_format is None:
                return MAIN_CHANNEL
            if cached_channel_id := player._player_channel_map.get(player_id):
                return cached_channel_id
            return player._resolve_channel_for_player(player_id, player._pcm_format)

        player._push_stream = player.api.group.start_stream(channel_resolver=channel_resolver)
        audio_source = player.mass.streams.get_stream(media, pcm_format)
        prepared_queue = asyncio.Queue(maxsize=4)
        prepare_task = asyncio.create_task(
            _prepare_commit_frames(player, audio_source, pcm_format, prepared_queue)
        )
        while True:
            prepared = await prepared_queue.get()
            if prepared is None:
                break
            if player._push_stream is None or player._push_stream.is_stopped:
                break

            stream_rel_start_us = stream_position_us
            stream_position_us += prepared.main_duration_us
            _inject_pending_historical_audio(player)
            player._push_stream.prepare_audio(prepared.main_pcm, sendspin_fmt)
            for channel_id, (processed_pcm, dsp_fmt) in prepared.dsp_pcm.items():
                player._push_stream.prepare_audio(processed_pcm, dsp_fmt, channel_id=channel_id)

            commit_start_us = await player._push_stream.commit_audio()
            if player._main_pcm_cache:
                last_chunk = player._main_pcm_cache[-1]
                main_start_us = last_chunk.timestamp_us + last_chunk.duration_us
            else:
                main_start_us = commit_start_us
            if first_main_start_us is None:
                first_main_start_us = main_start_us
            _append_main_pcm_cache(
                player,
                timestamp_us=main_start_us,
                duration_us=prepared.main_duration_us,
                pcm_data=prepared.main_pcm,
            )
            commit_count += 1
            if commit_count % 10 == 0 and first_main_start_us is not None:
                channel_debug = [
                    f"main(start_rel={main_start_us - first_main_start_us}us,"
                    f"dur={prepared.main_duration_us}us,bytes={len(prepared.main_pcm)})"
                ]
                for channel_id, (processed_pcm, dsp_fmt) in sorted(
                    prepared.dsp_pcm.items(), key=lambda item: str(item[0])
                ):
                    dsp_duration_us = pcm_duration_us(
                        len(processed_pcm),
                        dsp_fmt.sample_rate,
                        dsp_fmt.bit_depth,
                        dsp_fmt.channels,
                    )
                    channel_debug.append(
                        f"{channel_id}(start_rel={main_start_us - first_main_start_us}us,"
                        f"dur={dsp_duration_us}us,bytes={len(processed_pcm)})"
                    )
                dsp_debug = []
                sorted_dsps = sorted(
                    player._dsp_channels.values(),
                    key=lambda item: str(item.channel_id),
                )
                for dsp in sorted_dsps:
                    buffered_lag_us = max(0, dsp.input_us_total - dsp.raw_output_us_total)
                    dsp_debug.append(
                        f"{dsp.channel_id}(in={dsp.input_us_total}us/{dsp.input_bytes_total}B,"
                        f"raw_out={dsp.raw_output_us_total}us/{dsp.raw_output_bytes_total}B,"
                        f"deliv_out={dsp.delivered_output_us_total}us/"
                        f"{dsp.delivered_output_bytes_total}B,lag={buffered_lag_us}us,"
                        f"pending={len(dsp.pending)}B,peak={dsp.pending_peak_bytes}B,"
                        f"chunks={dsp.chunks_processed},fail={dsp.chunks_failed})"
                    )
                player.logger.debug(
                    "Commit %s stream_rel=%sus commit_start_rel=%sus %s ffmpeg=%s",
                    commit_count,
                    stream_rel_start_us,
                    main_start_us - first_main_start_us,
                    " ".join(channel_debug),
                    " ".join(dsp_debug) if dsp_debug else "none",
                )
            _prune_main_pcm_cache(player)
            await player._push_stream.sleep_to_limit_buffer(_MAX_BUFFER_AHEAD_US)

        if prepare_task is not None:
            await prepare_task

    except asyncio.CancelledError:
        player.logger.debug("Playback cancelled for player %s", player.display_name)
        raise
    except Exception:
        player.logger.exception("Error during playback for player %s", player.display_name)
        raise
    finally:
        if player._push_stream is not None and not player._push_stream.is_stopped:
            with suppress(Exception):
                await player.api.group.stop()
        if prepare_task is not None and not prepare_task.done():
            prepare_task.cancel()
            with suppress(asyncio.CancelledError):
                await prepare_task
        player._push_stream = None
        player._pcm_format = None
        if audio_source is not None:
            with suppress(Exception):
                await audio_source.aclose()
        for dsp in player._dsp_channels.values():
            with suppress(Exception):
                await dsp.ffmpeg.close()
        player._dsp_channels.clear()
        player._active_dsp_filter_keys.clear()
        player._player_channel_map.clear()
        player._player_filter_key_map.clear()
        player._pending_join_members.clear()
        player._pending_historical_injections.clear()
        player._main_pcm_cache.clear()
        if player._playback_task is asyncio.current_task():
            player._playback_task = None
        player._attr_playback_state = PlaybackState.IDLE
        player._attr_elapsed_time = 0
        player._attr_elapsed_time_last_updated = time.time()
        player._attr_current_media = None
        player.update_state()
