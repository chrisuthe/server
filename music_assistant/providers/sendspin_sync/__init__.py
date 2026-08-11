"""
Sendspin Sync plugin provider.

Declares a dependency on the Sendspin player provider, so it only loads once
Sendspin is up. It exposes a single AudioSource - the calibration chirp track -
which plays through the regular playback path so the latency measured from it
matches the latency of real music on the same player.
"""

from __future__ import annotations

import asyncio
from itertools import cycle
from typing import TYPE_CHECKING

from music_assistant_models.enums import ContentType, MediaType, ProviderFeature, StreamType
from music_assistant_models.errors import MediaNotFoundError
from music_assistant_models.media_items import AudioSource, ProviderMapping
from music_assistant_models.media_items.audio_format import AudioFormat
from music_assistant_models.streamdetails import StreamDetails, StreamMetadata

from music_assistant.constants import CONF_ENTRY_WARN_PREVIEW
from music_assistant.controllers.streams.audio import AUDIO_SOURCE_CHUNK_SECONDS
from music_assistant.models.plugin import PluginProvider

from .chirp import BIT_DEPTH, CHANNELS, SAMPLE_RATE, build_chirp_period

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from music_assistant_models.config_entries import ConfigEntry, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


SUPPORTED_FEATURES = {ProviderFeature.AUDIO_SOURCE}

# stable id of the single AudioSource this provider exposes; combined with the
# provider instance_id it forms the persistent browse/play uri
AUDIO_SOURCE_ID = "calibration"


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SendspinSyncProvider(mass, manifest, config, SUPPORTED_FEATURES)


class SendspinSyncProvider(PluginProvider):
    """Sendspin Sync plugin provider for Music Assistant."""

    # tracks which queue currently owns the exclusive AudioSource. Set in
    # on_source_selected (NOT in get_stream_details - that path also runs from
    # queue preload, where claiming would block a later cross-queue handoff).
    _in_use_by_queue: str | None = None
    # the active stream_session_id, paired with _in_use_by_queue: same-queue
    # reconnects refresh this token without changing _in_use_by_queue, so the
    # stream loop and its teardown guard on both still matching.
    _active_session_id: str | None = None

    _audio_format: AudioFormat
    _audio_source: AudioSource
    _period_chunks: list[bytes]

    async def get_config_entries(self) -> tuple[ConfigEntry, ...]:
        """Return the (options) config entries for this provider instance."""
        return (CONF_ENTRY_WARN_PREVIEW,)

    async def handle_async_init(self) -> None:
        """Handle async initialization of the provider."""
        self._audio_format = AudioFormat(
            content_type=ContentType.PCM_S16LE,
            sample_rate=SAMPLE_RATE,
            bit_depth=BIT_DEPTH,
            channels=CHANNELS,
        )
        self._audio_source = AudioSource(
            item_id=AUDIO_SOURCE_ID,
            provider=self.instance_id,
            name=self.name,
            provider_mappings={
                ProviderMapping(
                    item_id=AUDIO_SOURCE_ID,
                    provider_domain=self.domain,
                    provider_instance=self.instance_id,
                    audio_format=self._audio_format,
                )
            },
            # pausing would corrupt the phase the measurement reads, and seeking
            # is meaningless on an endless source
            can_play_pause=False,
            can_seek=False,
            can_next_previous=False,
            exclusive=True,
            allow_external_trigger=False,
            # the track is generated locally, so MA can start it on demand
            can_initiate=True,
        )
        # The track is one period generated once and then replayed verbatim:
        # re-synthesising per chunk would let the chirp spacing drift. Synthesis
        # is a tight per-frame math loop, so it runs off the event loop; slicing
        # it up front into the chunk size the streams controller paces
        # AudioSources at leaves the stream loop handing out ready-made chunks.
        period = await asyncio.to_thread(build_chirp_period)
        chunk_size = int(SAMPLE_RATE * AUDIO_SOURCE_CHUNK_SECONDS) * CHANNELS * (BIT_DEPTH // 8)
        self._period_chunks = [
            period[offset : offset + chunk_size] for offset in range(0, len(period), chunk_size)
        ]

    async def get_audio_sources(self) -> list[AudioSource]:
        """Return the AudioSources this plugin currently exposes."""
        return [self._audio_source]

    async def get_stream_details(self, item_id: str, media_type: MediaType) -> StreamDetails:
        """
        Return StreamDetails for the calibration AudioSource.

        Raises MediaNotFoundError for any other item_id.
        """
        if item_id != AUDIO_SOURCE_ID:
            raise MediaNotFoundError(f"Unknown AudioSource: {item_id}")
        return StreamDetails(
            provider=self.instance_id,
            item_id=item_id,
            audio_format=self._audio_format,
            media_type=MediaType.AUDIO_SOURCE,
            stream_type=StreamType.CUSTOM,
            stream_metadata=StreamMetadata(title=self.name),
        )

    async def get_audio_stream(
        self, streamdetails: StreamDetails, seek_position: int = 0
    ) -> AsyncGenerator[bytes]:
        """Yield the calibration track as raw PCM, looping the precomputed period."""
        consumer_queue = self._in_use_by_queue
        captured_session_id = self._active_session_id
        chunks = cycle(self._period_chunks)
        try:
            # Unlike every other AudioSource, nothing upstream rate-limits this
            # one - it is synthesised, not received. The streams controller paces
            # it instead, through realtime_pcm_pacer on the format-match path and
            # ffmpeg's -re on the resampling path, so the loop must not self-pace
            # on top of that.
            while (
                self._in_use_by_queue == consumer_queue
                and self._active_session_id == captured_session_id
            ):
                yield next(chunks)
            self.logger.debug(
                "Stopping calibration stream for queue %s: this session no longer owns the source",
                consumer_queue,
            )
        finally:
            if (
                self._in_use_by_queue == consumer_queue
                and self._active_session_id == captured_session_id
            ):
                self._in_use_by_queue = None

    async def on_source_selected(
        self, source_id: str, player_id: str, queue_id: str, stream_session_id: str
    ) -> None:
        """React to the calibration AudioSource being selected for playback on a player."""
        if source_id != AUDIO_SOURCE_ID:
            return
        self._in_use_by_queue = queue_id
        self._active_session_id = stream_session_id

    async def on_source_unselected(
        self, source_id: str, queue_id: str, stream_session_id: str
    ) -> None:
        """React to MA tearing down the calibration AudioSource's stream from a queue."""
        if source_id != AUDIO_SOURCE_ID:
            return
        if self._active_session_id != stream_session_id:
            return
        self._active_session_id = None
        if self._in_use_by_queue == queue_id:
            self._in_use_by_queue = None
