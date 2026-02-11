# Multi-Device DSP for Sendspin: Debugging Handoff

## Problem Statement

Per-player DSP (EQ, volume, channel mode, limiter) in grouped Sendspin playback.
When multiple speakers are grouped, each should get its own DSP-processed audio
through a shared ffmpeg process (players with identical DSP configs share one process).

**Current status:** Implementation complete, linting/type checking passes, but
**players are not in sync** during playback. Three test iterations have been done,
each fixing different issues but sync remains broken.

## Repository Layout

Both repos are in the same worktree:

- **Server (Music Assistant):** `/var/home/maximr/projects/music-assistant/sendspin-refactor/server/`
- **aiosendspin (library):** `/var/home/maximr/projects/music-assistant/sendspin-refactor/aiosendspin/`
- **Branch:** `feat/aiosendspin-4.0`

Commands:
- `./scripts/run-in-env.sh pre-commit run -a` (lint + type check)
- `./scripts/run-in-env.sh pytest tests/providers/sendspin/` (tests)

## Architecture Overview

```
Source stream (raw 48kHz/16-bit stereo PCM from Music Assistant)
     |
     +-- MAIN_CHANNEL (UUID zero): raw PCM (always present)
     |   Subscribed by: players with no DSP changes (no EQ, no limiter, etc.)
     |
     +-- DSP Channel A (uuid5 of filter_params): processed PCM
     |   ffmpeg process A (EQ + limiter)
     |   Subscribed by: player X, player Y (same DSP config = shared channel)
     |
     +-- DSP Channel B: processed PCM (mono)
         ffmpeg process B (EQ + pan=mono + limiter)
         Subscribed by: player Z (different DSP config)
```

### Main loop (in `_run_playback`)

```python
async for chunk in audio_source:
    sync_dsp_channels()           # handle departed members
    push_stream.prepare_audio(chunk, stereo_format)                    # MAIN_CHANNEL
    for dsp in dsp_channels.values():
        processed = await process_dsp_chunk(dsp, chunk)                # ffmpeg in/out
        push_stream.prepare_audio(processed, dsp_format, channel_id=dsp.channel_id)
    await push_stream.commit_audio()                                   # atomic commit
    await throttle_playback(...)                                       # stay ~6s ahead
```

### Channel resolver

Each player role asks `group.get_channel_for_player(client_id)` to determine which
channel it should receive audio from. The resolver is a callback set at `start_stream()`:

```python
def channel_resolver(player_id: str) -> UUID:
    return self._player_channel_map.get(player_id, MAIN_CHANNEL)
```

### Key timing concept

`PushStream._channel_timing[channel_id]` holds the **next chunk's start timestamp**
(in microseconds). This advances forward as chunks are committed. For a stream that's
been running for a while and is throttled to stay ~6s ahead of realtime:

- `_channel_timing[MAIN_CHANNEL]` ~ `now + 6_000_000` (microseconds)
- PCM cache is pruned to remove chunks where `timestamp_us + duration_us <= now_us`
- So PCM cache contains chunks from approximately `now` to `now + 6s`

## Files and Key Functions

### `server/music_assistant/providers/sendspin/player.py` (ALL changes are here)

The full git diff is available via `git diff HEAD -- music_assistant/providers/sendspin/player.py`.

Key additions:

| Symbol | Line | Description |
|--------|------|-------------|
| `_DSPChannel` | ~155 | Dataclass: channel_id, filter_params, ffmpeg, output_channels, pending bytearray |
| `_needs_dsp_channel()` | ~166 | Returns `tuple(filter_params)` or None |
| `_create_dsp_channel()` | ~554 | Creates FFMpeg process, returns `_DSPChannel` |
| `_setup_dsp_channels()` | ~587 | Initial setup at playback start for all group members |
| `_drain_stdout()` | ~601 | Reads ffmpeg stdout with 10ms timeout loop into `dsp.pending` |
| `_process_dsp_chunk()` | ~616 | `gather(write, drain_stdout)` then extract result from pending buffer |
| `_sync_dsp_channels()` | ~646 | Handles departed members (new members are pre-configured in `set_members`) |
| `_prepare_dsp_for_join()` | ~685 | **Critical**: Pre-configures DSP channel before `add_client()` |
| `_run_playback()` | ~759 | Main playback loop with DSP channel processing |
| `set_members()` | ~852 | Calls `_prepare_dsp_for_join()` before `add_client()` |

### `aiosendspin/aiosendspin/server/push_stream.py` (READ ONLY - understanding internals)

| Symbol | Line | Description |
|--------|------|-------------|
| `PushStream._channel_timing` | ~301 | `dict[UUID, int]` - next chunk start per channel |
| `PushStream._pcm_chunk_cache` | ~309 | `dict[int, deque[CachedPCMChunk]]` - raw PCM cache keyed by channel_id.int |
| `PushStream._role_chunk_cache` | ~307 | `defaultdict[TransformKey, list[CachedChunk]]` - encoded cache per transform |
| `prepare_audio()` | ~392 | Stores PCM for next commit |
| `commit_audio()` | ~475 | Processes historical buffers, then live; returns earliest play_start_us |
| `_resolve_channel_play_start()` | ~584 | New channels get `now + DEFAULT_INITIAL_DELAY_US` (250ms) |
| `_process_historical_buffers()` | ~621 | Auto-aligns: `timing = anchor - total_duration` if no explicit start_time_us |
| `on_role_join()` | ~1104 | Catch-up entry point: checks role cache, then PCM cache |
| `_do_role_join()` | ~1118 | Actual catch-up: tries role cache first, falls through to PCM cache catch-up |
| `_start_catchup_encoding()` | ~1343 | Encodes PCM cache into role cache for late joiners |
| `get_late_join_target_timestamp_us()` | ~459 | Returns minimum playback timestamp for late-join |
| `_rebase_first_join_channel_timing()` | ~1229 | Rebases stale channel timing for first joiner |
| `enable_pcm_cache_for_channel()` | ~358 | Enables raw PCM caching for a channel |
| `CachedPCMChunk` | ~254 | Frozen dataclass: timestamp_us, duration_us, pcm_data, sample_rate, bit_depth, channels |

### `aiosendspin/aiosendspin/server/group.py` (READ ONLY)

| Symbol | Line | Description |
|--------|------|-------------|
| `start_stream()` | ~146 | Creates PushStream, sets channel_resolver |
| `add_client()` | ~528 | Adds client to group, calls `on_role_join()` for active stream |
| `stop()` | ~301 | Stops playback, signals state change |
| `stop_stream()` | ~186 | Calls `push_stream.stop()` |

### `aiosendspin/aiosendspin/server/roles/player/v1.py` (READ ONLY)

| Symbol | Line | Description |
|--------|------|-------------|
| `get_audio_requirements()` | ~145 | Returns `AudioRequirements` with `channel_id` from resolver |
| `_ensure_audio_requirements()` | ~675 | Builds AudioRequirements using `group.get_channel_for_player()` |
| `on_group_changed()` | ~245 | Rebuilds audio requirements with `force=True` |

### `server/music_assistant/helpers/audio.py` (READ ONLY)

| Symbol | Line | Description |
|--------|------|-------------|
| `get_player_filter_params()` | ~1383 | Returns list of ffmpeg filter params for a player |
| `is_grouping_preventing_dsp()` | ~1337 | Returns False when `MULTI_DEVICE_DSP` in supported_features |

### `server/music_assistant/helpers/ffmpeg.py` (READ ONLY)

`FFMpeg(AsyncProcess)` at line 30. Wraps ffmpeg as an async process with stdin feeder,
stderr reader. `AsyncProcess` uses independent `_stdin_lock`/`_stdout_lock` for safe
concurrent read/write via `gather()`.

## Bug History

### Iteration 1: Initial implementation

**Symptom:** Players not in sync. Log showed `ts_gap_ms(avg=66.9 min=-4816.0 max=96.0)` -
timestamps going backwards by ~4.8 seconds for the late-joining player.

**Root cause:** `_bootstrap_dsp_channel` injected historical DSP audio with
`start_time_us=cached.timestamp_us` from MAIN_CHANNEL cache. But the player had already
received MAIN_CHANNEL catch-up from aiosendspin's built-in mechanism. Timestamps regressed.

**Fix:** Replaced `_bootstrap_dsp_channel` with `_align_dsp_channel_timing` using a
single silence sample via `prepare_historical_audio()` without explicit `start_time_us`
to trigger auto-alignment in `_process_historical_buffers`.

### Iteration 2: Silence-based alignment

**Symptom:** Three issues:
1. "First played without DSP (not good! imagine DSP makes signal much quieter!) then out
   of sync with DSP"
2. Log showed `ts_gap_ms(avg=370.2 min=96.0 max=3112.0)` - 3s forward timestamp gap
3. Stop button didn't work (intermittently still played, never reached stopped state)

**Root causes:**
1. **No DSP during catch-up**: `set_members` calls `add_client()` which triggers
   `on_role_join()` catch-up immediately. At that point, `_sync_dsp_channels` hasn't
   run yet (it runs on next audio loop iteration), so the channel resolver returns
   `MAIN_CHANNEL` for the new player. Catch-up serves raw (non-DSP) audio.
2. **3s timestamp gap**: After silence-based alignment, DSP channel timing was aligned to
   MAIN_CHANNEL position, but the catch-up role cache from MAIN_CHANNEL didn't extend
   all the way to the DSP channel's catch-up point.
3. **Stop not working**: In `finally` block, `dsp.ffmpeg.close()` (slow) was called
   BEFORE `stop_stream()`, so the stream continued playing during cleanup.

**Fix:** Major refactor:
- Moved DSP state (`_dsp_channels`, `_player_channel_map`, `_pcm_format`) to instance
  attributes so they're accessible from `set_members`.
- Created `_prepare_dsp_for_join()` called from `set_members` BEFORE `add_client()`.
- It creates the DSP channel, processes MAIN_CHANNEL PCM cache through ffmpeg, and
  directly populates PushStream internals (`_channel_timing`, `_pcm_chunk_cache`).
- Reordered `finally` block: stop stream first, then cleanup.

### Iteration 3 (CURRENT): Pre-configured DSP

**Symptom:** "Still not in sync"

**No detailed logs provided yet.** Analysis below identifies likely issues.

## Root Cause Analysis of Current Sync Issue

### Likely Issue: Catch-up skipped due to timing mismatch

In `_prepare_dsp_for_join()` (line ~735):
```python
self._push_stream._channel_timing[dsp.channel_id] = self._push_stream._channel_timing[MAIN_CHANNEL]
```

This sets the DSP channel timing to MAIN_CHANNEL's current position (~`now + 6s`).

Then `add_client()` triggers `on_role_join()` -> `_do_role_join()` which calls
`_start_catchup_encoding()`. Inside `_start_catchup_encoding` (push_stream.py line 1359):

```python
align_to_channel_tail = channel_id != MAIN_CHANNEL  # True for DSP channels
target_ts = self.get_late_join_target_timestamp_us(
    channel_id=channel_id,
    align_to_channel_tail=align_to_channel_tail,
)
```

With `align_to_channel_tail=True` and `channel_id` in `_channel_timing`, this returns:
```python
return max(now_us, self._channel_timing[channel_id])  # = now + 6s
```

Then the eligibility filter:
```python
eligible = [chunk for chunk in pcm_chunks if chunk.timestamp_us + chunk.duration_us > encode_start_ts]
```

Where `encode_start_ts = target_ts = now + 6s`. The DSP PCM cache has chunks timestamped
from `~now + 100ms` to `~now + 5.9s`. Since `5.9s < 6s`, **NO chunks are eligible!**

The code hits this early return (push_stream.py ~1151):
```python
if latest_cached_end_us <= late_join_target_us:
    if self._channel_timing:
        self._ensure_role_started(r)
    return
```

The player gets `on_stream_start()` but **no catch-up audio chunks**. It only receives
audio from the next live `commit_audio()` call, which means ~6 seconds of silence while
existing players have buffered audio.

### Secondary Issue: `_prepare_dsp_for_join` blocks the event loop

Processing ~6 seconds of cached MAIN_CHANNEL PCM through ffmpeg involves multiple
`await _process_dsp_chunk()` calls. Each call uses `gather(write, drain)` which takes
real time. During this processing:

- `_run_playback` continues interleaving (same event loop)
- MAIN_CHANNEL timing advances further
- The DSP channel is not yet in `_dsp_channels`, so live chunks aren't produced for it
- By the time `_prepare_dsp_for_join` finishes, there's a gap between the last cached
  DSP chunk and the current MAIN_CHANNEL timing

### Tertiary Issue: Private attribute access

`_prepare_dsp_for_join` directly accesses `push_stream._channel_timing` and
`push_stream._pcm_chunk_cache`. These are private attributes of `PushStream`. While
both codebases are controlled by the same developer, this is fragile.

## Potential Solutions

### Option A: Don't set channel timing in `_prepare_dsp_for_join`

Remove the line that copies MAIN_CHANNEL's timing. Let `_resolve_channel_play_start()`
in `commit_audio()` set it naturally when the first live DSP chunk arrives. The catch-up
mechanism would then use `align_to_channel_tail=False` (since timing doesn't exist yet)
and use `now + 100ms` as the target, which the PCM cache can easily reach.

**Risk:** The DSP channel timing would start at `now + 250ms` instead of matching
MAIN_CHANNEL. Need to verify this doesn't cause a permanent timing offset.

### Option B: Add public API to PushStream for channel seeding

Instead of poking at private attributes, add a method like:
```python
def seed_channel_from_pcm(
    self,
    channel_id: UUID,
    pcm_chunks: list[CachedPCMChunk],
) -> None:
    """Seed a new channel with pre-processed PCM cache.

    Sets channel timing to the end of the provided chunks and populates
    the PCM cache. Use for DSP channels that need catch-up data before
    roles join.
    """
```

This encapsulates the timing logic properly and can handle the edge cases
(like setting timing to the end of cached chunks rather than MAIN_CHANNEL's timing).

### Option C: Use `prepare_historical_audio()` properly

Instead of directly populating the PCM cache, use the existing
`prepare_historical_audio()` API. This would:
1. Queue DSP chunks as historical audio
2. Let `commit_audio()` process them with proper timestamp assignment
3. The auto-alignment logic in `_process_historical_buffers()` would handle timing

The challenge: `prepare_historical_audio()` raises ValueError if the channel already has
active timing. And the next `commit_audio()` call happens in the live loop, which might
commit before `add_client()` is called.

### Option D: Remove `_prepare_dsp_for_join` entirely

Instead, modify the catch-up flow to handle DSP channels differently:
1. Let the player join on MAIN_CHANNEL initially (no DSP during catch-up)
2. Once the live loop detects the new member in `_sync_dsp_channels`, switch
   the player to the DSP channel
3. The switch would require `on_role_format_changed()` to rebind

**Risk:** There's a brief period of non-DSP audio, which was the original complaint
from iteration 2.

## Key Invariants to Maintain

1. **Atomic commit**: All channels must be committed together in `commit_audio()` to
   maintain timestamp alignment.

2. **Pipe deadlock prevention**: ffmpeg stdin/stdout are ~64KB OS pipes. A 1-second chunk
   is ~192KB. `gather(write, drain_stdout)` prevents deadlock by running both concurrently.

3. **Channel sharing**: Players with identical `get_player_filter_params()` output must
   share one ffmpeg process and one channel to minimize overhead.

4. **Channel resolver consistency**: The resolver must return the correct channel BEFORE
   `add_client()` fires `on_role_join()`, otherwise catch-up serves wrong audio.

5. **Finally block order**: Stop the stream FIRST (so clients stop immediately), THEN
   clean up ffmpeg processes and audio source.

## How to Test

1. `./scripts/run-in-env.sh pre-commit run -a` - must pass
2. Start Music Assistant: `python -m music_assistant --log-level debug`
3. Group two Sendspin players and play music
4. Listen for: both players should play at exactly the same time
5. Test late-join: start playback on one player, then add another to the group
6. Test stop: press stop, both players should stop immediately

## Debugging Tips

- Look for `ts_gap_ms` in logs - it shows timestamp gap statistics per player
- Look for `DSP failed for channel` warnings
- Look for `Late join catch-up` debug messages
- Look for `Catch-up idle timeout` messages (indicates catch-up gave up)
- Add logging to `_prepare_dsp_for_join` to see cache sizes and timing values
- The key diagnostic: print `_channel_timing` values for all channels before and after
  `_prepare_dsp_for_join`, and check what `_start_catchup_encoding` receives as
  `target_ts` and `eligible` chunk count
