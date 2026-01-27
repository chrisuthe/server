# Local Radio Mode Implementation Plan

## Executive Summary

This plan outlines how to implement a "radio mode" for local tracks that creates continuous, similar-track playback from a single seed track without relying on streaming provider APIs. The system will leverage Last.fm's similarity data, existing MA library metadata, play history, and (future) genre functionality to build an intelligent recommendation engine for local-only content.

---

## Current State Analysis

### Existing Radio Mode (Streaming-Only)
- **Entry Point**: `PlayerQueuesController.play_media()` with `radio_mode=True`
- **Flow**: Seed → Base Tracks → Provider `similar_tracks()` → Queue Population
- **Pattern**: Alternating base tracks with dynamic similar tracks (B + 2D + B + 2D...)
- **Auto-refill**: Triggered when <5 tracks remain in queue
- **Provider Dependency**: Requires `ProviderFeature.SIMILAR_TRACKS` support

### Available Data Sources for Local Radio

**1. Last.fm API Endpoints**
- `track.getSimilar` - Get similar tracks by name/artist or MBID
- `artist.getSimilar` - Get similar artists by name or MBID
- Already integrated with caching (90-day TTL)
- Rate limited to 5 requests/second

**2. MA Library Data**
- `playlog` table: Complete play history with timestamps per user
- `play_count` & `last_played`: Aggregated statistics on all media items
- External IDs: MBIDs, ISRCs for Last.fm matching
- Metadata JSON: Contains enriched data (could include genres in future)
- Relationship tables: Track-Album-Artist mappings

**3. Existing Recommendation Infrastructure**
- `lastfm_recommendations` provider with matching logic
- `parsers.py`: Smart name/MBID/ISRC matching to local library
- Multi-tier caching system
- Concurrent provider search with deduplication

---

## Design Approach: Hybrid Similarity System

### Core Philosophy

Build a **multi-strategy recommendation engine** that:
1. **Primary**: Uses Last.fm similarity data matched against local library
2. **Fallback**: Uses metadata-based similarity (artist, album, genre)
3. **Intelligence**: Learns from play history and user preferences
4. **Adaptability**: Adjusts repetition rules based on library size

### Key Design Principles

1. **Quality First**: Prioritize track similarity over library coverage
2. **No Dead Ends**: Always have fallback strategies when Last.fm lacks data
3. **Respect History**: Avoid recent replays based on library size
4. **Progressive Enhancement**: Start simple, add genre/ML features later
5. **Performance**: Cache aggressively, pre-compute when possible

---

## Implementation Strategy

### Phase 1: Last.fm Similar Tracks for Local Library

**Objective**: Extend existing radio mode to support local-only tracks using Last.fm similarity data

#### Components to Build

**1.1 Local Radio Mode Detector**
```python
# In PlayerQueuesController.play_media()
def _is_local_only_radio(self, media_item: MediaItemType) -> bool:
    """Determine if radio mode should use local-only strategy."""
    # Check if item is from filesystem provider
    # Check if user has enabled local-only radio preference
    # Check if no streaming providers with SIMILAR_TRACKS are available
```

**1.2 Local Similar Tracks Service**
```python
# New file: music_assistant/helpers/local_radio.py

class LocalRadioService:
    """Service for generating radio playlists from local library."""

    async def get_similar_tracks(
        self,
        seed_track: Track,
        limit: int = 25,
        exclude_track_ids: set[str] | None = None,
        min_time_since_played: int = 0  # seconds
    ) -> list[Track]:
        """Get similar tracks from local library.

        Strategy:
        1. Query Last.fm for similar tracks
        2. Match Last.fm results to local library
        3. Fill gaps with artist-based matching
        4. Filter by play history constraints
        5. Sort by similarity score
        """
```

**1.3 Last.fm → Local Library Matcher**
- Reuse existing `lastfm_recommendations/parsers.py` logic
- Extend to work with single track queries (not just bulk recommendations)
- Add library-only filtering (don't search streaming providers)

**1.4 Play History Filter**
```python
async def filter_recently_played(
    self,
    candidates: list[Track],
    min_time_since_played: int,
    user_id: str
) -> list[Track]:
    """Remove tracks played within threshold.

    :param min_time_since_played: Minimum seconds since last play
    """
    # Query playlog table for recent plays by user_id
    # Remove candidates that were played within threshold
    # Return filtered list
```

#### Integration Points

**Modify `PlayerQueuesController._get_radio_tracks()`**
```python
async def _get_radio_tracks(
    self,
    queue: PlayerQueue,
    is_initial_radio_mode: bool = False,
) -> list[QueueItem]:
    """Generate radio tracks."""

    # NEW: Detect if local-only radio should be used
    if self._should_use_local_radio(queue):
        return await self._get_local_radio_tracks(queue, is_initial_radio_mode)

    # EXISTING: Use streaming provider similar tracks
    # ... existing code ...
```

**New Method: `_get_local_radio_tracks()`**
```python
async def _get_local_radio_tracks(
    self,
    queue: PlayerQueue,
    is_initial_radio_mode: bool = False,
) -> list[QueueItem]:
    """Generate radio tracks using local library only.

    Uses Last.fm similarity + local library matching + play history filtering.
    """
    # 1. Calculate minimum replay interval based on library size
    min_replay_seconds = self._calculate_min_replay_interval(queue.userid)

    # 2. Get base tracks from play history (or radio_source if initial)
    base_tracks = await self._get_base_tracks(queue, is_initial_radio_mode)

    # 3. For each base track, get similar tracks from local library
    local_radio_service = LocalRadioService(self.mass)
    similar_tracks = []

    for base_track in random.sample(base_tracks, min(5, len(base_tracks))):
        tracks = await local_radio_service.get_similar_tracks(
            seed_track=base_track,
            limit=10,
            exclude_track_ids=already_used_ids,
            min_time_since_played=min_replay_seconds
        )
        similar_tracks.extend(tracks)

        if len(similar_tracks) >= 50:
            break

    # 4. Organize into pattern (same as streaming radio)
    # ... pattern logic ...
```

#### Adaptive Repetition Rules

**Calculate Minimum Replay Interval**
```python
def _calculate_min_replay_interval(self, user_id: str) -> int:
    """Calculate minimum seconds between replays based on library size.

    Goal: Avoid playing same track more than once per day for large libraries,
    but allow more frequent replays for small libraries.

    :param user_id: User ID for library size calculation
    :return: Minimum seconds between plays
    """
    # Query library size for user
    library_track_count = await self.mass.music.tracks.count(
        extra_query="favorite = 1 OR provider = 'filesystem'"
    )

    # Calculate theoretical "once per day" threshold
    AVG_TRACK_DURATION = 210  # 3:30 in seconds
    SECONDS_PER_DAY = 86400

    # Tracks needed to fill 24 hours
    tracks_per_day = SECONDS_PER_DAY // AVG_TRACK_DURATION  # ~411

    if library_track_count >= tracks_per_day * 2:
        # Large library: enforce once per day
        return SECONDS_PER_DAY
    elif library_track_count >= tracks_per_day:
        # Medium library: once per 12 hours
        return SECONDS_PER_DAY // 2
    elif library_track_count >= tracks_per_day // 2:
        # Small library: once per 6 hours
        return SECONDS_PER_DAY // 4
    else:
        # Very small library: once per 3 hours
        return SECONDS_PER_DAY // 8
```

---

### Phase 2: Metadata-Based Fallback Strategies

**Objective**: Ensure radio mode never runs out of tracks, even when Last.fm has no data

#### 2.1 Artist-Based Similarity
```python
async def get_similar_by_artist(
    self,
    seed_track: Track,
    limit: int = 25
) -> list[Track]:
    """Find similar tracks by shared artist.

    Strategy:
    1. Get all artists from seed track
    2. Get all tracks by those artists from local library
    3. Exclude seed track and already-played
    4. Randomize with slight preference for same album
    5. Return limited results
    """
```

#### 2.2 Album-Based Similarity
```python
async def get_similar_by_album(
    self,
    seed_track: Track,
    limit: int = 25
) -> list[Track]:
    """Find similar tracks from same or similar albums.

    Strategy:
    1. Prefer other tracks from same album (excluding seed)
    2. Include tracks from albums by same artist
    3. Randomize to avoid predictability
    """
```

#### 2.3 Weighted Fallback Chain
```python
async def get_similar_tracks_with_fallback(
    self,
    seed_track: Track,
    limit: int = 25,
    **filters
) -> list[Track]:
    """Get similar tracks using multi-strategy approach.

    Tries strategies in order until enough tracks are found:
    1. Last.fm similar tracks (best quality)
    2. Artist-based similarity (good quality)
    3. Album-based similarity (acceptable quality)
    4. Random from library (last resort)

    Each strategy fills a portion:
    - 70% from best available strategy
    - 20% from next best strategy
    - 10% random for discovery
    """
```

---

### Phase 3: Genre-Enhanced Recommendations (Future)

**Objective**: Leverage genre data when available to improve similarity matching

#### 3.1 Genre Data Integration

**Prerequisites:**
- Genre controller fully implemented
- Genre metadata populated from:
  - Last.fm tags
  - MusicBrainz genres
  - ID3 tags from local files
  - User manual tagging

**Genre-Based Similarity:**
```python
async def get_similar_by_genre(
    self,
    seed_track: Track,
    limit: int = 25
) -> list[Track]:
    """Find similar tracks by shared genres.

    Strategy:
    1. Extract genres from seed track metadata
    2. Query library for tracks with overlapping genres
    3. Weight by genre overlap (more shared genres = higher score)
    4. Randomize within each score tier
    5. Exclude recently played
    """
```

#### 3.2 Multi-Factor Scoring

**Combine multiple signals:**
```python
def calculate_similarity_score(
    seed_track: Track,
    candidate_track: Track,
    factors: dict[str, float]
) -> float:
    """Calculate multi-factor similarity score.

    Factors (with configurable weights):
    - Last.fm similarity rank: 40%
    - Genre overlap: 30%
    - Artist match: 20%
    - Album match: 10%

    :return: Score between 0.0 and 1.0
    """
```

#### 3.3 Genre-Based Discovery Mode

```python
# New radio mode variant: "Genre Radio"
async def play_genre_radio(
    self,
    queue_id: str,
    seed_genres: list[str],
    radio_source: list[MediaItemType]
) -> None:
    """Start radio mode based on genre seeds.

    Different from track-based radio:
    - Explores genre rather than specific track similarity
    - Wider variety within genre boundaries
    - Good for mood-based listening
    """
```

---

### Phase 4: Smart Caching & Pre-Computation

**Objective**: Reduce latency and API calls through intelligent caching

#### 4.1 Similarity Matrix Cache

```python
# Store pre-computed similarity relationships
CACHE_CATEGORY_LOCAL_SIMILARITY = 10

async def build_similarity_cache(self) -> None:
    """Pre-compute similarity relationships for local library.

    Runs periodically (e.g., daily) to:
    1. Query Last.fm for all local tracks
    2. Build similarity matrix
    3. Store in database cache with 30-day TTL
    4. Enables instant radio mode startup
    """
```

**Database Schema:**
```sql
CREATE TABLE track_similarity(
    [track_id_a] INTEGER NOT NULL,
    [track_id_b] INTEGER NOT NULL,
    [similarity_score] REAL NOT NULL,
    [source] TEXT NOT NULL,  -- 'lastfm', 'artist', 'genre', etc.
    [timestamp_computed] INTEGER NOT NULL,
    PRIMARY KEY (track_id_a, track_id_b)
);

CREATE INDEX idx_track_similarity_a ON track_similarity(track_id_a, similarity_score DESC);
```

#### 4.2 Play History Optimization

```sql
-- Add index for efficient recent play queries
CREATE INDEX idx_playlog_user_timestamp ON playlog(userid, timestamp DESC);

-- Materialized view for per-track last played by user
CREATE TABLE track_last_played(
    [track_id] INTEGER NOT NULL,
    [userid] TEXT NOT NULL,
    [last_played] INTEGER NOT NULL,
    PRIMARY KEY (track_id, userid)
);
```

---

## Configuration & User Settings

### New Configuration Options

**Global Settings (in Music Controller):**
```python
CONF_ENABLE_LOCAL_RADIO = "enable_local_radio"  # bool, default: True
CONF_LOCAL_RADIO_FALLBACK_STRATEGY = "local_radio_fallback_strategy"
# Options: "lastfm_only", "lastfm_with_artist_fallback", "metadata_only"

CONF_LOCAL_RADIO_MIN_REPLAY_HOURS = "local_radio_min_replay_hours"  # int, 0 = auto
CONF_LOCAL_RADIO_CACHE_SIMILARITY = "local_radio_cache_similarity"  # bool, default: True
```

**Per-Queue Settings:**
```python
# Allow user to prefer local radio even when streaming providers available
CONF_PREFER_LOCAL_RADIO = "prefer_local_radio"  # bool, default: False
```

---

## Technical Implementation Details

### File Structure

```
music_assistant/
├── controllers/
│   ├── player_queues.py          # Modified: add local radio detection
│   └── music.py                   # Modified: add similarity queries
├── helpers/
│   ├── local_radio.py             # NEW: Core local radio service
│   ├── similarity.py              # NEW: Similarity calculation utilities
│   └── play_history.py            # NEW: Play history filtering utilities
└── providers/
    └── lastfm_recommendations/
        ├── parsers.py             # Modified: extract matching logic
        └── local_matcher.py       # NEW: Dedicated local library matcher
```

### Data Flow Diagram

```
User Selects Track (Local File)
         ↓
PlayerQueue.play_media(radio_mode=True)
         ↓
Detect Local-Only Radio Mode
         ↓
LocalRadioService.get_similar_tracks()
         ↓
    ┌────────────────┴────────────────┐
    ↓                                  ↓
Query Last.fm API              Check Similarity Cache
(track.getSimilar)                    ↓
    ↓                          Return Cached Results
Parse Last.fm Results                 ↓
    ↓                                  ↓
Match to Local Library ←──────────────┘
    ↓
Filter by Play History
(min_time_since_played)
    ↓
Fallback: Artist/Album/Genre Matching
(if not enough results)
    ↓
Sort by Similarity Score
    ↓
Return to PlayerQueue
    ↓
Organize into Pattern (B+2D+B+2D...)
    ↓
Populate Queue & Start Playback
    ↓
Auto-Refill when <5 tracks remain
```

---

## Testing Strategy

### Unit Tests

1. **Similarity Matching Tests**
   - Last.fm results → local library matching accuracy
   - MBID/ISRC matching vs name matching
   - Fallback strategy triggering

2. **Play History Filtering Tests**
   - Recent play exclusion logic
   - Library size → replay interval calculation
   - Edge cases (empty library, single track)

3. **Caching Tests**
   - Cache hit/miss behavior
   - TTL expiration handling
   - Cache invalidation on library changes

### Integration Tests

1. **End-to-End Radio Mode Tests**
   - Start radio from local track
   - Verify queue population
   - Verify auto-refill behavior
   - Verify no streaming provider calls

2. **Multi-Strategy Tests**
   - Last.fm available → uses Last.fm
   - Last.fm unavailable → uses artist fallback
   - Very small library → handles gracefully

3. **Play History Tests**
   - Track not replayed within threshold
   - Multiple users with separate histories
   - History respects across radio sessions

---

## Performance Considerations

### Expected Latencies

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| Initial radio start | <2s | Including Last.fm API call |
| Queue refill | <500ms | Using cached similarity |
| Similarity cache build | <30s per 1000 tracks | Background job |
| Play history filter | <100ms | Indexed query |

### Scalability Limits

| Library Size | Expected Performance | Mitigation |
|--------------|---------------------|------------|
| <1000 tracks | Excellent | No special handling |
| 1000-10000 tracks | Good | Enable similarity cache |
| 10000-50000 tracks | Fair | Pre-compute similarity matrix |
| >50000 tracks | Challenging | Limit cache to top 10K most played |

### API Rate Limiting

**Last.fm API:**
- Current limit: 5 requests/second (already implemented)
- Cache hit ratio target: >80%
- Strategy: Batch queries during off-peak hours

---

## Migration & Rollout Plan

### Phase 1: Core Local Radio (MVP)
**Outcome**: Working local radio mode that users can test with their local tracks

- [ ] Implement `LocalRadioService` class
- [ ] Extract matching logic from `lastfm_recommendations`
- [ ] Add play history filtering utilities
- [ ] Modify `PlayerQueuesController._get_radio_tracks()`
- [ ] Implement local radio detection
- [ ] Add configuration options
- [ ] Implement artist-based similarity fallback
- [ ] Implement album-based similarity fallback
- [ ] Add weighted fallback chain
- [ ] Unit tests for core components
- [ ] Integration tests
- [ ] Test with various library sizes

### Phase 2: Optimization
**Outcome**: Faster radio mode, especially for large libraries

- [ ] Implement similarity cache
- [ ] Add background cache builder
- [ ] Optimize database queries
- [ ] Performance testing & tuning

### Phase 3: Genre Enhancement (Future)
**Outcome**: Better recommendations using genre intelligence when genre data is available

- [ ] Wait for genre controller implementation
- [ ] Add genre-based similarity
- [ ] Implement multi-factor scoring
- [ ] Add genre radio mode variant

---

## Success Metrics

### Functional Metrics
- [ ] Radio mode works with 0 streaming providers configured
- [ ] Average queue refill time <500ms
- [ ] No track repetition within calculated threshold
- [ ] Fallback strategies trigger correctly

### Quality Metrics
- [ ] User satisfaction: Similar tracks feel "related" to seed
- [ ] Discovery factor: Users find new tracks in their library
- [ ] Session length: Users keep radio mode playing longer

### Technical Metrics
- [ ] Last.fm API cache hit ratio >80%
- [ ] Database query latency <100ms P95
- [ ] Memory usage <50MB for similarity cache

---

## Open Questions & Future Enhancements

### Open Questions

1. **Genre Priority**: When genre data is available, should it override Last.fm data or complement it?
   - **Recommendation**: Complement (weighted scoring)

2. **User Feedback Loop**: Should we track skip behavior to improve recommendations?
   - **Recommendation**: Yes, as Phase 6 enhancement

3. **Multi-Seed Radio**: Should users be able to seed radio with multiple tracks?
   - **Recommendation**: Yes, similar to existing album/artist radio

4. **Cross-User Recommendations**: For households, should we blend multiple users' histories?
   - **Recommendation**: No, keep per-user for privacy

### Future Enhancements

1. **Machine Learning Similarity Model**
   - Train on user play history
   - Learn personalized similarity beyond Last.fm
   - Requires significant play history data

2. **Audio Feature Analysis**
   - Use acousticbrainz, essentia, or local audio analysis
   - Match tracks by tempo, energy, mood
   - Requires additional dependencies

3. **Collaborative Filtering**
   - "Users who played X also played Y"
   - Requires opt-in data sharing
   - Privacy implications

4. **Smart Mixing**
   - Detect transitions between tracks
   - Crossfade between similar songs
   - Requires audio processing

---

## Risk Assessment & Mitigation

### High Risk

**Risk**: Last.fm API provides poor matches for obscure local tracks
- **Impact**: Low-quality radio experience
- **Mitigation**: Robust fallback to artist/album/genre matching
- **Detection**: Track similarity scores and skip rates

### Medium Risk

**Risk**: Large libraries cause slow query performance
- **Impact**: Latency spikes during queue refill
- **Mitigation**: Pre-computed similarity cache, database indexing
- **Detection**: Performance monitoring, P95 latency alerts

### Low Risk

**Risk**: Play history becomes stale for inactive users
- **Impact**: Suboptimal recommendations
- **Mitigation**: Fall back to global popularity when history is old
- **Detection**: Check last_played timestamps

---

## Conclusion

This plan provides a comprehensive roadmap for implementing local-only radio mode in Music Assistant. The approach is:

1. **Pragmatic**: Leverages existing Last.fm integration and recommendation infrastructure
2. **Robust**: Multiple fallback strategies ensure radio never runs out of tracks
3. **Adaptive**: Replay rules scale with library size
4. **Extensible**: Designed to incorporate genre data when available
5. **Performant**: Caching and pre-computation minimize latency

The phased approach delivers working functionality at each stage:
- **Phase 1** delivers a complete, testable local radio feature
- **Phase 2** enhances performance for large libraries
- **Phase 3** adds genre-based intelligence when available

**Key Success Factor**: Quality of Last.fm matching to local library determines recommendation quality. The existing `lastfm_recommendations` provider demonstrates this is achievable.
