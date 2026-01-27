# Last.fm Metadata Provider Refactor Plan

## Executive Summary

This plan outlines the refactoring of the Last.fm Recommendations provider from a standalone MusicProvider into a MetadataProvider with generic similarity methods. This architectural change makes similarity data available as a standard metadata capability that any provider can implement, while keeping all Last.fm-specific logic self-contained within the Last.fm provider.

---

## Current State Analysis

### Existing Implementation

**Last.fm Recommendations Provider**
- **Type**: `MusicProvider` (extends `music_assistant/models/music_provider.py`)
- **Location**: `music_assistant/providers/lastfm_recommendations/`
- **Supported Feature**: `ProviderFeature.RECOMMENDATIONS`
- **Primary Method**: `recommendations() -> list[RecommendationFolder]`

**Architecture:**
```
LastFMRecommendationsProvider (MusicProvider)
├── api_client.py - Last.fm API calls
├── recommendations.py - Recommendation folder generation
├── parsers.py - Last.fm → MA library matching
├── mbid_resolver.py - MBID → ISRC resolution
└── constants.py - Configuration constants
```

**Key Capabilities:**
- Fetches similar artists/tracks from Last.fm API
- Resolves Last.fm data to local library items
- Organizes results into recommendation folders
- Caches resolved items for performance
- Supports personalized, global, genre, and geo-based recommendations

### Problems with Current Architecture

1. **Feature Locked to Single Provider**: Similarity data is only accessible through Last.fm's recommendation folders
2. **Not Reusable**: Other systems (like radio mode) can't leverage Last.fm similarity without duplicating code
3. **Wrong Provider Type**: Last.fm is fundamentally a metadata/information service, not a music source
4. **Missed Opportunities**: Other metadata providers (MusicBrainz, TheAudioDB) could provide similarity data but have no standard interface

---

## Proposed Architecture

### Design Philosophy

**Make similarity a standard metadata capability:**
- Any MetadataProvider can optionally implement `get_similar_artists()` and `get_similar_tracks()`
- Consumers request similarity data through the standard interface
- Each provider implements the methods using their own data sources
- Implementation details stay self-contained within each provider

### New MetadataProvider Interface

**Add to `music_assistant/models/metadata_provider.py`:**

```python
class MetadataProvider(Provider):
    """Base representation of a Metadata Provider."""

    # ... existing methods ...

    async def get_similar_artists(
        self,
        artist: Artist,
        limit: int = 25
    ) -> list[Artist]:
        """Get similar artists from this metadata provider.

        :param artist: The seed artist to find similar artists for.
        :param limit: Maximum number of similar artists to return.
        :return: List of similar Artist objects, ordered by similarity (most similar first).
        """
        if ProviderFeature.SIMILAR_ARTISTS in self.supported_features:
            raise NotImplementedError
        return []

    async def get_similar_tracks(
        self,
        track: Track,
        limit: int = 25
    ) -> list[Track]:
        """Get similar tracks from this metadata provider.

        :param track: The seed track to find similar tracks for.
        :param limit: Maximum number of similar tracks to return.
        :return: List of similar Track objects, ordered by similarity (most similar first).
        """
        if ProviderFeature.SIMILAR_TRACKS in self.supported_features:
            raise NotImplementedError
        return []
```

### New Provider Features

**Add to `music_assistant_models.enums.ProviderFeature`:**
```python
class ProviderFeature(str, Enum):
    # ... existing features ...
    SIMILAR_ARTISTS = "similar_artists"
    SIMILAR_TRACKS = "similar_tracks"
```

### Refactored Last.fm Provider

**New Type**: `MetadataProvider` (extends `music_assistant/models/metadata_provider.py`)
**New Supported Features**:
- `ProviderFeature.SIMILAR_ARTISTS`
- `ProviderFeature.SIMILAR_TRACKS`
- `ProviderFeature.RECOMMENDATIONS` (optional, can be kept or removed)

**Implementation:**
```python
class LastFMMetadataProvider(MetadataProvider):
    """Last.fm Metadata Provider for Music Assistant.

    Provides similarity data and recommendations from Last.fm based on their
    global listening data and algorithms.
    """

    async def get_similar_artists(
        self,
        artist: Artist,
        limit: int = 25
    ) -> list[Artist]:
        """Get similar artists from Last.fm.

        Uses Last.fm's artist.getSimilar API endpoint and resolves results
        to available items in the Music Assistant library.
        """
        # Implementation stays within provider
        # Reuses existing api_client, parsers, mbid_resolver modules

    async def get_similar_tracks(
        self,
        track: Track,
        limit: int = 25
    ) -> list[Track]:
        """Get similar tracks from Last.fm.

        Uses Last.fm's track.getSimilar API endpoint and resolves results
        to available items in the Music Assistant library.
        """
        # Implementation stays within provider
        # Reuses existing api_client, parsers, mbid_resolver modules

    async def recommendations(self) -> list[RecommendationFolder]:
        """Get this provider's recommendations organized into folders.

        OPTIONAL: Can be kept for backward compatibility or removed if
        recommendations should be generated elsewhere using the similarity methods.
        """
        # Potentially refactored to use get_similar_artists/tracks
        # Or kept as-is for backward compatibility
```

---

## Implementation Strategy

### Phase 1: Add Generic Similarity Interface

**Objective**: Establish the standard interface without breaking existing functionality

#### 1.1 Update Enums

**File**: External package `music_assistant_models` (coordinate with upstream)
```python
# music_assistant_models/enums.py
class ProviderFeature(str, Enum):
    # ... existing ...
    SIMILAR_ARTISTS = "similar_artists"
    SIMILAR_TRACKS = "similar_tracks"
```

#### 1.2 Update MetadataProvider Base Class

**File**: `music_assistant/models/metadata_provider.py`
```python
async def get_similar_artists(
    self,
    artist: Artist,
    limit: int = 25
) -> list[Artist]:
    """Get similar artists from this metadata provider.

    :param artist: The seed artist to find similar artists for.
    :param limit: Maximum number of similar artists to return.
    :return: List of similar Artist objects, ordered by similarity.
    """
    if ProviderFeature.SIMILAR_ARTISTS in self.supported_features:
        raise NotImplementedError
    return []

async def get_similar_tracks(
    self,
    track: Track,
    limit: int = 25
) -> list[Track]:
    """Get similar tracks from this metadata provider.

    :param track: The seed track to find similar tracks for.
    :param limit: Maximum number of similar tracks to return.
    :return: List of similar Track objects, ordered by similarity.
    """
    if ProviderFeature.SIMILAR_TRACKS in self.supported_features:
        raise NotImplementedError
    return []
```

**Testing:**
- [ ] Verify no existing MetadataProvider implementations break
- [ ] Confirm default implementations return empty lists

---

### Phase 2: Refactor Last.fm Provider

**Objective**: Convert Last.fm from MusicProvider to MetadataProvider while preserving all functionality

#### 2.1 Change Provider Base Class

**File**: `music_assistant/providers/lastfm_recommendations/__init__.py`

**Before:**
```python
from music_assistant.models.music_provider import MusicProvider

class LastFMRecommendationsProvider(MusicProvider):
    """Last.fm Recommendations Provider for Music Assistant."""
```

**After:**
```python
from music_assistant.models.metadata_provider import MetadataProvider

class LastFMMetadataProvider(MetadataProvider):
    """Last.fm Metadata Provider for Music Assistant.

    Provides similarity data and recommendations from Last.fm.
    """
```

#### 2.2 Update Supported Features

**Before:**
```python
SUPPORTED_FEATURES = {
    ProviderFeature.RECOMMENDATIONS,
}
```

**After:**
```python
SUPPORTED_FEATURES = {
    ProviderFeature.SIMILAR_ARTISTS,
    ProviderFeature.SIMILAR_TRACKS,
    ProviderFeature.RECOMMENDATIONS,  # Optional: keep for backward compatibility
}
```

#### 2.3 Extract Similarity Methods

**New Method**: `get_similar_artists()`
```python
async def get_similar_artists(
    self,
    artist: Artist,
    limit: int = 25
) -> list[Artist]:
    """Get similar artists from Last.fm.

    :param artist: The seed artist to find similar artists for.
    :param limit: Maximum number of similar artists to return.
    :return: List of similar Artist objects from MA library.
    """
    # 1. Query Last.fm API for similar artists
    lastfm_data = await self.api.get_similar_artists(
        artist_name=artist.name,
        mbid=artist.mbid,
        limit=limit * 2  # Buffer for resolution failures
    )

    # 2. Resolve Last.fm results to MA library items
    resolved_artists = await self._resolve_artists(lastfm_data)

    # 3. Return limited results
    return resolved_artists[:limit]
```

**New Method**: `get_similar_tracks()`
```python
async def get_similar_tracks(
    self,
    track: Track,
    limit: int = 25
) -> list[Track]:
    """Get similar tracks from Last.fm.

    :param track: The seed track to find similar tracks for.
    :param limit: Maximum number of similar tracks to return.
    :return: List of similar Track objects from MA library.
    """
    # 1. Query Last.fm API for similar tracks
    lastfm_data = await self.api.get_similar_tracks(
        track_name=track.name,
        artist_name=track.artists[0].name if track.artists else None,
        mbid=track.mbid,
        limit=limit * 2  # Buffer for resolution failures
    )

    # 2. Resolve Last.fm results to MA library items
    resolved_tracks = await self._resolve_tracks(lastfm_data)

    # 3. Return limited results
    return resolved_tracks[:limit]
```

#### 2.4 Add API Client Methods

**File**: `music_assistant/providers/lastfm_recommendations/api_client.py`

**New Method**: `get_similar_artists()`
```python
async def get_similar_artists(
    self,
    artist_name: str,
    mbid: str | None = None,
    limit: int = 50,
    autocorrect: bool = True
) -> list[dict[str, Any]]:
    """Get similar artists from Last.fm.

    :param artist_name: Name of the artist.
    :param mbid: MusicBrainz ID of the artist (optional, more accurate).
    :param limit: Maximum number of results to return.
    :param autocorrect: Whether to autocorrect artist name.
    :return: List of similar artist data from Last.fm.
    """
    params: dict[str, Any] = {
        "limit": limit,
        "autocorrect": 1 if autocorrect else 0,
    }

    # Prefer MBID for accuracy
    if mbid:
        params["mbid"] = mbid
    else:
        params["artist"] = artist_name

    data = await self._get_data("artist.getSimilar", **params)

    # Extract similar artists from response
    similar_artists = data.get("similarartists", {}).get("artist", [])
    if not isinstance(similar_artists, list):
        similar_artists = [similar_artists]

    return similar_artists
```

**New Method**: `get_similar_tracks()`
```python
async def get_similar_tracks(
    self,
    track_name: str,
    artist_name: str | None = None,
    mbid: str | None = None,
    limit: int = 50,
    autocorrect: bool = True
) -> list[dict[str, Any]]:
    """Get similar tracks from Last.fm.

    :param track_name: Name of the track.
    :param artist_name: Name of the artist.
    :param mbid: MusicBrainz ID of the track (optional, more accurate).
    :param limit: Maximum number of results to return.
    :param autocorrect: Whether to autocorrect track/artist names.
    :return: List of similar track data from Last.fm.
    """
    params: dict[str, Any] = {
        "limit": limit,
        "autocorrect": 1 if autocorrect else 0,
    }

    # Prefer MBID for accuracy
    if mbid:
        params["mbid"] = mbid
    else:
        params["track"] = track_name
        if artist_name:
            params["artist"] = artist_name

    data = await self._get_data("track.getSimilar", **params)

    # Extract similar tracks from response
    similar_tracks = data.get("similartracks", {}).get("track", [])
    if not isinstance(similar_tracks, list):
        similar_tracks = [similar_tracks]

    return similar_tracks
```

#### 2.5 Add Resolution Helper Methods

**File**: `music_assistant/providers/lastfm_recommendations/__init__.py`

```python
async def _resolve_artists(
    self,
    lastfm_artists: list[dict[str, Any]]
) -> list[Artist]:
    """Resolve Last.fm artist data to MA library items.

    :param lastfm_artists: Raw artist data from Last.fm API.
    :return: List of resolved Artist objects from MA library.
    """
    resolved = []
    for artist_data in lastfm_artists:
        # Use existing parsers module
        artist = await parse_artist(
            mass=self.mass,
            lastfm_data=artist_data,
            mbid_resolver=self.mbid_resolver
        )
        if artist:
            resolved.append(artist)
    return resolved

async def _resolve_tracks(
    self,
    lastfm_tracks: list[dict[str, Any]]
) -> list[Track]:
    """Resolve Last.fm track data to MA library items.

    :param lastfm_tracks: Raw track data from Last.fm API.
    :return: List of resolved Track objects from MA library.
    """
    resolved = []
    for track_data in lastfm_tracks:
        # Use existing parsers module
        track = await parse_track(
            mass=self.mass,
            lastfm_data=track_data,
            mbid_resolver=self.mbid_resolver
        )
        if track:
            resolved.append(track)
    return resolved
```

#### 2.6 Update Manifest

**File**: `music_assistant/providers/lastfm_recommendations/manifest.json`

**Before:**
```json
{
  "type": "music",
  "domain": "lastfm_recommendations",
  "name": "Last.fm Recommendations",
  ...
}
```

**After:**
```json
{
  "type": "metadata",
  "domain": "lastfm",
  "name": "Last.fm",
  "description": "Get similarity data and recommendations from Last.fm",
  ...
}
```

**Note**: Changing `domain` will affect configuration - may need migration strategy

#### 2.7 Update Directory Name (Optional)

**Consider renaming:**
- `music_assistant/providers/lastfm_recommendations/` → `music_assistant/providers/lastfm/`

**Migration considerations:**
- User configurations reference provider by domain
- May need config migration or keep old domain for backward compatibility
- Decision: Keep as `lastfm_recommendations` for now, rename in major version

**Testing:**
- [ ] Provider loads successfully as MetadataProvider
- [ ] `get_similar_artists()` returns valid Artist objects
- [ ] `get_similar_tracks()` returns valid Track objects
- [ ] Existing `recommendations()` method still works (if kept)
- [ ] API calls are properly throttled
- [ ] Results are cached appropriately
- [ ] Resolution to library items works correctly

---

### Phase 3: Enable Consumption of Similarity Data

**Objective**: Make similarity data available to other parts of Music Assistant through controller methods

**IMPORTANT**: Controller methods are REQUIRED (moved to Phase 1). All consumers must access similarity data through the metadata controller, never by querying providers directly.

#### 3.1 Add Metadata Controller Methods (REQUIRED)

**File**: `music_assistant/controllers/metadata.py`

**New Method**: `get_similar_artists()`
```python
async def get_similar_artists(
    self,
    artist: Artist,
    limit: int = 25,
    provider_filter: list[str] | None = None
) -> list[Artist]:
    """Get similar artists from available metadata providers.

    Queries all metadata providers that support SIMILAR_ARTISTS feature
    and aggregates results.

    :param artist: The seed artist to find similar artists for.
    :param limit: Maximum number of similar artists to return.
    :param provider_filter: Optional list of provider IDs to query.
    :return: Aggregated list of similar artists from all providers.
    """
    all_similar = []

    # Get all metadata providers supporting SIMILAR_ARTISTS
    # Note: self.mass.music.providers contains all provider types (music, metadata, player, plugin)
    # The music controller is the central provider registry in Music Assistant
    providers = [
        prov for prov in self.mass.music.providers
        if isinstance(prov, MetadataProvider)
        and ProviderFeature.SIMILAR_ARTISTS in prov.supported_features
        and (not provider_filter or prov.instance_id in provider_filter)
    ]

    # Query each provider concurrently
    results = await asyncio.gather(
        *[prov.get_similar_artists(artist, limit) for prov in providers],
        return_exceptions=True
    )

    # Aggregate results
    for result in results:
        if isinstance(result, list):
            all_similar.extend(result)

    # Deduplicate and limit
    seen_ids = set()
    unique_similar = []
    for similar_artist in all_similar:
        if similar_artist.item_id not in seen_ids:
            seen_ids.add(similar_artist.item_id)
            unique_similar.append(similar_artist)

    return unique_similar[:limit]
```

**New Method**: `get_similar_tracks()`
```python
async def get_similar_tracks(
    self,
    track: Track,
    limit: int = 25,
    provider_filter: list[str] | None = None
) -> list[Track]:
    """Get similar tracks from available metadata providers.

    Queries all metadata providers that support SIMILAR_TRACKS feature
    and aggregates results.

    :param track: The seed track to find similar tracks for.
    :param limit: Maximum number of similar tracks to return.
    :param provider_filter: Optional list of provider IDs to query.
    :return: Aggregated list of similar tracks from all providers.
    """
    # Similar implementation to get_similar_artists
    # ...
```

#### 3.2 Use in Radio Mode (Future Enhancement)

**File**: `music_assistant/controllers/player_queues.py`

**Potential usage (showing correct pattern - always use controller):**
```python
async def _get_radio_tracks(
    self,
    queue: PlayerQueue,
    is_initial_radio_mode: bool = False,
) -> list[QueueItem]:
    """Generate radio tracks."""

    # Get base tracks
    base_tracks = await self._get_base_tracks(queue, is_initial_radio_mode)

    # Get similar tracks from metadata providers via controller (REQUIRED pattern)
    # NEVER query providers directly - always go through self.mass.metadata
    similar_tracks = []
    for base_track in base_tracks[:5]:  # Use first 5 base tracks
        # Query metadata controller for similar tracks
        tracks = await self.mass.metadata.get_similar_tracks(
            track=base_track,
            limit=10
        )
        similar_tracks.extend(tracks)

    # Fallback to streaming provider if no metadata results
    if not similar_tracks:
        similar_tracks = await self._get_similar_from_streaming_provider(...)

    # ... rest of radio logic ...
```

**Testing:**
- [ ] Metadata controller aggregates results from multiple providers
- [ ] Deduplication works correctly
- [ ] Radio mode can use similarity data (if implemented)

---

### Phase 4: Documentation and Migration

**Objective**: Document the new pattern and help users/developers adopt it

#### 4.1 Update Documentation

**Files to update:**
- `CLAUDE.md` - Add section on MetadataProvider similarity methods
- Provider README (if exists) - Document Last.fm as MetadataProvider
- API docs - Document new MetadataProvider methods

**Example addition to CLAUDE.md:**
```markdown
### MetadataProvider Similarity Methods

Metadata providers can optionally implement similarity methods to provide
recommendations and related items:

- `get_similar_artists(artist, limit)` - Get artists similar to seed artist
- `get_similar_tracks(track, limit)` - Get tracks similar to seed track

**Implementation:**
1. Add `ProviderFeature.SIMILAR_ARTISTS` and/or `SIMILAR_TRACKS` to supported features
2. Implement the methods with provider-specific logic
3. Return resolved MA library items (not external IDs)
4. Results should be ordered by similarity (most similar first)

**Example providers:**
- Last.fm - Uses Last.fm's similarity algorithms and global listening data
- (Future) MusicBrainz - Could use relationship data
- (Future) TheAudioDB - Could use genre/style similarity
```

#### 4.2 Migration Notes for Users

**Configuration:**
- Existing Last.fm Recommendations configurations should continue to work
- Provider domain remains `lastfm_recommendations` for backward compatibility
- No user action required for existing installations

**Breaking Changes:**
- None if `recommendations()` method is kept
- If removed, recommendation folders would disappear (not recommended for now)

#### 4.3 Developer Guide

**Creating a MetadataProvider with Similarity:**

```python
from music_assistant.models.metadata_provider import MetadataProvider
from music_assistant_models.enums import ProviderFeature

SUPPORTED_FEATURES = {
    ProviderFeature.SIMILAR_ARTISTS,
    ProviderFeature.SIMILAR_TRACKS,
}

class MyMetadataProvider(MetadataProvider):
    async def get_similar_artists(
        self,
        artist: Artist,
        limit: int = 25
    ) -> list[Artist]:
        # 1. Query your data source
        external_results = await self._query_external_api(artist)

        # 2. Resolve to MA library items
        resolved = []
        for result in external_results:
            ma_artist = await self._resolve_to_ma_library(result)
            if ma_artist:
                resolved.append(ma_artist)

        # 3. Return limited, ordered results
        return resolved[:limit]
```

**Testing:**
- [ ] Documentation is clear and complete
- [ ] Examples are accurate
- [ ] Migration path is documented

---

## File Structure Changes

### Before
```
music_assistant/
├── models/
│   ├── metadata_provider.py      # No similarity methods
│   └── music_provider.py
└── providers/
    └── lastfm_recommendations/    # MusicProvider
        ├── __init__.py            # LastFMRecommendationsProvider(MusicProvider)
        ├── api_client.py
        ├── recommendations.py
        ├── parsers.py
        ├── mbid_resolver.py
        ├── constants.py
        └── manifest.json          # type: "music"
```

### After
```
music_assistant/
├── models/
│   ├── metadata_provider.py      # + get_similar_artists/tracks methods
│   └── music_provider.py
├── controllers/
│   └── metadata.py               # + similarity aggregation methods (optional)
└── providers/
    └── lastfm_recommendations/    # MetadataProvider
        ├── __init__.py            # LastFMMetadataProvider(MetadataProvider)
        ├── api_client.py          # + get_similar_artists/tracks API calls
        ├── recommendations.py     # Potentially refactored
        ├── parsers.py
        ├── mbid_resolver.py
        ├── constants.py
        └── manifest.json          # type: "metadata"
```

---

## Testing Strategy

### Unit Tests

**File**: `tests/providers/lastfm_recommendations/test_similarity.py`

```python
async def test_get_similar_artists():
    """Test get_similar_artists returns valid Artist objects."""
    # Setup
    provider = await setup_lastfm_provider()
    seed_artist = create_test_artist(name="Radiohead")

    # Execute
    similar = await provider.get_similar_artists(seed_artist, limit=10)

    # Assert
    assert len(similar) <= 10
    assert all(isinstance(a, Artist) for a in similar)
    assert seed_artist.item_id not in [a.item_id for a in similar]

async def test_get_similar_tracks():
    """Test get_similar_tracks returns valid Track objects."""
    # Similar to above

async def test_similarity_caching():
    """Test that similarity results are cached appropriately."""

async def test_similarity_resolution():
    """Test that Last.fm results resolve to MA library items."""
```

### Integration Tests

```python
async def test_metadata_controller_aggregation():
    """Test that metadata controller aggregates similarity from multiple providers."""

async def test_backward_compatibility():
    """Test that existing recommendation folders still work."""
```

---

## Performance Considerations

### Expected Impact

| Aspect | Before | After | Notes |
|--------|--------|-------|-------|
| Provider load time | ~2s | ~2s | No change |
| Similarity query latency | N/A | ~500ms | New capability |
| Recommendation folder generation | ~20s | ~20s | No change if kept as-is |
| Memory usage | ~50MB | ~50MB | No significant change |
| Cache effectiveness | 80%+ | 80%+ | Reuses existing cache |

### Optimizations

1. **Caching**: Reuse existing cache infrastructure
2. **Concurrent Resolution**: Already implemented in parsers module
3. **API Throttling**: Already implemented in api_client module
4. **MBID Resolution**: Already optimized with caching

---

## Migration Path

### For Users

**No action required** - existing configurations continue to work

**If provider domain changes:**
1. Backup existing configuration
2. Remove old Last.fm Recommendations provider
3. Add new Last.fm provider with same API key
4. Configuration migrates automatically (ideally)

### For Developers

**If using Last.fm Recommendations:**
- Update imports if class/module names change
- Use new similarity methods if appropriate
- Existing `recommendations()` method remains available (if kept)

**If building new features:**
- Can now query similarity data through MetadataProvider interface
- Example: Radio mode can leverage Last.fm similarity
- Example: "Related artists" UI feature can use similarity data

---

## Decision Points

### 1. Keep `recommendations()` Method?

**Option A: Keep it**
- ✅ No breaking changes
- ✅ Backward compatible
- ✅ Familiar to existing users
- ❌ Duplicates some logic with similarity methods

**Option B: Remove it**
- ✅ Cleaner architecture
- ✅ Forces adoption of new pattern
- ❌ Breaking change for existing users
- ❌ Recommendations feature temporarily lost

**Recommendation**: Keep it for now, potentially deprecate in future major version

### 2. Provider Domain/Name Change?

**Option A: Keep `lastfm_recommendations`**
- ✅ No config migration needed
- ✅ No breaking changes
- ❌ Misleading name (it's now a metadata provider)

**Option B: Change to `lastfm`**
- ✅ Accurate naming
- ✅ Consistent with other metadata providers
- ❌ Requires config migration
- ❌ Potential user confusion

**Recommendation**: Keep `lastfm_recommendations` for backward compatibility, change in major version

### 3. Metadata Controller Integration?

**Option A: Add aggregation methods to metadata controller (REQUIRED)**
- ✅ Convenient central API
- ✅ Handles multi-provider scenarios
- ✅ Follows MA architecture (all provider access through controllers)
- ✅ Prevents providers from communicating directly
- ❌ Adds some complexity

**Option B: Let consumers query providers directly (ARCHITECTURALLY INCORRECT)**
- ❌ Violates MA architecture - providers must not be accessed directly
- ❌ Providers should never communicate with each other
- ❌ All provider access must go through controllers/core
- ❌ Not allowed in Music Assistant

**Recommendation**: **Option A is required**. Music Assistant architecture mandates that all provider access goes through controllers, not directly. Consumers (like radio mode, UI features, etc.) must always call controller methods, which in turn coordinate with providers.

---

## Rollout Plan

### Phase 1: Foundation (No Breaking Changes)
**Outcome**: New interface exists, Last.fm implements it, controller methods provide access, everything backward compatible

- [ ] Add `SIMILAR_ARTISTS` and `SIMILAR_TRACKS` to ProviderFeature enum (upstream)
- [ ] Add `get_similar_artists()` and `get_similar_tracks()` to MetadataProvider base
- [ ] **Add `get_similar_artists()` and `get_similar_tracks()` to Metadata controller (REQUIRED)**
- [ ] Change Last.fm base class from MusicProvider to MetadataProvider
- [ ] Update Last.fm supported features
- [ ] Implement `get_similar_artists()` in Last.fm provider
- [ ] Implement `get_similar_tracks()` in Last.fm provider
- [ ] Add API client methods for similarity
- [ ] Add resolution helper methods
- [ ] Keep existing `recommendations()` method as-is
- [ ] Update manifest type to "metadata"
- [ ] Unit tests for new methods
- [ ] Unit tests for controller aggregation
- [ ] Integration tests for backward compatibility
- [ ] Documentation updates

### Phase 2: Adoption (Optional)
**Outcome**: Other parts of Music Assistant use similarity data through controller

- [ ] Update radio mode to use similarity data via metadata controller
- [ ] Add "Related Artists" UI feature using metadata controller
- [ ] Additional metadata providers implement similarity (if desired)

### Phase 3: Optimization (Future)
**Outcome**: Streamlined implementation, possible deprecations

- [ ] Consider refactoring `recommendations()` to use similarity methods internally
- [ ] Consider deprecating `recommendations()` in favor of similarity methods
- [ ] Consider renaming provider domain to `lastfm`
- [ ] Consider merging with `lastfm_scrobble` provider (both use Last.fm API)

---

## Success Metrics

### Functional Metrics
- [ ] Last.fm provider loads successfully as MetadataProvider
- [ ] `get_similar_artists()` returns valid results
- [ ] `get_similar_tracks()` returns valid results
- [ ] Existing `recommendations()` method still works (if kept)
- [ ] No regressions in existing Last.fm functionality

### Quality Metrics
- [ ] Similarity results match Last.fm web UI quality
- [ ] Resolution accuracy >70% (similar to existing recommendations)
- [ ] Cache hit ratio >80%

### Technical Metrics
- [ ] Query latency <500ms P95
- [ ] No memory leaks
- [ ] API rate limits respected

---

## Risks and Mitigation

### High Risk

**Risk**: Breaking existing Last.fm Recommendations users
- **Impact**: Users lose recommendation features
- **Mitigation**: Keep `recommendations()` method, comprehensive testing
- **Detection**: User reports, integration tests

### Medium Risk

**Risk**: Provider domain change breaks configurations
- **Impact**: Users must reconfigure provider
- **Mitigation**: Keep domain as `lastfm_recommendations` for now
- **Detection**: Config loading errors

### Low Risk

**Risk**: Similarity methods are unused by other parts of MA
- **Impact**: Refactor provides no immediate value
- **Mitigation**: Document use cases, provide examples
- **Detection**: Code usage analysis

---

## Alternatives Considered

### Alternative 1: Keep as MusicProvider, Add Similarity Methods

**Approach**: Keep Last.fm as MusicProvider but add similarity methods directly

**Pros:**
- No breaking changes
- Simpler migration

**Cons:**
- Wrong abstraction (MusicProvider doesn't provide music)
- Can't standardize similarity across providers
- Violates single responsibility principle

**Rejected because**: Doesn't solve the architectural problem

### Alternative 2: Create New SimilarityProvider Type

**Approach**: Create a new provider type specifically for similarity

**Pros:**
- Very clear separation of concerns
- Could have additional similarity-specific features

**Cons:**
- More complex type hierarchy
- Similarity is fundamentally metadata
- Requires more boilerplate

**Rejected because**: Over-engineering, similarity fits naturally in MetadataProvider

### Alternative 3: Make Similarity a Service, Not Provider

**Approach**: Create a SimilarityService that aggregates from various sources

**Pros:**
- Centralized similarity logic
- Could implement advanced algorithms

**Cons:**
- Doesn't leverage existing provider architecture
- Harder to add new data sources
- Less flexible

**Rejected because**: Provider pattern already exists and works well

---

## Conclusion

This refactoring transforms Last.fm from a specialized recommendation provider into a standard metadata provider with similarity capabilities. The approach:

1. **Establishes a pattern**: Any metadata provider can now offer similarity data
2. **Maintains backward compatibility**: Existing functionality is preserved
3. **Enables future features**: Radio mode and other features can leverage similarity
4. **Stays self-contained**: All Last.fm logic remains within the Last.fm provider
5. **Follows MA architecture**: Uses existing provider, cache, and resolution patterns

The phased rollout ensures no breaking changes while enabling powerful new capabilities. Phase 1 delivers a complete, tested implementation that improves architecture without disrupting users.
