"""Sonic Similarity plugin for Music Assistant.

Provides similarity search and USearch indexing on top of signatures
produced by audio_analysis-type providers (default: sonic_analysis).
Engine:

  - 18-dim weighted Euclidean over engineered audio features
    (assembled via vectors.assemble_vector), with preset weights, MMR
    diversity, depth/branch_factor recursive expansion, and optional
    metadata reranking. Surfaced as sonic_similarity/similar.

The 1024-dim CLAP-embedding similarity engine lives in the separate
sonic_clap plugin, which manages the CLAP usearch index and reads the
persisted embeddings from audio_analysis.extra_data.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from music_assistant_models.media_items import Album

from music_assistant.models.audio_analysis import AudioAnalysisData
from music_assistant.models.plugin import PluginProvider
from music_assistant.providers.sonic_similarity.similarity import (
    apply_mmr,
    combine_seeds_centroid,
    expand_recursive,
    merge_union_results,
)
from music_assistant.providers.sonic_similarity.vectors import (
    VECTOR_DIMENSIONS,
    assemble_vector,
    compute_corpus_stats,
    compute_weighted_distance,
    normalize_features,
)

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigEntry, ConfigValueType, ProviderConfig
    from music_assistant_models.media_items import Track
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

USEARCH_INDEX_FILENAME_TPL = "sonic_signatures_{domain}.usearch"
CONF_AA_PROVIDER = "aa_provider_domain"
BACKGROUND_SCAN_TASK_ID = "audio_analysis_background_scan"

# Overlay registry: each aa_provider_domain listed here declares which
# AudioAnalysisData fields it owns. At vector-assembly time, rows from
# these sources are overlaid onto the primary provider's row. Skipped
# when the primary aa_provider_domain equals an overlay source (no
# self-overlay). Silent fallback when a track has no overlay row.
#
#   smart_fades — BPM from beat_this CNN, key/mode from S-KEY;
#                  strictly better than librosa's heuristic versions.
#
# Note: sonic_analysis used to have an overlay entry for CLAP zero-shot
# soft scalars (a separate clap_analysis provider). That provider has
# been merged into sonic_analysis itself — one audio load per track
# produces both the librosa measurements and the CLAP scalars, written
# to a single audio_analysis row. No cross-provider overlay needed.
OVERLAY_SOURCES: dict[str, tuple[str, ...]] = {
    "smart_fades": ("bpm", "key", "mode"),
}

# Scale factor applied to the genre/year metadata bonus in
# _apply_metadata_reranking. Without scaling the raw genre Jaccard +
# year-proximity terms reach magnitudes ~10-20x the audio-similarity
# distance and dominate ranking entirely, making preset weight changes
# invisible. 0.1 brings max combined bonus to ~0.2 - comparable to the
# audio-distance dynamic range, so categorical context still nudges
# results without overriding the audio sliders.
METADATA_BONUS_SCALE: float = 0.1

SIMILARITY_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "rhythm": 1.0,
        "loudness": 1.0,
        "timbre": 1.0,
        "regularity": 1.0,
        "mood": 1.0,
        "tonal": 1.0,
        "dynamics": 1.0,
    },
    "vibe": {
        "rhythm": 0.3,
        "loudness": 0.5,
        "timbre": 1.0,
        "regularity": 0.3,
        "mood": 1.0,
        "tonal": 0.5,
        "dynamics": 0.8,
    },
    "party": {
        "rhythm": 1.0,
        "loudness": 0.5,
        "timbre": 0.3,
        "regularity": 0.8,
        "mood": 0.5,
        "tonal": 0.2,
        "dynamics": 0.3,
    },
    "genre_era": {
        "rhythm": 0.5,
        "loudness": 0.5,
        "timbre": 0.5,
        "regularity": 0.5,
        "mood": 0.8,
        "tonal": 0.8,
        "dynamics": 0.5,
    },
    "discover": {
        "rhythm": 0.5,
        "loudness": 0.7,
        "timbre": 1.0,
        "regularity": 0.5,
        "mood": 0.8,
        "tonal": 0.8,
        "dynamics": 0.7,
    },
}


def _parse_weights(params: dict[str, Any]) -> dict[str, float]:
    """Parse similarity weights from API parameters."""
    preset_name = str(params.get("preset", "balanced"))
    preset = SIMILARITY_PRESETS.get(preset_name, SIMILARITY_PRESETS["balanced"])
    result = dict(preset)  # copy

    def _clamp(val: str, fallback: float) -> float:
        try:
            return max(0.0, min(1.0, float(val)))
        except (ValueError, TypeError):
            return fallback

    for group, default in result.items():
        key = f"{group}_weight"
        if key in params:
            result[group] = _clamp(params[key], default)

    return result


def _parse_similar_params(  # noqa: PLR0913
    item_id: str | None = None,
    item_ids: list[str] | None = None,
    limit: int = 25,
    depth: int = 1,
    branch_factor: int = 5,
    blend_mode: str = "centroid",
    seed_weights: list[float] | None = None,
    diversity: float = 0.0,
    preset: str = "balanced",
    candidates: int = 50,
    filter_genres: list[str] | None = None,
    filter_providers: list[str] | None = None,
    exclude_track_ids: list[str] | None = None,
    exclude_artists: list[str] | None = None,
    resolve: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Validate and normalize parameters for the similar endpoint."""
    if item_ids is None:
        if item_id is None:
            msg = "Either item_id or item_ids must be provided"
            raise ValueError(msg)
        item_ids = [item_id]

    limit = max(1, min(100, limit))
    depth = max(1, min(5, depth))
    diversity = max(0.0, min(1.0, diversity))

    if blend_mode not in ("centroid", "union"):
        blend_mode = "centroid"

    if seed_weights is not None and len(seed_weights) != len(item_ids):
        msg = f"seed_weights length ({len(seed_weights)}) must match item_ids ({len(item_ids)})"
        raise ValueError(msg)

    has_filters = any(
        x is not None for x in (filter_genres, filter_providers, exclude_track_ids, exclude_artists)
    )
    if has_filters:
        candidates = candidates * 2

    return {
        "item_ids": item_ids,
        "limit": limit,
        "depth": depth,
        "branch_factor": branch_factor,
        "blend_mode": blend_mode,
        "seed_weights": seed_weights,
        "diversity": diversity,
        "preset": preset,
        "candidates": candidates,
        "filter_genres": filter_genres,
        "filter_providers": filter_providers,
        "exclude_track_ids": exclude_track_ids,
        "exclude_artists": exclude_artists,
        "resolve": resolve,
        "kwargs": kwargs,
    }


def apply_filters(
    candidates: list[tuple[str, str, float]],
    seed_ids: set[str],
    exclude_track_ids: set[str] | None,
    filter_providers: set[str] | None,
) -> list[tuple[str, str, float]]:
    """Apply cheap post-ANN filters to candidate list.

    :param candidates: List of (item_id, provider, distance) tuples.
    :param seed_ids: Seed track IDs to exclude.
    :param exclude_track_ids: Additional track IDs to exclude.
    :param filter_providers: If set, only keep candidates from these providers.
    """
    result: list[tuple[str, str, float]] = []
    exclude = seed_ids | (exclude_track_ids or set())

    for item_id, provider, dist in candidates:
        if item_id in exclude:
            continue
        if filter_providers is not None and provider not in filter_providers:
            continue
        result.append((item_id, provider, dist))

    return result


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider instance with given configuration."""
    return SonicSimilarityPlugin(mass, manifest, config)


async def get_config_entries(
    mass: MusicAssistant,  # noqa: ARG001
    instance_id: str | None = None,  # noqa: ARG001
    action: str | None = None,  # noqa: ARG001
    values: dict[str, ConfigValueType] | None = None,  # noqa: ARG001
) -> tuple[ConfigEntry, ...]:
    """Return Config entries to setup this provider.

    :param mass: MusicAssistant instance.
    :param instance_id: id of an existing provider instance (None if new instance setup).
    :param action: action key called from config entries UI.
    :param values: the (intermediate) raw values for config entries sent with the action.
    """
    from music_assistant_models.config_entries import ConfigEntry  # noqa: PLC0415
    from music_assistant_models.enums import ConfigEntryType  # noqa: PLC0415

    return (
        ConfigEntry(
            key=CONF_AA_PROVIDER,
            type=ConfigEntryType.STRING,
            default_value="sonic_analysis",
            label="Analysis Provider",
            description="Which audio analysis provider's data to use for similarity vectors. "
            "Default: sonic_analysis (librosa + CLAP, on-device).",
        ),
    )


class SonicSimilarityPlugin(PluginProvider):
    """Plugin that provides similarity search over sonic analysis signatures."""

    def __init__(
        self,
        mass: MusicAssistant,
        manifest: ProviderManifest,
        config: ProviderConfig,
    ) -> None:
        """Initialize the Sonic Similarity plugin."""
        super().__init__(mass, manifest, config)
        self._aa_domain: str = "sonic_analysis"
        self._indexes: dict[str, Any] = {}
        self._corpus_stats: dict[str, tuple[list[float], list[float]]] = {}
        self._label_map: dict[int, tuple[str, str]] = {}
        self._reverse_label_map: dict[tuple[str, str], int] = {}
        self._next_label: int = 1
        self._signature_cache: dict[str, list[float]] = {}
        self.corpus_means: list[float] | None = None
        self.corpus_stds: list[float] | None = None
        self._search_index: Any = None
        self._signatures_since_rebuild: int = 0
        self._unregister_handles: list[Callable[[], None]] = []

    async def loaded_in_mass(self) -> None:
        """Register similarity API commands and build the 18-dim search index."""
        self._unregister_handles.append(
            self.mass.register_api_command("sonic_similarity/similar", self._handle_similar)
        )
        self._unregister_handles.append(
            self.mass.register_api_command("sonic_similarity/status", self._handle_status)
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "sonic_similarity/rebuild_index", self._handle_rebuild_index
            )
        )
        self._aa_domain = str(self.config.get_value(CONF_AA_PROVIDER) or "sonic_analysis")
        self.logger.info(
            "Sonic Similarity loaded (aa_provider=%s), rebuilding search index...",
            self._aa_domain,
        )
        await self._rebuild_search_index()
        self.logger.info(
            "Search index ready: %d signatures cached, corpus_stats=%s",
            len(self._signature_cache),
            self.corpus_means is not None,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Unregister API commands and save the search index."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        await asyncio.to_thread(self._save_search_index)
        for cached_domain, cached in self._indexes.items():
            if cached_domain == self._aa_domain:
                continue
            old_domain = self._aa_domain
            self._aa_domain = cached_domain
            self._search_index = cached["index"]
            await asyncio.to_thread(self._save_search_index)
            self._aa_domain = old_domain
        await super().unload(is_removed)

    # --- API handlers ---

    async def _handle_similar(  # noqa: PLR0913, PLR0915
        self,
        item_id: str | None = None,
        item_ids: list[str] | None = None,
        limit: int = 25,
        depth: int = 1,
        branch_factor: int = 5,
        blend_mode: str = "centroid",
        seed_weights: list[float] | None = None,
        diversity: float = 0.0,
        preset: str = "balanced",
        candidates: int = 50,
        filter_genres: list[str] | None = None,
        filter_providers: list[str] | None = None,
        exclude_track_ids: list[str] | None = None,
        exclude_artists: list[str] | None = None,
        resolve: bool = False,
        include_group_distances: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Find tracks similar to the given track(s)."""
        params = _parse_similar_params(
            item_id=item_id,
            item_ids=item_ids,
            limit=limit,
            depth=depth,
            branch_factor=branch_factor,
            blend_mode=blend_mode,
            seed_weights=seed_weights,
            diversity=diversity,
            preset=preset,
            candidates=candidates,
            filter_genres=filter_genres,
            filter_providers=filter_providers,
            exclude_track_ids=exclude_track_ids,
            exclude_artists=exclude_artists,
            resolve=resolve,
            **kwargs,
        )
        p_item_ids: list[str] = params["item_ids"]
        weights = _parse_weights({**params.get("kwargs", {}), "preset": params["preset"]})

        seed_sigs: list[list[float]] = []
        valid_seed_ids: list[str] = []
        for sid in p_item_ids:
            sig = self._signature_cache.get(sid)
            if sig is not None:
                seed_sigs.append(sig)
                valid_seed_ids.append(sid)
            else:
                self.logger.warning("Seed %s not in signature cache, skipping", sid)

        if not seed_sigs or self.corpus_means is None or self.corpus_stds is None:
            return {
                "analyzed": False,
                "seed_track_ids": p_item_ids,
                "blend_mode": params["blend_mode"],
                "depth": params["depth"],
                "items": [],
            }

        corpus_means = self.corpus_means
        corpus_stds = self.corpus_stds

        def _search_generation(
            seeds: list[list[float]],
            seen: set[str],
        ) -> list[tuple[str, str, list[float], float]]:
            if params["blend_mode"] == "union":
                all_neighborhoods: list[list[tuple[str, float]]] = []
                for seed_vec in seeds:
                    normalized = normalize_features(seed_vec, corpus_means, corpus_stds)
                    raw = self._query_index(normalized, params["candidates"])
                    neighborhood: list[tuple[str, float]] = []
                    for lbl, cos_dist in raw:
                        if lbl not in self._label_map:
                            continue
                        cand_id, _prov = self._label_map[lbl]
                        if cand_id not in seen:
                            neighborhood.append((cand_id, cos_dist))
                    all_neighborhoods.append(neighborhood)
                candidate_ids = merge_union_results(all_neighborhoods)
            else:
                centroid = combine_seeds_centroid(seeds, params.get("seed_weights"))
                normalized = normalize_features(centroid, corpus_means, corpus_stds)
                raw = self._query_index(normalized, params["candidates"])
                candidate_ids = []
                for lbl, cos_dist in raw:
                    if lbl not in self._label_map:
                        continue
                    cand_id, _prov = self._label_map[lbl]
                    if cand_id not in seen:
                        candidate_ids.append((cand_id, cos_dist))

            raw_tuples: list[tuple[str, str, float]] = []
            for cand_id, cos_dist in candidate_ids:
                cand_provider = "library"
                for key in self._reverse_label_map:
                    if key[0] == cand_id:
                        cand_provider = key[1]
                        break
                raw_tuples.append((cand_id, cand_provider, cos_dist))

            seed_id_set = set(valid_seed_ids)
            exclude_set = set(params["exclude_track_ids"]) if params["exclude_track_ids"] else None
            filter_prov_set = (
                set(params["filter_providers"]) if params["filter_providers"] else None
            )
            filtered = apply_filters(raw_tuples, seed_id_set | seen, exclude_set, filter_prov_set)

            original_centroid = combine_seeds_centroid(seed_sigs)
            orig_normalized = normalize_features(original_centroid, corpus_means, corpus_stds)

            results: list[tuple[str, str, list[float], float]] = []
            for cand_id, cand_provider, _cos_dist in filtered:
                cand_features = self._signature_cache.get(cand_id)
                if cand_features is None:
                    continue
                cand_normalized = normalize_features(cand_features, corpus_means, corpus_stds)
                dist = compute_weighted_distance(orig_normalized, cand_normalized, weights)
                results.append((cand_id, cand_provider, cand_features, dist))

            results.sort(key=lambda x: x[3])
            return results

        raw_results = expand_recursive(
            initial_seeds=seed_sigs,
            searcher=_search_generation,
            depth=params["depth"],
            branch_factor=params["branch_factor"],
        )

        if params["filter_genres"] or params["exclude_artists"]:
            raw_results = await self._apply_metadata_filters(
                raw_results,
                filter_genres=params["filter_genres"],
                exclude_artists=params["exclude_artists"],
            )

        if weights.get("tonal", 0.0) > 0:
            raw_results = await self._apply_metadata_reranking(
                valid_seed_ids[0],
                raw_results,
                weights,
            )

        if params["diversity"] > 0:
            original_centroid = combine_seeds_centroid(seed_sigs)
            orig_normalized = normalize_features(original_centroid, corpus_means, corpus_stds)
            mmr_candidates = [
                (r[0], normalize_features(r[2], corpus_means, corpus_stds), r[3])
                for r in raw_results
            ]
            mmr_result = apply_mmr(
                mmr_candidates,
                orig_normalized,
                params["diversity"],
                params["limit"],
                weights=weights,
            )
            result_lookup = {r[0]: r for r in raw_results}
            final_items: list[tuple[str, str, float, int]] = [
                (cid, result_lookup[cid][1], dist, result_lookup[cid][4])
                for cid, dist in mmr_result
            ]
        else:
            raw_results.sort(key=lambda x: x[3])
            final_items = [(r[0], r[1], r[3], r[4]) for r in raw_results]

        final_items = final_items[: params["limit"]]

        debug_breakdown_map: dict[str, dict[str, Any]] = {}
        if include_group_distances:
            from music_assistant.providers.sonic_similarity.vectors import (  # noqa: PLC0415
                build_debug_breakdown,
            )

            original_centroid = combine_seeds_centroid(seed_sigs)
            orig_normalized = normalize_features(original_centroid, corpus_means, corpus_stds)
            for cid, _prov, displayed_dist, _gen in final_items:
                cand_features = self._signature_cache.get(cid)
                if cand_features is not None:
                    cand_normalized = normalize_features(cand_features, corpus_means, corpus_stds)
                    debug_breakdown_map[cid] = build_debug_breakdown(
                        orig_normalized, cand_normalized, weights, displayed_dist
                    )

        if params["resolve"]:
            items = await self._resolve_results(
                final_items,
                debug_breakdown_map if include_group_distances else None,
            )
        else:
            items = []
            for cid, prov, dist, gen in final_items:
                entry: dict[str, Any] = {
                    "item_id": cid,
                    "provider": prov,
                    "distance": round(dist, 4),
                    "generation": gen,
                }
                if include_group_distances and cid in debug_breakdown_map:
                    entry.update(debug_breakdown_map[cid])
                items.append(entry)

        return {
            "analyzed": True,
            "seed_track_ids": valid_seed_ids,
            "blend_mode": params["blend_mode"],
            "depth": params["depth"],
            "items": items,
        }

    async def _handle_status(self) -> dict[str, Any]:
        """Return current analysis status."""
        index_size = len(self._search_index) if self._search_index is not None else 0
        return {
            "index_size": index_size,
            "has_corpus_stats": self.corpus_means is not None,
            "cached_signatures": len(self._signature_cache),
            "aa_provider_domain": self._aa_domain,
        }

    async def _handle_rebuild_index(self) -> dict[str, Any]:
        """Rebuild the USearch index from stored analysis data."""
        await self._rebuild_search_index()
        index_size = len(self._search_index) if self._search_index is not None else 0
        return {"status": "rebuilt", "index_size": index_size}

    def _init_search_index(self) -> None:
        """Create or load a USearch HNSW index."""
        from usearch.index import (  # type: ignore[attr-defined]  # noqa: PLC0415
            Index,
            MetricKind,
            ScalarKind,
        )

        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME_TPL.format(
            domain=self._aa_domain
        )
        self._search_index = Index(
            ndim=VECTOR_DIMENSIONS,
            metric=MetricKind.Cos,
            dtype=ScalarKind.F32,
        )
        if index_path.exists():
            try:
                self._search_index.load(str(index_path))
                if self._search_index.ndim != VECTOR_DIMENSIONS:
                    self.logger.warning(
                        "Index dimension mismatch (%d vs %d), discarding stale index",
                        self._search_index.ndim,
                        VECTOR_DIMENSIONS,
                    )
                    index_path.unlink()
                    self._search_index = Index(
                        ndim=VECTOR_DIMENSIONS,
                        metric=MetricKind.Cos,
                        dtype=ScalarKind.F32,
                    )
                else:
                    self.logger.debug("Loaded USearch index from %s", index_path)
            except Exception:
                self.logger.warning("Failed to load index file, starting fresh")
                index_path.unlink(missing_ok=True)
                self._search_index = Index(
                    ndim=VECTOR_DIMENSIONS,
                    metric=MetricKind.Cos,
                    dtype=ScalarKind.F32,
                )

    def _save_search_index(self) -> None:
        """Persist the USearch index to disk."""
        if self._search_index is None:
            return
        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME_TPL.format(
            domain=self._aa_domain
        )
        self._search_index.save(str(index_path))
        self.logger.debug("Saved USearch index to %s", index_path)

    def _add_to_index(self, label: int, normalized_features: list[float]) -> None:
        """Add a vector to the search index.

        :param label: Integer label for the vector.
        :param normalized_features: Z-score normalized feature vector.
        """
        if self._search_index is None:
            self._init_search_index()
        vec = np.array(normalized_features, dtype=np.float32)
        # Remove existing entry if present (handles dimension change rebuilds)
        if label in self._search_index:
            self._search_index.remove(label)
        self._search_index.add(label, vec)

    def _query_index(self, normalized_features: list[float], k: int) -> list[tuple[int, float]]:
        """Search the index for the k nearest neighbors.

        :param normalized_features: Z-score normalized query vector.
        :param k: Number of neighbors to return.
        """
        if self._search_index is None or len(self._search_index) == 0:
            return []
        vec = np.array(normalized_features, dtype=np.float32)
        results = self._search_index.search(vec, k)
        return [
            (int(lbl), float(dist))
            for lbl, dist in zip(results.keys, results.distances, strict=False)
        ]

    # --- Index rebuild ---

    async def _load_overlay_overrides(
        self,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """Load overlay rows from every source in OVERLAY_SOURCES, merged.

        :returns: Map of (item_id, provider) → {field: value}. Populated fields
            come from each overlay source's declared field list. Sources are
            processed in OVERLAY_SOURCES registration order; later entries win
            on field conflicts (no current conflicts).
        """
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for source_domain, fields in OVERLAY_SOURCES.items():
            if self._aa_domain == source_domain:
                continue
            rows = await self.mass.streams.audio_analysis.get_audio_analysis_rows(source_domain)
            for row in rows:
                try:
                    data = AudioAnalysisData.from_dict(json.loads(row["analysis_data"]))
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
                per_track = merged.setdefault((row["item_id"], row["provider"]), {})
                for field in fields:
                    value = getattr(data, field, None)
                    if value is not None:
                        per_track[field] = value
        return merged

    @staticmethod
    def _apply_overlay_overrides(data: AudioAnalysisData, override: dict[str, Any] | None) -> None:
        """Overlay per-field values from overlay sources onto analysis data in place."""
        if not override:
            return
        for field, value in override.items():
            setattr(data, field, value)

    async def _rebuild_search_index(self) -> None:  # noqa: PLR0915
        """Rebuild the search index from all stored analysis rows."""
        rows = await self.mass.streams.audio_analysis.get_audio_analysis_rows(self._aa_domain)
        if not rows:
            self.logger.info("No analysis rows found in database, skipping index rebuild")
            return

        overlay_overrides = await self._load_overlay_overrides()

        # Reset state before rebuilding
        self._label_map.clear()
        self._reverse_label_map.clear()
        self._next_label = 1
        self._signature_cache.clear()

        # Collect signatures, deduplicating by (item_id, provider)
        all_features: list[list[float]] = []
        seen: set[tuple[str, str]] = set()
        row_entries: list[tuple[str, str, list[float]]] = []
        overlay_applied_count = 0
        for row in rows:
            try:
                data = AudioAnalysisData.from_dict(json.loads(row["analysis_data"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            item_id = row["item_id"]
            provider = row["provider"]
            per_track_override = overlay_overrides.get((item_id, provider))
            if per_track_override:
                overlay_applied_count += 1
            self._apply_overlay_overrides(data, per_track_override)
            vec = assemble_vector(data)
            if vec is None or len(vec) != VECTOR_DIMENSIONS:
                continue
            key = (item_id, provider)
            if key in seen:
                continue
            seen.add(key)
            all_features.append(vec)
            row_entries.append((item_id, provider, vec))
            self._signature_cache[item_id] = vec

        if not all_features:
            # Help the user diagnose the frustrating "250 rows, 0 signatures" case.
            # Peek at up to 3 rows and report which required fields they are missing.
            missing_report: list[str] = []
            for row in rows[:3]:
                try:
                    data = AudioAnalysisData.from_dict(json.loads(row["analysis_data"]))
                except (json.JSONDecodeError, TypeError, KeyError):
                    missing_report.append(f"{row['item_id']}: row unparsable")
                    continue
                missing = [
                    f
                    for f in (
                        "bpm",
                        "energy",
                        "danceability",
                        "loudness_integrated",
                        "loudness_range",
                        "brightness",
                        "harmonic_complexity",
                        "roughness",
                        "rhythmic_regularity",
                        "key",
                        "mode",
                    )
                    if getattr(data, f, None) is None
                ]
                missing_report.append(f"{row['item_id']}: missing {missing}")
            self.logger.info(
                "No valid signatures assembled from %d rows in domain=%s, "
                "skipping index rebuild. Sample diagnostics: %s. "
                "Common cause: current aa_provider_domain lacks required scalar fields — "
                "switch Similarity Source to sonic_analysis (which populates all "
                "required hard scalars).",
                len(rows),
                self._aa_domain,
                "; ".join(missing_report),
            )
            return

        self.corpus_means, self.corpus_stds = compute_corpus_stats(all_features)

        # Delete old index file and rebuild in a thread to avoid blocking the event loop
        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME_TPL.format(
            domain=self._aa_domain
        )
        index_path.unlink(missing_ok=True)
        self._search_index = None

        def _build_and_save() -> None:
            assert self.corpus_means is not None
            assert self.corpus_stds is not None
            # Create a fresh empty index (don't load from disk — file was already deleted)
            from usearch.index import Index, MetricKind, ScalarKind  # noqa: PLC0415

            self._search_index = Index(
                ndim=VECTOR_DIMENSIONS,
                metric=MetricKind.Cos,
                dtype=ScalarKind.F32,
            )
            for item_id, provider, features in row_entries:
                label = self._get_or_assign_label(item_id, provider)
                normalized = normalize_features(features, self.corpus_means, self.corpus_stds)
                self._add_to_index(label, normalized)
            self._save_search_index()

        await asyncio.to_thread(_build_and_save)
        self._signatures_since_rebuild = 0
        self.logger.info(
            "Rebuilt search index with %d signatures (%d with overlay fields applied)",
            len(row_entries),
            overlay_applied_count,
        )

    # --- Label mapping ---

    def _get_or_assign_label(self, item_id: str, provider: str) -> int:
        """Return (or create) a unique integer label for an item_id/provider pair.

        :param item_id: The track's item ID.
        :param provider: The music provider domain or instance ID.
        """
        key = (item_id, provider)
        if key in self._reverse_label_map:
            return self._reverse_label_map[key]
        label = self._next_label
        self._next_label += 1
        self._label_map[label] = key
        self._reverse_label_map[key] = label
        return label

    # --- Similarity helpers ---

    async def _apply_metadata_filters(
        self,
        results: list[tuple[str, str, list[float], float, int]],
        filter_genres: list[str] | None = None,
        exclude_artists: list[str] | None = None,
    ) -> list[tuple[str, str, list[float], float, int]]:
        """Apply metadata-based filters that require track resolution."""
        if not filter_genres and not exclude_artists:
            return results

        genre_set = {g.lower() for g in filter_genres} if filter_genres else None
        artist_set = {a.lower() for a in exclude_artists} if exclude_artists else None

        filtered: list[tuple[str, str, list[float], float, int]] = []
        for item_id, provider, features, dist, gen in results:
            try:
                track = await self.mass.music.tracks.get(item_id, provider)
            except Exception:  # noqa: S112
                continue

            if genre_set:
                track_genres = set()
                if track.metadata and track.metadata.genres:
                    track_genres = {g.lower() for g in track.metadata.genres}
                if not track_genres & genre_set:
                    continue

            if artist_set:
                track_artists = {a.name.lower() for a in (track.artists or [])}
                if track_artists & artist_set:
                    continue

            filtered.append((item_id, provider, features, dist, gen))
        return filtered

    async def _apply_metadata_reranking(
        self,
        seed_item_id: str,
        results: list[tuple[str, str, list[float], float, int]],
        weights: dict[str, float],
    ) -> list[tuple[str, str, list[float], float, int]]:
        """Apply genre and year bonuses to re-rank candidates."""
        try:
            seed_prov = "library"
            for key in self._reverse_label_map:
                if key[0] == seed_item_id:
                    seed_prov = key[1]
                    break
            seed_track: Track = await self.mass.music.tracks.get(seed_item_id, seed_prov)
        except Exception:
            return results

        seed_genres: set[str] = set()
        if seed_track.metadata and seed_track.metadata.genres:
            seed_genres = seed_track.metadata.genres

        seed_year: int | None = None
        if isinstance(seed_track.album, Album) and seed_track.album.year:
            seed_year = seed_track.album.year

        scored: list[tuple[str, str, list[float], float, int]] = []
        for item_id, provider, features, dist, gen in results:
            bonus = 0.0
            try:
                cand_track: Track = await self.mass.music.tracks.get(item_id, provider)
            except Exception:
                scored.append((item_id, provider, features, dist, gen))
                continue

            genre_weight = weights.get("tonal", 0.0)
            year_weight = weights.get("tonal", 0.0)
            if genre_weight > 0 and seed_genres:
                cand_genres: set[str] = set()
                if cand_track.metadata and cand_track.metadata.genres:
                    cand_genres = cand_track.metadata.genres
                if cand_genres:
                    intersection = len(seed_genres & cand_genres)
                    union_size = len(seed_genres | cand_genres)
                    if union_size > 0:
                        bonus -= METADATA_BONUS_SCALE * genre_weight * (intersection / union_size)

            if year_weight > 0 and seed_year is not None:
                cand_year: int | None = None
                if isinstance(cand_track.album, Album) and cand_track.album.year:
                    cand_year = cand_track.album.year
                if cand_year is not None:
                    year_diff = abs(seed_year - cand_year)
                    bonus -= METADATA_BONUS_SCALE * year_weight * (1.0 / (1.0 + year_diff * 0.1))

            scored.append((item_id, provider, features, dist + bonus, gen))

        scored.sort(key=lambda x: x[3])
        return scored

    async def _resolve_results(
        self,
        items: list[tuple[str, str, float, int]],
        debug_breakdown_map: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Resolve track metadata for result items.

        :param items: List of (item_id, provider, distance, generation) tuples to resolve.
        :param debug_breakdown_map: Optional per-track debug breakdown (weighted_distance,
            metadata_bonus, group_distances) keyed by item_id.
        """

        async def _resolve_one(
            item_id: str,
            provider: str,
            dist: float,
            gen: int,
        ) -> dict[str, Any]:
            entry: dict[str, Any] = {
                "item_id": item_id,
                "provider": provider,
                "distance": round(dist, 4),
                "generation": gen,
            }
            try:
                track = await self.mass.music.tracks.get(item_id, provider)
                artists = ", ".join(a.name for a in getattr(track, "artists", []) or [])
                entry["name"] = track.name
                entry["artist"] = artists
            except Exception:
                entry["name"] = "(unknown)"
                entry["artist"] = ""
            if debug_breakdown_map and item_id in debug_breakdown_map:
                entry.update(debug_breakdown_map[item_id])
            return entry

        return list(
            await asyncio.gather(
                *[_resolve_one(cid, prov, dist, gen) for cid, prov, dist, gen in items]
            )
        )
