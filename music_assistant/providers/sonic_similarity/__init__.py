"""Sonic Similarity plugin for Music Assistant.

Provides similarity search, backfill, USearch indexing, and a debug UI
on top of signatures produced by the sonic_analysis provider.
"""

from __future__ import annotations

import asyncio
import json
import time
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

USEARCH_INDEX_FILENAME = "sonic_signatures.usearch"
CONF_MAX_CONCURRENT = "max_concurrent_analyses"

SIMILARITY_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "rhythm": 1.0,
        "loudness": 1.0,
        "timbre": 1.0,
        "regularity": 1.0,
        "tonal": 1.0,
        "dynamics": 1.0,
    },
    "vibe": {
        "rhythm": 0.3,
        "loudness": 0.5,
        "timbre": 1.0,
        "regularity": 0.3,
        "tonal": 0.5,
        "dynamics": 0.8,
    },
    "party": {
        "rhythm": 1.0,
        "loudness": 0.5,
        "timbre": 0.3,
        "regularity": 0.8,
        "tonal": 0.2,
        "dynamics": 0.3,
    },
    "genre_era": {
        "rhythm": 0.5,
        "loudness": 0.5,
        "timbre": 0.5,
        "regularity": 0.5,
        "tonal": 0.8,
        "dynamics": 0.5,
    },
    "discover": {
        "rhythm": 0.5,
        "loudness": 0.7,
        "timbre": 1.0,
        "regularity": 0.5,
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
            key=CONF_MAX_CONCURRENT,
            type=ConfigEntryType.INTEGER,
            default_value=1,
            label="Max concurrent analyses",
            description="Maximum number of tracks to analyze simultaneously during backfill. "
            "Higher values speed up backfill but use more CPU and memory.",
            range=(1, 8),
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
        """Register API commands and rebuild the search index."""
        self._unregister_handles.append(
            self.mass.register_api_command("sonic_analysis/similar", self._handle_similar)
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "sonic_analysis/trigger_backfill", self._handle_trigger_backfill
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command("sonic_analysis/status", self._handle_status)
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "sonic_analysis/rebuild_index", self._handle_rebuild_index
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command(
                "sonic_analysis/analyzed_tracks", self._handle_analyzed_tracks
            )
        )
        self._unregister_handles.append(
            self.mass.register_api_command("sonic_analysis/clear_all", self._handle_clear_all)
        )
        self.mass.webserver.register_dynamic_route("/sonic_analysis/debug", self._serve_debug_page)
        self.logger.info("Sonic Similarity loaded, rebuilding search index from database...")
        await self._rebuild_search_index()
        self.logger.info(
            "Search index ready: %d signatures cached, corpus_stats=%s",
            len(self._signature_cache),
            self.corpus_means is not None,
        )

    async def unload(self, is_removed: bool = False) -> None:
        """Unregister API commands, dynamic routes, and save the search index."""
        for unregister in self._unregister_handles:
            unregister()
        self._unregister_handles.clear()
        self.mass.webserver.unregister_dynamic_route("/sonic_analysis/debug")
        await asyncio.to_thread(self._save_search_index)
        await super().unload(is_removed)

    # --- API handlers ---

    async def _serve_debug_page(self, request: Any) -> Any:
        """Serve the debug console HTML page."""
        from aiohttp import web  # noqa: PLC0415

        return web.Response(text=_DEBUG_HTML, content_type="text/html")

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

        if params["resolve"]:
            items = await self._resolve_results(final_items)
        else:
            items = [
                {"item_id": cid, "provider": prov, "distance": round(dist, 4), "generation": gen}
                for cid, prov, dist, gen in final_items
            ]

        return {
            "analyzed": True,
            "seed_track_ids": valid_seed_ids,
            "blend_mode": params["blend_mode"],
            "depth": params["depth"],
            "items": items,
        }

    async def _handle_trigger_backfill(self) -> dict[str, str]:
        """Start a background task to analyze unanalyzed library tracks."""
        self.mass.tasks.run_background_task(
            task_id="sonic_analysis_backfill",
            name="Sonic Analysis: library backfill",
            handler=self._backfill_library,
            allow_cancel=True,
            allow_retry=True,
        )
        return {"status": "backfill_started"}

    async def _handle_status(self) -> dict[str, Any]:
        """Return current analysis status."""
        index_size = len(self._search_index) if self._search_index is not None else 0
        return {
            "index_size": index_size,
            "has_corpus_stats": self.corpus_means is not None,
            "cached_signatures": len(self._signature_cache),
        }

    async def _handle_rebuild_index(self) -> dict[str, Any]:
        """Rebuild the USearch index from stored analysis data."""
        await self._rebuild_search_index()
        index_size = len(self._search_index) if self._search_index is not None else 0
        return {"status": "rebuilt", "index_size": index_size}

    async def _handle_clear_all(self) -> dict[str, str]:
        """Delete all stored signatures and reset the search index."""
        assert self.mass.music.database is not None
        await self.mass.music.database.execute(
            "DELETE FROM audio_analysis WHERE aa_provider_domain = :domain",
            {"domain": "sonic_analysis"},
        )
        await self.mass.music.database.commit()
        self._label_map.clear()
        self._reverse_label_map.clear()
        self._next_label = 1
        self._signature_cache.clear()
        self.corpus_means = None
        self.corpus_stds = None
        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME
        index_path.unlink(missing_ok=True)
        self._search_index = None
        self._init_search_index()
        self.logger.info("Cleared all sonic analysis data")
        return {"status": "cleared"}

    async def _handle_analyzed_tracks(
        self,
        search: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return analyzed tracks with metadata, optionally filtered by search term."""
        assert self.mass.music.database is not None
        rows = await self.mass.music.database.get_rows(
            "audio_analysis",
            {"aa_provider_domain": "sonic_analysis"},
            limit=0,
        )

        # collect unique (item_id, provider) pairs with valid signatures
        seen: set[tuple[str, str]] = set()
        entries: list[tuple[str, str]] = []
        for row in rows:
            try:
                data = AudioAnalysisData.from_dict(json.loads(row["analysis_data"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            if assemble_vector(data) is None:
                continue
            key = (row["item_id"], row["provider"])
            if key not in seen:
                seen.add(key)
                entries.append(key)

        async def _resolve(item_id: str, provider: str) -> dict[str, Any]:
            try:
                t = await self.mass.music.tracks.get(item_id, provider)
                artists = ", ".join(a.name for a in getattr(t, "artists", []) or [])
                return {"item_id": item_id, "name": t.name, "artist": artists}
            except Exception:
                return {"item_id": item_id, "name": "(unknown)", "artist": ""}

        if search:
            resolved = await asyncio.gather(*[_resolve(iid, prov) for iid, prov in entries])
            q = search.lower()
            tracks = [
                t
                for t in resolved
                if q in t["name"].lower() or q in t["artist"].lower() or q in t["item_id"]
            ]
            total = len(tracks)
            page = tracks[offset : offset + limit]
        else:
            total = len(entries)
            page_entries = entries[offset : offset + limit]
            page = list(await asyncio.gather(*[_resolve(iid, prov) for iid, prov in page_entries]))

        return {"total": total, "offset": offset, "limit": limit, "items": page}

    # --- Index management ---

    def _init_search_index(self) -> None:
        """Create or load a USearch HNSW index."""
        from usearch.index import (  # type: ignore[attr-defined]  # noqa: PLC0415
            Index,
            MetricKind,
            ScalarKind,
        )

        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME
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
        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME
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

    async def _rebuild_search_index(self) -> None:
        """Rebuild the search index from all stored analysis rows."""
        rows = await self.mass.music.database.get_rows(
            "audio_analysis", {"aa_provider_domain": "sonic_analysis"}, limit=0
        )
        if not rows:
            self.logger.info("No analysis rows found in database, skipping index rebuild")
            return

        # Reset state before rebuilding
        self._label_map.clear()
        self._reverse_label_map.clear()
        self._next_label = 1
        self._signature_cache.clear()

        # Collect signatures, deduplicating by (item_id, provider)
        all_features: list[list[float]] = []
        seen: set[tuple[str, str]] = set()
        row_entries: list[tuple[str, str, list[float]]] = []
        for row in rows:
            try:
                data = AudioAnalysisData.from_dict(json.loads(row["analysis_data"]))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
            vec = assemble_vector(data)
            if vec is None or len(vec) != VECTOR_DIMENSIONS:
                continue
            item_id = row["item_id"]
            provider = row["provider"]
            key = (item_id, provider)
            if key in seen:
                continue
            seen.add(key)
            all_features.append(vec)
            row_entries.append((item_id, provider, vec))
            self._signature_cache[item_id] = vec

        if not all_features:
            self.logger.info(
                "No valid signatures found in %d rows, skipping index rebuild", len(rows)
            )
            return

        self.corpus_means, self.corpus_stds = compute_corpus_stats(all_features)

        # Delete old index file and rebuild in a thread to avoid blocking the event loop
        index_path = Path(self.mass.storage_path) / USEARCH_INDEX_FILENAME
        index_path.unlink(missing_ok=True)
        self._search_index = None

        def _build_and_save() -> None:
            assert self.corpus_means is not None
            assert self.corpus_stds is not None
            self._init_search_index()
            for item_id, provider, features in row_entries:
                label = self._get_or_assign_label(item_id, provider)
                normalized = normalize_features(features, self.corpus_means, self.corpus_stds)
                self._add_to_index(label, normalized)
            self._save_search_index()

        await asyncio.to_thread(_build_and_save)
        self._signatures_since_rebuild = 0
        self.logger.info("Rebuilt search index with %d signatures", len(row_entries))

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
                        bonus -= genre_weight * (intersection / union_size)

            if year_weight > 0 and seed_year is not None:
                cand_year: int | None = None
                if isinstance(cand_track.album, Album) and cand_track.album.year:
                    cand_year = cand_track.album.year
                if cand_year is not None:
                    year_diff = abs(seed_year - cand_year)
                    bonus -= year_weight * (1.0 / (1.0 + year_diff * 0.1))

            scored.append((item_id, provider, features, dist + bonus, gen))

        scored.sort(key=lambda x: x[3])
        return scored

    async def _resolve_results(
        self,
        items: list[tuple[str, str, float, int]],
    ) -> list[dict[str, Any]]:
        """Resolve track metadata for result items."""

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
            return entry

        return list(
            await asyncio.gather(
                *[_resolve_one(cid, prov, dist, gen) for cid, prov, dist, gen in items]
            )
        )

    # --- Backfill ---

    async def _backfill_library(self) -> None:  # noqa: PLR0915
        """Analyze all library tracks that don't have a signature yet."""
        from music_assistant_models.enums import MediaType  # noqa: PLC0415

        from music_assistant.controllers.tasks.context import (  # noqa: PLC0415
            update_current_task_progress,
            update_current_task_progress_text,
        )

        update_current_task_progress_text("Collecting unanalyzed tracks...")
        all_tracks = await self.mass.music.tracks.library_items(limit=50000, offset=0)
        to_analyze: list[tuple[str, str]] = []

        for track in all_tracks:
            mapping = next(iter(track.provider_mappings), None)
            if mapping is None:
                continue
            version = await self.mass.streams.audio_analysis.get_audio_analysis_version(
                mapping.item_id, mapping.provider_instance, "sonic_analysis"
            )
            if version is not None and version >= 1:
                continue
            to_analyze.append((mapping.item_id, mapping.provider_instance))

        total = len(to_analyze)
        if total == 0:
            self.logger.info("Backfill: all tracks already analyzed")
            return

        self.logger.info("Backfill: %d tracks to analyze", total)
        analyzed = 0
        conf_val = self.config.get_value(CONF_MAX_CONCURRENT)
        max_concurrent = int(conf_val) if isinstance(conf_val, (int, float, str)) else 1
        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        async def _analyze_one(prov_item_id: str, prov_instance: str) -> bool:
            async with semaphore:
                try:
                    t_start = time.monotonic()
                    prov = self.mass.get_provider(prov_instance)
                    if prov is None or not hasattr(prov, "get_stream_details"):
                        return False
                    stream_details = await prov.get_stream_details(prov_item_id, MediaType.TRACK)
                    file_path = getattr(stream_details, "path", None)
                    if file_path is None:
                        return False
                    import librosa  # noqa: PLC0415

                    audio, _sr = await asyncio.to_thread(
                        librosa.load, str(file_path), sr=22050, mono=True
                    )
                    if len(audio) < 22050:
                        return False
                    from music_assistant.providers.sonic_analysis.helpers import (  # noqa: PLC0415
                        collapse_to_analysis,
                        extract_block_features,
                    )

                    bf = await asyncio.to_thread(extract_block_features, audio, 22050)
                    if bf is None:
                        return False
                    analysis = await asyncio.to_thread(collapse_to_analysis, bf, 22050)
                    await self.mass.streams.audio_analysis.set_audio_analysis(
                        item_id=prov_item_id,
                        provider_instance_id_or_domain=prov_instance,
                        aa_provider_domain="sonic_analysis",
                        analysis=analysis,
                        analysis_version=1,
                    )
                    vec = assemble_vector(analysis)
                    if vec is not None:
                        self._signature_cache[prov_item_id] = vec
                    self.logger.debug(
                        "Backfill analyzed %s (%.1fs)",
                        prov_item_id,
                        time.monotonic() - t_start,
                    )
                    return True
                except Exception as exc:
                    self.logger.debug("Backfill failed for %s: %s", prov_item_id, exc)
                    return False

        batch_size = 10
        for batch_start in range(0, total, batch_size):
            batch = to_analyze[batch_start : batch_start + batch_size]
            results = await asyncio.gather(
                *[_analyze_one(prov_item_id, prov_inst) for prov_item_id, prov_inst in batch]
            )
            analyzed += sum(1 for r in results if r)
            done = min(batch_start + batch_size, total)
            update_current_task_progress(int(done / total * 100))
            update_current_task_progress_text(f"Analyzed {analyzed}/{total} tracks...")

        await self._rebuild_search_index()
        self.logger.info("Backfill complete: analyzed %d/%d tracks", analyzed, total)


_DEBUG_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sonic Analysis Debug Console</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #e0e0e0; padding: 20px; }
  h1 { color: #7b68ee; margin-bottom: 20px; }
  h2 { color: #9b8afb; margin: 15px 0 10px; font-size: 1.1em; }
  .panel { background: #16213e; border-radius: 8px; padding: 15px; margin-bottom: 15px;
           border: 1px solid #2a2a4a; }
  label { display: block; margin-bottom: 4px; font-size: 0.85em; color: #aaa; }
  input, select { width: 100%%; padding: 8px; border-radius: 4px; border: 1px solid #3a3a5a;
                  background: #0f3460; color: #e0e0e0; font-family: monospace;
                  margin-bottom: 8px; }
  input[type="range"] { padding: 0; }
  button { padding: 8px 16px; border-radius: 4px; border: none; cursor: pointer;
           font-weight: bold; margin-right: 6px; margin-bottom: 6px; }
  .btn-primary { background: #7b68ee; color: white; }
  .btn-primary:hover { background: #6a5acd; }
  .btn-success { background: #2ecc71; color: white; }
  .btn-success:hover { background: #27ae60; }
  .btn-danger { background: #e74c3c; color: white; }
  .btn-danger:hover { background: #c0392b; }
  .btn-info { background: #3498db; color: white; }
  .btn-info:hover { background: #2980b9; }
  .btn-warn { background: #e67e22; color: white; }
  .btn-warn:hover { background: #d35400; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status { padding: 6px 12px; border-radius: 4px; font-size: 0.85em;
            display: inline-block; margin-bottom: 8px; }
  .status-connected { background: #27ae60; color: white; }
  .status-disconnected { background: #e74c3c; color: white; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
  @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  table { width: 100%%; border-collapse: collapse; margin-top: 10px; }
  th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #2a2a4a;
           font-size: 0.85em; }
  th { color: #9b8afb; }
  .slider-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
  .slider-row label { width: 70px; margin: 0; flex-shrink: 0; }
  .slider-row input[type="range"] { flex: 1; margin: 0; }
  .slider-row .slider-val { width: 35px; text-align: right; font-family: monospace;
                            font-size: 0.85em; color: #7b68ee; }
  #statusInfo { font-family: monospace; font-size: 0.9em; padding: 8px;
                background: #0f3460; border-radius: 4px; min-height: 40px; }
  #log { background: #0a0a1a; border: 1px solid #2a2a4a; border-radius: 4px; padding: 10px;
         max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.8em;
         white-space: pre-wrap; word-break: break-all; }
  .log-send { color: #3498db; }
  .log-recv { color: #2ecc71; }
  .log-error { color: #e74c3c; }
  .log-info { color: #f39c12; }
</style>
</head>
<body>

<h1>Sonic Analysis Debug Console</h1>

<!-- Connection -->
<div class="panel">
  <h2>Connection</h2>
  <label for="serverUrl">Server URL</label>
  <input type="text" id="serverUrl" placeholder="http://your-server:8095">
  <label for="authToken">Auth Token</label>
  <input type="text" id="authToken" placeholder="Paste your long-lived token here">
  <button class="btn-primary" id="btnConnect">Connect</button>
  <button class="btn-danger" id="btnDisconnect" disabled>Disconnect</button>
  <span id="connStatus" class="status status-disconnected">Disconnected</span>
</div>

<div class="grid">
  <!-- Status & Actions -->
  <div class="panel">
    <h2>Status</h2>
    <button class="btn-info" id="btnStatus" disabled>Fetch Status</button>
    <div id="statusInfo"></div>

    <h2>Actions</h2>
    <button class="btn-warn" id="btnBackfill" disabled>Trigger Backfill</button>
    <button class="btn-warn" id="btnRebuild" disabled>Rebuild Index</button>
    <button class="btn-danger" id="btnClearAll" disabled>Clear All</button>
    <div id="actionResult" style="font-family:monospace;font-size:0.85em;margin-top:8px;"></div>
  </div>

  <!-- Similar Tracks -->
  <div class="panel">
    <h2>Find Similar Tracks</h2>
    <label for="itemId">Item ID (library track ID)</label>
    <input type="text" id="itemId" placeholder="e.g. 42">
    <label for="preset">Preset</label>
    <select id="preset">
      <option value="balanced">balanced</option>
      <option value="vibe">vibe</option>
      <option value="party">party</option>
      <option value="genre_era">genre_era</option>
      <option value="discover">discover</option>
    </select>
    <h2>Weights</h2>
    <div id="sliders"></div>
    <h2>Search Options</h2>
    <label for="limit">Limit</label>
    <input type="number" id="limit" value="25" min="1" max="100">
    <label for="depth">Depth (1-5)</label>
    <input type="number" id="depth" value="1" min="1" max="5">
    <label for="branchFactor">Branch Factor</label>
    <input type="number" id="branchFactor" value="5" min="1" max="20">
    <label for="blendMode">Blend Mode</label>
    <select id="blendMode">
      <option value="centroid">centroid</option>
      <option value="union">union</option>
    </select>
    <div class="slider-row">
      <label>diversity</label>
      <input type="range" id="diversity" min="0" max="100" value="0">
      <span class="slider-val" id="diversityVal">0</span>
    </div>
    <button class="btn-success" id="btnSearch" disabled>Search Similar</button>
    <div id="results"></div>
  </div>
</div>

<!-- Analyzed Tracks -->
<div class="panel">
  <h2>Analyzed Tracks</h2>
  <div style="display:flex;gap:8px;margin-bottom:8px;">
    <input type="text" id="trackSearch" placeholder="Search by name, artist, or ID..."
           style="flex:1;margin:0;">
    <button class="btn-info" id="btnLoadTracks" disabled>Load</button>
  </div>
  <div id="trackTableWrap" style="max-height:400px;overflow-y:auto;"></div>
  <div id="trackPaging" style="margin-top:8px;font-size:0.85em;"></div>
</div>

<!-- Log -->
<div class="panel">
  <h2>Activity Log</h2>
  <button class="btn-danger" id="btnClearLog">Clear Log</button>
  <div id="log"></div>
</div>

<script>
(function() {
  var ws = null;
  var msgId = 0;
  var pending = new Map();

  var PRESETS = {
    balanced: {
      timbre: 100, harmony: 100, texture: 100,
      rhythm: 100, energy: 100, genre: 0, year: 0
    },
    vibe: {
      timbre: 80, harmony: 50, texture: 60,
      rhythm: 30, energy: 100, genre: 0, year: 0
    },
    party: {
      timbre: 30, harmony: 20, texture: 30,
      rhythm: 100, energy: 80, genre: 0, year: 0
    },
    genre_era: {
      timbre: 50, harmony: 50, texture: 50,
      rhythm: 50, energy: 50, genre: 80, year: 60
    },
    discover: {
      timbre: 100, harmony: 80, texture: 70,
      rhythm: 50, energy: 70, genre: 0, year: 0
    }
  };

  var WEIGHT_NAMES = ['timbre', 'harmony', 'texture', 'rhythm', 'energy', 'genre', 'year'];

  var slidersDiv = document.getElementById('sliders');
  WEIGHT_NAMES.forEach(function(name) {
    var row = document.createElement('div');
    row.className = 'slider-row';
    var lbl = document.createElement('label');
    lbl.textContent = name;
    var slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '100';
    slider.value = String(PRESETS.balanced[name]);
    slider.id = 'w_' + name;
    var val = document.createElement('span');
    val.className = 'slider-val';
    val.id = 'v_' + name;
    val.textContent = slider.value;
    slider.addEventListener('input', function() {
      val.textContent = slider.value;
    });
    row.appendChild(lbl);
    row.appendChild(slider);
    row.appendChild(val);
    slidersDiv.appendChild(row);
  });

  document.getElementById('diversity').addEventListener('input', function() {
    document.getElementById('diversityVal').textContent = this.value;
  });

  document.getElementById('preset').addEventListener('change', function() {
    var p = PRESETS[this.value] || PRESETS.balanced;
    WEIGHT_NAMES.forEach(function(name) {
      var s = document.getElementById('w_' + name);
      var v = document.getElementById('v_' + name);
      s.value = String(p[name]);
      v.textContent = String(p[name]);
    });
  });

  function logMsg(msg, cls) {
    var el = document.getElementById('log');
    var entry = document.createElement('div');
    entry.className = cls || '';
    entry.textContent = '[' + new Date().toLocaleTimeString() + '] ' + msg;
    el.appendChild(entry);
    el.scrollTop = el.scrollHeight;
  }

  function setConnected(connected) {
    var st = document.getElementById('connStatus');
    st.textContent = connected ? 'Connected' : 'Disconnected';
    st.className = 'status ' + (connected ? 'status-connected' : 'status-disconnected');
    document.getElementById('btnConnect').disabled = connected;
    document.getElementById('btnDisconnect').disabled = !connected;
    var btns = ['btnStatus', 'btnBackfill', 'btnRebuild', 'btnClearAll',
      'btnSearch', 'btnLoadTracks'];
    btns.forEach(function(id) { document.getElementById(id).disabled = !connected; });
  }

  function sendCommand(command, args) {
    return new Promise(function(resolve, reject) {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        reject(new Error('Not connected'));
        return;
      }
      msgId++;
      var msg = { message_id: String(msgId), command: command };
      if (args !== undefined) { msg.args = args; }
      logMsg('SEND: ' + command + ' ' + JSON.stringify(args || {}), 'log-send');
      pending.set(String(msgId), { resolve: resolve, reject: reject });
      ws.send(JSON.stringify(msg));
    });
  }

  function connect() {
    var url = document.getElementById('serverUrl').value.trim().replace(/\\/$/, '');
    var token = document.getElementById('authToken').value.trim();
    if (!url) { logMsg('Enter a server URL', 'log-error'); return; }
    var wsUrl = url.replace(/^http/, 'ws') + '/ws';
    logMsg('Connecting to ' + wsUrl + '...', 'log-info');
    ws = new WebSocket(wsUrl);
    ws.onopen = function() {
      logMsg('WebSocket opened, authenticating...', 'log-info');
      var authArgs = {};
      if (token) { authArgs.token = token; }
      sendCommand('auth', authArgs).then(function() {
        logMsg('Authenticated', 'log-info');
        setConnected(true);
        fetchStatus();
      }).catch(function(err) {
        logMsg('Auth failed: ' + err, 'log-error');
      });
    };
    ws.onmessage = function(event) {
      var data;
      try { data = JSON.parse(event.data); } catch(e) { return; }
      if (data.message_id && pending.has(data.message_id)) {
        var p = pending.get(data.message_id);
        pending.delete(data.message_id);
        if (data.error_code) {
          var em = 'ERROR [' + data.message_id + ']: ';
          logMsg(em + (data.details || data.error_code), 'log-error');
          p.reject(new Error(data.details || data.error_code));
        } else {
          var rm = 'RECV [' + data.message_id + ']: ';
          var rs = JSON.stringify(data.result || {});
          logMsg(rm + rs.substring(0, 200), 'log-recv');
          p.resolve(data.result);
        }
      }
    };
    ws.onclose = function() {
      logMsg('WebSocket closed', 'log-info');
      setConnected(false);
      ws = null;
    };
    ws.onerror = function() {
      logMsg('WebSocket error', 'log-error');
    };
  }

  function disconnect() {
    if (ws) { ws.close(); }
  }

  function fetchStatus() {
    sendCommand('sonic_analysis/status').then(function(result) {
      var el = document.getElementById('statusInfo');
      el.textContent = 'Index size: ' + (result.index_size || 0) +
        ' | Has corpus stats: ' + (result.has_corpus_stats || false) +
        ' | Cached signatures: ' + (result.cached_signatures || 0);
    }).catch(function(err) {
      document.getElementById('statusInfo').textContent = 'Error: ' + err;
    });
  }

  function triggerBackfill() {
    sendCommand('sonic_analysis/trigger_backfill').then(function(result) {
      document.getElementById('actionResult').textContent = JSON.stringify(result);
    }).catch(function(err) {
      document.getElementById('actionResult').textContent = 'Error: ' + err;
    });
  }

  function rebuildIndex() {
    sendCommand('sonic_analysis/rebuild_index').then(function(result) {
      document.getElementById('actionResult').textContent = JSON.stringify(result);
      fetchStatus();
    }).catch(function(err) {
      document.getElementById('actionResult').textContent = 'Error: ' + err;
    });
  }

  function clearAll() {
    if (!confirm('Delete ALL sonic analysis data? You will need to re-analyze your library.')) {
      return;
    }
    sendCommand('sonic_analysis/clear_all').then(function(result) {
      document.getElementById('actionResult').textContent = JSON.stringify(result);
      fetchStatus();
    }).catch(function(err) {
      document.getElementById('actionResult').textContent = 'Error: ' + err;
    });
  }

  function searchSimilar() {
    var itemId = document.getElementById('itemId').value.trim();
    if (!itemId) { logMsg('Enter an Item ID', 'log-error'); return; }
    var preset = document.getElementById('preset').value;
    var limit = parseInt(document.getElementById('limit').value, 10) || 25;
    var depth = parseInt(document.getElementById('depth').value, 10) || 1;
    var branchFactor = parseInt(document.getElementById('branchFactor').value, 10) || 5;
    var blendMode = document.getElementById('blendMode').value;
    var diversity = parseInt(document.getElementById('diversity').value, 10) / 100;
    var args = { item_id: itemId, preset: preset, limit: limit,
      depth: depth, branch_factor: branchFactor, blend_mode: blendMode,
      diversity: diversity, resolve: true };
    WEIGHT_NAMES.forEach(function(name) {
      var v = parseInt(document.getElementById('w_' + name).value, 10);
      args[name + '_weight'] = (v / 100).toFixed(2);
    });
    var resultsEl = document.getElementById('results');
    while (resultsEl.firstChild) { resultsEl.removeChild(resultsEl.firstChild); }
    var searching = document.createElement('em');
    searching.textContent = 'Searching...';
    resultsEl.appendChild(searching);
    sendCommand('sonic_analysis/similar', args).then(function(result) {
      while (resultsEl.firstChild) { resultsEl.removeChild(resultsEl.firstChild); }
      if (!result.analyzed) {
        var msg = document.createElement('em');
        msg.textContent = 'Track not analyzed yet.';
        resultsEl.appendChild(msg);
        return;
      }
      var items = result.items || [];
      if (items.length === 0) {
        var noResults = document.createElement('em');
        noResults.textContent = 'No similar tracks found.';
        resultsEl.appendChild(noResults);
        return;
      }
      var tbl = document.createElement('table');
      var thead = document.createElement('tr');
      ['#', 'Name', 'Artist', 'Distance', 'Gen'].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        thead.appendChild(th);
      });
      tbl.appendChild(thead);
      items.forEach(function(item, i) {
        var tr = document.createElement('tr');
        var td1 = document.createElement('td'); td1.textContent = String(i + 1);
        var td2 = document.createElement('td'); td2.textContent = item.name || item.item_id;
        var td3 = document.createElement('td'); td3.textContent = item.artist || '-';
        var td4 = document.createElement('td'); td4.textContent = String(item.distance);
        var td5 = document.createElement('td'); td5.textContent = item.generation != null ? String(item.generation) : '-';
        tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4); tr.appendChild(td5);
        tbl.appendChild(tr);
      });
      resultsEl.appendChild(tbl);
    }).catch(function(err) {
      while (resultsEl.firstChild) { resultsEl.removeChild(resultsEl.firstChild); }
      var errMsg = document.createElement('em');
      errMsg.textContent = 'Error: ' + err;
      resultsEl.appendChild(errMsg);
    });
  }

  var trackOffset = 0;
  var trackLimit = 50;

  function loadAnalyzedTracks(offset) {
    trackOffset = offset || 0;
    var search = document.getElementById('trackSearch').value.trim();
    var args = { limit: trackLimit, offset: trackOffset };
    if (search) { args.search = search; }
    var wrap = document.getElementById('trackTableWrap');
    wrap.textContent = 'Loading...';
    sendCommand('sonic_analysis/analyzed_tracks', args).then(function(result) {
      wrap.textContent = '';
      var items = result.items || [];
      if (items.length === 0) {
        wrap.textContent = 'No analyzed tracks found.';
        document.getElementById('trackPaging').textContent = '';
        return;
      }
      var tbl = document.createElement('table');
      var thead = document.createElement('tr');
      ['ID', 'Name', 'Artist', ''].forEach(function(h) {
        var th = document.createElement('th');
        th.textContent = h;
        thead.appendChild(th);
      });
      tbl.appendChild(thead);
      items.forEach(function(item) {
        var tr = document.createElement('tr');
        var td1 = document.createElement('td'); td1.textContent = item.item_id;
        var td2 = document.createElement('td'); td2.textContent = item.name || '-';
        var td3 = document.createElement('td'); td3.textContent = item.artist || '-';
        var td4 = document.createElement('td');
        var btn = document.createElement('button');
        btn.className = 'btn-info';
        btn.textContent = 'Find Similar';
        btn.style.padding = '3px 8px';
        btn.style.fontSize = '0.8em';
        btn.addEventListener('click', function() {
          document.getElementById('itemId').value = item.item_id;
          searchSimilar();
          window.scrollTo({top: 0, behavior: 'smooth'});
        });
        td4.appendChild(btn);
        tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
        tbl.appendChild(tr);
      });
      wrap.appendChild(tbl);
      var total = result.total || 0;
      var paging = document.getElementById('trackPaging');
      var page = Math.floor(trackOffset / trackLimit) + 1;
      var pages = Math.ceil(total / trackLimit);
      paging.textContent = 'Showing ' + (trackOffset + 1) + '-' +
        Math.min(trackOffset + trackLimit, total) + ' of ' + total;
      if (trackOffset > 0) {
        var prevLink = document.createElement('a');
        prevLink.href = '#';
        prevLink.textContent = ' | Prev';
        prevLink.addEventListener('click', function(e) {
          e.preventDefault();
          loadAnalyzedTracks(Math.max(0, trackOffset - trackLimit));
        });
        paging.appendChild(prevLink);
      }
      if (trackOffset + trackLimit < total) {
        var nextLink = document.createElement('a');
        nextLink.href = '#';
        nextLink.textContent = ' | Next';
        nextLink.addEventListener('click', function(e) {
          e.preventDefault();
          loadAnalyzedTracks(trackOffset + trackLimit);
        });
        paging.appendChild(nextLink);
      }
    }).catch(function(err) {
      wrap.textContent = 'Error: ' + err;
    });
  }

  document.getElementById('btnConnect').addEventListener('click', connect);
  document.getElementById('btnDisconnect').addEventListener('click', disconnect);
  document.getElementById('btnStatus').addEventListener('click', fetchStatus);
  document.getElementById('btnBackfill').addEventListener('click', triggerBackfill);
  document.getElementById('btnRebuild').addEventListener('click', rebuildIndex);
  document.getElementById('btnClearAll').addEventListener('click', clearAll);
  document.getElementById('btnSearch').addEventListener('click', searchSimilar);
  document.getElementById('btnLoadTracks').addEventListener('click', function() {
    loadAnalyzedTracks(0);
  });
  document.getElementById('trackSearch').addEventListener('keyup', function(e) {
    if (e.key === 'Enter') { loadAnalyzedTracks(0); }
  });
  document.getElementById('btnClearLog').addEventListener('click', function() {
    var el = document.getElementById('log');
    while (el.firstChild) { el.removeChild(el.firstChild); }
  });
})();
</script>
</body>
</html>
"""
