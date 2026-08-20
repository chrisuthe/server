"""Tests for Audiobookshelf provider initialization."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aioaudiobookshelf.exceptions import LoginError as AbsLoginError
from music_assistant_models.config_entries import ProviderConfig
from music_assistant_models.enums import MediaType, ProviderFeature, ProviderType
from music_assistant_models.errors import LoginFailed

from music_assistant.models.music_provider import MusicProvider
from music_assistant.providers.audiobookshelf import Audiobookshelf
from music_assistant.providers.audiobookshelf.helpers import LibraryHelper

if TYPE_CHECKING:
    from aioaudiobookshelf.client.session_configuration import (
        SessionConfiguration as AbsSessionConfiguration,
    )


def test_supported_features_before_async_init(provider: Audiobookshelf) -> None:
    """Provider features are available before asynchronous initialization."""
    assert ProviderFeature.PLAYLIST_CREATE_AUDIOBOOKS in provider.supported_features


@pytest.mark.asyncio
async def test_verify_ssl_defaults_to_on(provider: Audiobookshelf) -> None:
    """A config with no stored verify_ssl must still verify certificates."""
    # nothing collected by a setup flow and no declared config entry to fall back on, as for
    # a config migrated from before setup flows existed
    provider.mass.config.get = Mock(return_value={})  # type: ignore[method-assign]
    provider.config = ProviderConfig(
        values={},
        type=ProviderType.MUSIC,
        domain="audiobookshelf",
        instance_id="audiobookshelf--test123",
    )
    session_configs: list[AbsSessionConfiguration] = []

    async def _capture(session_config: AbsSessionConfiguration, **_kwargs: object) -> object:
        session_configs.append(session_config)
        raise AbsLoginError("stop here")

    with (
        patch(
            "music_assistant.providers.audiobookshelf.aioabs.get_user_and_socket_client", _capture
        ),
        pytest.raises(LoginFailed),
    ):
        await provider.handle_async_init()

    assert session_configs
    assert session_configs[0].verify_ssl is True


@pytest.mark.asyncio
async def test_sync_library_clears_stale_library_ids(provider: Audiobookshelf) -> None:
    """sync_library removes stale library IDs that no longer exist on ABS."""
    assert "lib1" in provider.libraries.audiobooks
    provider.libraries.podcasts["pod_stale"] = LibraryHelper(name="Stale Podcast")

    mock_library = Mock()
    mock_library.id_ = "pod_lib_1"
    mock_library.name = "Podcasts"
    mock_library.media_type = "podcast"
    provider._client.get_all_libraries = AsyncMock(return_value=[mock_library])  # type: ignore[method-assign]
    provider._client.get_library_narrators = AsyncMock(return_value=[])  # type: ignore[method-assign]
    provider._client.get_my_user = AsyncMock()  # type: ignore[method-assign]
    provider.mass.cache.set = AsyncMock()  # type: ignore[method-assign]
    provider.mass.music.get_playlog_provider_item_ids = AsyncMock(return_value=set())  # type: ignore[method-assign]

    with patch.object(MusicProvider, "sync_library", AsyncMock()):
        await provider.sync_library(MediaType.AUDIOBOOK)

    assert len(provider.libraries.audiobooks) == 0
    # Only audiobooks are cleared — podcasts dict is untouched by an AUDIOBOOK sync
    assert "pod_stale" in provider.libraries.podcasts


@pytest.mark.asyncio
async def test_sync_library_keys_prunes_stale_and_adds_new(provider: Audiobookshelf) -> None:
    """_sync_library_keys removes stale keys and adds new ones, preserving item_ids."""
    provider.libraries.audiobooks["stale_book"] = LibraryHelper(name="Stale Book")
    provider.libraries.audiobooks["existing_book"] = LibraryHelper(
        name="Existing Book", item_ids={"id_a", "id_b"}
    )
    provider.libraries.podcasts["stale_pod"] = LibraryHelper(name="Stale Podcast")

    current_book = Mock()
    current_book.id_ = "existing_book"
    current_book.name = "Existing Book"
    current_book.media_type = "book"
    current_pod = Mock()
    current_pod.id_ = "new_pod"
    current_pod.name = "New Podcast"
    current_pod.media_type = "podcast"

    provider._sync_library_keys([current_book, current_pod])

    assert list(provider.libraries.audiobooks.keys()) == ["existing_book"]
    assert provider.libraries.audiobooks["existing_book"].item_ids == {"id_a", "id_b"}
    assert list(provider.libraries.podcasts.keys()) == ["new_pod"]
