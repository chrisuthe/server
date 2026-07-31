"""
Integration tests for the PlaylistController.

Uses a database-only MusicAssistant instance with a real SQLite database in a
temporary directory to verify that a playlist's ``translation_key`` survives the
library round-trip, and that ``created_by_userid`` records who created a playlist.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from music_assistant_models.auth import User, UserRole
from music_assistant_models.enums import ProviderFeature
from music_assistant_models.media_items import Playlist, ProviderMapping

from music_assistant.constants import DB_TABLE_PLAYLISTS
from music_assistant.controllers.music.media.playlists import PlaylistController
from music_assistant.controllers.webserver.helpers.auth_middleware import set_current_user
from music_assistant.mass import MusicAssistant


@pytest.fixture(scope="class", name="mass")
def mass_fixture(music_mass_class: MusicAssistant) -> MusicAssistant:
    """Return the class-scoped database-only Music Assistant fixture."""
    return music_mass_class


@pytest.fixture(scope="class")
async def playlist_ctrl(mass: MusicAssistant) -> PlaylistController:
    """Get the playlist controller from a running MusicAssistant instance."""
    return mass.music.playlists


@pytest.fixture(autouse=True)
def unauthenticated_session() -> None:
    """Start every test with no session user, so each test states the user it needs."""
    set_current_user(None)


def _make_playlist(
    item_id: str,
    name: str,
    *,
    translation_key: str | None = None,
    translation_params: list[str] | None = None,
) -> Playlist:
    """Create a provider-mapped Playlist for adding to the library."""
    return Playlist(
        item_id=item_id,
        provider="builtin",
        name=name,
        translation_key=translation_key,
        translation_params=translation_params,
        provider_mappings={
            ProviderMapping(item_id=item_id, provider_domain="builtin", provider_instance="builtin")
        },
        owner="Music Assistant",
        is_editable=False,
    )


def _make_user(user_id: str) -> User:
    """Create a User as the auth middleware would put it in the session context."""
    return User(user_id=user_id, username=user_id, role=UserRole.USER)


async def _creator_of(playlist_ctrl: PlaylistController, db_id: int) -> str | None:
    """Return the creator recorded for the given library playlist."""
    rows = await playlist_ctrl.mass.music.database.get_rows_from_query(
        f"SELECT created_by_userid FROM {DB_TABLE_PLAYLISTS} WHERE item_id = :item_id",
        {"item_id": db_id},
    )
    creator: str | None = rows[0]["created_by_userid"]
    return creator


def _stub_provider(item_id: str, name: str) -> MagicMock:
    """Return a music provider stub whose create_playlist yields a new provider playlist."""
    provider = MagicMock()
    provider.name = "Builtin"
    provider.supported_features = {ProviderFeature.PLAYLIST_CREATE}
    provider.create_playlist = AsyncMock(return_value=_make_playlist(item_id, name))
    return provider


class TestPlaylistTranslationKey:
    """The translation_key column survives the library round-trip (mirrors genres)."""

    async def test_parameterless_key_survives_round_trip(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """A static (parameterless) translation_key is persisted and read back."""
        created = await playlist_ctrl.add_item_to_library(
            _make_playlist("infinite_mix", "Infinite Mix (library)", translation_key="infinite_mix")
        )
        fetched = await playlist_ctrl.get_library_item(int(created.item_id))
        assert fetched.translation_key == "infinite_mix"

    async def test_key_and_params_survive_round_trip(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """A key + translation_params (e.g. Spotify's per-account Liked Songs) both persist."""
        created = await playlist_ctrl.add_item_to_library(
            _make_playlist(
                "liked_songs",
                "Liked Songs Alice",
                translation_key="liked_songs",
                translation_params=["Alice"],
            )
        )
        fetched = await playlist_ctrl.get_library_item(int(created.item_id))
        assert fetched.translation_key == "liked_songs"
        assert fetched.translation_params == ["Alice"]

    async def test_update_backfills_key_on_resync(self, playlist_ctrl: PlaylistController) -> None:
        """A row added without a key adopts one when the provider later supplies it."""
        created = await playlist_ctrl.add_item_to_library(
            _make_playlist("recently_played", "Recently played tracks")
        )
        assert (await playlist_ctrl.get_library_item(int(created.item_id))).translation_key is None
        # re-sync: same provider item, now carrying a translation_key -> update path adopts it
        await playlist_ctrl.add_item_to_library(
            _make_playlist(
                "recently_played", "Recently played tracks", translation_key="recently_played"
            )
        )
        fetched = await playlist_ctrl.get_library_item(int(created.item_id))
        assert fetched.translation_key == "recently_played"


class TestPlaylistCreator:
    """create_playlist records the session user; every other path stays household-wide."""

    async def test_creator_recorded_for_authenticated_user(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """A playlist created by an authenticated user records that user's id."""
        set_current_user(_make_user("alice"))
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_a", "Alice's Mix")
        ):
            created = await playlist_ctrl.create_playlist("Alice's Mix")
        assert await _creator_of(playlist_ctrl, int(created.item_id)) == "alice"

    async def test_no_session_user_stays_household_wide(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """Creating without a session user records no creator instead of raising."""
        set_current_user(None)
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_b", "Sync Mix")
        ):
            created = await playlist_ctrl.create_playlist("Sync Mix")
        assert await _creator_of(playlist_ctrl, int(created.item_id)) is None

    async def test_library_add_stays_household_wide(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """A playlist reaching the library via add_item_to_library records no creator."""
        set_current_user(_make_user("alice"))
        created = await playlist_ctrl.add_item_to_library(_make_playlist("synced", "Synced Mix"))
        assert await _creator_of(playlist_ctrl, int(created.item_id)) is None

    async def test_first_creator_wins_on_existing_row(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """Re-creating a playlist that already exists does not reassign its creator."""
        set_current_user(_make_user("alice"))
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_c", "Shared Mix")
        ):
            created = await playlist_ctrl.create_playlist("Shared Mix")
            set_current_user(_make_user("bob"))
            await playlist_ctrl.create_playlist("Shared Mix")
        assert await _creator_of(playlist_ctrl, int(created.item_id)) == "alice"

    async def test_clear_created_by_user_reverts_to_household_wide(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """Clearing a user's playlists unsets only that user's rows."""
        set_current_user(_make_user("alice"))
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_d", "Alice Only")
        ):
            alice_playlist = await playlist_ctrl.create_playlist("Alice Only")
        set_current_user(_make_user("bob"))
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_e", "Bob Only")
        ):
            bob_playlist = await playlist_ctrl.create_playlist("Bob Only")

        await playlist_ctrl.clear_created_by_user("alice")

        assert await _creator_of(playlist_ctrl, int(alice_playlist.item_id)) is None
        assert await _creator_of(playlist_ctrl, int(bob_playlist.item_id)) == "bob"

    async def test_clear_created_by_user_without_playlists_is_a_no_op(
        self, playlist_ctrl: PlaylistController
    ) -> None:
        """Clearing a user that created no playlists does not raise or touch other rows."""
        set_current_user(_make_user("alice"))
        with patch.object(
            playlist_ctrl.mass, "get_provider", return_value=_stub_provider("mix_f", "Kept Mix")
        ):
            kept = await playlist_ctrl.create_playlist("Kept Mix")

        await playlist_ctrl.clear_created_by_user("nobody")

        assert await _creator_of(playlist_ctrl, int(kept.item_id)) == "alice"
