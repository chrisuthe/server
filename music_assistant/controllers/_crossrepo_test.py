"""TESTING ONLY — cross-repo review check. Delete this file; do not merge."""

from dataclasses import dataclass

from mashumaro.mixins.orjson import DataClassORJSONMixin

from music_assistant.helpers.api import api_command


@dataclass
class GuestPlaybackStats(DataClassORJSONMixin):
    """Serialized stats about guest playback, returned to API clients."""

    total_guest_plays: int
    last_guest_id: str | None = None


class GuestStatsController:
    """Exposes guest playback stats over the API."""

    @api_command("players/guest_stats")
    async def get_guest_stats(self) -> GuestPlaybackStats:
        """Return guest playback stats (new client-facing API command)."""
        return GuestPlaybackStats(total_guest_plays=0)
