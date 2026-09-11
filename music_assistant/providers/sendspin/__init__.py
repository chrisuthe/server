"""
Player Provider for the Sendspin Audio Protocol.

https://github.com/Sendspin-Protocol/spec
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiosendspin.server.connection import SendspinConnection
from aiosendspin.server.roles.registry import register_role
from aiosendspin.server.roles.source import SourceV1Role

from music_assistant.providers.sendspin.provider import SendspinProvider

if TYPE_CHECKING:
    from aiosendspin.models.types import ClientMessage
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

# DEV ONLY, not for merging: lets unpaired, unencrypted clients use the source role, so source
# clients that cannot pair yet (sendspin-cpp's source-role branch) can be tested end to end.
# aiosendspin registers source@v1 as pairing-only, and 9.1.1 predates the client-stream/*
# message rename (Sendspin spec #163) those clients already send.
register_role("source@v1", SourceV1Role, requires_pairing=False)

_deserialize_client_message_org = SendspinConnection._deserialize_client_message


def _patched_deserialize_client_message(
    _cls: type[SendspinConnection], raw_message: str
) -> ClientMessage:
    """Deserialize an inbound client message, also accepting the client-stream/* spelling."""
    if '"client-stream/' in raw_message:
        raw_message = raw_message.replace('"client-stream/', '"client_stream/')
    return _deserialize_client_message_org(raw_message)


SendspinConnection._deserialize_client_message = classmethod(  # type: ignore[method-assign,assignment]
    _patched_deserialize_client_message
)


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SendspinProvider(mass, manifest, config)
