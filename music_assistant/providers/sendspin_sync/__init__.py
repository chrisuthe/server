"""
Sendspin Sync plugin provider.

Declares a dependency on the Sendspin player provider, so it only loads once
Sendspin is up. It declares no provider features and no config entries, so
loading it has no observable effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant.models.plugin import PluginProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.enums import ProviderFeature
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize provider(instance) with given configuration."""
    return SendspinSyncProvider(mass, manifest, config, SUPPORTED_FEATURES)


class SendspinSyncProvider(PluginProvider):
    """Sendspin Sync plugin provider for Music Assistant."""
