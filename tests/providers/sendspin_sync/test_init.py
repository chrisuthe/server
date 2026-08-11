"""Tests for the Sendspin Sync plugin provider entry point."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from music_assistant_models.provider import ProviderManifest

from music_assistant.providers import sendspin_sync
from music_assistant.providers.sendspin_sync import SendspinSyncProvider, setup

MANIFEST_PATH = pathlib.Path(sendspin_sync.__file__).parent / "manifest.json"


async def test_setup_returns_the_plugin_provider() -> None:
    """setup() hands Music Assistant a SendspinSyncProvider instance."""
    manifest = MagicMock()
    config = MagicMock()
    # both are load-bearing: Provider.__init__ names its logger after the domain and
    # passes the log level through str(), which a bare MagicMock turns into a repr
    # that setLevel rejects
    manifest.domain = "sendspin_sync"
    config.get_value = MagicMock(return_value="GLOBAL")
    provider = await setup(MagicMock(), manifest, config)
    assert isinstance(provider, SendspinSyncProvider)


async def test_manifest_declares_its_sendspin_dependency() -> None:
    """The shipped manifest couples the plugin to the sendspin provider."""
    manifest = await ProviderManifest.parse(str(MANIFEST_PATH))
    assert manifest.domain == "sendspin_sync"
    assert manifest.depends_on == "sendspin"
