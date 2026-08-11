# Sendspin Sync Plugin

An experimental plugin provider that depends on the Sendspin player provider.

This is a scaffold: `SUPPORTED_FEATURES` is empty and the provider declares no
config entries, so adding it has no observable effect. The package ships the
manifest, the `SendspinSyncProvider` class and its `setup()` entry point.

`manifest.json` declares `"depends_on": "sendspin"`, so Music Assistant defers
setup until the Sendspin player provider is loaded. Sendspin is a builtin
provider that cannot be disabled, so in practice it is always present.

## File Structure

```
sendspin_sync/
├── __init__.py       SendspinSyncProvider + setup()
├── manifest.json     Experimental plugin manifest
├── strings.json      Translatable manifest description
├── icon.svg          Provider icon (light)
├── icon_dark.svg     Provider icon (dark)
└── README.md         This file
```
