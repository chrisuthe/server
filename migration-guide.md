# API Migration Guide: feat/major-refactor

This document outlines all breaking API changes from the current HEAD (109be50) to the `feat/major-refactor` branch.

## Table of Contents

- [Client API Changes](#client-api-changes)
- [Role API Changes](#role-api-changes)
- [Connection API Changes](#connection-api-changes)
- [Event System Refactoring](#event-system-refactoring)
- [Group API Changes](#group-api-changes)
- [Models API Changes](#models-api-changes)
- [Migration Checklist](#migration-checklist)

---

## Client API Changes

### `SendspinClient` Method Renames

#### `try_send_binary()` → `send_binary()`

**Before:**
```python
client.try_send_binary(
    data=chunk,
    role_family="player@v1",
    timestamp_us=ts,
    message_type=0x04,
) -> bool  # Returns success indicator
```

**After:**
```python
client.send_binary(
    data=chunk,
    role_family="player@v1",
    timestamp_us=ts,
    message_type=0x04,
) -> None  # No return value
```

**Breaking Changes:**
- Method name changed (snake_case consistency)
- Return type changed from `bool` to `None` (fire-and-forget semantics)
- Method is now a no-op when client is disconnected instead of returning `False`

**Migration:**
- Remove code that checks the return value
- If you need to verify delivery, use event listeners instead

---

### `ensure_role_state()` → `get_or_create_role_state()`

**Before:**
```python
state = client.ensure_role_state("player@v1", PlayerClientState)
```

**After:**
```python
state = client.get_or_create_role_state("player@v1", PlayerClientState)
```

**Breaking Changes:**
- Method renamed for clarity (ambiguous verb "ensure" → explicit "get_or_create")

**Migration:**
- Simple rename throughout your codebase

---

### New Property: `cleanup_on_mdns_removal`

**Added:**
```python
@property
def cleanup_on_mdns_removal(self) -> bool:
    """Whether this retained client should be removed when mDNS record disappears."""
```

**When it matters:**
- Indicates a client that received an `ANOTHER_SERVER` goodbye but is retained for potential recovery
- Read-only property for observing client lifetime behavior

---

### Internal Transport Handling Changes

**Removed Methods:**
- `on_transport_attach()` - No longer called during connection setup
- `on_transport_detach(goodbye_reason)` - No longer called during disconnection

**Why:**
- These lifecycle hooks were redundant with `on_connect()`/`on_disconnect()`
- Transport state is now managed entirely by the connection layer

---

## Role API Changes

### Lifecycle Hook Changes

#### Removed: `on_transport_attach()` and `on_transport_detach()`

**Before:**
```python
class MyRole(Role):
    def on_transport_attach(self) -> None:
        """Handle WebSocket connect/reconnect."""
        # Set up connection-specific state
        pass

    def on_transport_detach(self, goodbye_reason: GoodbyeReason | None = None) -> None:
        """Handle WebSocket disconnect."""
        # Clean up connection-specific state
        pass
```

**After:**
```python
class MyRole(Role):
    # These methods are removed
    # Use on_connect() and on_disconnect() instead
    pass
```

**Migration:**
- Move transport-specific setup to `on_connect()`
- Move transport-specific cleanup to `on_disconnect()`
- If you need the goodbye reason, add a hook for that in `on_disconnect()`

---

### `on_audio_chunk()` Return Type Change

**Before:**
```python
def on_audio_chunk(self, chunk: AudioChunk) -> bool:
    """Receive audio chunk. Return True if accepted, False for backpressure."""
    return True
```

**After:**
```python
def on_audio_chunk(self, chunk: AudioChunk) -> None:
    """Receive audio chunk."""
```

**Breaking Changes:**
- Return type changed from `bool` to `None`
- Return value is no longer used for backpressure signaling
- Backpressure is now handled at the connection/queue level instead

**Migration:**
- Remove return statements from `on_audio_chunk()` implementations
- If you were using return value for logic, migrate to connection-level backpressure handling

---

### `on_stream_request_format()` Signature Change

**Before:**
```python
def on_stream_request_format(
    self,
    payload: StreamRequestFormatPayload,
    *,
    stream_active: bool | None = None,
) -> None:
    """Handle stream/request-format payload."""
```

**After:**
```python
def on_stream_request_format(
    self,
    payload: StreamRequestFormatPayload,
) -> None:
    """Handle stream/request-format payload."""
```

**Breaking Changes:**
- Removed `stream_active` keyword-only parameter
- Stream active state must be determined from context

**Migration:**
- Remove the `stream_active` parameter from your implementations
- Use other signals (like `on_stream_start()` / `on_stream_end()`) to track state

---

### New Method: `has_connection()`

**Added:**
```python
def has_connection(self) -> bool:
    """Return True when the client currently has an active transport."""
```

**Replaces:**
- Internal `_has_transport` attribute (now encapsulated)

**Migration:**
- Use `role.has_connection()` instead of checking `role._has_transport`

---

### Removed Internal Attribute: `_has_transport`

**Was:**
```python
_has_transport: bool  # Internal state tracking
```

**Now:**
- Managed internally; use `has_connection()` method for checking state
- Transport state synchronization simplified

---

## Connection API Changes

### `SendspinConnection.try_send_binary()` → `send_binary()`

**Before:**
```python
connection.try_send_binary(
    data=frame,
    role="player@v1",
    timestamp_us=ts,
    message_type=0x04,
) -> bool
```

**After:**
```python
connection.send_binary(
    data=frame,
    role="player@v1",
    timestamp_us=ts,
    message_type=0x04,
) -> None
```

**Breaking Changes:**
- Method renamed for consistency
- Return type changed from `bool` to `None`
- Semantics: enqueue or silently drop (not fire-and-forget on disconnect)

**Migration:**
- Rename all calls to `send_binary()`
- Remove code that checks the return value

---

### New Methods for Connection State

#### `should_retry_server_initiated_connection()` → `bool`

**Added:**
```python
def should_retry_server_initiated_connection(self) -> bool:
    """Check if connection should be retried after server-initiated reconnect."""
```

**Use case:**
- Determines retry behavior for server-initiated connection changes

---

#### `goodbye_reason` → `property`

**Added:**
```python
@property
def goodbye_reason(self) -> GoodbyeReason | None:
    """The reason for the most recent disconnect, if applicable."""
```

**Migration:**
- Use this property instead of tracking disconnect reasons manually

---

### Connection Writer Refactoring

**Refactored methods (internal implementation details):**
- `_writer()` - Now broken into smaller methods for clarity
  - `_process_priority_messages()`
  - `_process_normal_messages()`
  - `_process_binary_role_messages()`
  - `_process_role_messages()`
  - `_wait_for_writer_work()`

**Impact:**
- Internal implementation; no API surface change for most users
- May affect custom connection subclasses

---

## Event System Refactoring

### Event Base Classes Reorganized

Events are now defined in a unified location (`aiosendspin.server.events`) with clearer hierarchy:

**New Event Hierarchy:**
```
ClientEvent (base)
├── ClientGroupChangedEvent
└── ClientRoleEvent (new base for role-emitted events)
    └── VolumeChangedEvent (player role)

GroupEvent (new base)
├── GroupStateChangedEvent
├── GroupMemberAddedEvent
├── GroupMemberRemovedEvent
├── GroupDeletedEvent
└── GroupRoleEvent (new base for role-emitted events)
    ├── ControllerEvent (controller role)
    ├── PlayerGroupEvent (player role)
    │   ├── PlayerGroupVolumeChangedEvent
    │   └── PlayerGroupMuteChangedEvent
    ├── MetadataEvent (metadata role)
    │   ├── MetadataUpdatedEvent
    │   └── MetadataClearedEvent
    └── ArtworkEvent (artwork role)
        ├── ArtworkUpdatedEvent
        └── ArtworkClearedEvent
```

### New Exported Events

**From `aiosendspin.server`:**

Added to `__all__`:
- `ClientRoleEvent`
- `GroupRoleEvent`
- `GroupEvent`
- `GroupStateChangedEvent`
- `GroupMemberAddedEvent`
- `GroupMemberRemovedEvent`
- `GroupDeletedEvent`

**From role modules:**

- `aiosendspin.server.roles.player`:
  - `PlayerGroupEvent`
  - `PlayerGroupVolumeChangedEvent`
  - `PlayerGroupMuteChangedEvent`

- `aiosendspin.server.roles.metadata`:
  - `MetadataEvent`
  - `MetadataUpdatedEvent`
  - `MetadataClearedEvent`

- `aiosendspin.server.roles.artwork`:
  - `ArtworkEvent`
  - `ArtworkUpdatedEvent`
  - `ArtworkClearedEvent`

### Migration

**Before:**
```python
from aiosendspin.server.group import GroupStateChangedEvent

listener = group.add_event_listener(
    lambda g, evt: print(f"State: {evt.state}")
)
```

**After:**
```python
from aiosendspin.server.events import GroupStateChangedEvent

listener = group.add_event_listener(
    lambda g, evt: print(f"State: {evt.state}")
)
```

---

## Group API Changes

### Removed Public Methods

#### Volume/Mute Control Methods Removed

**Removed:**
```python
# These are now handled by PlayerGroupRole
group.volume: int  # property
group.muted: bool  # property
group.set_volume(level: int) -> None
group.set_mute(muted: bool) -> None
```

**Why:**
- Delegated to role system (specifically `PlayerGroupRole`)
- Cleaner separation of concerns

**Migration:**
```python
# Before
group.set_volume(80)
current_vol = group.volume

# After
player_group_role = group.get_group_role("player")
if player_group_role:
    player_group_role.set_volume(80)
    current_vol = player_group_role.volume
```

---

#### `set_supported_commands()` Removed

**Removed:**
```python
group.set_supported_commands(commands: list[MediaCommand]) -> None
```

**Why:**
- Delegated to `ControllerGroupRole`

**Migration:**
```python
# Before
group.set_supported_commands([MediaCommand.PLAY, MediaCommand.PAUSE])

# After
controller_role = group.get_group_role("controller")
if controller_role:
    controller_role.set_supported_commands([MediaCommand.PLAY, MediaCommand.PAUSE])
```

---

#### `register_group_role()` Removed

**Removed:**
```python
group.register_group_role(role: GroupRole) -> None
```

**Why:**
- Group roles are now managed automatically by the role registry
- This was an internal implementation detail

**Migration:**
- Remove calls to this method; group roles are initialized automatically

---

#### `_send_stream_end_msg()` Removed

**Removed:**
```python
group._send_stream_end_msg(client: SendspinClient, roles: list[str] | None = None) -> None
```

**Why:**
- Now handled by role lifecycle hooks (`on_stream_end()`)
- Cleaner abstraction

**Migration:**
- Remove manual stream end messages
- Stream end is now handled automatically by role system

---

#### Delayed Stop Methods Removed

**Removed:**
```python
group._schedule_delayed_stop(
    stop_time_us: int,
    active: bool,
    needs_cleanup: bool,
) -> bool

group.stop(stop_time_us: int | None = None) -> bool
```

**Replaced with:**
```python
async def stop(self) -> bool:
    """Stop playback immediately."""
```

**Why:**
- Simplified API: no delayed stops
- Immediate stop semantics are clearer

**Migration:**
```python
# Before
group.stop(stop_time_us=future_timestamp)

# After
# Stop immediately instead
await group.stop()
```

---

### Other Group Changes

#### `stop()` Now Nullifies `_push_stream`

**Before:**
```python
self._push_stream.stop()
# _push_stream remained for potential reuse
```

**After:**
```python
self._push_stream.stop()
self._push_stream = None  # Fully cleaned up
```

**Impact:**
- More explicit cleanup
- Prevents stale references

---

#### Stream Cleanup on Client Removal

**Changed:**
```python
# Before: Manual stream end message sent if no roles
if not handled:
    self._send_stream_end_msg(client)

# After: Role lifecycle hooks handle this
for role in client.active_roles:
    role.on_stream_end()  # Automatic
```

---

## Models API Changes

### `ServerMessage.merge()` Method Added

**Added:**
```python
class ServerMessage(DataClassORJSONMixin):
    def merge(self, _other: ServerMessage) -> ServerMessage | None:
        """Merge two messages of the same type when safe, else return None."""
        return None
```

**Purpose:**
- Enables intelligent message coalescing (e.g., combining state updates)
- Default implementation returns `None` (no merge)
- Subclasses can override for custom merge behavior

**Impact:**
- Minimal for most users; primarily internal optimization
- If you have custom `ServerMessage` subclasses, consider implementing this

---

## GroupRole API Changes

### New Method: `emit_group_event()`

**Added:**
```python
def emit_group_event(self, event: GroupRoleEvent) -> None:
    """Emit a GroupRole event on the owning group's event stream."""
```

**Purpose:**
- Unified event emission from group roles
- Replaces per-role event listener patterns

**Migration Example (ControllerGroupRole):**

```python
# Before
self._signal_event(ControllerVolumeEvent(volume=cmd.volume))

# After
self.emit_group_event(ControllerVolumeEvent(volume=cmd.volume))
```

---

### ControllerGroupRole Event Listener Removed

**Removed:**
```python
controller_role.add_event_listener(callback: Callable[[ControllerEvent], None]) -> Callable[[], None]
controller_role._signal_event(event: ControllerEvent) -> None
```

**Why:**
- Unified into group event system via `emit_group_event()`
- Events now flow through `SendspinGroup.add_event_listener()` instead

**Migration:**
```python
# Before
unsub = controller_role.add_event_listener(
    lambda evt: print(f"Command: {evt}")
)

# After
unsub = group.add_event_listener(
    lambda g, evt: isinstance(evt, ControllerEvent) and print(f"Command: {evt}")
)
```

---

## Summary of Breaking Changes

| Category | What Changed | Impact | Migration Path |
|----------|-------------|--------|-----------------|
| Method Names | `try_send_binary()` → `send_binary()` | Return value removed | Remove return value checks |
| Method Names | `ensure_role_state()` → `get_or_create_role_state()` | Semantic clarity | Rename calls |
| Method Signatures | `on_audio_chunk()` return type | `bool` → `None` | Remove return statements |
| Method Signatures | `on_stream_request_format()` | Removed `stream_active` param | Remove parameter usage |
| Lifecycle Hooks | `on_transport_attach/detach()` | Removed | Use `on_connect/disconnect()` |
| Group Methods | Volume/mute controls | Moved to `PlayerGroupRole` | Access via role object |
| Group Methods | `set_supported_commands()` | Moved to `ControllerGroupRole` | Access via role object |
| Events | Base class hierarchy | Reorganized | Update imports from `events.py` |
| Events | `ControllerGroupRole` events | Changed to group events | Use `group.add_event_listener()` |

---

## Migration Checklist

### For Role Implementations

- [ ] Remove `on_transport_attach()` method
- [ ] Remove `on_transport_detach()` method
- [ ] Update `on_audio_chunk()` to return `None` instead of `bool`
- [ ] Remove `stream_active` parameter from `on_stream_request_format()`
- [ ] Use `has_connection()` instead of checking `_has_transport`
- [ ] Update `GroupRole` implementations to use `emit_group_event()`

### For Client Code

- [ ] Rename `try_send_binary()` calls to `send_binary()`
- [ ] Remove code that checks `send_binary()` return value
- [ ] Rename `ensure_role_state()` to `get_or_create_role_state()`
- [ ] Update group volume/mute access to use `PlayerGroupRole`
- [ ] Update `set_supported_commands()` calls to use `ControllerGroupRole`
- [ ] Update event listener imports to use `aiosendspin.server.events`
- [ ] Update event listeners for controller events to use group event stream

### For Connection Code

- [ ] Rename `try_send_binary()` to `send_binary()`
- [ ] Remove return value handling for binary sends

### For Tests

- [ ] Update mocks that implement role methods
- [ ] Update mocks that check `send_binary()` return values
- [ ] Update event listener registrations
- [ ] Update assertions on group volume/mute properties

---

## Questions?

For detailed information about specific changes, refer to the commit history or reach out to the development team.
