# Sendspin Sync Plugin

An experimental plugin provider that depends on the Sendspin player provider.

`manifest.json` declares `"depends_on": "sendspin"`, so Music Assistant defers
setup until the Sendspin player provider is loaded. Sendspin is a builtin
provider that cannot be disabled, so in practice it is always present.

## The calibration track

The plugin exposes a single AudioSource under "Live Inputs": a calibration track
that plays through the regular playback path — same codec, resampler and DSP as
music — so latency measured from it transfers to real playback.

The track is a chirp train at a fixed period. Riding one uninterrupted stream,
that train is a metronome running on the server clock, which lets a listener
recover a speaker's latency as a phase offset against it without knowing the
server's absolute time. Two properties therefore matter:

- **The period is exact.** One 1 s period is generated once at load and then
  replayed verbatim; jitter in the chirp spacing would corrupt the measurement.
- **The period is long enough.** Its length is the measurement's unambiguous
  range: an arrival more than half a period late is numbered onto the following
  chirp and comes back folded, so 1 s resolves speakers spread across 500 ms.
- **The pulse is a swept sine, not a noise burst.** Matched-filter correlation
  against a sweep compresses room reverberation into a sharp peak, where onset
  detection on noise degrades badly in a reverberant room.

`chirp.py` holds the signal and its specification: a 60 ms Hann-windowed
logarithmic sweep from 500 Hz to 8000 Hz at -6 dBFS, then a -60 dBFS noise bed
for the rest of the period, as 48 kHz 16-bit stereo PCM.

## Calibration sessions

A session commandeers a set of Sendspin speakers so the track can be measured on
each of them in turn. It is driven through the API:

| Command | Scope | Purpose |
| --- | --- | --- |
| `sendspin_sync/eligible_players` | `players.read` | The speakers a session can run against, and whether MA can write each one's delay |
| `sendspin_sync/session` | `players.read` | State of the running session, or `null` |
| `sendspin_sync/start_session` | `players.control` | Take the given speakers over and start the track |
| `sendspin_sync/solo_player` | `players.control` | Make one member of the session audible |
| `sendspin_sync/apply_measurements` | `config.players.write` | Turn the measured offsets into static delays and apply the ones MA can write |
| `sendspin_sync/stop_session` | `players.control` | End the session and restore every speaker |

The speakers are grouped onto a **hidden Sendspin virtual player** that owns the
queue the track plays on. That anchor never renders audio itself, which buys two
things: no user's queue is ever replaced, and every speaker is reached through
the same sync path — a real sync leader would render through the leader path
instead, and its latency would not be comparable to the members'.

`start_session` starts the track **once**. `solo_player` only ever touches
per-player mute and volume, never the queue, because the stream is the phase
reference every measurement is taken against; restarting it would invalidate the
measurements already banked. Isolation mutes the other members and issues no mute
command against the speaker being measured — clients that stop feeding their DAC
while muted shift their timing on resume, which is exactly the number being
measured. A speaker that an earlier `solo_player` had silenced is brought back
first, before the others go down.

That ordering is the longest settling window a single call can give, and it is
only as long as the mute commands that follow it. **The probe consuming this
session is therefore expected to discard the first chirp periods after a
`solo_player`** rather than treat the speaker as settled the moment the command
returns.

Speakers that advertise no mute control are silenced by driving their volume to
zero instead, and speakers with neither control are not eligible at all. The
restore undoes only the control the session actually drove, so a volume the user
changed on a merely-muted speaker mid-session is left alone.

Because the session never claims a mute or a zero volume the user set themselves,
a speaker that is already silent could be soloed but never heard — so
`start_session` refuses one outright rather than hand back a session with a
member that yields no measurement. `force` does not cover that; it only covers a
speaker that is playing or paused.

A session snapshots mute, volume, group membership, power and playback for
everything it takes over, and restores all of it when it ends — including when it
ends by exception, by the stream dying underneath it, by provider unload, or by
the inactivity timeout that stops a phone that walked away from leaving a house
muted. Each speaker's restore steps are attempted independently, so one that has
disconnected can not leave the others silent or grouped. Grouping powers a
speaker on, so one that was off is switched back off rather than resumed.

### Which player a session holds

Only web and app players are Sendspin players in their own right. Every physical
speaker — an ESPHome device, a Windows client, a MultiRoom-Audio box — is
registered **twice**: a hidden `PlayerType.PROTOCOL` Sendspin player that speaks the
protocol, and the visible player MA presents the device as, which is either a
Universal Player wrapping it or a native player its identifiers matched.

A session offers and holds the **visible** player, because that is the level
everything it does works at:

- **Grouping.** The anchor's `can_group_with` carries visible ids — protocol ids are
  translated to their parents — and both `SharedPlaybackSession.can_listen_in` and the
  player controller gate membership on it. A protocol id would be refused; a visible
  id is translated back to the Sendspin player, so the group is still one native
  Sendspin group carrying one uninterrupted stream.
- **Volume, mute and name.** A visible player's `volume_control` and `mute_control`
  resolve through its linked protocol player, so isolation drives the Sendspin client
  either way, and the name a user picked from is the one they know the speaker by.
- **The static delay** is the exception: it lives on the Sendspin player alone, so an
  apply resolves each member to it (`resolve_sendspin_player`) before reading or
  writing, while the result stays keyed by the member the client asked about. A member
  whose Sendspin side has gone is refused rather than reported as one to adjust by
  hand — folding its delay in as 0 would move the reference every other speaker in the
  group is corrected against.

Eligibility therefore asks whether a player *renders over Sendspin*, not which
provider it belongs to. A hidden protocol player is never offered, and neither is a
session anchor: a Sendspin virtual player is typed `PlayerType.PLAYER` just like a web
player, so `is_remote_session_host` is what excludes it.

## Applying the result

`apply_measurements` takes one number per session member: the offset at which the
chirp train arrived from that speaker, as the probe measured it. Those numbers are
**relative** — the probe measures every speaker against a baseline it picked for
itself — so only the differences between them carry meaning.

They are resolved into Sendspin static delays, and the ones MA is allowed to write
are applied — see [Speakers that cannot accept a delay](#speakers-that-cannot-accept-a-delay)
for the rest. A static delay *advances* a player:
the client subtracts it from the server timestamp, so a larger value makes the
sound leave that speaker earlier, and the protocol carries no negative value.
Equalising a group therefore means leaving the earliest speaker where it is and
pulling every later one forward to meet it:

```
total_i = offsets_ms[i] + current_static_delay_ms[i]
delay_i = round(total_i - min_j(total_j))
```

The earliest arrival is the one with the smallest `total`, so it gets 0 and does not
move; the latest gets the largest advance and moves the most. The group converges on
the earliest uncorrected arrival.

Folding in the delay a player already carries is what makes this **idempotent**:
re-running a calibration over an already corrected group returns the same delays
rather than stacking a second correction on top of the first, and a delay the user
set by hand for an amp is part of the baseline instead of being flattened to zero.

Every member of the session must be measured. Correcting a subset would leave the
rest normalised against a different baseline, which is worse than not correcting at
all, so a partial result is refused. A measurement that is not a finite number, or
one that implies a delay beyond the 5000 ms Sendspin carries, is likewise refused
rather than clamped — that is not a speaker latency, and the largest plausible
value would bury the bad measurement instead of reporting it. Nothing is written
until every value has been computed and accepted, so a refusal never leaves the
group half corrected. The writes themselves are not atomic: a speaker that drops off
partway through leaves the ones before it corrected and raises, rather than reporting
a success it did not achieve.

Applying does **not** end the session. The static delay config entry is
`immediate_apply` and carries no `requires_reload`, so saving it pushes the value
straight to the client rather than restarting the queue — which would end the very
stream the measurements are phased against. That makes *measure → apply → measure
again* possible inside one session, and verification has to happen there: ending
the session ungroups every speaker, changing the sync path the numbers describe.
The inactivity timeout still applies, so a client has to apply within it.

Unlike the other five commands, `apply_measurements` requires
`config.players.write` rather than `players.control`. The rest are transient and
fully restored when the session ends, but this one persists a player config value —
the same mutation `config/players/save` is gated on. `players.control` is held by
guests, and `config.players.write` by admin and service accounts only, so a plain
user can drive a session but not apply its result. An in-process call into the
Sendspin provider is not re-checked against the caller's scopes, so the command
registration is the only place that guard exists.

That holds even for a session over speakers MA cannot write to, which reaches the
scope check and then writes nothing — so a guest can drive such a session but not
read back its answer. The scope is on the command, not on what a particular run
turns out to touch, which fails closed.

## Speakers that cannot accept a delay

A Sendspin client only takes a static delay if it advertises `SET_STATIC_DELAY`.
ESPHome's Sendspin component implements one but gates it behind
`static_delay_adjustable`, which defaults to `false`, so a house of capable
hardware can turn out to have very few speakers MA is allowed to correct.

Those speakers are measured anyway. `eligible_players` returns them with
`adjustable: false`, and `apply_measurements` splits its result accordingly:

```
{"applied": {player_id: delay_ms}, "manual": {player_id: delay_ms}}
```

`applied` is what was written through the Sendspin provider. `manual` is the
delay resolved for each speaker MA could not write to, for the user to apply on
the device itself — by setting `static_delay_adjustable: true` and re-running, by
putting the value into `initial_static_delay` in the YAML, or on the delay control
of an outboard amp.

**The two halves are not the same kind of number.** An `applied` value is
absolute: it replaces the delay that speaker carried. A `manual` value is how much
further forward the speaker has to come, so it is *added* to any delay the device
already applies — that one is already inside the arrival the probe measured, and
MA cannot read it. For the usual device with no delay configured the two coincide,
but a device that already carries `initial_static_delay: 20ms` and comes back with
`manual: 43` needs `63ms`, not `43ms`.

The normalisation does not split. Offsets are *relative*, so every measured
speaker is in the `min()` that picks the reference, including the ones that
cannot be written — a non-adjustable speaker may well be the earliest arrival and
define the 0, which is fine and needs no special case. The range refusal applies
to all of them too: a measurement implying more than 5000 ms is a bad measurement
whoever is going to apply it. Dropping a speaker from the maths instead would not
merely deny it a correction, it would remove it from the picture the others are
normalised against, and the user could not even learn how far out it is.

A speaker MA cannot write to also contributes 0 to the `current_static_delay_ms`
term. MA holds no static delay for it — there is no config entry to hold one —
and whatever its firmware applies is already inside the arrival that was just
measured, so 0 keeps the sum honest and stable across re-runs.

`adjustable` is on `eligible_players` rather than discovered at apply time on
purpose: the user is told which speakers they will have to correct by hand while
picking them, not after walking the house.

A member that has *left* is refused outright rather than reported under `manual`.
Sendspin answers the same "no static delay" for a speaker that has disconnected as
for one that refuses one, so `apply_measurements` resolves each member before it
splits anything — otherwise a speaker that dropped mid-session would come back as
one for the user to go and adjust on a device that is not there.

## File Structure

```
sendspin_sync/
├── __init__.py       SendspinSyncProvider + setup() + the session API commands
├── chirp.py          Calibration signal generation
├── session.py        Calibration session orchestration (grouping, isolation, restore)
│                     plus the offset-to-static-delay normalisation
├── manifest.json     Experimental plugin manifest
├── strings.json      Translatable manifest description + session error messages
├── icon.svg          Provider icon (light)
├── icon_dark.svg     Provider icon (dark)
└── README.md         This file
```
