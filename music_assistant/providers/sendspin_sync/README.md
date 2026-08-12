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

- **The period is exact.** One 500 ms period is generated once at load and then
  replayed verbatim; jitter in the chirp spacing would corrupt the measurement.
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
| `sendspin_sync/eligible_players` | `players.read` | The Sendspin speakers a session can run against |
| `sendspin_sync/session` | `players.read` | State of the running session, or `null` |
| `sendspin_sync/start_session` | `players.control` | Take the given speakers over and start the track |
| `sendspin_sync/solo_player` | `players.control` | Make one member of the session audible |
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

## File Structure

```
sendspin_sync/
├── __init__.py       SendspinSyncProvider + setup() + the session API commands
├── chirp.py          Calibration signal generation
├── session.py        Calibration session orchestration (grouping, isolation, restore)
├── manifest.json     Experimental plugin manifest
├── strings.json      Translatable manifest description + session error messages
├── icon.svg          Provider icon (light)
├── icon_dark.svg     Provider icon (dark)
└── README.md         This file
```
