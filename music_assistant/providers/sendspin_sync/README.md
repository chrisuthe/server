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

## File Structure

```
sendspin_sync/
├── __init__.py       SendspinSyncProvider + setup()
├── chirp.py          Calibration signal generation
├── manifest.json     Experimental plugin manifest
├── strings.json      Translatable manifest description
├── icon.svg          Provider icon (light)
├── icon_dark.svg     Provider icon (dark)
└── README.md         This file
```
