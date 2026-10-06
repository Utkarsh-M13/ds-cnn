# Finding: on-device mic is far too quiet for live-audio keyword tests

Date: 2026-10-06. Device: Nordic Thingy:53 (serial DE8FE3011F02821B).
Firmware: thingy53-tflm (current PTQ model), live-mic snapshot path.

## What we measured
Captured on-device snapshots (full 32000-sample PCM dumped over serial, verified
complete) and ran compare_snapshot.py to get the recorded loudness:

| attempt                         | RMS      | raw PCM peak | device top score |
|---------------------------------|----------|--------------|------------------|
| "set a timer" spoken, ~once/sec | -57.5 dBFS | n/a        | help 8.6% (junk) |
| loud continuous sound ON the mic| -45.6 dBFS | +/- 76      | n/a (junk)       |

Training audio target is -20.0 dBFS. So the device records roughly 25-37 dB
quieter than training, and even a loud sound held against the mic only reaches
raw PCM peaks of +/- 76 (out of +/- 32768). That is the mic/PDM gain path, not
speaking volume or timing: no amount of louder/closer speech closes a 25 dB gap.

## Why it matters
At these levels the model sees near-silence, so every live prediction is low and
scattered. You cannot measure the device-vs-PC int8 gap (the thing QAT fixes) on
near-silence, because there is no confident decision to compare.

## What we do instead
Use the firmware's built-in ORACLE: inject a known clip (oracle_pcm.h =
set_a_timer at -19.1 dBFS, exactly training loudness) through the real on-device
int8 pipeline. Same known input for BEFORE (PTQ) and AFTER (QAT), so the only
variable is the model. This isolates the int8 rounding gap cleanly. See results.md.

## Separately (not this task)
The quiet mic is the team's known audio problem. Fixes to consider later: raise
PDM gain in firmware, and/or the planned training-time gain augmentation so the
model is robust to quiet input. Tracked apart from the QAT work.

Raw logs: raw/live_mic_set_a_timer.txt (live, -57 dBFS), raw/mic_loud_test.txt (loud, -45 dBFS).
