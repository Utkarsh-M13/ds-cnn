# Research: mic confidence, decibel mismatch, and normalization

Branch: `mic-norm-research` (local only, nothing pushed).
Covers this week's to-dos #1 (confidence increase), #4 (mic confidence / decibel
inaccuracies), and #5 (normalization because there are amplitude differences).
To-do #2 (Bisti accuracy discrepancy) waits on her new dataset. To-do #3 (power)
waits on the power profiler.

## 1. What we have (the two pipelines)

There are two kinds of normalization to think about: per-clip loudness (amplitude)
and post-MFCC feature standardization. Neither training nor the device does the
loudness one; both do the feature one.

**Loudness normalization (per clip, amplitude) - not done anywhere.** Training does
not loudness-normalize at all.
- `_load_waveform` (`src/train_tf.py:78`) reads each clip as recorded (mono,
  resample, pad/truncate) with no RMS or dBFS scaling.
- `config.yaml` defines `audio.target_dbfs: -20.0` and `trim_top_db: 30.0`, but
  `target_dbfs` is never used by the training code. The only RMS scaling in training
  is peak-scaling the noise mixes during noise augmentation, not the clips themselves.
- The clips are not normalized to a single level. Measured on the five held-out
  clips, they sit around -20 to -27 dBFS: naturally loud, but spread, not pinned to
  -20. So the model trains on loud audio, just not at one fixed loudness.

**Feature normalization (global mean/std).** After MFCC, the trainer standardizes
features with a single scalar mean and std.
- `src/train_tf.py:1030-1032`: `mean = X.mean()`, `std = X.std()`, `X = (X-mean)/std`.
- Stored at `src/train_tf.py:1232` as `feature_norm`, baked into the firmware as
  `MFCC_FEATURE_MEAN` and `MFCC_FEATURE_STD` in `dsp_constants.h`
  (original model: mean -7.331, std 30.233).

**On device (`thingy53-tflm`, branch `qat-oracle`).**
- PCM to float: `audio_f32[j] = audio_in[i+j] / 32768.0f`
  (`MLInferenceCtrl.cpp:289` and `:321`). This matches training's [-1, 1] range.
- MFCC: `arm_mfcc_f32(...)` (`:293`, `:325`).
- Normalize: `(mfcc - MFCC_FEATURE_MEAN) / MFCC_FEATURE_STD` (`:298`, `:332`), then
  quantize to int8.
- There is **no per-utterance loudness normalization** anywhere in the device path.
- Microphone: `dmic_cfg` sets PDM clock and channel map but **no gain**
  (`MicrophoneCtrl.cpp:23-26`).

## 2. The symptom we measured

From `baseline_evidence/MIC_FINDING.md` (device DE8FE3011F02821B):

| input                      | recorded RMS     | raw PCM peak | result        |
|----------------------------|------------------|--------------|---------------|
| "set a timer" spoken       | -57.5 dBFS       | n/a          | junk (8.6%)   |
| loud sound on the mic      | -45.6 dBFS       | +/- 76       | junk          |
| training clips (natural)   | -20 to -27 dBFS  | -            | -             |

The device records 25 to 37 dB quieter than the training clips. Even a loud sound held on
the mic only reaches raw PCM peaks of +/- 76 out of +/- 32768. That is a gain
problem in the mic/PDM path, not speaking volume or timing.

## 3. Root cause (why confidence collapses on live mic)

1. The device feeds the model audio at -45 to -57 dBFS; the model only ever saw the
   training clips at their natural loud level (~-20 to -27 dBFS). MFCC is log energy
   based, so a 25 to 37 dB amplitude drop pushes the features (the c0 log-energy term
   most of all) far below the training distribution.
2. The only thing standing between the mic and the model is the global scalar
   `(x - mean) / std`. A single mean and std shift and scale the whole feature set
   uniformly; they cannot correct a per-utterance amplitude offset. So the quiet
   input lands in a region the int8 model never trained on, and every live
   prediction is low and scattered.
3. Two independent contributors stack here: the hardware PDM gain is too low (the
   +/- 76 peak), and there is no per-clip loudness normalization on either side. The
   training clips are naturally loud, so the model never had to cope with quiet input;
   the mic produces exactly that and nothing scales it back up.

The repo already flags the fix direction: `oracle_test.py:156-159` says to either
re-normalize the training data or "add gain normalization in the MFCC path (scale
audio to target RMS before MFCC)."

This is also why we test with the oracle (a known -19 dBFS clip): it bypasses the
mic so the device-vs-PC int8 gap (the QAT target) can be measured on a confident
decision. The mic problem is orthogonal to QAT.

## 4. Possible issues, ranked by how much they likely matter

1. **No per-utterance loudness normalization anywhere (software).** Neither training
   nor the device loudness-normalizes. The training clips are naturally loud (~-20 to
   -27 dBFS) and the mic is 25 to 37 dB quieter, so the model sees a level it never
   trained on. The clean fix is to normalize to a common RMS target on both sides,
   training and live input. This is the biggest and most directly fixable mismatch.
2. **PDM/DMIC hardware gain too low (hardware/firmware).** +/- 76 raw peak even on
   loud sound means the analog-to-PCM path is starved, so the signal sits in the
   bottom few bits and carries extra quantization noise. No gain is set in
   `dmic_cfg`.
3. **Global scalar feature mean/std cannot absorb amplitude offset.** Inherent to a
   fixed mean/std; it is correct for the design, it just cannot do loudness
   normalization's job.
4. **No training-time gain augmentation.** The model was trained only on the clips'
   natural loud levels (~-20 to -27 dBFS), so it is brittle to any loudness it did not
   see. Even with a gain fix, augmentation would make it robust to the residual.
5. **c0 (log energy) dominates amplitude sensitivity.** The amplitude error
   concentrates in the first cepstral coefficient. Per-utterance cepstral mean
   normalization (CMN) or de-emphasizing c0 would reduce the sensitivity, but it
   changes the feature definition and needs a retrain.
6. **Loudness-norm reference region.** When we do add normalization, the RMS should be
   measured over the speech region, not the whole window. A naive on-device RMS over
   the full 2 s window (which includes silence and padding) would be thrown off by how
   much silence a clip has. `config.yaml trim_top_db: 30` is the trimming knob to mirror.
   Noted as an implementation detail for the fix below.

Ruled out: the PCM-to-float convention. The device divides by 32768 to match
soundfile's [-1, 1] (`MLInferenceCtrl.cpp:321`), so this is an amplitude/gain
problem, not a scaling-convention bug.

## 5. Possible fixes

| # | Fix | Where | Pros | Cons / risks |
|---|-----|-------|------|--------------|
| A | Per-utterance RMS loudness normalization to a common target, on both training and live input | training + firmware | Directly closes the gap, cheap, works even if hardware gain stays low, puts train and inference at the same level | Both sides must use the same RMS target and reference region (speech, not silence); normalizing a silent window amplifies noise, so it needs a floor or a VAD gate |
| B | Raise PDM/DMIC hardware gain | firmware / devicetree | Fixes the root hardware deficit, less quantization noise | Zephyr `dmic` API may not expose gain; may need the nrfx PDM GAIN register or a devicetree property; needs the board to tune |
| C | Training-time gain augmentation (randomize clip loudness) | training | Makes the model robust to any residual loudness error | Needs a retrain; best folded into the next run (and Bisti's new data) |
| D | Per-utterance CMN or de-emphasize c0 | training + firmware | Removes amplitude sensitivity at the feature level | Changes the feature definition, needs retrain + new dsp_constants + firmware change |

A and B are independent and complementary. A alone should recover most of the
confidence even without touching hardware. C hardens it. D is the heaviest and
only if A through C are not enough.

## 6. Proposed plan

**Phase 0 - quantify on PC (no device needed). DONE - results below.**
Take an oracle clip at -19 dBFS, synthetically attenuate it by 25 to 37 dB, and run
it through the exact MFCC + `(x-mean)/std` + int8 model path on the PC. Measure how
far confidence falls and confirm that re-normalizing the attenuated clip back to
-20 dBFS restores it. This proves amplitude is the cause, sets the target, and
needs no board. `compare_snapshot.py` already has the dBFS math to reuse.

### Phase 0 results (done, PC only, no board)

Ran an amplitude sweep on the five held-out oracle clips: normalize each to a loud
reference level (-20 dBFS, near where the training clips and the oracle sit),
attenuate it in steps down to the mic's real range (-45 to -57 dBFS), run the full
PC int8 path (tf.signal MFCC, `(x-mean)/std`, int8 model `kws_tf_int8.tflite`), then
re-normalize the attenuated clip back to the reference level and re-run. Script:
`phase0_amplitude.py` (scratchpad).

Confidence of the true label, reference level vs the two mic levels:

| clip        | -20 dBFS (ref) | -45 dBFS (mic, loud) | -57 dBFS (mic, spoken) | re-normalized |
|-------------|----------------|----------------------|------------------------|---------------|
| five        | 80.5% (five)   | 43.0% -> **call**    | 40.6% -> **call**      | 80.5% (five)  |
| yes         | 71.5% (yes)    | 39.1% (yes)          | 22.3% -> **help**      | 71.5% (yes)   |
| volume_down | 93.4%          | 75.8%                | 64.5%                  | 93.4%         |
| search      | 96.1%          | 94.9%                | 93.0%                  | 96.1%         |
| set_a_timer | 82.8%          | 87.9%                | 91.0%                  | 82.8%         |

What this proves:
1. **Amplitude alone moves predictions.** Just attenuating the clip (nothing else
   changed) dropped `five` from 80.5% to ~41% and flipped it to "call", and
   dropped `yes` from 71.5% to 22.3% and flipped it to "help" at the spoken-mic
   level. `volume_down` lost ~29 points. The global `(x-mean)/std` does not absorb
   this, exactly as predicted.
2. **Fix A works, completely, on PC.** Re-normalizing the attenuated clip back to the
   reference level restored every clip to its exact reference confidence and correct
   label. A per-utterance loudness-normalization step fully undoes the amplitude
   damage. This is the strongest evidence that fix A is the right primary fix.
3. **Caveat: the effect is not uniform across words.** `search` barely moved and
   `set_a_timer` actually rose as it got quieter (it sits near an int8 decision
   boundary). Real speech is more than a pure gain change, so on-device recovery
   will not be identical for every word, but the direction is clear and the fix is
   validated. The words that flip (five, yes) are the ones that matter.

Takeaway: implement fix A (software loudness normalization) first. Phases 2-4 stay
as written below.

**Phase 1 - software loudness normalization (fix A).**
Add per-utterance RMS normalization on both sides to a common target: in training
(so the model learns at one consistent level) and in the device path before MFCC
(so live input lands at that same level), with a silence floor so a quiet window is
not blown up into noise. Measure RMS over the speech region, trimming silence first
as `trim_top_db` does, and use the identical function and target in both places so
parity holds. This is the direction the prof wants, done with Bisti on the training
side.

**Phase 2 - hardware gain (fix B).**
Set the PDM gain so the raw signal uses more of the int16 range, reducing
quantization noise at the input. Tune against measured dBFS on the board.

**Phase 3 - training robustness (fix C).**
Add gain augmentation to the next training run so the model tolerates a range of
input levels. Fold this into the run on Bisti's new dataset.

**Phase 4 - verify on device.**
Once a board is in hand, re-run the live-mic `capture_snapshot` path and confirm
live confidence recovers toward the oracle/PC level. This is the real test of
to-do #1.

**Open questions to resolve before coding.**
- Which reference region to normalize over when we add it (whole clip vs trimmed speech),
  used identically in training and on device.
- Whether Zephyr's `dmic` exposes PDM gain, or if it needs nrfx / devicetree.
- Target level for both sides: a fixed RMS target (e.g. -20 dBFS), or an AGC-style
  running estimate on device.

To-do #2 (accuracy discrepancy) stays parked for Bisti's new dataset; to-do #3
(power) stays parked for the power profiler.
