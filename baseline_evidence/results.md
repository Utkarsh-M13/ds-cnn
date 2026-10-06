# BEFORE baseline - on-device vs PC (current PTQ model: kws_tf_int8.tflite)

Device: Nordic Thingy:53 (serial DE8FE3011F02821B), port /dev/cu.usbmodem101
Firmware: thingy53-tflm build/zephyr/dfu_application.zip (current PTQ model)
Model under test: kws_tf_int8.tflite (228248 bytes)
Captured by: Utkarsh

Each word: say it clearly ~15-20 cm from the mic. Raw serial log in raw/,
PC recomputation in compare_<word>.txt. "device %" is the firmware's own
confidence; "PC %" is compare_snapshot.py recomputing the same pipeline.

| word         | device % | PC %  | captured (local time) | raw log |
|--------------|----------|-------|-----------------------|---------|
| set_a_timer (oracle, -19.1 dBFS) | 35.9 | 91.0 | 2026-10-06 02:05 | raw/before_oracle_set_a_timer.txt |

## BEFORE baseline locked (2026-10-06 02:05)
Measured via the on-device ORACLE (firmware injects oracle_pcm.h = set_a_timer at
-19.1 dBFS) so the input is identical and repeatable, independent of the mic.
Model on device: kws_tf_int8.tflite (current PTQ).

- set_a_timer: device = 35.9%, PC = 91.0%. Both rank set_a_timer #1 (correct), but
  the device confidence is ~55 points lower on the SAME input. This is the int8
  rounding gap (CMSIS-NN requantize vs desktop TFLite) that QAT targets.
- Full per-label device-vs-PC table: compare_before_oracle_set_a_timer.txt.

AFTER (QAT) will repeat this exact capture with kws_tf_qat_int8.tflite swapped in
(oracle stays on, firmware otherwise identical). Success = device % climbs toward 91%.

## BEFORE trio locked (2026-10-06 02:35) - PTQ model (kws_tf_int8.tflite)
Captured via on-device oracle (mic-independent), 3 clips, full 2000-sample dumps.

| word        | device % | PC %  | gap (dev-PC) | device pick |
|-------------|----------|-------|--------------|-------------|
| set_a_timer | 35.9     | 91.0  | -55.1        | correct     |
| volume_down | 78.5     | 96.9  | -18.4        | correct     |
| yes (ctrl)  | 90.6     | 85.9  | +4.7         | correct     |

Reading: the device-vs-PC int8 gap is severe on set_a_timer, moderate on
volume_down, and negligible on the control word yes. QAT should lift the two
failing words toward their PC values while leaving yes intact.
Per-label tables: compare_before_{set_a_timer,volume_down,yes}.txt.

## AFTER trio locked (2026-10-06 02:55) - QAT model (kws_tf_qat_int8.tflite)
Same on-device oracle clips, QAT model flashed, firmware otherwise identical.

## HEADLINE: BEFORE vs AFTER (device confidence, same input)
| word        | BEFORE device % | AFTER device % | PC % (QAT) | device-PC gap: before -> after |
|-------------|-----------------|----------------|------------|--------------------------------|
| set_a_timer | 35.9            | 99.6           | 99.6       | -55.1  ->  0.0                 |
| volume_down | 78.5            | 99.6           | 99.6       | -18.4  ->  0.0                 |
| yes (ctrl)  | 90.6            | 97.3           | 97.7       |  +4.7  -> -0.4                 |

Result: QAT lifted both failing words to 99.6% on-device and the device now
MATCHES the PC int8 output exactly (same int8 value 127). The control word yes
stayed high (not broken). The device-vs-PC int8 rounding gap QAT targeted is
effectively eliminated.
Per-label tables: compare_after_{set_a_timer,volume_down,yes}.txt.

## HELD-OUT validation (2026-10-06 04:29) - the rigorous retest
Addresses two threats: (1) the trio above used TRAINING clips, (2) the two headline
words saturated int8 (both device and PC pinned at 127, hiding any rounding gap).

Setup: pulled 5 clips from train_tf's TEST split (never trained on; tools/pick_heldout_clips.py),
including search/five which land mid-confidence and do NOT saturate. Source wavs in
baseline_evidence/heldout_source/. Measured device-vs-PC for PTQ and QAT on the same clips.

### Tier 1 - PC accuracy on the full held-out test split (no overfitting)
PTQ: train 93.7% / val 93.6% / test 93.5%, test mean-conf 83.2%
QAT: train 93.7% / val 93.1% / test 93.0%, test mean-conf 89.5%
=> QAT did NOT overfit (train ~= test for both) and did NOT improve accuracy
   (93.5 -> 93.0, flat/slightly down). Its effect is higher confidence (+6.3 pts on
   unseen clips), i.e. wider margins, not better classification.

### Tier 2 - on-device device-vs-PC on held-out clips
| word        | PTQ dev% | PTQ PC% | PTQ gap | QAT dev% | QAT PC% | QAT gap | saturated |
|-------------|----------|---------|---------|----------|---------|---------|-----------|
| set_a_timer | 37.5     | 91.8    | -54.3   | 99.6     | 99.6    |  0.0    | yes (127) |
| volume_down | 71.5     | 94.1    | -22.7   | 99.6     | 99.6    |  0.0    | yes (127) |
| search      | 80.1     | 89.5    |  -9.4   | 98.1     | 98.1    |  0.0    | near-top  |
| five        | 84.0     | 93.4    |  -9.4   | 91.0     | 91.0    |  0.0    | NO (int8 105) |
| yes         | 83.6     | 76.2    |  +7.4   | 83.2     | 82.4    | +0.8    | NO (int8 85)  |

Key: 'five' at int8 105 (91%, not saturated) matches device==PC exactly under QAT;
'search' at int8 123 matches exactly. So the gap closing is a genuine device-vs-PC
convergence across the confidence range, not a saturation artifact. Honest residual:
yes keeps a ~0.8% gap (device int8 85 vs PC 83), so "near-zero", not literally zero.

### Conclusion
QAT reliably closes the CMSIS-NN-vs-desktop int8 rounding gap, on training AND held-out
clips, saturated AND non-saturated, at no accuracy cost. It does not improve accuracy.
Raw logs: raw/heldout_{ptq,qat}_*.txt; per-label: compare_heldout_{ptq,qat}_*.txt.
