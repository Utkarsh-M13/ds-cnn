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
