# 2025–26 Deployment and Inference Review

Review for [issue #22](https://github.com/macexo/exoskeleton-ml/issues/22).

**Scope:** `archive/2025-26/`  
**Contributors:** Yug Vashisth and Krish Bhagirath

## Summary

Last year's deployment code exports a trained PyTorch Temporal Convolutional Network (TCN) to ONNX and runs it on the Raspberry Pi's CPU. Two separate scripts accept sensor history and return the latest four joint-moment predictions: an [HTTP server](../../archive/2025-26/scripts/server.py) and a [serial endpoint](../../archive/2025-26/scripts/endpoint_tcn.py). They stop at returning predictions; conversion to physical torque and motor actuation are described in the [draft Controls interface](../../archive/2025-26/docs/ml_control_interface.md), but are not implemented in either script.

## How it works

### Export the checkpoint

[`export_onnx.py`](../../archive/2025-26/scripts/export_onnx.py) loads `best_model.pt` and its adjacent `config.yaml`, rebuilds the model, restores its weights, and switches it to evaluation mode. It creates an example input shaped `(1, effective_history, 28)`, exports the model with `torch.onnx.export(..., dynamo=True)`, and saves it to `outputs/model.onnx` by default.

The same script checks the ONNX graph and compares PyTorch and ONNX Runtime outputs for that one random input, using `rtol=1e-3` and `atol=1e-4`. A mismatch prints a warning but leaves the exported file in place. This checks conversion consistency, not prediction accuracy on test data. The script also prints an SCP command for copying the model to the Pi.

### Prepare the sensor data

Each timestep has 28 features: acceleration XYZ and gyroscope XYZ from four IMUs, followed by four joint angles. The configured IMU order is left thigh, left shank, right thigh, right shank; the configured angle order is `[hip_l, hip_r, knee_l, knee_r]`. See [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml).

During training, inputs 24–27 came from each trial’s `*_angle_filt.csv`, which contains filtered hip and knee angles derived from motion capture. The Controls draft assigns those same inputs to exoskeleton joint encoders. Even with the joints in the correct order, the measurements can differ because motion capture estimates the wearer’s joint angles while the encoders measure rotation at the exoskeleton joints. [Dataset preparation](../../archive/2025-26/scripts/prepare_hf_dataset.py#L103-L161) · [Georgia Tech study methods](https://doi.org/10.1038/s41586-024-08157-7) · [Controls angle inputs](../../archive/2025-26/docs/ml_control_interface.md#L70-L77)

The [dataset loader](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py) can normalize both inputs and targets using training statistics. Neither deployment script loads those statistics or reverses target normalization. For a checkpoint trained with normalized targets, its raw predictions therefore need an inverse transform before they represent Nm/kg, even though the HTTP response labels them with that unit. The [default model configuration](../../archive/2025-26/configs/model/tcn.yaml) leaves input normalization to the dataset.

### HTTP: send a full window with each request

[`server.py`](../../archive/2025-26/scripts/server.py) loads the ONNX model once at startup with `CPUExecutionProvider`. The [deployment proposal](../../archive/2025-26/docs/deployment_options.md#local-python-server) places sensor collection in a C++ process that calls this local Python service.

1. The caller sends a complete **187 × 28** sensor window. History must be maintained by the caller; the server retains no sensor buffer.
2. `POST /predict` accepts JSON containing a `window` array and validates its dimensions. `POST /predict_msgpack` instead accepts a flat MessagePack array of 5,236 values and reshapes it.
3. The server converts the window to `float32` with shape `(1, 187, 28)` and runs ONNX inference.
4. It takes only the newest timestep, `outputs[0][0, -1, :]`, and returns `hip_left`, `hip_right`, `knee_left`, and `knee_right`.

The response also contains `inference_ms`, which times only `session.run()`. It excludes transport, input validation, and response construction. `GET /health` reports whether the model session is loaded. These routes and their timing logic are all in [`server.py`](../../archive/2025-26/scripts/server.py).

### Serial: keep the window in Python

[`endpoint_tcn.py`](../../archive/2025-26/scripts/endpoint_tcn.py) runs independently of the HTTP server. It loads `outputs/model.onnx` and opens `/dev/ttyACM0` at 115,200 baud.

1. It fills a `deque(maxlen=100)` with zero samples.
2. It reads a newline-terminated message containing 28 comma-separated floats. Messages with the wrong feature count are skipped.
3. It appends the sample, dropping the oldest, and builds a `(1, 100, 28)` input. Predictions during startup include zero-filled history.
4. It runs ONNX inference, takes the newest timestep's four outputs, and writes them back as a comma-separated line with four decimal places.

The response contains no joint names or unit conversion. The sender must know what each output position means.

The two window sizes need reconciling. The [standard configuration](../../archive/2025-26/configs/model/tcn.yaml) declares 187 samples, and the [dedicated exporter](../../archive/2025-26/scripts/export_onnx.py) uses that value without requesting dynamic input dimensions. A 100-sample serial input cannot be assumed compatible with that export. There is a further discrepancy: the [implemented TCN](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py) has two kernel-7 convolutions per block across five blocks with dilations 1, 2, 4, 8, 16. Those layers imply a full receptive field of `1 + 2 × 6 × (1 + 2 + 4 + 8 + 16) = 373` samples, rather than the configured 187.

### Pass the result to Controls

The [interface draft](../../archive/2025-26/docs/ml_control_interface.md#output-specification) proposes converting Nm/kg to Nm by multiplying by participant mass. If targets were normalized during training, they first need their training mean and standard deviation restored. Neither the [server](../../archive/2025-26/scripts/server.py) nor the [serial endpoint](../../archive/2025-26/scripts/endpoint_tcn.py) performs these steps or commands motors.

Joint order also differs between code and specification. The server maps output positions to `[hip_l, hip_r, knee_l, knee_r]`, matching the targets in [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml). The [interface draft](../../archive/2025-26/docs/ml_control_interface.md) specifies `[hip_l, knee_l, hip_r, knee_r]` for outputs and joint-angle inputs. The serial endpoint sends outputs unchanged. Both input and output order must be agreed with Controls before integration; otherwise the right hip and left knee could be exchanged.

### Check inference and timing

The archive includes several supporting tools:

- [`inference_realtime.py`](../../archive/2025-26/scripts/inference_realtime.py) demonstrates PyTorch inference with an initial warmup, a rolling buffer, optional input normalization, and sample or chunk processing. Its main program streams a dataset trial rather than hardware readings.
- [`sample_inference.py`](../../archive/2025-26/scripts/sample_inference.py) compares PyTorch, ONNX, and traced TorchScript outputs and can benchmark them after warmup. It uses synthetic input and defaults to 100 timesteps.
- [`test_server.py`](../../archive/2025-26/scripts/test_server.py) and [`test_endpoint.cpp`](../../archive/2025-26/scripts/test_endpoint.cpp) send synthetic windows to the HTTP server and compare JSON and MessagePack timing. Despite its name, the C++ client does not test the serial endpoint.

The training data uses a 200 Hz timeline, with one sample every 5 ms. At that rate the standard 187-sample input represents about 0.94 seconds of movement (187 ÷ 200). At the 100 Hz rate proposed by Controls, the same input represents 1.87 seconds. Feeding 100 Hz readings directly to the model would double the time between adjacent samples, so a movement of a given duration would occupy half as many input rows as it did during training. [Dataset sampling rate](../../archive/2025-26/docs/georgia_dataset_analysis.md#L111-L118) · [Standard model window](../../archive/2025-26/configs/model/tcn.yaml) · [Controls timing specification](../../archive/2025-26/docs/ml_control_interface.md#timing--real-time-constraints)

The Controls draft allocates less than 5 ms to inference within its 10 ms loop. The HTTP clients report a pass when the average request round trip takes less than 10 ms. Sensor-to-motor timing remains unverified in this review; measured inference times are below.

### Measurements

These were run locally for this review. Nothing was run on exoskeleton hardware.

**Inference latency.** A checkpoint was exported with [`export_onnx.py`](../../archive/2025-26/scripts/export_onnx.py) (fixed `[1, 187, 28]` input, about 45.6k parameters) and timed over 500 runs on a laptop (Intel CPU, 1 thread).

| Runtime | Mean | p99 (slowest 1%) |
|---|---|---|
| ONNX Runtime | 2.3 ms | 5.9 ms |
| PyTorch | 9.0 ms | 14.2 ms |

A teammate's benchmark of last year's model on a **Raspberry Pi 5**, comparing PyTorch, TorchScript, and ONNX Runtime (the same comparison [`sample_inference.py`](../../archive/2025-26/scripts/sample_inference.py) makes), reported ONNX Runtime at **1.63 ms mean / 5.96 ms p99**, versus 10.03 ms mean for PyTorch: about 6× faster. The window length used there was not recorded; `sample_inference.py` defaults to 100 timesteps.

**Training run.** 3-epoch run of [`train_tcn.py`](../../archive/2025-26/scripts/train_tcn.py) with the default configuration (`outputs/2026-11-03/19-35-31`), evaluated on test participants BT15–BT17:

| Metric | Value |
|---|---|
| Test R² | 0.556 |
| Test RMSE | 0.555 scaled units (≈0.22 Nm/kg after undoing target normalization) |
| Test MAE | 0.385 scaled units |
| Per-joint test RMSE (hip_l, hip_r, knee_l, knee_r) | 0.553, 0.548, 0.580, 0.536 scaled units |

The training script prints these as "Nm/kg" ([`train_tcn.py#L311-L318`](../../archive/2025-26/scripts/train_tcn.py#L311-L318)), but the [dataset loader](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L304-L311) z-scores the targets, so the logged values are in standard deviations. The Nm/kg figure uses the target standard deviations in `normalization_stats.json`.

**Window shape.** Feeding a 100-sample window to the 187-sample export fails in ONNX Runtime with `INVALID_ARGUMENT`, confirming that the serial endpoint's input ([`endpoint_tcn.py#L17`](../../archive/2025-26/scripts/endpoint_tcn.py#L17)) is incompatible with the standard export. Perturbing past inputs showed that the newest output depends on about the last **360** samples, close to the 373-sample theoretical receptive field above.

## Advantages

- **ONNX Runtime was a good choice.** It is about 4× faster than PyTorch on a laptop and about 6× faster on a Pi 5 (see [Measurements](#measurements)), and the Pi does not need a full PyTorch install. [`server.py`](../../archive/2025-26/scripts/server.py)
- **The export script double-checks the converted model.** It runs PyTorch and ONNX on the same input and compares the outputs, and the check is easy to make stricter. [`export_onnx.py#L76-L87`](../../archive/2025-26/scripts/export_onnx.py#L76-L87)
- **The HTTP server is well designed.** It keeps no state between requests, has a `/health` check, rejects wrongly sized inputs, offers MessagePack for speed, and reports model time separately. [`server.py`](../../archive/2025-26/scripts/server.py)
- **The latency tests separate model time from overhead**, in both Python and C++. [`test_server.py`](../../archive/2025-26/scripts/test_server.py) · [`test_endpoint.cpp`](../../archive/2025-26/scripts/test_endpoint.cpp)
- **`inference_realtime.py` comes closest to handling live data correctly, but it is only a demo.**
  - [`inference_realtime.py`](../../archive/2025-26/scripts/inference_realtime.py) (demo only): keeps a rolling buffer, warms up on real samples, and scales inputs. It isn't connected to sensors or motors, doesn't convert outputs back to Nm/kg, and its 187-sample buffer is shorter than the ~360 samples the model uses.
  - [`endpoint_tcn.py`](../../archive/2025-26/scripts/endpoint_tcn.py) (serial deployment): has a rolling buffer, but it starts filled with zeros and never scales inputs.
  - [`server.py`](../../archive/2025-26/scripts/server.py) (HTTP deployment): has no buffer and doesn't scale. Both are left to the C++ control code, which was never written.
- **The model is small and causal.** At about 45.6k parameters it easily fits on a Pi, and it uses only past readings. [`tcn.py`](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py)

## Disadvantages

- **Neither deployment path scales inputs or outputs.** The model was trained on z-scores, but both paths send raw sensor values in and return z-scores labelled as Nm/kg, so the torques are wrong as written. [`datasets.py#L304-L311`](../../archive/2025-26/src/exoskeleton_ml/data/datasets.py#L304-L311) · [`server.py`](../../archive/2025-26/scripts/server.py) · [`endpoint_tcn.py`](../../archive/2025-26/scripts/endpoint_tcn.py)
  - *Recommend:* scale on the Pi with `normalization_stats.json`, since ML owns those statistics.
- **The joint order disagrees with the spec, which is a safety risk.** The code and training data use `[hip_l, hip_r, knee_l, knee_r]`; the Controls draft says `[hip_l, knee_l, hip_r, knee_r]`. Wired by the spec, the right-hip torque would drive the left-knee motor. The angle means in `normalization_stats.json` (hips ≈16.7, knees ≈−29.8 in the last four input positions) confirm the code's order. [`phase1.yaml`](../../archive/2025-26/configs/data/phase1.yaml) · [`server.py`](../../archive/2025-26/scripts/server.py) · [`ml_control_interface.md`](../../archive/2025-26/docs/ml_control_interface.md)
  - *Recommend:* keep the training order, fix the spec, and label the serial output. Changing the code would mean retraining.
- **The serial endpoint crashes on every call.** The ONNX model only accepts 187 samples and the endpoint sends 100. [`endpoint_tcn.py#L17`](../../archive/2025-26/scripts/endpoint_tcn.py#L17) · [`export_onnx.py`](../../archive/2025-26/scripts/export_onnx.py)
- **Every path uses the wrong history length.** The model looks back about 360 samples, but the scripts supply 187 or 100. [`tcn.yaml`](../../archive/2025-26/configs/model/tcn.yaml) · [`tcn.py`](../../archive/2025-26/src/exoskeleton_ml/models/tcn.py)
  - *Recommend:* re-export with a variable-length input and keep a ~360-sample buffer, so the full history is covered.
- **The spec assumes 100 Hz, but the data is 200 Hz.** Movements would look twice as fast to the model, and the 10 ms timing check is too loose. At 200 Hz the whole loop has 5 ms, and the Pi 5 ONNX p99 (5.96 ms) already exceeds that. [`georgia_dataset_analysis.md#L115-L116`](../../archive/2025-26/docs/georgia_dataset_analysis.md#L115-L116) · [`ml_control_interface.md`](../../archive/2025-26/docs/ml_control_interface.md#timing--real-time-constraints)
  - *Recommend:* stream at 200 Hz and test against 5 ms, because the model only fits its training rate.
- **The serial link is too slow.** At 115,200 baud (~11,520 bytes/s), a text line of 28 floats (~240 bytes) allows only about 48 readings/s (calculated). [`endpoint_tcn.py#L15`](../../archive/2025-26/scripts/endpoint_tcn.py#L15)
- **There is no safety fallback.** Neither path reuses the last torque or sends zero on an error. [`server.py`](../../archive/2025-26/scripts/server.py) · [`endpoint_tcn.py`](../../archive/2025-26/scripts/endpoint_tcn.py)
- **The export check is weak.** It uses one random input, and a mismatch only prints a warning. [`export_onnx.py#L78-L87`](../../archive/2025-26/scripts/export_onnx.py#L78-L87)
- **Testing only checks speed, not correctness, and the serial path is untested.** Tests use random or zero inputs, so wrong torques go unnoticed, and nothing tests `endpoint_tcn.py`, which is why its crash was missed. [`test_server.py`](../../archive/2025-26/scripts/test_server.py) · [`test_endpoint.cpp`](../../archive/2025-26/scripts/test_endpoint.cpp)
- **The angle units are likely mismatched.** The spec says radians, but the training angles look like degrees (means of roughly 17 and −30). [`ml_control_interface.md`](../../archive/2025-26/docs/ml_control_interface.md) · `normalization_stats.json`
  - *Recommend:* confirm with Controls and convert on the Pi.
- **Training RMSE is mislabelled as Nm/kg.** It is in scaled units; in real units it is about 0.22 Nm/kg. [`train_tcn.py#L311-L318`](../../archive/2025-26/scripts/train_tcn.py#L311-L318)

### Recommendations summary

| Issue | Last year | Recommend |
|---|---|---|
| Joint order | Code and spec disagree | Keep training order, fix spec, label outputs |
| Scaling | None in either path | Scale on the Pi with training stats |
| Window size | Fixed 187 export, serial sends 100 | Variable-length export, ~360-sample buffer |
| Sample rate | Spec assumes 100 Hz | Run at 200 Hz against a 5 ms budget |
| Serving path | Two half-finished paths | Serial, once fixed: it keeps buffering, scaling, and joint order on the ML side |

## Open questions

None of the five open questions in [`ml_control_interface.md#L238-L247`](../../archive/2025-26/docs/ml_control_interface.md#L238-L247) were answered, and the spec is still marked as a draft.

**For Controls**

- **Was Controls going to use the HTTP server or the serial endpoint?** They need different code from Controls, and the repo doesn't say which.
- **Did Controls ever write their C++/Arduino side?** If they did, its joint order and scaling need checking against ours.
- **Who multiplies by body mass to get Nm, and where does the mass come from?** This was never decided.
- **Does the hardware send degrees or radians, and which way do the IMU axes point?** The spec says radians, but the training data looks like degrees.

**Hardware testing**

- **How fast is a prediction on the target Pi with real data?** The Pi 5 benchmark used synthetic input and an unrecorded window length. It hasn't been timed with a ~360-sample window or recorded walking data.
- **What rate does our exoskeleton actually sample at: 55, 100, or 200 Hz?** The model only works correctly at the rate it was trained on (200 Hz).

**ML team**

- **How much accuracy is lost on real walking data with a 100- or 187-sample history instead of ~360?** It was only tested on random data.
- **Which ONNX file was the serial path tested with?** The standard export can't accept its 100-sample window.

### Next steps

1. Confirm joint order, sample rate, and units with Controls, then update the spec.
2. Add input scaling and output conversion to Nm/kg in the deployed path.
3. Re-export with a variable-length input and read the window size from the model.
4. Time inference on the target Pi with recorded walking data.
5. Add a safety fallback and correctness tests using real trials.
