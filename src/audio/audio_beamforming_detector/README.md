# Array-Agnostic Beamforming Detector

This package adds array-agnostic preprocessing, frequency-domain Delay-and-Sum
beamforming, optional light post-filtering, raw/beam YAMNet-RKNN fusion, and
5-window alarm logic in front of the existing YAMNet-RKNN model.

The current priority path is ReSpeaker4 realtime validation. ReSpeaker4 uses the
firmware hardware DOA angle as the default beam direction and falls back to
SRP-PHAT only when the hardware angle is unavailable. SMD16 and software-only
DOA remain reserved through config/backend interfaces, so replacing the capture
backend later does not require rewriting YAMNet-RKNN or alarm logic.

Default realtime entry:

```bash
python audio_beamforming_detector/run_realtime_respeaker.py \
  --config audio_beamforming_detector/configs/respeaker4.yaml \
  --model-path ../drone_yamnet.rknn \
  --scene 室外空旷 \
  --distance-m 20 \
  --background-noise 中噪声 \
  --target-present 1
```

Useful overrides:

```bash
python audio_beamforming_detector/run_realtime_respeaker.py \
  --config audio_beamforming_detector/configs/respeaker4.yaml \
  --model-path ../drone_yamnet.rknn \
  --doa-source respeaker_hardware

python audio_beamforming_detector/run_realtime_respeaker.py \
  --config audio_beamforming_detector/configs/respeaker4.yaml \
  --model-path ../drone_yamnet.rknn \
  --doa-source srp_phat
```

Mock hardware smoke test:

```bash
python audio_beamforming_detector/run_realtime_respeaker.py --mock-audio --model-path ""
```

Offline smoke test:

```bash
python audio_beamforming_detector/run_offline_eval.py \
  --config audio_beamforming_detector/configs/respeaker4.yaml \
  --input test/respeaker.wav \
  --scene 室内近距离 \
  --distance-m 5 \
  --background-noise 低噪声 \
  --target-present 1
```

Paper table aggregation from Qt metrics logs:

```bash
python scripts/generate_paper_tables.py \
  --metrics-root QT/QT_UI_Project_Hikvision_Modified/QT_UI_Project_Hikvision/build_windows/metrics_logs
```

The aggregation script fills tables 8.2, 8.3, and 8.4 when experiment labels
are present. Set `target_present=1` for drone runs and `target_present=0` for
negative/noise runs; otherwise Accuracy, Recall, F1, false alarm rate, and miss
rate are left as pending collection instead of being guessed.
