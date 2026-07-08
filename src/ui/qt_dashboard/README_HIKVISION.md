# Hikvision PTZ UAV Fusion Dashboard

这个版本把原来的 `/rk3588_vision` 共享内存视频输入，改为直接读取海康威视 RTSP：

```text
RTSP_URL_HERE
```

视觉识别结果仍通过 UDP `127.0.0.1:5005` 接入，音频雷达保持 UDP `5006`，原音频雷达显示逻辑保留。

## 编译运行

```bash
cd QT_UI_Project_Hikvision
rm -rf build
cmake -S . -B build
cmake --build build -j4
./build/qt_dashboard
```

## 视觉 UDP JSON 支持字段

推荐识别程序发送：

```json
{
  "frame_id": 123,
  "bbox_x": 100,
  "bbox_y": 120,
  "bbox_w": 50,
  "bbox_h": 40,
  "frame_w": 704,
  "frame_h": 576,
  "conf": 0.83,
  "state": "TRACK"
}
```

如果不发送 `frame_w/frame_h`，UI 会默认按当前 RTSP 视频帧尺寸缩放。
