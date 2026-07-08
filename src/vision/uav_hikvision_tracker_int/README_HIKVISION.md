# Hikvision RKNN YOLO UAV Tracker

这是从原来的 `/dev/video61 + 环境学习期` 版本改出来的海康球机 RTSP 版本。

## 主要修改

1. 输入源从 `/dev/video61` 改为海康 RTSP：
   `RTSP_URL_HERE
2. 取消环境适应学习、死物黑名单、启动 30 秒学习期。
3. 取消 ROI 裁剪跟踪。云台在动时 ROI 裁剪容易失效，所以始终全图 letterbox 到 640x640。
4. 保留 YOLO + NMS + Kalman 稳定跟踪。
5. 输出 UDP JSON 到 `127.0.0.1:5005`，包含原始海康画面坐标：
   - `frame_w`, `frame_h`
   - `bbox_x`, `bbox_y`, `bbox_w`, `bbox_h`
   - `cx`, `cy`, `dx`, `dy`
   - `conf`, `tracking`
6. 保留共享内存 `/rk3588_vision`，写入 640x640 letterbox BGR 图，兼容之前的显示逻辑。

## 编译

```bash
cd <repo_root>/src/vision/uav_hikvision_tracker_int
mkdir -p build
cd build
cmake ..
make -j4
```

## 运行前先配置海康网络

```bash
cd <repo_root>/src/vision/uav_hikvision_tracker_int
bash scripts/run_yolo.sh --variant int --check-only
```

## 运行

默认子码流 102：

```bash
cd <repo_root>/src/vision/uav_hikvision_tracker_int
./uav_hik_tracker
```

使用主码流 101：

```bash
./uav_hik_tracker "RTSP_URL_HERE"
```

## 重要参数

在 `main.cpp` 里：

```cpp
const bool ROTATE_180 = true;
const float SEARCH_CONF = 0.25f;
const float TRACK_CONF  = 0.15f;
const float TRACK_GATE_PX = 180.0f;
```

如果误检多，先提高 `SEARCH_CONF`，比如 `0.35`。
如果漏检多，先降低 `TRACK_CONF`，比如 `0.12`。
如果云台运动后容易丢目标，可以适当增大 `TRACK_GATE_PX`，比如 `220`。
