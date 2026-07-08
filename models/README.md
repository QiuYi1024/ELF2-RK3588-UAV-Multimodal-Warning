# 模型文件说明

- `best_uav_headless_i8.rknn`：RK3588 NPU 视觉无人机检测模型。
- `drone_yamnet.rknn`：RK3588 NPU 无人机声音识别模型。

运行时优先读取 `ANTI_UAV_YOLO_MODEL` 和 `ANTI_UAV_YAMNET_RKNN`，未设置时使用本目录模型。