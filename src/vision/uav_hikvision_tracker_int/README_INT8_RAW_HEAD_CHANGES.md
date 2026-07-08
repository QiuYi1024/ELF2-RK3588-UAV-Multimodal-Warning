# YOLO11 RKNN INT8 Raw-Head 控制程序适配说明

- 日期：2026-06-04
- 执行者：Codex
- 适用目录：`uav_hikvision_tracker_int`
- 目标平台：RK3588 / RKNN Runtime / 海康球机 YOLO 控制程序

## 结论

本目录已经适配新版 YOLO11 RKNN INT8 raw-head 模型。程序现在支持两种输出结构：

```text
旧 FP/单输出模型：
  output0 = [1, 5, 8400]

新 INT8/raw-head 模型：
  output0 reg_s8   [1,64,80,80]
  output1 cls_s8   [1, 1,80,80]
  output2 reg_s16  [1,64,40,40]
  output3 cls_s16  [1, 1,40,40]
  output4 reg_s32  [1,64,20,20]
  output5 cls_s32  [1, 1,20,20]
```

默认模型已经切换为：

```text
model/best_uav_headless_i8.rknn
```

该模型来自当前训练模型的 raw-head INT8 转换结果，已经通过 RK3588 正负实拍视频验证：

```text
正样本 drone_real_upright.mp4：1382 帧中 1001 帧通过采集阈值检测
负样本 huanjing_2026_5_23.mp4：2734 帧中 0 帧通过采集阈值误报
平均推理耗时：约 21 ms/frame
总体判定：PASS
```

模型 SHA256：

```text
C00C6DB810C148FDB204F3BDC7F142CCB313B80F8367CDCE829D243AFB1353AA
```

## 本次改动文件

### `model/best_uav_headless_i8.rknn`

新增已验证通过的 INT8 raw-head 模型。后续部署到板子时，需要确保它位于：

```text
<repo_root>
```

### `main.cpp`

主要改动：

```text
1. 导出包默认视觉模型路径为 models/best_uav_headless_i8.rknn，可通过 ANTI_UAV_YOLO_MODEL 覆盖。
2. 新增 g_rknn_output_count，用于记录 RKNN 实际输出数量。
3. rknn_init 后调用 rknn_query(RKNN_QUERY_IN_OUT_NUM) 查询输出数量。
4. 启动时打印每个输出的 dims 和 size，方便确认模型结构。
5. 推理线程不再固定取 1 个输出，而是按实际 output_count 调用 rknn_outputs_get。
6. output_count == 1 时继续走旧 decode_yolo11_single_output。
7. output_count == 6 时走新增 decode_yolo11_raw_head_outputs。
8. 新增 `--model-info-only`，只加载模型并打印输出信息，不启动 RTSP、显示和云台线程。
9. 新增 `--model-self-test`，用一帧黑图跑一次 RKNN 推理和 C++ YOLO 解码，不启动 RTSP、显示和云台线程。
```

输入预处理没有改，仍然保持和原程序一致：

```cpp
inputs[0].type = RKNN_TENSOR_UINT8;
inputs[0].fmt = RKNN_TENSOR_NHWC;
inputs[0].size = 640 * 640 * 3;
```

### `yolo_decoder.h`

新增 raw-head 解码函数声明：

```cpp
std::vector<ObjectBox> decode_yolo11_raw_head_outputs(
    rknn_output* outputs,
    int output_count,
    int img_size = 640
);
```

旧的单输出解码函数仍然保留，便于临时回退 FP 模型：

```cpp
std::vector<ObjectBox> decode_yolo11_single_output(rknn_output* outputs, int img_size = 640);
```

### `yolo_decoder.cpp`

新增 YOLO11 raw-head 解码逻辑：

```text
1. 读取 reg/cls 三个尺度输出。
2. 对 cls logit 做 sigmoid 得到单类别置信度。
3. 使用 MIN_DECODE_CONF=0.005 做早期候选过滤。
4. 对 reg 的 4 个方向分别做 DFL softmax。
5. 用 reg_max=16 的期望值还原 left/top/right/bottom。
6. 按 stride=8/16/32 还原 640x640 letterbox 坐标。
7. 转成 ObjectBox{cx, cy, w, h, conf}，交给 main.cpp 后续 NMS、几何过滤、候选态和跟踪态逻辑。
```

## 编译方式

在 RK3588 板子上编译：

```bash
cd <repo_root>
rm -rf build
mkdir build
cd build
cmake ..
make -j$(nproc)
```

如果 `rknn_api.h` 或 `librknnrt.so` 不在系统默认路径，按板子实际路径修改 `CMakeLists.txt` 里的 include 或 link 路径。

## 运行方式

使用默认 INT8 模型运行：

```bash
cd <repo_root>
./uav_hik_tracker
```

只检查模型结构，不启动控制程序：

```bash
cd <repo_root>
./uav_hik_tracker --model-info-only
```

只验证 RKNN 推理和 C++ 解码链路，不启动控制程序：

```bash
cd <repo_root>
./uav_hik_tracker --model-self-test
```

显式指定 INT8 模型：

```bash
./uav_hik_tracker \
  --model <repo_root>/models/best_uav_headless_i8.rknn
```

临时回退旧 FP 模型：

```bash
./uav_hik_tracker \
  --model <repo_root>/models/best_uav_headless_i8.rknn
```

也可以通过环境变量指定模型：

```bash
export ANTI_UAV_YOLO_MODEL=<repo_root>/models/best_uav_headless_i8.rknn
./uav_hik_tracker
```

## 正常启动日志

INT8 raw-head 模型正常时，应看到类似日志：

```text
模型：<repo_root>/models/best_uav_headless_i8.rknn
✅ RKNN 输出数量：6
   output0 dims=[1,64,80,80] ...
   output1 dims=[1,1,80,80] ...
   output2 dims=[1,64,40,40] ...
   output3 dims=[1,1,40,40] ...
   output4 dims=[1,64,20,20] ...
   output5 dims=[1,1,20,20] ...
[YOLO11-RAW-DECODE] max_conf=..., conf>0.005=..., returned=...
```

`--model-self-test` 正常时，应看到类似日志：

```text
[SELFTEST] output0 size=1638400 want_float=1
[SELFTEST] output1 size=25600 want_float=1
[SELFTEST] output2 size=409600 want_float=1
[SELFTEST] output3 size=6400 want_float=1
[SELFTEST] output4 size=102400 want_float=1
[SELFTEST] output5 size=1600 want_float=1
[YOLO11-RAW-DECODE] max_conf=..., conf>0.005=..., returned=...
[SELFTEST] decoded_boxes=...
```

如果输出数量是 `1`，说明当前加载的是旧单输出模型，程序会自动走旧解码。

如果输出数量不是 `1` 或 `6`，程序会打印：

```text
[RKNN] unsupported output_count=...
```

这说明模型结构和当前解码器不匹配。

## 阈值关系

raw-head 解码阶段只做低阈值候选过滤：

```text
RAW_HEAD_MIN_DECODE_CONF = 0.005
```

真正进入控制逻辑仍然使用 `main.cpp` 原有阈值：

```text
TRACK_CONF   = 0.30
ACQUIRE_CONF = 0.55
IOU_THRESH   = 0.40
```

注意：验证结果里负样本有少量 `TRACK_CONF=0.30` 级别候选，但没有任何帧达到 `ACQUIRE_CONF=0.55`。所以后续不要把低阈值候选直接当成首次采集目标，否则会增加误触发。

## 常见问题

### 1. 启动后提示 `RKNN 输出数量：1`

说明加载的是旧 FP 或旧单输出模型。确认 `--model` 或 `ANTI_UAV_YOLO_MODEL` 指向：

```text
best_uav_headless_i8.rknn
```

### 2. 输出数量是 6，但没有检测框

优先检查日志：

```text
[YOLO11-RAW-DECODE] max_conf=..., conf>0.005=..., returned=...
```

如果 `max_conf` 长期为 0，说明模型或输出顺序异常。

如果 `conf>0.005` 有值但 `returned=0`，说明 DFL 解码后的框被几何过滤掉，重点检查输出顺序是否仍为：

```text
reg_s8, cls_s8, reg_s16, cls_s16, reg_s32, cls_s32
```

### 3. 出现 `Query dynamic range failed`

这是 RKNN 静态 shape 模型常见 warning。只要 `load_rknn`、`rknn_init`、`rknn_run` 成功，并且 raw-head score 非零，不需要因为这条 warning 回退。

### 4. 负样本误报变多

先确认首次采集仍要求：

```text
conf >= ACQUIRE_CONF
```

不要只用 `TRACK_CONF` 进入候选态。当前实测通过是建立在 `ACQUIRE_CONF=0.55` 的采集阈值上。

## 后续建议

1. 先在板子上只编译运行程序，确认启动日志显示 `RKNN 输出数量：6`。
2. 用 `--model-info-only` 确认模型输出结构，不直接启动 RTSP 和云台。
3. 用 `--model-self-test` 确认 C++ 可以完成一次 RKNN 推理和 raw-head 解码。
4. 使用固定实拍视频或 RTSP 回放验证检测效果，再接入云台闭环。
5. 如果要继续换模型，保持 raw-head 6 输出结构不变，C++ 侧不需要再改解码器。
