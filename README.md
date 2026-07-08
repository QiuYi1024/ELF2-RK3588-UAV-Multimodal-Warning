# 基于RK3588的声光多模态融合低空无人机预警系统

## 项目简介

本项目面向低空无人机预警场景，基于飞凌ELF2（RK3588）构建边缘人工智能声光多模态系统。系统接入网络球机和多通道麦克风阵列，在 RK3588 端完成视觉检测、音频识别、DOA 方向感知、云台控制、Qt 界面显示和实验数据保存。

## 赛题方向

边缘人工智能应用。

## 运行平台

主控平台为飞凌ELF2 / RK3588。Windows 端仅作为远程显示、调试、录像和数据检查辅助。

## 主要功能

- YOLO/RKNN 无人机视觉检测与 ByteTrack-lite 跟踪
- 海康网络球机 RTSP 接入和 PTZ 控制框架
- ReSpeaker 多通道音频采集、YAMNet/RKNN 声音识别和 DOA 方位角读取
- 声学方向引导视觉搜索、视觉确认与融合控制策略
- RID900 Remote ID 状态读取
- RK3588 端 Qt MIPI 界面和 Windows 端 Qt 远程调试界面
- session 数据保存、manifest、summary 和校验文件生成逻辑

## 目录结构

```text
configs/           脱敏 example 配置
scripts/           导出结构适配后的一键启动、停止、验证和导出脚本
src/vision/        YOLO/RKNN、海康 RTSP、后处理、跟踪和视觉 PTZ 控制源码
src/audio/         ReSpeaker、YAMNet/RKNN、DOA、音频预处理源码
src/fusion/        声光融合和控制策略源码
src/ptz/           PTZ 控制、自动跟踪、自动变焦相关源码
src/rid/           RID900 读取器、解析器和 demo 测试样例
src/ui/            Qt 主界面、RadarWidget、VideoWidget、DataManager 相关源码
src/data_manager/  session、manifest、summary 和数据目录管理逻辑
models/            RKNN 模型
examples/          脱敏 demo 样例
docs/design/       真实脱敏设计文件待补
docs/report/       脱敏作品设计报告待补
demo/              脱敏演示视频或链接待补
photos/            开发板和系统实物照片待补
```

## 快速启动

1. 根据 `configs/*.example.yaml` 创建本地真实配置，真实配置不得提交。
2. 确认 `models/best_uav_headless_i8.rknn` 和 `models/drone_yamnet.rknn` 存在。
3. C++/Qt 模块需先按各自 `CMakeLists.txt` 编译生成可执行文件。
4. 可通过环境变量覆盖模型和可执行文件路径：`ANTI_UAV_YOLO_MODEL`、`ANTI_UAV_YAMNET_RKNN`、`ANTI_UAV_YOLO_BIN`、`ANTI_UAV_QT_BIN`。
5. 运行：

```bash
bash scripts/install_deps.sh
bash scripts/verify_all.sh
bash scripts/run_all.sh
```

## 数据保存说明

默认数据根目录为 `/home/elf/AntiUAV_Data`。公开仓库不包含完整原始数据集、大视频、大音频、旧日志和缓存。

## 隐私说明

本仓库不包含真实学校信息、指导老师信息、个人身份信息、设备密码、访问令牌、完整原始数据集和未脱敏测试数据。RID 样例使用 `DEMO_SERIAL_001`、`DEMO_DEVICE_001`、`DEMO_DEVICE_002` 和虚构 demo 坐标。

## 后续待补材料

`docs/design`、`docs/report`、`demo`、`photos` 仅为开源仓库预留目录。当前未生成假报告、假视频、假图片或假照片，后续由作者补充真实且脱敏的材料。

## 开源协议

本导出版本默认使用 MIT License。第三方依赖遵循其各自协议。