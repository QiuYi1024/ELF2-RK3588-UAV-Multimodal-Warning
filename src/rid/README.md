# RID900 Remote ID 接入

日期：2026-06-25  
执行者：Codex  
协议依据：`D:/codex/ELF2/AntiUAV/RID900无人机探测模块使用说明书.docx` V1.0

## 硬件连接

RID900 的 USB Type-C 接口同时提供 5V 供电和虚拟串口通信。接到 ELF2 的 USB 3.0 口可以正常使用，但实际链路是 115200 bps 串口，USB 2.0 带宽也足够。

当前实机枚举：

```text
USB VID:PID: 303a:1001
串口驱动: cdc_acm
稳定设备名: /dev/antiuav-rid900
原始设备: /dev/ttyACM0
串口参数: 115200, 8 data bits, no parity, 1 stop bit, no flow control
```

板端 udev 规则：

```text
/etc/udev/rules.d/99-antiuav-rid900.rules
```

## 软件链路

```text
RID900 USB 虚拟串口
-> RID900/rid900_reader.py
-> UDP 127.0.0.1:5009
-> Qt Remote ID 页面
```

`rid900_reader.py` 支持：

- 自动发现 `/dev/serial/by-id/*`、`/dev/ttyUSB*`、`/dev/ttyACM*`
- 多串口并存时拒绝猜测，要求配置明确设备路径
- 串口断开后自动重连
- 显式设置 DTR/RTS，恢复 USB CDC 偶发的“端口已打开但无输出”
- 无报文时每 2 秒发送串口在线状态，8 秒无任何字节后自动重连
- 解析 `$GBRID` 无人机报文
- 解析 `$HBRID` 模块心跳
- 保留说明书中的校验文本，但不按标准 NMEA XOR 错误丢包
- 将无人机序列号、厂商、型号、位置、高度、速度、航向和飞手位置转换为 JSON

## 配置

私有 `configs/runtime.env`：

```text
ANTI_UAV_RID900_ENABLED=1
ANTI_UAV_RID900_DEVICE=/dev/antiuav-rid900
ANTI_UAV_RID900_BAUD=115200
ANTI_UAV_RID900_HOST=127.0.0.1
ANTI_UAV_RID900_PORT=5009
ANTI_UAV_RID900_DTR=0
ANTI_UAV_RID900_RTS=0
ANTI_UAV_RID900_SETTLE_SEC=1.2
ANTI_UAV_RID900_STATUS_INTERVAL_SEC=2.0
ANTI_UAV_RID900_IDLE_RECONNECT_SEC=8.0
ANTI_UAV_RID900_RECONNECT_SEC=2
```

一键启动、停止和状态检查已包含 RID900：

```bash
./bin/run_all.sh
./bin/status_all.sh
./bin/stop_all.sh
```

## 验证

解析器单元测试：

```bash
cd <repo_root>/src/rid
python -m unittest -v test_rid900_reader.py
```

说明书样例回放：

```bash
cd <repo_root>/src/rid
./bin/run_rid900.sh \
  --input-file RID900/manual_samples.txt \
  --replay-interval-ms 1000
```

实机当前已读取到：

```text
RID900 心跳正常
设备号: DEMO_DEVICE_002
目标数: 0（验证时无人机已关闭）
```

## “已接 USB 但仍等待模块”

先区分两个状态：

- `已连接 / 等待报文`：USB、驱动和串口正常，但暂未收到 `$HBRID/$GBRID`。
- `RID900 心跳正常`：已经持续收到模块心跳。

2026-06-25 实机曾出现 `/dev/ttyACM0` 已打开但模块无输出。短时只读测试确认
四种 DTR/RTS 组合均可恢复 `$HBRID`，设备、线缆、驱动和 115200 8N1 均正常。
读取器现已自动执行控制线设置和无数据重连，无需手工拔插。
