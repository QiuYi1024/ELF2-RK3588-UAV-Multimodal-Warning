#!/usr/bin/env python3
"""RID900 virtual-serial reader and UDP bridge."""

from __future__ import annotations

import argparse
import glob
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


def _to_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _split_checksum(line: str) -> tuple[str, str]:
    body, marker, checksum = line.strip().partition("*")
    return body, checksum.strip() if marker else ""


@dataclass
class RidTarget:
    serial_number: str
    vendor: str
    product_type: str
    detect_rssi_dbm: int
    detect_snr_db: int
    drone_type: int
    drone_class: int
    state_info: int
    drone_longitude: float
    drone_latitude: float
    height_m: float
    altitude_m: float
    pressure_height_m: float
    yaw_deg: int
    horizontal_speed_mps: float
    vertical_speed_mps: float
    position_error_m: float
    height_error_m: float
    pilot_longitude: float
    pilot_latitude: float
    pilot_height_m: float


def parse_gbrid(line: str) -> dict:
    body, checksum = _split_checksum(line)
    fields = body.split(",")
    if len(fields) < 22 or fields[0] != "$GBRID":
        raise ValueError(f"invalid GBRID field count/header: {len(fields)}")

    target = RidTarget(
        serial_number=fields[3].strip(),
        vendor=fields[4].strip(),
        product_type=fields[5].strip(),
        detect_rssi_dbm=_to_int(fields[1]),
        detect_snr_db=_to_int(fields[2]),
        drone_type=_to_int(fields[6]),
        drone_class=_to_int(fields[7]),
        state_info=_to_int(fields[8]),
        drone_longitude=_to_float(fields[9]),
        drone_latitude=_to_float(fields[10]),
        height_m=_to_float(fields[11]),
        altitude_m=_to_float(fields[12]),
        pressure_height_m=_to_float(fields[13]),
        yaw_deg=_to_int(fields[14]),
        horizontal_speed_mps=_to_float(fields[15]),
        vertical_speed_mps=_to_float(fields[16]),
        position_error_m=_to_float(fields[17]),
        height_error_m=_to_float(fields[18]),
        pilot_longitude=_to_float(fields[19]),
        pilot_latitude=_to_float(fields[20]),
        pilot_height_m=_to_float(fields[21]),
    )
    payload = asdict(target)
    payload.update(
        {
            "type": "rid900_target",
            "checksum_text": checksum,
            "raw": line.strip(),
        }
    )
    return payload


def parse_hbrid(line: str) -> dict:
    body, checksum = _split_checksum(line)
    fields = body.split(",")
    if len(fields) < 9 or fields[0] != "$HBRID":
        raise ValueError(f"invalid HBRID field count/header: {len(fields)}")
    return {
        "type": "rid900_heartbeat",
        "heartbeat": fields[1].strip(),
        "device_id": fields[2].strip(),
        # 说明书未定义第 3、4 字段语义，保留原值，避免错误解释。
        "metric_1": _to_float(fields[3]),
        "metric_2": _to_float(fields[4]),
        "module_longitude": _to_float(fields[5]),
        "module_latitude": _to_float(fields[6]),
        "module_altitude_m": _to_float(fields[7]),
        "fix_status": _to_int(fields[8]),
        "checksum_text": checksum,
        "raw": line.strip(),
    }


def parse_line(line: str) -> dict:
    line = line.strip()
    if line.startswith("$GBRID,"):
        return parse_gbrid(line)
    if line.startswith("$HBRID,"):
        return parse_hbrid(line)
    raise ValueError("unsupported RID900 sentence")


def discover_serial_port(explicit: str) -> str:
    if explicit and explicit.lower() != "auto":
        return explicit

    candidates: list[str] = []
    for pattern in ("/dev/serial/by-id/*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        candidates.extend(glob.glob(pattern))
    candidates = sorted(dict.fromkeys(os.path.realpath(path) for path in candidates))
    if not candidates:
        raise FileNotFoundError("no /dev/ttyUSB*, /dev/ttyACM* or /dev/serial/by-id device")
    if len(candidates) > 1:
        raise RuntimeError(
            "multiple serial devices found; set ANTI_UAV_RID900_DEVICE explicitly: "
            + ", ".join(candidates)
        )
    return candidates[0]


class UdpPublisher:
    def __init__(self, host: str, port: int) -> None:
        self.destination = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_summary_monotonic = 0.0

    def send(self, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("ts_ms", int(time.time() * 1000))
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.destination)
        now = time.monotonic()
        if now - self._last_summary_monotonic < 5.0:
            return
        self._last_summary_monotonic = now
        message_type = payload.get("type", "")
        if message_type == "rid900_target":
            print(
                "[RID900] target "
                f"serial={payload.get('serial_number', '--')} "
                f"model={payload.get('vendor', '--')} {payload.get('product_type', '--')} "
                f"rssi={payload.get('detect_rssi_dbm', 0)}",
                flush=True,
            )
        elif message_type == "rid900_heartbeat":
            print(
                "[RID900] heartbeat "
                f"device={payload.get('device_id', '--')} "
                f"source={payload.get('source', '--')}",
                flush=True,
            )
        elif message_type == "rid900_status":
            print(
                "[RID900] status "
                f"connected={payload.get('connected', 0)} "
                f"message={payload.get('message', '--')} "
                f"source={payload.get('source', '--')}",
                flush=True,
            )


def publish_lines(
    lines: Iterable[str],
    publisher: UdpPublisher,
    source: str,
    interval_sec: float = 0.0,
) -> int:
    parsed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = parse_line(line)
        except ValueError as exc:
            publisher.send(
                {
                    "type": "rid900_parse_error",
                    "source": source,
                    "error": str(exc),
                    "raw": line,
                }
            )
            continue
        payload["source"] = source
        publisher.send(payload)
        parsed += 1
        if interval_sec > 0:
            time.sleep(interval_sec)
    return parsed


def run_file(path: Path, publisher: UdpPublisher, interval_sec: float) -> int:
    publisher.send({"type": "rid900_status", "connected": 1, "source": str(path), "mode": "replay"})
    with path.open("r", encoding="utf-8") as handle:
        count = publish_lines(handle, publisher, str(path), interval_sec)
    publisher.send(
        {
            "type": "rid900_status",
            "connected": 0,
            "source": str(path),
            "mode": "replay",
            "message": "replay_complete",
            "parsed_messages": count,
        }
    )
    return count


def run_serial(args: argparse.Namespace, publisher: UdpPublisher) -> None:
    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("pyserial is required for RID900 serial mode") from exc

    while True:
        port = ""
        try:
            port = discover_serial_port(args.device)
            publisher.send(
                {
                    "type": "rid900_status",
                    "connected": 0,
                    "source": port,
                    "mode": "serial",
                    "message": "opening",
                }
            )
            with serial.Serial(
                port=port,
                baudrate=args.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5,
            ) as device:
                # RID900 的 Espressif USB CDC 偶发出现“端口已打开但无输出”。
                # 显式设置控制线可让模块从该暂态恢复，且不涉及固件修改。
                device.dtr = bool(args.dtr)
                device.rts = bool(args.rts)
                time.sleep(args.settle_sec)
                publisher.send(
                    {
                        "type": "rid900_status",
                        "connected": 1,
                        "source": port,
                        "mode": "serial",
                        "message": "connected_waiting_data",
                    }
                )
                last_data_monotonic = time.monotonic()
                last_status_monotonic = last_data_monotonic
                while True:
                    raw = device.readline()
                    if not raw:
                        now = time.monotonic()
                        if now - last_status_monotonic >= args.status_interval_sec:
                            publisher.send(
                                {
                                    "type": "rid900_status",
                                    "connected": 1,
                                    "source": port,
                                    "mode": "serial",
                                    "message": "connected_waiting_data",
                                    "idle_ms": int((now - last_data_monotonic) * 1000),
                                }
                            )
                            last_status_monotonic = now
                        if now - last_data_monotonic >= args.idle_reconnect_sec:
                            raise TimeoutError(
                                f"no RID900 serial data for {args.idle_reconnect_sec:.1f}s"
                            )
                        continue
                    last_data_monotonic = time.monotonic()
                    last_status_monotonic = last_data_monotonic
                    line = raw.decode("ascii", errors="replace").strip()
                    publish_lines((line,), publisher, port)
        except KeyboardInterrupt:
            publisher.send(
                {
                    "type": "rid900_status",
                    "connected": 0,
                    "source": port,
                    "mode": "serial",
                    "message": "stopped",
                }
            )
            return
        except Exception as exc:
            publisher.send(
                {
                    "type": "rid900_status",
                    "connected": 0,
                    "source": port or args.device,
                    "mode": "serial",
                    "message": "reconnecting",
                    "error": str(exc),
                }
            )
            print(f"[RID900] {exc}", file=sys.stderr, flush=True)
            time.sleep(args.reconnect_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RID900 serial/NMEA to UDP bridge")
    parser.add_argument("--device", default=os.getenv("ANTI_UAV_RID900_DEVICE", "auto"))
    parser.add_argument("--baud", type=int, default=int(os.getenv("ANTI_UAV_RID900_BAUD", "115200")))
    parser.add_argument("--udp-host", default=os.getenv("ANTI_UAV_RID900_HOST", "127.0.0.1"))
    parser.add_argument("--udp-port", type=int, default=int(os.getenv("ANTI_UAV_RID900_PORT", "5009")))
    parser.add_argument("--dtr", type=int, choices=(0, 1),
                        default=int(os.getenv("ANTI_UAV_RID900_DTR", "0")))
    parser.add_argument("--rts", type=int, choices=(0, 1),
                        default=int(os.getenv("ANTI_UAV_RID900_RTS", "0")))
    parser.add_argument("--settle-sec", type=float,
                        default=float(os.getenv("ANTI_UAV_RID900_SETTLE_SEC", "1.2")))
    parser.add_argument("--status-interval-sec", type=float,
                        default=float(os.getenv("ANTI_UAV_RID900_STATUS_INTERVAL_SEC", "2.0")))
    parser.add_argument("--idle-reconnect-sec", type=float,
                        default=float(os.getenv("ANTI_UAV_RID900_IDLE_RECONNECT_SEC", "8.0")))
    parser.add_argument(
        "--reconnect-sec",
        type=float,
        default=float(os.getenv("ANTI_UAV_RID900_RECONNECT_SEC", "2")),
    )
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--replay-interval-ms", type=int, default=50)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    publisher = UdpPublisher(args.udp_host, args.udp_port)
    if args.input_file:
        return 0 if run_file(args.input_file, publisher, args.replay_interval_ms / 1000.0) > 0 else 1
    run_serial(args, publisher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
