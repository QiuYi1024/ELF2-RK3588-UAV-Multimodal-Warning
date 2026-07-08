from __future__ import annotations

import math

import numpy as np


def circular_array_positions(num_mics: int, radius_m: float, start_angle_deg: float = 0.0) -> np.ndarray:
    coords = []
    for i in range(int(num_mics)):
        a = math.radians(float(start_angle_deg) + 360.0 * i / float(num_mics))
        coords.append([float(radius_m) * math.cos(a), float(radius_m) * math.sin(a), 0.0])
    return np.asarray(coords, dtype=np.float64)


def build_mic_positions(array_config: dict) -> np.ndarray:
    num_mics = int(array_config.get("num_mics", 0))
    geometry = str(array_config.get("geometry", "custom")).lower()

    if "mic_positions" in array_config:
        positions = np.asarray(array_config["mic_positions"], dtype=np.float64)
    elif geometry == "circular":
        positions = circular_array_positions(
            num_mics=num_mics,
            radius_m=float(array_config.get("radius_m", 0.08)),
            start_angle_deg=float(array_config.get("start_angle_deg", 0.0)),
        )
    else:
        raise ValueError("custom geometry requires array.mic_positions in the YAML config")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"mic_positions must have shape [num_mics, 3], got {positions.shape}")
    if num_mics and positions.shape[0] != num_mics:
        raise ValueError(f"array.num_mics={num_mics} but mic_positions has {positions.shape[0]} rows")
    return positions


def direction_vector_2d(azimuth_deg: float) -> np.ndarray:
    a = math.radians(float(azimuth_deg))
    return np.array([math.cos(a), math.sin(a), 0.0], dtype=np.float64)


def far_field_delays(mic_positions: np.ndarray, azimuth_deg: float, sound_speed: float = 343.0) -> np.ndarray:
    direction = direction_vector_2d(azimuth_deg)
    delays = np.asarray(mic_positions, dtype=np.float64).dot(direction) / float(sound_speed)
    return delays - float(np.mean(delays))
