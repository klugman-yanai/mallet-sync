# src/config.py

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from numpy.typing import DTypeLike


@dataclass(frozen=True)
class DeviceConfig:
    """Configuration for an audio recording device"""
    device_id: int
    name: str
    channels: int = 9
    sample_rate: int = 16000
    chunk_size: int = 1024
    dtype: DTypeLike = "int16"

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.device_id}, Channels: {self.channels})"


@dataclass(frozen=True)
class RecordingSession:
    """Configuration for a recording session"""
    devices: list[DeviceConfig]
    base_path: Path = field(default_factory=lambda: Path("recordings"))
    output_format: str = "WAV"
    subtype: str = "PCM_16"
    timestamp_format: str = "%H_%M_%S_%Y%m%d"  # 14_30_45_20240115

    def __post_init__(self):
        """Convert base_path to Path if string provided"""
        if isinstance(self.base_path, str):
            object.__setattr__(self, 'base_path', Path(self.base_path))
        self.base_path.mkdir(exist_ok=True)
