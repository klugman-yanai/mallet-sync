# src/config.py

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from numpy.typing import DTypeLike


@dataclass
class AudioConfig:
    """Configuration for audio recording settings."""
    sample_rate: int = 16000
    dtype: DTypeLike = "int16"
    channels: int = 9
    chunk_size: int = 1024
    recordings_dir: Path = field(default_factory=lambda: Path("recordings"))

    def __post_init__(self):
        """Ensure recordings directory exists."""
        self.recordings_dir.mkdir(exist_ok=True)


@dataclass
class DeviceConfig:
    """Configuration for a single audio device."""
    name: str
    device_id: int
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.device_id}, Channels: {self.channels})"


@dataclass
class RecordingSession:
    """Configuration for a recording session."""
    duration: float
    devices: dict[int, DeviceConfig]
    output_format: str = 'WAV'
    subtype: str = 'PCM_16'
    timestamp_format: str = "%Y%m%d_%H%M%S"

    def get_filename(self, device_id: int, channel: int | None = None) -> Path:
        """
        Generate filename path for recording based on device and channel.

        Args:
            device_id: ID of the recording device
            channel: Optional channel number for multi-channel recordings

        Returns:
            Path object for the recording file
        """
        from datetime import datetime

        if device_id not in self.devices:
            raise ValueError(f"Device ID {device_id} not found in recording session")

        device = self.devices[device_id]
        timestamp = datetime.now().strftime(self.timestamp_format)
        ext = self.output_format.lower()

        if channel is not None:
            filename = f"{device.name}_ch{channel}_{timestamp}.{ext}"
        else:
            filename = f"{device.name}_{timestamp}.{ext}"

        return Path(filename)
