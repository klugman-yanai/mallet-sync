# src/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from numpy.typing import DTypeLike

if TYPE_CHECKING:
    from mallet_sync.audio import SoundDeviceStreamRecorder


DEFAULT_FILENAME_TEMPLATE = 'mallet_{role}_{context}.wav'
DEFAULT_OUTPUT_DIR = Path('./output')
DEFAULT_INPUT_DIR = Path('./recordings')


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_low_input_latency: float
    default_low_output_latency: float
    default_high_input_latency: float
    default_high_output_latency: float
    default_samplerate: float


@dataclass(frozen=True)
class DeviceConfig:
    """Configuration for an audio recording device"""

    index: int
    name: str
    system_role: str = field(init=False)
    sample_rate: int = field(init=False, default=16000)
    chunk_size: int = field(init=False, default=1024)
    dtype: DTypeLike = field(init=False, default='int16')
    max_input_channels: int = field(init=False, default=9)
    max_output_channels: int = field(init=False, default=4)

    def __str__(self) -> str:
        return f'{self.name} (ID: {self.index}, Channels: {self.max_input_channels})'

    def to_dict(self) -> dict[str, Any]:
        return {
            'index': self.index,
            'name': self.name,
            'system_role': self.system_role,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'dtype': self.dtype,
            'max_input_channels': self.max_input_channels,
            'max_output_channels': self.max_output_channels,
        }

    def __post_init__(self):
        # Can't directly set frozen fields, so we need to use object.__setattr__
        object.__setattr__(self, 'system_role', self._determine_system_role())

    def _determine_system_role(self) -> str:
        """Determine the logical role of this device in the system."""
        return DeviceConfig.get_device_role(self.index)

    @staticmethod
    def get_device_role(index: int) -> str:
        """Map device index to logical system role."""
        roles = {
            1: 'main',  # Main mallet device
            3: 'wired',  # Wired mallet device
        }
        return roles.get(index, f'device_{index}')  # Fallback for unknown indices


@dataclass(frozen=True, slots=True)
class RecordingSession:
    """
    Session metadata and configuration for a recording session.
    Optionally manages SoundDeviceStreamRecorder instances for all devices.
    """

    devices: list[DeviceConfig]
    recorders: list[SoundDeviceStreamRecorder]
    output_dir: Path
    duration_sec: int = -1
    created_at: str = field(init=False, default_factory=lambda: datetime.now().isoformat())
    output_format: str = field(init=False, default='WAV')
    subtype: str = field(init=False, default='PCM_16')

    def to_dict(self) -> dict[str, Any]:
        return {
            'created_at': self.created_at,
            'output_dir': str(self.output_dir),
            'devices': [d.to_dict() for d in self.devices],
            'duration_sec': self.duration_sec,
            'output_format': self.output_format,
            'subtype': self.subtype,
        }

    def __post_init__(self):
        """Convert output_dir to Path if string provided"""
        if isinstance(self.output_dir, str):
            object.__setattr__(self, 'output_dir', Path(self.output_dir))
        self.output_dir.mkdir(exist_ok=True)
