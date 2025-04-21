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


def build_session(
    devices: list[DeviceConfig],
    recorders: list[SoundDeviceStreamRecorder],
    output_dir: Path,
) -> RecordingSession:
    """Build a recording session from devices, recorders, and output directory.

    Args:
        devices: List of device configurations
        recorders: List of recorder instances
        output_dir: Directory to save recordings

    Returns:
        Initialized RecordingSession instance
    """
    return RecordingSession(
        devices=devices,
        recorders=recorders,
        output_dir=output_dir,
    )


@dataclass(frozen=True, slots=True)
class RecordingSession:
    """
    Session metadata and configuration for a recording session.
    Optionally manages SoundDeviceStreamRecorder instances for all devices.

    Tracks comprehensive information about the recording environment,
    equipment, and technical parameters for better debugging and analysis.
    """

    # Core configuration (required at init)
    devices: list[DeviceConfig]
    recorders: list[SoundDeviceStreamRecorder]
    output_dir: Path

    # Recording parameters
    duration_sec: float = 0.0
    input_files: list[str] = field(default_factory=list)
    session_name: str = field(default_factory=lambda: f'session_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

    # Technical metadata (automatic)
    created_at: str = field(init=False, default_factory=lambda: datetime.now().isoformat())
    completed_at: str = field(init=False, default='')
    output_format: str = field(init=False, default='WAV')
    subtype: str = field(init=False, default='PCM_16')
    channel_count: dict[str, int] = field(init=False, default_factory=dict)
    sample_rate: int = field(init=False, default=16000)

    # Status tracking
    recordings_completed: list[str] = field(default_factory=list)
    status: str = field(init=False, default='initialized')
    errors: list[str] = field(default_factory=list)

    def mark_complete(self) -> None:
        """Mark the session as complete and record the completion time.

        Also updates the total session duration based on start and end times.
        """
        # Using object.__setattr__ since the dataclass is frozen
        completion_time = datetime.now()
        object.__setattr__(self, 'completed_at', completion_time.isoformat())
        object.__setattr__(self, 'status', 'completed')

        # Calculate total duration if not already set
        if self.duration_sec <= 0.0:
            try:
                start_time = datetime.fromisoformat(self.created_at)
                duration = (completion_time - start_time).total_seconds()
                object.__setattr__(self, 'duration_sec', duration)
            except (ValueError, TypeError):
                # If there's any issue parsing dates, don't update duration
                pass

    def add_recording(self, context_name: str, duration: float = 0.0) -> None:
        """Add a recording to the list of completed recordings.

        Args:
            context_name: The name/identifier of the recording context
            duration: Duration of the recording in seconds
        """
        # Update recordings list
        recordings = self.recordings_completed.copy()
        recordings.append(context_name)
        object.__setattr__(self, 'recordings_completed', recordings)

        # Update total duration
        if duration > 0.0:
            new_duration = self.duration_sec + duration
            object.__setattr__(self, 'duration_sec', new_duration)

    def add_error(self, error_message: str) -> None:
        """Add an error message to the session."""
        error_list = self.errors.copy()
        error_list.append(error_message)
        object.__setattr__(self, 'status', 'error')
        object.__setattr__(self, 'errors', error_list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the session to a dictionary for serialization."""
        return {
            # Core information
            'session_name': self.session_name,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'status': self.status,
            # Configuration
            'output_dir': str(self.output_dir),
            'devices': [d.to_dict() for d in self.devices],
            'device_count': len(self.devices),
            # Recording details
            'duration_sec': self.duration_sec,
            'recordings_completed': self.recordings_completed,
            'input_files': self.input_files,
            # Technical details
            'output_format': self.output_format,
            'subtype': self.subtype,
            'channel_count': self.channel_count,
            'sample_rate': self.sample_rate,
            # Status information
            'errors': self.errors,
        }

    def __post_init__(self):
        """Initialize derived fields and ensure output directory exists."""
        # Convert output_dir to Path if string provided
        if isinstance(self.output_dir, str):
            object.__setattr__(self, 'output_dir', Path(self.output_dir))

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize channel count mapping
        channel_counts = {}
        for device in self.devices:
            channel_counts[device.name] = device.max_input_channels
        object.__setattr__(self, 'channel_count', channel_counts)

        # Set sample rate from first device (assuming all use same rate)
        if self.devices:
            object.__setattr__(self, 'sample_rate', self.devices[0].sample_rate)
