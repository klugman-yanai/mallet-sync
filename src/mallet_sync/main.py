"""
mallet_sync

Assumptions:
- Python 3.13+ is required.
- Custom logger is used.
- Audio I/O will use sounddevice (not pyaudio).

"""

import sys

from collections.abc import Callable
from datetime import datetime
from os import getcwd, path
from pathlib import Path

import soundfile as sf

from mallet_sync.audio import SoundDeviceAudioPlayer, SoundDeviceStreamRecorder
from mallet_sync.config import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DeviceConfig,
    RecordingSession,
    get_logger,
)
from mallet_sync.utils.audio_utils import record_ambient_noise, record_tests, record_zones
from mallet_sync.utils.device_utils import get_mallet_devices
from mallet_sync.utils.file_utils import (
    create_output_directory,
    get_recording_path,
    save_session_metadata,
)

logger = get_logger(__name__)


def main() -> None:
    output_dir = setup_output_dir()
    devices = detect_devices()
    recorders = create_recorders(devices, output_dir)
    session = build_session(devices, recorders, output_dir)

    # Record ambient noise
    record_ambient_noise(session)

    # Record zone calibrations
    record_zones(session)

    # Record test files
    record_tests(session)

    # Save session metadata
    save_session_metadata(output_dir, session)


# ---- Application Workflow Functions ----------------------------------------


def setup_output_dir() -> Path:
    return create_output_directory(DEFAULT_OUTPUT_DIR)


def detect_devices() -> list[DeviceConfig]:
    """Detect mallet devices and return a list of device configs.

    Returns the first two devices found, prioritizing main and wired roles.
    """
    mallet_1, mallet_2 = get_mallet_devices(display=True)
    return [mallet_1, mallet_2]


def create_recorders(
    devices: list[DeviceConfig],
    output_dir: Path,
    recording_context: str = 'ambient',
) -> list[SoundDeviceStreamRecorder]:

    return [
        SoundDeviceStreamRecorder.from_device_config(
            dev,
            output_file=get_recording_path(dev, output_dir, recording_context),
        )
        for dev in devices
    ]


def build_session(
    devices: list[DeviceConfig],
    recorders: list[SoundDeviceStreamRecorder],
    output_dir: Path,
) -> RecordingSession:
    return RecordingSession(
        devices=devices,
        recorders=recorders,
        output_dir=output_dir,
    )


# ---- Main Application Workflow ---------------------------------------------


if __name__ == '__main__':
    main()
