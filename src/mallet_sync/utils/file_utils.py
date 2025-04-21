"""
File and path utilities for mallet-sync.
"""

import json
import os

from datetime import datetime
from pathlib import Path

from mallet_sync.config import DEFAULT_FILENAME_TEMPLATE, DeviceConfig, RecordingSession, get_logger

logger = get_logger(__name__)


def get_recording_path(
    device: DeviceConfig,
    output_dir: Path,
    context: str,
    template: str = DEFAULT_FILENAME_TEMPLATE,
) -> Path:
    """Generate a recording path given a device, directory, and context."""
    filename = template.format(
        role=device.system_role,
        context=context,
    )
    return output_dir / filename


def create_output_directory(
    base_dir: Path,
    timestamp_format: str = '%Y%m%d_%H%M%S',
) -> Path:
    """
    Create timestamped output directory for recordings.

    Args:
        base_dir: Base directory for recordings
        timestamp_format: Format for timestamp in directory name

    Returns:
        Path to created directory
    """
    # Create parent directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Create timestamped subdirectory
    timestamp = datetime.now().strftime(timestamp_format)
    output_dir = base_dir / timestamp
    output_dir.mkdir(exist_ok=True)

    logger.info(f'Created output directory: {output_dir}')
    return output_dir


def build_output_filename(
    device_index: int,
    device_name: str,
    prefix: str = 'recording',
    time_format: str = '%Y%m%d_%H%M%S',
) -> str:
    """Build a standardized output filename for recordings."""
    ts = datetime.now().strftime(time_format)
    # Remove invalid chars from device name
    safe_name = ''.join(c if c.isalnum() else '_' for c in device_name)
    return f'{prefix}_{device_index}_{safe_name}_{ts}.wav'


def save_session_metadata(output_dir: Path, session: RecordingSession) -> None:
    """Save session metadata to JSON file."""
    import platform

    metadata = {
        'session': session.to_dict(),
        'system': {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
        },
    }

    metadata_path = output_dir / 'session_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f'Saved session metadata to {metadata_path}')
