"""
File and path utilities for mallet-sync.
"""

import json
import logging
import os

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import soundfile as sf

from mallet_sync.config import (
    DEFAULT_FILENAME_TEMPLATE,
    DEFAULT_OUTPUT_DIR,
    DeviceConfig,
    RecordingSession,
    get_logger,
)

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


def setup_output_dir() -> Path:
    """Create the default output directory for recordings.

    Returns:
        Path to the created output directory
    """
    return create_output_directory(DEFAULT_OUTPUT_DIR)


def create_output_directory(
    base_dir: Path | str,
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
    output_dir = base_dir / Path(timestamp)
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


@dataclass
class AudioHardwareInfo:
    """Streamlined information about audio hardware."""

    default_input: int
    mallet_devices: list[str]
    host_api: str | None = None
    query_error: str | None = None


@dataclass
class SystemInfo:
    """Essential system information."""

    platform: str
    python_version: str
    timestamp: str


@dataclass
class EnvironmentInfo:
    """Minimal environment information."""

    cwd: str
    virtual_env: str


@dataclass
class FileSizeStats:
    """Statistics about file sizes for recordings."""

    min: int
    max: int
    total: int
    count: int
    average: float = 0.0

    def update(self, file_size: int) -> None:
        """Update statistics with a new file size."""
        self.min = min(self.min, file_size)
        self.max = max(self.max, file_size)
        self.total += file_size
        self.count += 1
        self.average = self.total / self.count if self.count > 0 else 0.0


@dataclass
class RecordingStats:
    """Statistics about WAV recordings in a directory."""

    total_duration_sec: float = 0.0
    formats: dict[str, int] = field(default_factory=dict)
    sample_rates: dict[int, int] = field(default_factory=dict)
    file_sizes: dict[str, FileSizeStats] = field(default_factory=dict)


@dataclass
class FileStatistics:
    """Essential statistics about recordings."""

    file_count: int
    recordings: dict[str, Any]


@dataclass
class SessionMetadata:
    """Complete session metadata."""

    session: dict[str, Any]  # We use dict here since RecordingSession already has to_dict
    system: SystemInfo
    environment: EnvironmentInfo
    audio: AudioHardwareInfo | dict[str, Any]
    files: FileStatistics


def save_session_safely(output_dir: Path, session: RecordingSession) -> None:
    """Save session metadata with error handling.

    Attempts to save the metadata even if there was an error during recording.
    This ensures we capture as much information as possible about the session.

    Args:
        output_dir: Directory to save metadata in
        session: Session to save metadata for
    """
    try:
        logger.info("Saving comprehensive session metadata")
        save_session_metadata(output_dir, session)
    except Exception:
        logger.exception("Failed to save metadata")


def save_session_metadata(output_dir: Path, session: RecordingSession) -> None:
    """Save comprehensive session metadata to JSON file.

    Includes session details, system info, audio hardware specs,
    and recording statistics for better analysis and debugging.
    """
    import os
    import platform
    import time

    from datetime import datetime

    import sounddevice as sd

    # Mark the session as complete
    session.mark_complete()

    # Get audio hardware info
    audio_info: AudioHardwareInfo | dict[str, Any] = {}
    try:
        try:
            if hasattr(sd.default, 'device') and hasattr(sd.default.device, '__getitem__'):
                input_val = sd.default.device[0]
                if input_val is not None:
                    input_device = int(input_val)
        except (IndexError, TypeError, ValueError, AttributeError):
            pass

        # Get host API list safely
        host_apis = []
        try:
            apis = sd.query_hostapis()
            if apis is not None:
                for api in apis:
                    if isinstance(api, dict) and 'name' in api:
                        host_apis.append(api['name'])
        except Exception:
            logger.exception('Failed to query host APIs')
            pass

        # Create the audio hardware info - streamlined version
        audio_info = AudioHardwareInfo(
            default_input=input_device,
            mallet_devices=[d.name for d in session.devices if 'Mallet' in d.name],
            host_api=host_apis[0] if host_apis else None,
            query_error=None,
        )
    except Exception as e:
        # Fallback to simple dict if hardware query fails
        audio_info = {'query_error': str(e)}

    # Create streamlined metadata structure with proper typing
    metadata = SessionMetadata(
        # Session info
        session=session.to_dict(),
        # Essential system info
        system=SystemInfo(
            platform=platform.platform().split('-')[0],  # Just the OS name
            python_version=platform.python_version(),
            timestamp=datetime.now().isoformat(),
        ),
        # Minimal environment info
        environment=EnvironmentInfo(
            cwd=os.getcwd(),
            virtual_env=os.environ.get('VIRTUAL_ENV', ''),
        ),
        # Focused audio hardware info
        audio=audio_info,
        # Essential file statistics only
        files=FileStatistics(
            file_count=len(list(output_dir.glob('*.wav'))),
            recordings=_get_recording_stats(output_dir),
        ),
    )

    # Convert dataclasses to dictionaries and save to JSON file
    metadata_path = output_dir / 'session_metadata.json'
    with open(metadata_path, 'w') as f:
        # Use dataclasses.asdict or a custom serialization function to handle dataclasses
        json.dump(dataclass_to_dict(metadata), f, indent=2, default=str)

    logger.info(f'Saved session metadata to {metadata_path}')


def dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclasses to dictionaries recursively for JSON serialization.

    Handles nested dataclasses, collections, and special types for proper
    serialization to avoid type errors.
    """
    # Handle None values
    if obj is None:
        return None

    # Handle dataclasses by recursively converting their fields
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for field_name, field_value in obj.__dict__.items():
            result[field_name] = dataclass_to_dict(field_value)
        return result

    # Handle dictionary types
    if isinstance(obj, dict):
        return {str(key): dataclass_to_dict(value) for key, value in obj.items()}

    # Handle sequence types
    if isinstance(obj, (list, tuple, set)):
        return [dataclass_to_dict(item) for item in obj]

    # Handle Path objects
    if isinstance(obj, Path):
        return str(obj)

    # Handle other objects that might not be JSON serializable
    try:
        # This will verify if the object is JSON serializable
        json.dumps(obj)
    except (TypeError, OverflowError):
        # If not serializable, convert to string
        return str(obj)
    else:
        return obj


def _get_directory_size(directory: Path) -> int:
    """Calculate the total size of all files in a directory."""
    total_size = 0
    for file_path in directory.glob('**/*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    return total_size


def _get_recording_stats(output_dir: Path) -> dict[str, Any]:
    """Calculate statistics for recorded files in the output directory.
    Returns a dictionary with essential recording stats."""
    total_duration = 0.0
    formats = {}
    sample_rates = {}
    file_sizes = {}
    for wav_file in output_dir.glob('*.wav'):
        try:
            # Get file info
            info = sf.info(wav_file)
            file_size = wav_file.stat().st_size

            # Track duration
            total_duration += info.duration

            # Count formats
            format_key = f'{info.format} ({info.subtype})'
            formats[format_key] = formats.get(format_key, 0) + 1

            # Count sample rates
            sample_rate = int(info.samplerate)
            sample_rates[sample_rate] = sample_rates.get(sample_rate, 0) + 1

            # Track file sizes by prefix
            prefix = wav_file.name.split('_')[0] if '_' in wav_file.name else 'unknown'
            if prefix not in file_sizes:
                file_sizes[prefix] = {
                    'min': file_size,
                    'max': file_size,
                    'total': file_size,
                    'count': 1,
                    'average': float(file_size),
                }
            else:
                # Update existing stats
                current = file_sizes[prefix]
                current['min'] = min(current['min'], file_size)
                current['max'] = max(current['max'], file_size)
                current['total'] += file_size
                current['count'] += 1
                current['average'] = current['total'] / current['count']

        except Exception as e:
            # Skip files that can't be read, but log the error
            logger.debug(f'Failed to process audio file {wav_file}: {e}')
            continue

    # Return dictionary with all stats
    return {
        'total_duration_sec': total_duration,
        'formats': formats,
        'sample_rates': sample_rates,
        'file_sizes': file_sizes,
    }
