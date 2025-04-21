"""
Device detection and configuration utilities for mallet-sync.
"""

from pathlib import Path
from typing import Any

import sounddevice as sd

from rapidfuzz.fuzz import partial_ratio

from mallet_sync.audio import SoundDeviceStreamRecorder
from mallet_sync.config import DeviceConfig, DeviceInfo, get_logger
from mallet_sync.utils.file_utils import get_recording_path

logger = get_logger(__name__)


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
    """Create recorder instances for all devices for a specific recording context.

    Args:
        devices: List of device configurations
        output_dir: Directory to save recordings
        recording_context: Context name for recordings (e.g., 'ambient', 'zone_1')

    Returns:
        List of recorder instances, one per device
    """
    return [
        SoundDeviceStreamRecorder.from_device_config(
            dev,
            output_file=get_recording_path(dev, output_dir, recording_context),
        )
        for dev in devices
    ]


TARGET_KEYWORDS: tuple[str, ...] = ('kardome', 'mallet', 'kt')


def _get_device_info(device_index: int) -> DeviceInfo | None:
    """
    Retrieve device information for a given index.

    Args:
        device_index: Index of the device to query.

    Returns:
        DeviceInfo dataclass instance if successful; None if retrieval fails.

    Side Effects:
        Logs errors if device info cannot be retrieved.
    """
    try:
        device_info = sd.query_devices(device_index)
        if not isinstance(device_info, dict):
            device_info = dict(device_info)
        return DeviceInfo(**device_info)
    except Exception:
        logger.exception(f'Error querying device {device_index}')
        return None


def find_devices_by_keyword(
    keywords: tuple[str, ...] = TARGET_KEYWORDS,
    min_score: int = 80,
    max_devices: int = 2,
) -> tuple[DeviceConfig, ...]:
    """
    Find up to `max_devices` whose names fuzzily match any keyword.

    Args:
        keywords: Keywords to match device names (case-insensitive).
        min_score: Minimum fuzzy match score (0-100).
        max_devices: Maximum number of devices to find.

    Returns:
        Tuple of DeviceConfig objects (index and name only).

    Raises:
        RuntimeError: If fewer than `max_devices` are found.

    Side Effects:
        Logs debug and error messages.
    """
    devices = sd.query_devices()
    matches: list[DeviceConfig] = []

    for idx in range(len(devices)):
        device_info: DeviceInfo | None = _get_device_info(idx)
        if device_info is None:
            continue

        name: str = device_info.name.lower()
        for keyword in keywords:
            score: float = partial_ratio(name, keyword.lower())
            if score < min_score:
                continue
            logger.debug(
                f'Found device: {name}, index: {idx}, score: {score} with keyword: {keyword}',
            )
            try:
                matches.append(DeviceConfig(idx, device_info.name))
            except TypeError:
                logger.exception(f'Error creating DeviceConfig, device_info: {device_info}')
                raise
            break  # Only one match per device
        if len(matches) >= max_devices:
            break

    if len(matches) < max_devices:
        error_message = f'Found only {len(matches)} devices, but {max_devices} are required.'
        logger.error(error_message)
        raise RuntimeError(error_message)

    return tuple(matches)


def get_mallet_devices(*, display: bool = True) -> tuple[DeviceConfig, ...]:
    try:
        device_configs = find_devices_by_keyword()
    except RuntimeError as e:
        logger.panel(str(e), title='Device Detection Failed', border_style='red')
        return tuple()

    # Display detected devices
    if display:
        device_panel = '\n'.join(f'Index: {cfg.index} | Name: {cfg.name}' for cfg in device_configs)
        logger.panel(device_panel, title='Detected Mallet Devices', border_style='green')
    logger.debug('Detected device configs: %s', device_configs)
    return device_configs
