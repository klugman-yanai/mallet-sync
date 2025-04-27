# utils.py
"""Consolidated utilities for file operations and device detection."""

from datetime import datetime
from pathlib import Path

import sounddevice as sd

from rapidfuzz.fuzz import partial_ratio

# Use absolute imports
from mallet_sync.config import (
    FILENAME_TEMPLATE,
    MALLET_CHANNELS,
    MALLET_KEYWORDS,
    MALLET_ROLES,
    NUM_MALLETS,
    DeviceInfo,
    get_logger,
)

logger = get_logger(__name__)


# === Device detection and selection utilities ===


def check_maya44_output_device() -> bool:
    """
    Check if the default output device is a MAYA44 device.

    Returns:
        True if MAYA44 is the default output, False otherwise
    """
    try:
        # Get the default output device
        default_output = sd.query_devices(kind='output')

        if not isinstance(default_output, dict):
            default_output = dict(default_output)

        device_name = default_output.get('name', '').lower()
        logger.debug(f'Default output device: {device_name}')

        # Check if it's a MAYA44 device
        is_maya44 = 'maya44' in device_name

        if not is_maya44:
            logger.error(
                f'Default output device is not a MAYA44 device: {default_output.get("name")}. '
                'Please set MAYA44 as your default playback device in the Sound control panel.',
            )

    except Exception:
        logger.exception('Error checking output device')
        return False
    else:
        return is_maya44


def _get_device_info(device_index: int) -> DeviceInfo | None:
    """Retrieve simplified device information."""
    try:
        idx = int(device_index)
        device_info_raw = sd.query_devices(idx)
        if not isinstance(device_info_raw, dict):
            device_info_raw = dict(device_info_raw)

        return DeviceInfo(
            name=str(device_info_raw.get('name', 'Unknown')),
            index=idx,
            hostapi=int(device_info_raw.get('hostapi', -1)),
            max_input_channels=int(device_info_raw.get('max_input_channels', 0)),
            max_output_channels=int(device_info_raw.get('max_output_channels', 0)),
            default_samplerate=float(device_info_raw.get('default_samplerate', 0.0)),
            supported_samplerates=device_info_raw.get('supported_samplerates', []),
        )
    except Exception:
        logger.exception(f'Error querying device {device_index}')
        return None


def is_mallet_device(device: DeviceInfo) -> bool:
    """Determine if a device is a Mallet device based on name matching."""
    if device.max_input_channels < MALLET_CHANNELS:
        return False

    dev_name_lower = device.name.lower()
    return any(partial_ratio(dev_name_lower, keyword) >= 80 for keyword in MALLET_KEYWORDS)


def find_mallet_devices() -> list[tuple[DeviceInfo, str]]:
    """
    Finds and labels Mallet input devices based on discovery order.
    First device found becomes 'main', second 'wired', third 'hmtc', etc.
    based on the MALLET_ROLES configuration.
    """
    try:
        devices_raw = sd.query_devices()
    except Exception as e:
        logger.critical(f'Failed to query audio devices: {e}')
        return []

    logger.debug('Scanning for Mallet input devices...')

    # Get all potential Mallet devices
    all_mallet_matches = [_get_device_info(idx) for idx in range(len(devices_raw))]
    all_mallet_matches = [dev for dev in all_mallet_matches if dev and is_mallet_device(dev)]

    logger.debug(f'Found {len(all_mallet_matches)} potential Mallet devices. Selecting targets...')

    if len(all_mallet_matches) < NUM_MALLETS:
        logger.error(f'Found only {len(all_mallet_matches)} Mallet devices, need at least {NUM_MALLETS}.')
        return []

    # Sort by device index for consistent selection
    all_mallet_matches.sort(key=lambda dev: dev.index)

    # Assign roles based on discovery order using the role names from config
    result: list[tuple[DeviceInfo, str]] = []
    for i, dev in enumerate(all_mallet_matches[:NUM_MALLETS]):
        # Use the role name from config or fallback to a numbered role if we have more mallets than defined roles
        role = MALLET_ROLES[i] if i < len(MALLET_ROLES) else f'mallet_{i + 1}'
        result.append((dev, role))
        logger.debug(f"Selected '{role}' Mallet: [{dev.index}] {dev.name}")

    return result


# === File operations utilities ===


def scan_audio_files(input_dir: Path, *, exclude_calibration: bool = False) -> list[Path]:
    """
    Scans subdirectories for WAV files to process in a specific order.

    Args:
        input_dir: Base directory containing the subdirectories
        exclude_calibration: If True, excludes calibration directories (ambient, kardome_*)
                            If False, includes all directories

    Order of processing (when all included):
    ambient -> kardome_afe -> kardome_bio -> kardome_kws -> test_audio
    """
    files_to_process: list[Path] = []

    # All subdirectories in processing order
    calibration_dirs = ['ambient', 'kardome_afe', 'kardome_bio', 'kardome_kws']
    test_dirs = ['test_audio']

    # Select which directories to process based on the flag
    subdirs = test_dirs if exclude_calibration else calibration_dirs + test_dirs

    logger.debug(f"Scanning for WAV files in subdirectories of '{input_dir.resolve()}'...")
    if exclude_calibration:
        logger.info('Processing test files only (calibration files excluded)')

    for subdir in subdirs:
        subdir_path = input_dir / subdir
        if not subdir_path.exists():
            logger.warning(f"Subdirectory '{subdir}' does not exist. Skipping.")
            continue

        try:
            found = sorted(list(subdir_path.glob('*.wav')))
            files_to_process.extend(found)
            logger.debug(f"Found {len(found)} WAV files in '{subdir}'")
        except Exception:
            logger.exception(f"Error scanning subdirectory '{subdir}'")

    if not files_to_process:
        logger.warning(f'No WAV files found in any of the subdirectories: {subdirs}')
    else:
        logger.info(f'Ready to process {len(files_to_process)} WAV files')

    return files_to_process


def create_output_dir(base_dir: Path) -> Path:
    """Creates a timestamp directory with date and 24-hour time."""
    now = datetime.now()

    # Create a format with year-month-day and 24-hour time: "250424_1030"
    date_part = now.strftime('%y%m%d')  # Year-Month-Day
    time_part = now.strftime('%H%M')  # 24-hour-Minute

    # Create a folder name that's easy to locate by time
    folder_name = f'{date_part}_{time_part}'

    # Add seconds for uniqueness if needed
    unique_id = now.strftime('%S')

    # Create directory path
    output_dir = base_dir / folder_name

    # Handle duplicate directory names by adding the seconds as a suffix if needed
    if output_dir.exists():
        output_dir = base_dir / f'{folder_name}_{unique_id}'

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Created output directory: {output_dir.resolve()}')
    except OSError:
        logger.critical(f'Failed to create output directory {output_dir}')
        raise

    return output_dir


def generate_output_path(output_dir: Path, role: str, context: str) -> Path:
    """Generates the full path for an output recording file."""
    filename = FILENAME_TEMPLATE.format(role=role, context=context)
    return output_dir / filename


if __name__ == '__main__':

    mallet1, mallet2, mallet3 = find_mallet_devices()

    from pprint import pprint

    pprint(mallet1)
    pprint(mallet2)
    pprint(mallet3)
