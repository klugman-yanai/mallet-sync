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
    MALLET_SAMPLE_RATE,
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
    """Retrieve simplified device information with comprehensive error handling.

    Args:
        device_index: Index of the device to query

    Returns:
        DeviceInfo object with device details or None if retrieval failed
    """
    try:
        idx = int(device_index)
        device_info_raw = sd.query_devices(idx)
        if not isinstance(device_info_raw, dict):
            device_info_raw = dict(device_info_raw)

        name = str(device_info_raw.get('name', 'Unknown'))
        max_inputs = int(device_info_raw.get('max_input_channels', 0))

        # Log detailed device information to help with debugging
        logger.debug(
            f'Device {idx} details: {name}, '
            f'inputs={max_inputs}, '
            f'hostapi={device_info_raw.get("hostapi", -1)}, '
            f'default_rate={device_info_raw.get("default_samplerate", 0.0)}',
        )

        # Validate device has expected capabilities
        if max_inputs == 0:
            logger.warning(f'Device {idx} ({name}) has no input channels')

        return DeviceInfo(
            name=name,
            index=idx,
            hostapi=int(device_info_raw.get('hostapi', -1)),
            max_input_channels=max_inputs,
            max_output_channels=int(device_info_raw.get('max_output_channels', 0)),
            default_samplerate=float(device_info_raw.get('default_samplerate', 0.0)),
            supported_samplerates=device_info_raw.get('supported_samplerates', []),
        )
    except Exception:
        logger.exception(f'Error querying device {device_index}')
        return None


def is_mallet_device(device: DeviceInfo) -> bool:
    """Determine if a device is a Mallet device based on name matching and channel requirements.

    A device is considered a Mallet device if:
    1. It has at least MALLET_CHANNELS input channels
    2. Its name matches one of the MALLET_KEYWORDS with a fuzzy match score >= 80

    Args:
        device: Device information to check

    Returns:
        True if device meets Mallet criteria, False otherwise
    """
    # First check channel requirements
    if device.max_input_channels < MALLET_CHANNELS:
        logger.debug(
            f'Device {device.index} ({device.name}) rejected: insufficient channels '
            f'(has {device.max_input_channels}, need {MALLET_CHANNELS})',
        )
        return False

    # Check name match using fuzzy matching
    dev_name_lower = device.name.lower()

    # Calculate match scores for logging
    match_scores = [(keyword, partial_ratio(dev_name_lower, keyword)) for keyword in MALLET_KEYWORDS]
    best_match = max(match_scores, key=lambda x: x[1]) if match_scores else ('none', 0)

    # A device is a match if any keyword has a match score >= 80
    is_match = any(score >= 80 for _, score in match_scores)

    if is_match:
        logger.debug(
            f'Device {device.index} ({device.name}) identified as Mallet device '
            f"(best match: '{best_match[0]}' with score {best_match[1]})",
        )
    else:
        logger.debug(
            f"Device {device.index} ({device.name}) rejected: name doesn't match "
            f"(best match: '{best_match[0]}' with score {best_match[1]})",
        )

    return is_match


def find_mallet_devices() -> list[tuple[DeviceInfo, str]]:
    """
    Finds and labels Mallet input devices based on discovery order with robust error handling.

    This function attempts to discover all connected Mallet devices and assign appropriate roles.
    First device found becomes 'main', second 'wired', third 'hmtc', etc. based on the MALLET_ROLES
    configuration. If insufficient devices are found, returns an empty list.

    Returns:
        List of (device_info, role) tuples for discovered Mallet devices, or empty list if
        requirements cannot be met
    """
    try:
        # Get all audio devices
        devices_raw = sd.query_devices()
        total_devices = len(devices_raw)
        logger.info(f'Found {total_devices} total audio devices. Scanning for Mallet devices...')

        # Log detailed information about all detected audio devices
        for i, dev in enumerate(devices_raw):
            # Extract device properties safely handling different object types
            try:
                # Try different approaches to safely extract device properties
                if (
                    hasattr(dev, 'name')
                    and hasattr(dev, 'max_input_channels')
                    and hasattr(dev, 'max_output_channels')
                ):
                    # Object with attributes
                    name = str(getattr(dev, 'name', 'Unknown'))
                    max_inputs = int(getattr(dev, 'max_input_channels', 0))
                    max_outputs = int(getattr(dev, 'max_output_channels', 0))
                elif isinstance(dev, dict):
                    # Standard dictionary
                    name = str(dev.get('name', 'Unknown'))
                    max_inputs = int(dev.get('max_input_channels', 0))
                    max_outputs = int(dev.get('max_output_channels', 0))

                logger.debug(f'Device {i}: {name} (inputs={max_inputs}, outputs={max_outputs})')
            except (TypeError, KeyError, AttributeError):
                logger.warning(f'Could not extract properties from device {i}')
    except Exception as e:
        logger.critical(f'Failed to query audio devices: {e}')
        return []

    # Collect device information with error handling for each device
    device_infos = []
    for idx in range(total_devices):
        try:
            dev_info = _get_device_info(idx)
            if dev_info is not None:
                device_infos.append(dev_info)
        except Exception:
            logger.exception(f'Error processing device {idx}')

    # Find devices that match Mallet criteria
    all_mallet_matches = [dev for dev in device_infos if is_mallet_device(dev)]

    logger.info(
        f'Found {len(all_mallet_matches)} potential Mallet devices out of {len(device_infos)} total devices',
    )

    # Log details of all matched devices
    for i, dev in enumerate(all_mallet_matches):
        logger.debug(f'Mallet device {i + 1}: [{dev.index}] {dev.name} (channels={dev.max_input_channels})')

    if len(all_mallet_matches) < NUM_MALLETS:
        logger.error(
            f'Insufficient Mallet devices found. Have {len(all_mallet_matches)}, need {NUM_MALLETS}. '
            f'Check device connections and ensure they match the expected keywords: {MALLET_KEYWORDS}',
        )
        return []

    # Sort by device index for consistent selection across runs
    all_mallet_matches.sort(key=lambda dev: dev.index)

    # Assign roles based on discovery order using the role names from config
    result: list[tuple[DeviceInfo, str]] = []
    for i, dev in enumerate(all_mallet_matches[:NUM_MALLETS]):
        # Use the role name from config or fallback to a numbered role if we have more mallets than defined roles
        role = MALLET_ROLES[i] if i < len(MALLET_ROLES) else f'mallet_{i + 1}'
        result.append((dev, role))
        logger.info(f"Selected '{role}' Mallet device: [{dev.index}] {dev.name}")

    # Validate selected devices meet minimum requirements for the configuration
    sample_rate_issues = []
    for dev, role in result:
        # Check sample rate compatibility
        if MALLET_SAMPLE_RATE not in dev.supported_samplerates:
            sample_rate_issues.append(f"{dev.name} ({role}) doesn't support {MALLET_SAMPLE_RATE}Hz")

    # Warn about any potential issues
    if sample_rate_issues:
        logger.warning(f'Sample rate compatibility issues detected: {", ".join(sample_rate_issues)}')

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


def create_session_dir(base_dir: Path) -> Path:
    """Creates a timestamp-based session directory under the base output directory."""
    now = datetime.now()
    date_part = now.strftime('%y%m%d')  # Year-Month-Day
    time_part = now.strftime('%H%M')  # 24-hour-Minute
    folder_name = f'{date_part}_{time_part}'
    session_dir = base_dir / folder_name

    if session_dir.exists():
        unique_id = now.strftime('%S')
        session_dir = base_dir / f'{folder_name}_{unique_id}'

    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f'Created session directory: {session_dir.resolve()}')
    except OSError:
        logger.critical(f'Failed to create session directory {session_dir}')
        raise

    return session_dir


def create_role_dirs(session_dir: Path, roles: list[str]) -> dict[str, Path]:
    """Creates subdirectories for each role under the session directory."""
    role_dirs = {}
    for role in roles:
        role_dir = session_dir / role
        role_dir.mkdir(exist_ok=True)
        logger.debug(f'Created role directory: {role_dir.resolve()}')
        role_dirs[role] = role_dir
    return role_dirs


def validate_device_availability(devices: list[tuple[DeviceInfo, str]]) -> bool:
    """Validate that devices are still available and ready for recording.

    This function re-checks each device to confirm it's still connected and operational.
    Useful for validating devices right before recording begins, as USB devices can
    disconnect/reconnect between detection and use.

    Args:
        devices: List of (device_info, role) tuples to validate

    Returns:
        True if all devices are available and ready, False otherwise
    """
    if not devices:
        logger.error('No devices to validate')
        return False

    logger.debug(f'Validating availability of {len(devices)} devices')
    all_valid = True

    for device, role in devices:
        try:
            # Re-fetch the current device info
            current_info = _get_device_info(device.index)

            # Check if device is still present and has expected properties
            if current_info is None:
                logger.error(f'Device {device.index} ({role}: {device.name}) is no longer available')
                all_valid = False
                continue

            # Verify name hasn't changed (indication device was replaced)
            if current_info.name != device.name:
                logger.warning(
                    f"Device {device.index} changed name from '{device.name}' to '{current_info.name}'. "
                    f'This may indicate a USB reconnection event.',
                )

            # Verify channel count hasn't changed
            if current_info.max_input_channels < MALLET_CHANNELS:
                logger.error(
                    f'Device {device.index} ({role}: {device.name}) now has {current_info.max_input_channels} '
                    f'input channels, which is less than the required {MALLET_CHANNELS}',
                )
                all_valid = False
                continue

            logger.debug(f'Device {device.index} ({role}: {device.name}) is valid and ready')

        except Exception:
            logger.exception(f'Error validating device {device.index} ({role}: {device.name})')
            all_valid = False

    if all_valid:
        logger.info('All devices validated successfully')
    else:
        logger.error('Some devices failed validation')

    return all_valid


def generate_output_path(role_dirs: dict[str, Path], role: str, context: str) -> Path:
    """Generates the full path for an output recording file, organized in the role subdirectory."""
    output_path = role_dirs[role] / FILENAME_TEMPLATE.format(role=role, context=context)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f'Generated output path: {output_path}')
    return output_path


if __name__ == '__main__':
    mallet1, mallet2, mallet3 = find_mallet_devices()

    from pprint import pprint

    pprint(mallet1)
    pprint(mallet2)
    pprint(mallet3)
