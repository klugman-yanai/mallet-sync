"""
Device detection and configuration utilities for mallet-sync.
"""

from typing import Any

import sounddevice as sd

from rapidfuzz.fuzz import partial_ratio

from mallet_sync.config import DeviceConfig, DeviceInfo, get_logger

logger = get_logger(__name__)

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
        error_message = (
            f'Found only {len(matches)} devices, but {max_devices} are required.'
        )
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
