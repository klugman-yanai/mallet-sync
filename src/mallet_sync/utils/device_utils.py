# utils/device_utils.py
import sounddevice as sd

from rapidfuzz.fuzz import partial_ratio

from mallet_sync.config import (
    MAIN_MALLET_INDEX,
    MALLET_CHANNELS,
    MALLET_KEYWORDS,
    WIRED_MALLET_INDEX,
    DeviceInfo,
    get_logger,
)

logger = get_logger(__name__)


def _get_device_info(device_index: int) -> DeviceInfo | None:
    """Retrieve simplified device information."""
    # (Implementation remains the same as before)
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


def find_mallet_devices() -> list[tuple[DeviceInfo, str]]:
    """
    Selects exactly two Mallet input devices, prioritizing indices
    1 ('main') and 3 ('wired').
    """
    # (Implementation remains the same as before)
    all_mallet_matches: list[DeviceInfo] = []
    selected_mallets_with_roles: list[tuple[DeviceInfo, str]] = []
    try:
        devices_raw = sd.query_devices()
    except Exception as e:
        logger.critical(f'Failed to query audio devices: {e}')
        return []
    logger.info('Scanning for Mallet input devices...')
    for idx in range(len(devices_raw)):
        dev_info = _get_device_info(idx)
        if not dev_info:
            continue
        if dev_info.max_input_channels >= MALLET_CHANNELS:
            dev_name_lower = dev_info.name.lower()
            for keyword in MALLET_KEYWORDS:
                score = partial_ratio(dev_name_lower, keyword)
                if score >= 80:
                    logger.debug(
                        f"Potential Mallet match: [{dev_info.index}] {dev_info.name} (score {score} with '{keyword}')",
                    )
                    all_mallet_matches.append(dev_info)
                    break
    logger.info(f'Found {len(all_mallet_matches)} potential Mallet devices. Selecting targets...')
    main_mallet: DeviceInfo | None = None
    wired_mallet: DeviceInfo | None = None
    remaining_matches = []
    for dev in all_mallet_matches:
        if dev.index == MAIN_MALLET_INDEX:
            main_mallet = dev
            logger.info(f"Selected 'main' Mallet: [{dev.index}] {dev.name}")
        elif dev.index == WIRED_MALLET_INDEX:
            wired_mallet = dev
            logger.info(f"Selected 'wired' Mallet: [{dev.index}] {dev.name}")
        else:
            remaining_matches.append(dev)
    selected_count = (1 if main_mallet else 0) + (1 if wired_mallet else 0)
    needed = 2 - selected_count
    if needed > 0 and remaining_matches:
        remaining_matches.sort(key=lambda d: d.index)
        take_from_remaining = min(needed, len(remaining_matches))
        logger.info(
            f'Specific indices not found/filled. Taking {take_from_remaining} from remaining {len(remaining_matches)} matches.',
        )
        for i in range(take_from_remaining):
            dev = remaining_matches[i]
            if main_mallet is None:
                main_mallet = dev
                logger.info(f"Selected 'main' Mallet (fallback): [{dev.index}] {dev.name}")
            elif wired_mallet is None:
                wired_mallet = dev
                logger.info(f"Selected 'wired' Mallet (fallback): [{dev.index}] {dev.name}")
    if main_mallet:
        selected_mallets_with_roles.append((main_mallet, 'main'))
    if wired_mallet:
        selected_mallets_with_roles.append((wired_mallet, 'wired'))
    if len(selected_mallets_with_roles) != 2:
        logger.error(f'Expected 2 Mallet input devices, but selected {len(selected_mallets_with_roles)}.')
        return []
    logger.info(f'Successfully selected {len(selected_mallets_with_roles)} target Mallet devices.')
    return selected_mallets_with_roles
