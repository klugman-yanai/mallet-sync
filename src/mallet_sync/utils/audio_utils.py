"""
Audio processing and recording utilities for mallet-sync.
"""

import re
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any

import sounddevice as sd
import soundfile as sf

from rich.progress import TaskID

from mallet_sync.audio import SoundDeviceAudioPlayer, SoundDeviceStreamRecorder
from mallet_sync.config import DEFAULT_INPUT_DIR, DeviceConfig, RecordingSession, get_logger
from mallet_sync.utils.file_utils import get_recording_path

logger = get_logger(__name__)


def _find_ambient_files(audio_dir: Path) -> dict[str, Path]:
    """Find ambient noise files in the audio directory."""
    result: dict[str, Path] = {}

    ambient_files = list(audio_dir.glob('ambient*.wav'))
    for index, file_path in enumerate(sorted(audio_dir.glob('ambient*.wav'))):
        # If there's just one file, use 'ambient' as the key
        # Otherwise, number them: ambient_1, ambient_2, etc.
        key = 'ambient' if index == 0 and len(ambient_files) == 1 else f'ambient_{index + 1}'

        result[key] = file_path
        logger.debug(f'Found ambient file: {key} -> {file_path}')

    return result


def _find_matching_files(
    audio_dir: Path,
    pattern: str,
    regex: str,
    context_prefix: str,
) -> dict[str, Path]:
    """
    Find all files matching a pattern and extract context keys using regex.

    Args:
        audio_dir: Directory to search in
        pattern: Glob pattern to match files
        regex: Regular expression to extract ID from filename
        context_prefix: Prefix for context keys (e.g., 'zone_', 'test_')

    Returns:
        Dictionary mapping context keys to file paths
    """

    result: dict[str, Path] = {}

    for file_path in sorted(audio_dir.glob(pattern)):
        match = re.search(regex, file_path.stem)
        if not match:
            continue

        identifier = match.group(1)
        context_key = f'{context_prefix}{identifier}'

        # We'll include all files, adding a numeric suffix for duplicates
        if context_key in result:
            # Find a unique key by adding a numeric suffix
            suffix = 1
            while f'{context_key}_{suffix}' in result:
                suffix += 1
            unique_key = f'{context_key}_{suffix}'

            logger.debug(f'Multiple files for {context_key}, using key {unique_key} for {file_path}')
            result[unique_key] = file_path
        else:
            result[context_key] = file_path
            logger.debug(f'Found {context_key} file: {file_path}')

    return result


def scan_input_file_directory() -> dict[str, Path]:
    """
    Scan the input file directory and return a dictionary of context names to wav paths.
    Handles flexible naming patterns and includes all matching files.
    """
    input_file_dir = DEFAULT_INPUT_DIR
    if not input_file_dir.exists():
        logger.error(f'Input file directory not found: {input_file_dir}')
        return {}

    # Find different types of files and merge the results
    contexts = {}

    # Find all ambient files
    contexts.update(_find_ambient_files(input_file_dir))

    # Find all zone files with no limit (e.g., zone_1_hello.wav → zone_1)
    zone_files = _find_matching_files(
        input_file_dir,
        'zone_*.wav',
        r'zone_(\d+)',
        'zone_',
    )
    contexts.update(zone_files)

    # Find all test files with no limit (e.g., test_2_blahblah.wav → test_2)
    test_files = _find_matching_files(
        input_file_dir,
        'test_*.wav',
        r'test_([^_]+)',
        'test_',
    )
    contexts.update(test_files)

    return contexts


def create_all_recorders(
    devices: list[DeviceConfig],
    output_dir: Path,
) -> dict[str, list[SoundDeviceStreamRecorder]]:
    """
    Create recorders for all devices and all recording contexts.
    Returns a dictionary mapping context names to lists of recorders.
    """
    contexts = scan_input_file_directory()
    if not contexts:
        return {}

    recorders_by_context = {}

    for context in contexts:
        recorders = [
            SoundDeviceStreamRecorder.from_device_config(
                dev,
                output_file=get_recording_path(dev, output_dir, context),
            )
            for dev in devices
        ]
        recorders_by_context[context] = recorders

    return recorders_by_context


def record_with_progress(
    session: RecordingSession,
    wav_path: str | Path | None = None,
    duration: float = 7.0,
    description: str = 'Recording',
    context_name: str = '',
) -> None:
    """
    Record audio with progress tracking.

    Args:
        session: The recording session with devices and recorders
        wav_path: Optional path to WAV file to play while recording
        duration: Duration to record if no WAV file is provided
        description: User-friendly description for the progress bar
        context_name: Recording context name for logging
    """
    if wav_path is not None:
        # Record while playing a WAV file
        info = sf.info(str(wav_path))
        actual_duration: float = info.duration
        total_steps = 100  # Use fixed 100 steps for percentage-based progress
        wav_filename = Path(wav_path).name
        logger.info(f'Recording {context_name} with {wav_filename} (duration: {actual_duration:.2f}s)')

        # Create progress tracking with StandardLogger using file_progress mode
        with logger.progress(description=description, file_progress=True) as progress:
            # Create a task with fixed total steps
            task_id = progress.add_task(description, total=total_steps)

            # Start all recorders
            for recorder in session.recorders:
                recorder.start()

            # Play WAV with non-blocking method so we can update progress
            player = SoundDeviceAudioPlayer(str(wav_path))
            player.start()

            # Update progress while playing
            start_time = time.time()
            last_percent = -1

            # Keep updating until playback finishes
            while player._is_playing:
                # Calculate progress percentage (0-100)
                elapsed = time.time() - start_time
                percent = min(99, int(99 * elapsed / actual_duration))

                # Only advance if percentage has increased
                if percent > last_percent:
                    # How many steps to advance
                    steps_to_advance = percent - last_percent
                    if steps_to_advance > 0:
                        progress.update(TaskID(task_id), advance=steps_to_advance)
                        last_percent = percent

                time.sleep(0.05)

            # Final update to ensure 100% completion
            remaining_steps = total_steps - last_percent - 1
            if remaining_steps > 0:
                progress.update(TaskID(task_id), advance=remaining_steps)

            # Stop all recorders when playback finishes
            for recorder in session.recorders:
                recorder.stop()
    else:
        # Record silence for the specified duration
        logger.info(f'Recording {context_name} for {duration:.2f}s')

        # Create progress tracking with StandardLogger
        with logger.progress(description=description, file_progress=True) as progress:
            # Use 100 steps for percentage-based progress
            total_steps = 100
            task_id = progress.add_task(description, total=total_steps)

            # Start all recorders
            for recorder in session.recorders:
                recorder.start()

            # Update progress continuously during recording
            start_time = time.time()
            last_percent = -1

            # Keep updating until duration completes
            while time.time() - start_time < duration:
                # Calculate progress percentage (0-100)
                elapsed = time.time() - start_time
                percent = min(99, int(99 * elapsed / duration))

                # Only advance if percentage has increased
                if percent > last_percent:
                    # How many steps to advance
                    steps_to_advance = percent - last_percent
                    if steps_to_advance > 0:
                        progress.update(TaskID(task_id), advance=steps_to_advance)
                        last_percent = percent

                time.sleep(0.05)

            # Final update to ensure 100% completion
            remaining_steps = total_steps - last_percent - 1
            if remaining_steps > 0:
                progress.update(TaskID(task_id), advance=remaining_steps)

            # Stop all recorders
            for recorder in session.recorders:
                recorder.stop()

    logger.info(f'Finished recording {context_name or "audio"}.')


def record_ambient_noise(
    session: RecordingSession,
    wav_path: str | Path | None = None,
    duration: float = 7.0,
) -> None:
    """
    Record ambient noise for the session. If wav_path is provided, record while playing the WAV file
    (duration is determined automatically). If wav_path is None, record silence for the specified duration.
    Displays a progress bar during recording.
    """
    description = 'Recording ambient noise'
    if wav_path is not None:
        wav_filename = Path(wav_path).name
        description = f'Recording ambient noise with {wav_filename}'

    record_with_progress(
        session=session,
        wav_path=wav_path,
        duration=duration,
        description=description,
        context_name='ambient noise',
    )


def record_zones(session: RecordingSession) -> None:
    """
    Record all zone calibration files found in the audio directory.
    Creates unique recorders for each zone to ensure separate output files.
    """
    audio_files = scan_input_file_directory()
    zone_files = {k: v for k, v in audio_files.items() if k.startswith('zone_')}

    if not zone_files:
        logger.warning('No zone calibration files found')
        return

    for zone_name, zone_file in zone_files.items():
        logger.info(f'Recording {zone_name} calibration')

        # Create new recorders for this specific zone
        zone_recorders = [
            SoundDeviceStreamRecorder.from_device_config(
                dev,
                output_file=get_recording_path(dev, session.output_dir, zone_name),
            )
            for dev in session.devices
        ]

        # Create a temporary session with these new recorders
        zone_session = RecordingSession(
            devices=session.devices,
            recorders=zone_recorders,
            output_dir=session.output_dir,
        )

        # Record using this specialized session with progress tracking
        description = f'Recording {zone_name} calibration'
        record_with_progress(
            session=zone_session,
            wav_path=zone_file,
            description=description,
            context_name=zone_name,
        )

    logger.info(f'Completed recording {len(zone_files)} zone calibrations')


def record_tests(session: RecordingSession) -> None:
    """
    Record all test files found in the audio directory.
    Creates unique recorders for each test to ensure separate output files.
    """
    audio_files = scan_input_file_directory()
    test_files = {k: v for k, v in audio_files.items() if k.startswith('test_')}

    if not test_files:
        logger.warning('No test files found')
        return

    for test_name, test_file in test_files.items():
        logger.info(f'Recording {test_name}')

        # Create new recorders for this specific test
        test_recorders = [
            SoundDeviceStreamRecorder.from_device_config(
                dev,
                output_file=get_recording_path(dev, session.output_dir, test_name),
            )
            for dev in session.devices
        ]

        # Create a temporary session with these new recorders
        test_session = RecordingSession(
            devices=session.devices,
            recorders=test_recorders,
            output_dir=session.output_dir,
        )

        # Record using this specialized session with progress tracking
        description = f'Recording {test_name}'
        record_with_progress(
            session=test_session,
            wav_path=test_file,
            description=description,
            context_name=test_name,
        )

    logger.info(f'Completed recording {len(test_files)} tests')
