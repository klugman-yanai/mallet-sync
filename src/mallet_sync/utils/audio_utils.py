"""
Audio processing and recording utilities for mallet-sync.

This module provides utilities for:
1. Finding and scanning audio files in the input directory
2. Creating and managing audio recorders
3. Tracking recording progress
4. Recording different types of audio (ambient, zones, tests)
"""

import re
import time

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import sounddevice as sd
import soundfile as sf

from rich.progress import TaskID

from mallet_sync.audio.sd_player import SoundDeviceAudioPlayer
from mallet_sync.audio.sd_recorder import SoundDeviceStreamRecorder
from mallet_sync.config import DEFAULT_INPUT_DIR, DeviceConfig, get_logger
from mallet_sync.config.config import RecordingSession
from mallet_sync.utils.file_utils import get_recording_path

logger = get_logger(__name__)


# --------------------------------------------------------------------------------
# MODULE CONSTANTS
# --------------------------------------------------------------------------------

DEFAULT_RECORDING_DURATION = 7.0
PROGRESS_UPDATE_INTERVAL = 0.05  # seconds
PROGRESS_TOTAL_STEPS = 100  # Steps for percentage-based progress tracking


# --------------------------------------------------------------------------------
# FILE SCANNING FUNCTIONS
# --------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------
# RECORDER MANAGEMENT FUNCTIONS
# --------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------
# PROGRESS TRACKING FUNCTIONS
# --------------------------------------------------------------------------------


def _track_wav_playback_progress(
    progress_context: Any,
    player: SoundDeviceAudioPlayer,
    task_id: TaskID | int,
    duration: float,
) -> None:
    """
    Track progress during WAV file playback.

    Args:
        progress_context: The progress context from logger.progress
        player: The audio player instance
        task_id: Progress tracking task ID
        duration: Duration of the audio file in seconds
    """
    start_time = time.time()
    last_percent = -1

    # Keep updating until playback finishes
    while player._is_playing:
        # Calculate progress percentage (0-99 to leave room for final update)
        elapsed = time.time() - start_time
        percent = min(99, int(99 * elapsed / duration))

        # Only advance if percentage has increased
        if percent > last_percent:
            # How many steps to advance
            steps_to_advance = percent - last_percent
            if steps_to_advance > 0:
                progress_context.update(TaskID(task_id), advance=steps_to_advance)
                last_percent = percent

        time.sleep(PROGRESS_UPDATE_INTERVAL)

    # Final update to ensure 100% completion
    remaining_steps = PROGRESS_TOTAL_STEPS - last_percent - 1
    if remaining_steps > 0:
        progress_context.update(TaskID(task_id), advance=remaining_steps)


def _track_silent_recording_progress(
    progress_context: Any,
    task_id: TaskID | int,
    duration: float,
) -> None:
    """
    Track progress during silent recording.

    Args:
        progress_context: The progress context from logger.progress
        task_id: Progress tracking task ID
        duration: Recording duration in seconds
    """
    start_time = time.time()
    last_percent = -1

    # Keep updating until duration completes
    while time.time() - start_time < duration:
        # Calculate progress percentage (0-99)
        elapsed = time.time() - start_time
        percent = min(99, int(99 * elapsed / duration))

        # Only advance if percentage has increased
        if percent > last_percent:
            # How many steps to advance
            steps_to_advance = percent - last_percent
            if steps_to_advance > 0:
                progress_context.update(TaskID(task_id), advance=steps_to_advance)
                last_percent = percent

        time.sleep(PROGRESS_UPDATE_INTERVAL)

    # Final update to ensure 100% completion
    remaining_steps = PROGRESS_TOTAL_STEPS - last_percent - 1
    if remaining_steps > 0:
        progress_context.update(TaskID(task_id), advance=remaining_steps)


# --------------------------------------------------------------------------------
# ERROR HANDLING FUNCTIONS
# --------------------------------------------------------------------------------


def handle_recording_error(session: RecordingSession, error_context: str) -> None:
    """Handle recording errors by logging and updating session metadata.

    Args:
        session: The recording session to update
        error_context: Context description for the error
    """
    error_msg = f'Error during {error_context}'
    logger.exception(error_msg)
    session.add_error(error_msg)


# --------------------------------------------------------------------------------
# CORE RECORDING FUNCTIONS
# --------------------------------------------------------------------------------


def record_with_progress(
    session: RecordingSession,
    wav_path: Path | None = None,
    duration: float = DEFAULT_RECORDING_DURATION,
    description: str = 'Recording',
    context_name: str = '',
) -> float:
    """
    Record audio with progress tracking.

    Args:
        session: The recording session with devices and recorders
        wav_path: Optional path to WAV file to play while recording
        duration: Duration to record if no WAV file is provided
        description: User-friendly description for the progress bar
        context_name: Recording context name for logging
    """
    # Input validation
    if session.recorders is None or len(session.recorders) == 0:
        raise ValueError('Recording session must have at least one recorder')

    # Convert string path to Path object if needed
    if isinstance(wav_path, str):
        wav_path = Path(wav_path)

    if wav_path is not None:
        # Record while playing a WAV file
        info = sf.info(str(wav_path))
        actual_duration: float = info.duration
        wav_filename = wav_path.name
        logger.debug(f'Recording {context_name} with {wav_filename} (duration: {actual_duration:.2f}s)')

        # Create progress tracking with StandardLogger using file_progress mode
        with logger.progress(description=description, file_progress=True) as progress:
            # Create a task with fixed total steps
            task_id: int = progress.add_task(description, total=PROGRESS_TOTAL_STEPS)

            # Start all recorders
            for recorder in session.recorders:
                recorder.start()

            # Play WAV with non-blocking method so we can update progress
            player = SoundDeviceAudioPlayer(str(wav_path))
            player.start()

            # Track progress during playback
            _track_wav_playback_progress(progress, player, task_id, actual_duration)

            # Stop all recorders when playback finishes
            for recorder in session.recorders:
                recorder.stop()

        # Return the actual duration from the WAV file
        return actual_duration
    # Record silence for the specified duration
    logger.debug(f'Recording {context_name} for {duration:.2f}s')

    # Create progress tracking with StandardLogger
    with logger.progress(description=description, file_progress=True) as progress:
        # Create task with steps for percentage-based progress
        task_id = progress.add_task(description, total=PROGRESS_TOTAL_STEPS)

        # Start all recorders
        for recorder in session.recorders:
            recorder.start()

        # Track progress during silent recording
        _track_silent_recording_progress(progress, task_id, duration)

        # Stop all recorders
        for recorder in session.recorders:
            recorder.stop()

    logger.debug(f'Finished recording {context_name or "audio"}')
    # Return the specified duration for silent recording
    return duration


def record_ambient_noise(
    session: RecordingSession,
    wav_path: Path | None = None,
    duration: float = DEFAULT_RECORDING_DURATION,
) -> None:
    """
    Record ambient noise for the session.

    Automatically searches for an ambient_noise.wav file in the default input directory.
    If wav_path is provided, it uses that file instead. If no ambient noise file is found,
    it falls back to recording silence for the specified duration.

    Displays a progress bar during recording and tracks input files in session metadata.

    Error handling is built in, with exceptions logged and recorded in session metadata.

    Args:
        session: The recording session with devices and recorders
        wav_path: Optional path to WAV file to play while recording (overrides auto-detection)
        duration: Duration to record if no WAV file is provided/found
    """
    try:
        logger.info('Starting ambient noise recording')
        description = 'Recording ambient noise'

        # Auto-detect ambient noise file if not explicitly provided
        if wav_path is None:
            # Look for ambient_noise.wav in the default input directory
            default_ambient_file = DEFAULT_INPUT_DIR / 'ambient_noise.wav'
            if default_ambient_file.exists():
                wav_path = default_ambient_file
                logger.debug(f"Found ambient noise file: {wav_path}")

        # Update description if using a WAV file
        if wav_path is not None and wav_path.exists():
            wav_filename = Path(wav_path).name
            description = f'Recording ambient noise with {wav_filename}'

            # Track the input file in session metadata
            input_files = session.input_files.copy()
            if str(wav_path) not in input_files:
                input_files.append(str(wav_path))
                # Use object.__setattr__ to modify the frozen dataclass
                object.__setattr__(session, 'input_files', input_files)

        # Record and get duration
        actual_duration = record_with_progress(
            session=session,
            wav_path=wav_path,
            duration=duration,
            description=description,
            context_name='ambient noise',
        )

        # Update session with recording info
        session.add_recording('ambient', actual_duration)
    except Exception:
        error_msg = 'Error during ambient noise recording'
        logger.exception(error_msg)
        session.add_error(error_msg)


# --------------------------------------------------------------------------------
# MULTI-CONTEXT RECORDING FUNCTIONS
# --------------------------------------------------------------------------------


def _record_for_context_files(
    session: RecordingSession,
    context_files: dict[str, Path],
    context_type: str,
) -> bool:
    """
    Record audio for a set of context files (zones or tests).
    Creates unique recorders for each context to ensure separate output files.

    Args:
        session: Base recording session with devices
        context_files: Dictionary mapping context names to WAV file paths
        context_type: Type of context (zone/test) for logging
    """
    if not context_files:
        logger.warning(f'No {context_type} files found')
        return False

    for context_name, file_path in context_files.items():
        # Create new recorders for this specific context
        context_recorders = [
            SoundDeviceStreamRecorder.from_device_config(
                dev,
                output_file=get_recording_path(dev, session.output_dir, context_name),
            )
            for dev in session.devices
        ]

        # Create a temporary session with these new recorders
        context_session = RecordingSession(
            devices=session.devices,
            recorders=context_recorders,
            output_dir=session.output_dir,
        )

        # Track the input file path
        input_files = session.input_files.copy()
        if str(file_path) not in input_files:
            input_files.append(str(file_path))
            object.__setattr__(session, 'input_files', input_files)

        # Record using this specialized session with progress tracking
        description = f'Recording {context_name}'
        actual_duration = record_with_progress(
            session=context_session,
            wav_path=file_path,
            description=description,
            context_name=context_name,
        )

        # Update the session with this recording
        session.add_recording(context_name, actual_duration)

    logger.debug(f'Completed recording {len(context_files)} {context_type}s')
    return True


def record_zones(session: RecordingSession) -> None:
    """
    Record all zone calibration files found in the audio directory.
    Creates unique recorders for each zone to ensure separate output files.

    Error handling is built in, with exceptions logged and recorded in session metadata.

    Args:
        session: The recording session with devices and recorders
    """
    try:
        logger.info('Starting zone calibration recording')
        # Get zone files and create context-specific recorders for each
        audio_files = scan_input_file_directory()
        zone_files = {k: v for k, v in audio_files.items() if k.startswith('zone_')}

        # Only mark as complete if files were actually processed
        if _record_for_context_files(session, zone_files, 'zone calibration'):
            session.add_recording('zones')
    except Exception:
        error_msg = 'Error during zone calibration recording'
        logger.exception(error_msg)
        session.add_error(error_msg)


def record_tests(session: RecordingSession) -> None:
    """
    Record all test files found in the audio directory.
    Creates unique recorders for each test to ensure separate output files.

    Error handling is built in, with exceptions logged and recorded in session metadata.

    Args:
        session: The recording session with devices and recorders
    """
    try:
        logger.info('Starting test recording')
        # Get test files and create context-specific recorders for each
        audio_files = scan_input_file_directory()
        test_files = {k: v for k, v in audio_files.items() if k.startswith('test_')}

        # Only mark as complete if files were actually processed
        if _record_for_context_files(session, test_files, 'test'):
            session.add_recording('tests')
    except Exception:
        error_msg = 'Error during test recording'
        logger.exception(error_msg)
        session.add_error(error_msg)
