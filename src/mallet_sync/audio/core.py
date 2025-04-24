# audio/core.py
import contextlib
import tempfile
import threading
import time

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# Use absolute imports
from mallet_sync.audio.sd_player import AudioPlayer
from mallet_sync.audio.sd_recorder import AudioRecorder
from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    RECORDER_CHUNK_SIZE,
    DeviceInfo,
    get_logger,
)
from mallet_sync.utils import generate_output_path

logger = get_logger(__name__)


def play_silence(
    mallet_devices: list[tuple[DeviceInfo, str]],
    output_dir: Path,
    duration_sec: int = 10,
    sample_rate: int = MALLET_SAMPLE_RATE,
) -> None:
    """
    Play silence directly by streaming zeros to output device and record from mallet devices.

    Args:
        mallet_devices: List of mallet devices to record from
        output_dir: Directory to save recordings
        duration_sec: Duration of silence in seconds
        sample_rate: Sample rate for audio streaming
    """
    _silence_start_time = time.time()
    logger.info(f'Starting ambient noise calibration: {duration_sec} seconds of silence')

    # Create a StreamPlayer that produces silence
    class SilenceStreamPlayer:
        """A simple class that mimics AudioPlayer but streams zeros."""

        def __init__(self, duration_sec: int, sample_rate: int):
            self.duration_sec = duration_sec
            self.sample_rate = sample_rate
            self.is_playing = False
            self._stop_event = threading.Event()
            self._thread: threading.Thread | None = None

        def start(self) -> None:
            """Start streaming silence."""
            if self.is_playing:
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._streaming_silence_task,
                daemon=True,
            )
            self._thread.start()
            self.is_playing = True
            logger.debug('Started silence streaming')

        def _streaming_silence_task(self) -> None:
            """Stream zeros to the output device."""
            chunk_size = 1024
            total_frames = self.sample_rate * self.duration_sec
            frames_played = 0

            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,  # Mono silence
                blocksize=chunk_size,
                dtype='float32',
            ) as stream:
                while not self._stop_event.is_set() and frames_played < total_frames:
                    # Calculate chunk size for this iteration (might be smaller at the end)
                    current_chunk = min(chunk_size, total_frames - frames_played)
                    if current_chunk <= 0:
                        break

                    # Generate silence chunk
                    data = np.zeros((current_chunk, 1), dtype=np.float32)

                    # Write to stream
                    stream.write(data)
                    frames_played += current_chunk

        def stop(self) -> None:
            """Stop streaming silence."""
            if not self.is_playing:
                return

            self._stop_event.set()
            self.is_playing = False
            logger.debug('Requested silence streaming stop')

        def join(self) -> None:
            """Wait for the streaming thread to complete."""
            if self._thread and self._thread.is_alive():
                self._thread.join()
                logger.debug('Silence streaming thread joined')
            self._thread = None

        @property
        def duration(self) -> float:
            """Duration in seconds."""
            return self.duration_sec

    # Create silence player
    silence_player = SilenceStreamPlayer(duration_sec, sample_rate)

    # Use a specialized version of play_and_record_cycle with our silence player
    play_and_record_cycle_with_silence(
        mallet_devices=mallet_devices,
        silence_player=silence_player,
        output_dir=output_dir,
    )


def check_ambient_files(input_dir: Path) -> bool:
    """
    Checks for ambient WAV files.

    Args:
        input_dir: Base directory containing audio subdirectories

    Returns:
        True if ambient files exist, False otherwise
    """
    # Check for ambient files
    ambient_dir: Path = input_dir / 'ambient'
    ambient_files: list[Path] = list(ambient_dir.glob('*.wav')) if ambient_dir.exists() else []

    has_ambient_files = len(ambient_files) > 0
    logger.debug(f'Ambient directory: {ambient_dir.resolve()}')
    logger.debug(f'Found {len(ambient_files)} ambient files')

    if not has_ambient_files:
        logger.info('No ambient noise files found. Will play 10 seconds of silence first.')

    return has_ambient_files


# This function is no longer needed since we're streaming silence directly
# Keeping as a stub for compatibility
def cleanup_temp_file(file_path: Path | None = None) -> None:
    """Function kept for compatibility but no longer needed"""
    pass


def play_and_record_cycle_with_silence(
    mallet_devices: list[tuple[DeviceInfo, str]],
    silence_player: Any,
    output_dir: Path,
) -> None:
    """
    Specialized version of play_and_record_cycle for streaming silence.

    Args:
        mallet_devices: List of mallet devices to record from
        silence_player: Player object that streams silence
        output_dir: Directory to save recordings
    """
    try:
        silence_duration = silence_player.duration
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f'Recording ambient silence for calibration ({silence_duration:.1f} seconds)')

        # Initialize recorders for each mallet device
        recorders = []
        for i, (device, friendly_name) in enumerate(mallet_devices):
            # Create output filename with device info
            output_path = (
                output_dir / f'silence_{timestamp}_device{i}_{friendly_name.replace(" ", "_")}.wav'
            )
            logger.debug(f'Creating silence recorder for device {device.index} to {output_path.name}')

            # Create recorder for this device
            recorder = AudioRecorder(
                output_file=output_path,
                device_index=device.index,
            )
            recorders.append(recorder)

        # Start all recorders first
        logger.debug('Starting all recorders')
        for recorder in recorders:
            recorder.start()

        # Start player
        logger.debug('Starting silence player')
        silence_player.start()

        # Wait for player to finish
        silence_player.join()

        # Stop all recorders
        logger.debug('Stopping all recorders')
        for recorder in recorders:
            recorder.stop()
            recorder.join()

        logger.info(
            f'Successfully played {silence_duration:.2f} seconds of silence and recorded from {len(recorders)} devices',
        )
    except Exception:
        logger.exception('Error during silence play and record cycle')


def process_audio_batch(
    files: list[Path],
    mallet_devices: list[tuple[DeviceInfo, str]],
    output_dir: Path,
    sleep_time_sec: float,
    *,  # Force keyword arguments after this point
    needs_silence: bool = False,
) -> None:
    """
    Process a batch of audio files with optional silence first.

    Args:
        files: List of audio files to process
        mallet_devices: List of mallet devices to record from
        output_dir: Directory to save recordings
        sleep_time_sec: Time to sleep between files
        needs_silence: Whether to play silence before processing files
    """
    start_time = time.time()
    total_files = len(files)

    # Log batch processing start with device info
    logger.info(f'Starting batch processing of {total_files} files using {len(mallet_devices)} devices')
    device_names = [f'{name} (ID: {info.index})' for info, name in mallet_devices]
    for device in device_names:
        logger.debug(f'Device: {device}')

    # Play silence directly if needed
    if needs_silence:
        silence_start = time.time()
        play_silence(mallet_devices, output_dir)
        silence_duration = time.time() - silence_start
        logger.info(f'Ambient silence recording completed in {silence_duration:.1f} seconds')
        logger.debug(f'Sleeping for {sleep_time_sec} seconds before starting file processing')
        time.sleep(sleep_time_sec)

    # Process all regular files
    for idx, wav_file in enumerate(files, 1):
        file_start = time.time()
        logger.info(f'Processing file {idx}/{total_files}: {wav_file.name}')
        play_and_record_cycle(mallet_devices, wav_file, output_dir)
        file_duration = time.time() - file_start

        # Format as minutes:seconds if over 60 seconds
        if file_duration >= 60:
            minutes = int(file_duration // 60)
            seconds = file_duration % 60
            duration_str = f'{minutes}m {seconds:.1f}s'
        else:
            duration_str = f'{file_duration:.1f} seconds'

        logger.info(f'File {idx}/{total_files} processed in {duration_str}')

        if idx < total_files:
            logger.debug(f'Sleeping for {sleep_time_sec} seconds before next file')
            time.sleep(sleep_time_sec)

    # Log completion with human-readable total time
    total_duration = time.time() - start_time
    if total_duration >= 60:
        minutes = int(total_duration // 60)
        seconds = total_duration % 60
        duration_str = f'{minutes}m {seconds:.1f}s'
    else:
        duration_str = f'{total_duration:.1f} seconds'

    logger.info(f'All processing finished in {duration_str}')
    logger.info(f'Recordings saved in: {output_dir.resolve()}')


def play_and_record_cycle(
    mallet_devices: list[tuple[DeviceInfo, str]],
    wav_path: Path,
    output_dir: Path,
) -> None:
    """
    Plays one WAV file on default output and records from Mallet devices using streaming.

    Following the exact threading flow:
    1. Recorder.start (starts both recording and file writing threads)
    2. Player.start (begins streaming playback)
    3. Player.join (waits for playback to complete)
    4. Recorder.stop (stops recording but continues file writing)
    5. Recorder.join (waits for any remaining data to be written to files)

    All processes work with chunks to ensure continuous processing without frame drops.
    """
    cycle_start_time = time.time()

    # Get file size in human-readable format
    file_size_bytes = wav_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    logger.info(f'--- Processing: {wav_path.name} ({file_size_mb:.2f} MB) ---')
    logger.debug(f'Starting process cycle for {wav_path} at {cycle_start_time}')
    context_name = wav_path.stem

    # Setup player with chunked streaming
    player = AudioPlayer(wav_path, chunk_size=RECORDER_CHUNK_SIZE)
    if not player.can_play:
        logger.error(f'Cannot prepare player for {wav_path.name}. Skipping cycle.')
        return

    # Setup recorders with streaming processing
    recorders: list[AudioRecorder] = []
    for dev_info, role in mallet_devices:
        out_path = generate_output_path(output_dir, role, context_name)
        recorder = AudioRecorder(
            device_index=dev_info.index,
            channels=MALLET_CHANNELS,
            sample_rate=MALLET_SAMPLE_RATE,
            chunk_size=RECORDER_CHUNK_SIZE,
            output_file=out_path,
            dtype=MALLET_DTYPE,
        )
        recorders.append(recorder)

    if not recorders:
        logger.error('No recorders could be created. Skipping cycle.')
        return

    active_recorders: list[AudioRecorder] = []

    try:
        # STEP 1: Start recorders (begins both recording and file writing)
        logger.info('Starting streaming recorders...')
        for recorder in recorders:
            try:
                recorder.start()
                active_recorders.append(recorder)
                logger.debug(f'Started streaming recorder for {recorder.output_file.name}')
            except Exception:
                logger.exception(f'Failed to start recorder {recorder.output_file.name}')

        if not active_recorders:
            logger.error('No recorders started successfully. Skipping cycle.')
            return

        # Allow recorders a moment to initialize
        logger.debug('Brief pause for recorder initialization')
        time.sleep(0.2)

        # STEP 2: Start streaming playback
        playback_start_time = time.time()
        formatted_duration = (
            f'{int(player.duration // 60)}:{player.duration % 60:04.1f}'
            if player.duration >= 60
            else f'{player.duration:.1f}s'
        )
        logger.info(f'Playing {wav_path.name} (duration: {formatted_duration})')
        logger.debug(f'Starting streaming playback at {playback_start_time}')
        try:
            player.start()
        except Exception:
            logger.exception(f'Failed to start playback for {wav_path.name}')
            # Continue recording to capture silence

        # STEP 3: Wait for playback to complete
        logger.debug('Waiting for playback to complete...')
        player.join()
        playback_time = time.time() - playback_start_time
        formatted_playback_time = (
            f'{int(playback_time // 60)}:{playback_time % 60:04.1f}'
            if playback_time >= 60
            else f'{playback_time:.1f}s'
        )
        logger.info(f'Playback of {wav_path.name} complete in {formatted_playback_time}')

        # STEP 4: Stop recording (but file writing continues)
        logger.debug('Stopping recorders (file writing continues)...')
        for recorder in active_recorders:
            try:
                recorder.stop()
            except Exception:
                logger.exception(f'Error stopping recorder {recorder.output_file.name}')

        # STEP 5: Wait for any remaining data to be written to files
        logger.debug('Waiting for file writing to complete...')
        file_writing_start = time.time()
        for recorder in active_recorders:
            try:
                # Wait for both recording and writing to complete
                recorder.join()
            except Exception:
                logger.exception(f'Error joining recorder {recorder.output_file.name}')
        file_writing_time = time.time() - file_writing_start
        logger.debug(f'File writing completed in {file_writing_time:.1f} seconds')

    except Exception:
        logger.exception(f'Unhandled error during recording cycle for {wav_path.name}')

        # Emergency cleanup if we're exiting due to an exception
        if player.is_playing:
            logger.warning('Forcing audio stop via sd.stop()')
            try:
                sd.stop()
            except Exception:
                logger.exception('Error calling sd.stop()')

        # Ensure we attempt to stop all recorders
        for recorder in active_recorders:
            if recorder.is_recording:
                with contextlib.suppress(Exception):
                    recorder.stop()

    total_cycle_time = time.time() - cycle_start_time
    if total_cycle_time >= 60:
        minutes = int(total_cycle_time // 60)
        seconds = total_cycle_time % 60
        cycle_time_str = f'{minutes}m {seconds:.1f}s'
    else:
        cycle_time_str = f'{total_cycle_time:.1f} seconds'

    # Calculate and log output file sizes
    output_sizes = []
    for recorder in recorders:
        if recorder.output_file.exists():
            size_bytes = recorder.output_file.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            output_sizes.append((recorder.output_file.name, size_mb))

    logger.info(f'--- Finished processing: {wav_path.name} in {cycle_time_str} ---')
    for name, size in output_sizes:
        logger.debug(f'  Output: {name} ({size:.2f} MB)')

    # Log a summary of the outputs at info level
    total_output_size = sum(size for _, size in output_sizes)
    logger.info(f'Created {len(output_sizes)} output files ({total_output_size:.2f} MB total)')
