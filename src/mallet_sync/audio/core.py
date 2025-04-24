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
    logger.info(f'Streaming {duration_sec} seconds of silence and recording')

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
        logger.info(f'Playing silence for {silence_duration:.2f} seconds')

        # Initialize recorders for each mallet device
        recorders = []
        for i, (device, friendly_name) in enumerate(mallet_devices):
            # Create output filename with device info
            output_path = (
                output_dir / f'silence_{timestamp}_device{i}_{friendly_name.replace(" ", "_")}.wav'
            )

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

    # Play silence directly if needed
    if needs_silence:
        play_silence(mallet_devices, output_dir)
        time.sleep(sleep_time_sec)

    # Process all regular files
    for wav_file in files:
        play_and_record_cycle(mallet_devices, wav_file, output_dir)
        time.sleep(sleep_time_sec)

    # Log completion
    end_time = time.time()
    logger.info(f'All processing finished in {end_time - start_time:.2f} seconds.')
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
    logger.info(f'--- Processing: {wav_path.name} ---')
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
        time.sleep(0.2)

        # STEP 2: Start streaming playback
        logger.info(f'Starting streaming playback of {wav_path.name} ({player.duration:.2f}s)')
        try:
            player.start()
        except Exception:
            logger.exception(f'Failed to start playback for {wav_path.name}')
            # Continue recording to capture silence

        # STEP 3: Wait for playback to complete
        logger.info('Waiting for playback to complete...')
        player.join()
        logger.info(f'Playback of {wav_path.name} complete.')

        # STEP 4: Stop recording (but file writing continues)
        logger.info('Stopping recorders (file writing continues)...')
        for recorder in active_recorders:
            try:
                recorder.stop()
            except Exception:
                logger.exception(f'Error stopping recorder {recorder.output_file.name}')

        # STEP 5: Wait for any remaining data to be written to files
        logger.info('Waiting for file writing to complete...')
        for recorder in active_recorders:
            try:
                # Wait for both recording and writing to complete
                recorder.join()
            except Exception:
                logger.exception(f'Error joining recorder {recorder.output_file.name}')

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

    logger.info(f'--- Finished processing: {wav_path.name} ---')
