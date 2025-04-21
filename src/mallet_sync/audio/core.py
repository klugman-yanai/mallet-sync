# audio/core.py
import threading
import time

from pathlib import Path

import sounddevice as sd  # Keep for sd.stop() if needed

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
from mallet_sync.utils.file_utils import generate_output_path

logger = get_logger(__name__)


def play_and_record_cycle(
    mallet_devices: list[tuple[DeviceInfo, str]],
    wav_path: Path,
    output_dir: Path,
):
    """
    Plays one WAV file on default output and records from Mallet devices
    using dedicated Player and Recorder classes.
    """
    logger.info(f'--- Processing: {wav_path.name} ---')
    context_name = wav_path.stem

    # --- Player Setup ---
    player = AudioPlayer(wav_path)
    if not player.can_play:
        logger.error(f'Cannot prepare player for {wav_path.name}. Skipping cycle.')
        return  # Skip if player couldn't load audio

    # --- Recorder Setup ---
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

    # --- Execute Playback and Recording ---
    active_recorders = []
    playback_started = False
    try:
        # Start recorders first
        for r in recorders:
            try:
                r.start()
                active_recorders.append(r)
            except Exception:
                # Error logged within start() or here
                logger.exception(f'Failed to start recorder {r.output_file.name}')

        if not active_recorders:
            logger.error('No recorders started successfully. Skipping cycle.')
            return

        # Give recorders a moment to initialize
        time.sleep(0.2)

        # Start playback
        try:
            player.start()
            playback_started = True
        except Exception:
            logger.exception(f'Failed to start playback for {wav_path.name}')
            # Continue to record silence for the duration

        # --- Wait for Playback or Simulate Duration ---
        if playback_started:
            logger.info(f'Waiting for playback of {wav_path.name} ({player.duration:.2f}s) to finish...')
            player.join()  # Wait for the player's thread to complete
            logger.info(f'Playback finished for {wav_path.name}.')
        else:
            # Simulate duration if playback didn't start
            duration_to_wait = (
                player.duration if player.duration > 0 else 1.0
            )  # Default wait if duration unknown
            logger.info(
                f'Playback failed or skipped. Recording silence for estimated duration: {duration_to_wait:.2f}s',
            )
            time.sleep(max(0.0, duration_to_wait))
            logger.info('Simulated duration finished.')

    except Exception:
        logger.exception(f'Unhandled error during main recording cycle for {wav_path.name}')
    finally:
        # --- Stop Recorders ---
        logger.info(f'Stopping recorders for {context_name}...')
        # Use a separate loop to signal stop first (less critical with blocking play)
        # for r in active_recorders:
        #     try: r.stop() # Stop now handles join and write
        #     except Exception: logger.exception(f"Error stopping recorder {r.output_file.name}")

        # Ensure recorders are stopped and files written even if errors occurred
        # Call stop individually to handle potential write errors per file
        for r in active_recorders:
            try:
                r.stop()  # stop() includes join() and _write_file()
            except Exception:
                # Errors during stop/join/write are logged within stop()
                logger.exception(f'Error during final stop/write for {r.output_file.name}')

        # Check if playback thread unexpectedly hung (unlikely with blocking play in thread)
        if playback_started and player.is_playing:
            logger.warning(
                f'Playback thread for {wav_path.name} still seems active after join? Attempting sd.stop().',
            )
            try:
                sd.stop()
            except Exception:
                logger.exception('Error calling sd.stop()')

        logger.info(f'--- Finished processing: {wav_path.name} ---')
