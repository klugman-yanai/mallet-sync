# audio/core.py
import contextlib
import threading
import time

from pathlib import Path
from typing import Optional

import sounddevice as sd

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
