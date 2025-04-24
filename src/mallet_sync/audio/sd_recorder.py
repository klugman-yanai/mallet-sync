# audio/recorder.py
import queue
import threading
import time

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from numpy._typing import DTypeLike

# Use absolute imports
from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    RECORDER_CHUNK_SIZE,
    get_logger,
)

logger = get_logger(__name__)


class AudioRecorder:
    """Records audio from a specified device to a WAV file using streaming approach."""

    def __init__(
        self,
        output_file: Path,
        device_index: int,
        channels: int = MALLET_CHANNELS,
        sample_rate: int = MALLET_SAMPLE_RATE,
        chunk_size: int = RECORDER_CHUNK_SIZE,
        dtype: DTypeLike = MALLET_DTYPE,
    ):
        self.device_index = device_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.output_file = output_file
        self.dtype = dtype

        # Thread control
        self._record_stop_event = threading.Event()
        self._write_stop_event = threading.Event()
        self._record_thread: threading.Thread | None = None
        self._write_thread: threading.Thread | None = None
        self._exception: Exception | None = None

        # Data pipeline elements
        self._data_queue = queue.Queue(maxsize=100)  # Buffer between recording and writing
        self._is_recording = False
        self._is_writing = False

        # Statistics
        self._frames_recorded = 0
        self._frames_written = 0

    @property
    def is_recording(self) -> bool:
        """Checks if the recorder thread is active."""
        return self._is_recording and self._record_thread is not None and self._record_thread.is_alive()

    @property
    def is_writing(self) -> bool:
        """Checks if the file writing thread is active."""
        return self._is_writing and self._write_thread is not None and self._write_thread.is_alive()

    def start(self):
        """Starts both recording and file writing threads."""
        if self.is_recording:
            logger.error(f'Recording already in progress for {self.output_file.name}.')
            raise RuntimeError(f'Recording already in progress for {self.output_file.name}.')

        # Ensure parent directory exists before starting
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Reset state
        self._record_stop_event.clear()
        self._write_stop_event.clear()
        self._exception = None
        self._frames_recorded = 0
        self._frames_written = 0

        # Clear queue if needed
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
            except queue.Empty:
                break

        # Start file writing thread first
        self._is_writing = True
        self._write_thread = threading.Thread(
            target=self._file_writing_loop,
            daemon=True,
            name=f'Write-{self.output_file.stem}',
        )
        self._write_thread.start()
        logger.debug(f'Started file writing thread for {self.output_file.name}')

        # Then start recording thread
        self._is_recording = True
        self._record_thread = threading.Thread(
            target=self._recording_loop,
            daemon=True,
            name=f'Rec-{self.output_file.stem}',
        )
        self._record_thread.start()
        logger.info(f'Started streaming recording on device {self.device_index} -> {self.output_file.name}')

    def join(self, timeout: float = 5.0) -> bool:
        """
        Wait for the recording and file writing threads to complete.

        Args:
            timeout: Maximum time to wait for thread completion in seconds

        Returns:
            bool: True if all threads completed, False if timed out
        """
        result = True

        # First, join the recording thread if it's active
        if self.is_recording and self._record_thread is not None:
            logger.debug(f'Joining recording thread for {self.output_file.name}')
            self._record_thread.join(timeout=timeout)

            if self._record_thread.is_alive():
                logger.warning(f'Recording thread join timed out for {self.output_file.name}')
                result = False
            else:
                self._is_recording = False
                logger.debug(
                    f'Recording thread joined for {self.output_file.name} ({self._frames_recorded} frames)',
                )

        # Then, join the writing thread if it's active
        if self.is_writing and self._write_thread is not None:
            logger.debug(f'Joining write thread for {self.output_file.name}')
            self._write_thread.join(timeout=timeout)

            if self._write_thread.is_alive():
                logger.warning(f'Write thread join timed out for {self.output_file.name}')
                result = False
            else:
                self._is_writing = False
                logger.debug(
                    f'Write thread joined for {self.output_file.name} ({self._frames_written} frames)',
                )

        return result

    def stop(self):
        """Signals the recording thread to stop but allows the file writing to continue."""
        if not self.is_recording:
            logger.debug(f'Stop called but not recording: {self.output_file.name}')
            return

        logger.debug(f'Signaling stop for recorder: {self.output_file.name}')
        self._record_stop_event.set()

        # Wait briefly for recording to stop
        if self._record_thread is not None:
            logger.debug(f'Waiting for recording thread to finish: {self.output_file.name}')
            self._record_thread.join(timeout=1.0)

            if self._record_thread.is_alive():
                logger.warning(f'Recording thread did not stop gracefully for {self.output_file.name}')
            else:
                logger.info(f'Recording thread stopped for {self.output_file.name}')
                self._is_recording = False
                self._record_thread = None

        # Log exception if one occurred in the thread
        if self._exception:
            logger.error(f'Exception during recording for {self.output_file.name}: {self._exception}')

        logger.info(f'Stopped recording device {self.device_index}. File writing continues.')

    def stop_writing(self):
        """Signals the file writing thread to stop after processing all queued data."""
        if not self.is_writing:
            logger.debug(f'Stop writing called but not writing: {self.output_file.name}')
            return

        logger.debug(f'Signaling stop for file writer: {self.output_file.name}')
        self._write_stop_event.set()

        # Wait briefly for any in-progress writes to complete
        if self._write_thread is not None:
            logger.debug(f'Waiting for write thread to finish: {self.output_file.name}')
            self._write_thread.join(timeout=1.0)

            if self._write_thread.is_alive():
                logger.warning(f'Write thread is still processing data for {self.output_file.name}')
            else:
                logger.info(f'Write thread completed for {self.output_file.name}')
                self._is_writing = False
                self._write_thread = None

    def wait_for_write_complete(self, timeout: float = 30.0) -> bool:
        """
        Wait for the file writing to complete with a timeout.
        Returns True if writing completed successfully, False if it timed out.
        """
        if not self.is_writing or self._write_thread is None:
            return True

        logger.debug(f'Waiting for file writing to complete for {self.output_file.name}')
        self._write_thread.join(timeout=timeout)

        if self._write_thread.is_alive():
            logger.warning(f'File writing did not complete within timeout for {self.output_file.name}')
            return False

        logger.debug(f'File writing completed for {self.output_file.name}')
        self._is_writing = False
        self._write_thread = None
        return True

    def _recording_loop(self):
        """Streaming recording loop that places audio chunks directly into a queue."""
        try:
            with sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
            ) as stream:
                logger.debug(f'Input stream opened for device {self.device_index}')

                while not self._record_stop_event.is_set():
                    # Read a chunk of audio data
                    data, overflowed = stream.read(self.chunk_size)

                    if overflowed:
                        logger.warning(f'Input overflowed on device {self.device_index}')

                    if data.size > 0:
                        try:
                            # Put the data into the queue for the writer thread to process
                            # Use a timeout to ensure we can check the stop event regularly
                            self._data_queue.put(data.copy(), timeout=0.5)
                            self._frames_recorded += 1
                        except queue.Full:
                            logger.warning(f'Queue full for {self.output_file.name}, dropping audio chunk')
                    else:
                        # Rare case of empty data, add a tiny sleep
                        time.sleep(0.001)

        except sd.PortAudioError as pae:
            logger.exception(f'PortAudioError recording device {self.device_index}')
            self._exception = pae
        except Exception as exc:
            logger.exception(f'Error during recording loop for device {self.device_index}')
            self._exception = exc
        finally:
            # Signal the writing thread that recording is complete
            self._is_recording = False
            logger.debug(
                f'Recording loop finished for device {self.device_index} ({self._frames_recorded} frames)',
            )

    def _file_writing_loop(self):
        """Continuously writes audio chunks from the queue to a WAV file."""
        try:
            # Open the output file for streaming writing
            with sf.SoundFile(
                file=str(self.output_file),
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                format='WAV',
            ) as sound_file:
                logger.debug(f'Opened output file for streaming: {self.output_file.name}')

                # Keep writing until explicitly stopped and queue is empty
                while True:
                    # Check if we should stop and the queue is empty
                    if (
                        self._write_stop_event.is_set()
                        and self._data_queue.empty()
                        and not self.is_recording
                    ):
                        logger.debug(f'Write stop requested and queue empty for {self.output_file.name}')
                        break

                    try:
                        # Get data from the queue with a timeout to avoid blocking forever
                        chunk = self._data_queue.get(timeout=0.5)

                        # Write the chunk to the file
                        sound_file.write(chunk)
                        self._frames_written += 1

                        # Notify queue that this item is processed
                        self._data_queue.task_done()

                    except queue.Empty:
                        # No data available yet, check if recording has stopped
                        if not self.is_recording and self._data_queue.empty():
                            # If recording stopped and no more data, we're done
                            logger.debug(f'Recording stopped and queue empty for {self.output_file.name}')
                            break
                        continue  # Try again

            logger.info(f'Successfully wrote {self._frames_written} chunks to {self.output_file.name}')

        except sf.SoundFileError as sfe:
            logger.exception(f'SoundFile error writing to {self.output_file.name}')
            self._exception = sfe
        except Exception as exc:
            logger.exception(f'Failed to write audio file {self.output_file.name}')
            self._exception = exc
        finally:
            self._is_writing = False
            logger.debug(f'File writing loop finished for {self.output_file.name}')
