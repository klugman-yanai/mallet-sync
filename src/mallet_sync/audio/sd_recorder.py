# audio/recorder.py
import threading
import time

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# Use absolute imports
from mallet_sync.config import get_logger

logger = get_logger(__name__)


class AudioRecorder:
    """Records audio from a specified device to a WAV file using a thread."""

    def __init__(
        self,
        device_index: int,
        channels: int,
        sample_rate: int,
        chunk_size: int,
        output_file: Path,
        dtype: str,
    ):
        self.device_index = device_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.output_file = output_file
        self.dtype = dtype

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None
        self._frames: list[np.ndarray] = []
        self._is_recording = False

    @property
    def is_recording(self) -> bool:
        """Checks if the recorder thread is active."""
        return self._is_recording and self._thread is not None and self._thread.is_alive()

    def start(self):
        """Starts the recording thread."""
        if self.is_recording:
            # Use logger.error or raise specific exception
            logger.error(f'Recording already in progress for {self.output_file.name}.')
            raise RuntimeError(f'Recording already in progress for {self.output_file.name}.')

        self._stop_event.clear()
        self._exception = None
        self._frames = []
        self._is_recording = True  # Set state before starting thread
        self._thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
            name=f'Rec-{self.output_file.stem}',
        )
        self._thread.start()
        logger.info(f'Started recording device {self.device_index} -> {self.output_file.name}')

    def stop(self):
        """Signals the recording thread to stop, joins it, and writes the file."""
        if not self._is_recording or self._thread is None:
            logger.debug(f'Stop called but not recording: {self.output_file.name}')
            return

        logger.debug(f'Signaling stop for recorder: {self.output_file.name}')
        self._stop_event.set()

        logger.debug(f'Joining recorder thread: {self.output_file.name}')
        self._thread.join(timeout=3.0)  # Increased timeout slightly

        if self._thread.is_alive():
            logger.warning(f'Recorder thread for device {self.device_index} did not stop gracefully.')
            # Decide policy: should we attempt write anyway? For now, yes.

        self._is_recording = False
        logger.info(f'Stopped recording device {self.device_index}.')
        self._thread = None

        # Log exception if one occurred in the thread
        if self._exception:
            logger.error(f'Exception during recording for {self.output_file.name}: {self._exception}')
            # Optionally re-raise if needed: raise self._exception

        # Write the file after stopping and logging potential errors
        self._write_file()

    def _record_loop(self):
        """The main loop executed by the recording thread."""
        try:
            # Ensure parent directory exists before opening stream
            self.output_file.parent.mkdir(parents=True, exist_ok=True)

            with sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
            ) as stream:
                logger.debug(f'Input stream opened for device {self.device_index}')
                while not self._stop_event.is_set():
                    data, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        logger.warning(f'Input overflowed on device {self.device_index}')
                    if data.size > 0:
                        self._frames.append(data.copy())
                    else:
                        # stream.read blocks, so this sleep is likely redundant
                        # but harmless as a fallback.
                        time.sleep(0.001)

        except sd.PortAudioError as pae:
            logger.exception(f'PortAudioError recording device {self.device_index}')
            self._exception = pae
        except Exception as exc:
            # Use logger.exception to include traceback
            logger.exception(f'Error during recording loop for device {self.device_index}')
            self._exception = exc
        finally:
            logger.debug(f'Recording loop finished for device {self.device_index}')

    def _write_file(self):
        """Writes the collected audio frames to the output WAV file."""
        if not self._frames:
            logger.warning(f'No frames recorded for {self.output_file.name}. Skipping file write.')
            return
        try:
            duration_s = (
                len(self._frames) * self.chunk_size / self.sample_rate if self.sample_rate > 0 else 0
            )
            logger.info(
                f'Writing {len(self._frames)} chunks ({duration_s:.2f}s) to {self.output_file.name}',
            )

            arr = np.concatenate(self._frames, axis=0)
            sf.write(
                file=str(self.output_file),
                data=arr,
                samplerate=self.sample_rate,
                subtype=None,  # Auto from dtype
                format='WAV',
            )
            logger.info(f'Successfully wrote {self.output_file.name}')
        except Exception:
            logger.exception(f'Failed to write audio file {self.output_file.name}')
            # Maybe store this exception as well?
            # self._exception = self._exception or e # Keep first exception
