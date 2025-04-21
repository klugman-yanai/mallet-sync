# audio/player.py
import threading
import time

from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from mallet_sync.config.config import get_logger

logger = get_logger(__name__)


class AudioPlayer:
    """Plays an audio file on the default output device using a thread."""

    def __init__(self, wav_path: Path):
        self.wav_path = wav_path
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None
        self._playback_data: np.ndarray | None = None
        self._sample_rate: int | None = None
        self._duration: float = 0.0
        self._can_play: bool = False
        self._is_playing: bool = False  # Added state

        self._load_audio()

    def _load_audio(self):
        """Loads audio data and info from the WAV file."""
        try:
            # Get info first for duration
            info = sf.info(str(self.wav_path))
            self._duration = info.duration
            self._sample_rate = info.samplerate
            logger.debug(
                f'Player loading {self.wav_path.name} ({self._duration:.2f}s, {self._sample_rate} Hz)',
            )

            # Read data
            self._playback_data, sr_read = sf.read(self.wav_path, dtype='float32', always_2d=True)

            if sr_read != self._sample_rate:
                logger.warning(
                    f'Player: File info SR ({self._sample_rate}) != read SR ({sr_read}). Using {sr_read}.',
                )
                self._sample_rate = sr_read  # Trust the read sample rate

            self._can_play = True
            logger.debug(f'Player: Audio data loaded successfully ({self._playback_data.shape}).')

        except Exception as e:
            logger.exception(f'Player: Error reading audio file {self.wav_path.name}')
            self._can_play = False
            self._exception = e  # Store load exception

    @property
    def duration(self) -> float:
        """Returns the duration of the audio file in seconds."""
        return self._duration

    @property
    def can_play(self) -> bool:
        """Returns True if the audio file was loaded successfully."""
        return self._can_play

    @property
    def is_playing(self) -> bool:
        """Checks if the playback thread is active."""
        return self._is_playing and self._thread is not None and self._thread.is_alive()

    def start(self):
        """Starts playback in a background thread."""
        if not self._can_play or self._playback_data is None or self._sample_rate is None:
            logger.error(f'Player: Cannot start playback, audio not loaded for {self.wav_path.name}')
            return  # Or raise error

        if self.is_playing:
            logger.error(f'Player: Playback already in progress for {self.wav_path.name}')
            raise RuntimeError('Playback already in progress.')

        self._exception = None  # Clear previous exceptions
        self._is_playing = True  # Set state
        self._thread = threading.Thread(
            target=self._play_task,
            daemon=True,
            name=f'Play-{self.wav_path.stem}',
        )
        self._thread.start()
        logger.info(f'Player: Started playback of {self.wav_path.name} on default output.')

    def join(self, timeout: float | None = None):
        """Waits for the playback thread to complete."""
        if self._thread is not None:
            logger.debug(f'Player: Joining playback thread for {self.wav_path.name}')
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(
                    f'Player: Playback thread join timed out or did not complete for {self.wav_path.name}',
                )
            else:
                logger.debug(f'Player: Playback thread joined for {self.wav_path.name}')
        else:
            logger.debug(f'Player: Join called but no active thread for {self.wav_path.name}')

        # Re-raise exception if one occurred during playback
        if self._exception:
            # Avoid re-raising load errors if join is called later
            # We might only want to raise playback errors here
            logger.error(
                f'Player: Exception occurred during playback for {self.wav_path.name}: {self._exception}',
            )
            # raise self._exception

    def _play_task(self):
        """Task executed by the playback thread."""
        if self._playback_data is None or self._sample_rate is None:
            return  # Should not happen if start checks work

        try:
            # sd.play blocks the calling thread (this worker thread)
            sd.play(self._playback_data, self._sample_rate, blocking=True)
            logger.debug(f'Player: sd.play completed for {self.wav_path.name}')
        except sd.PortAudioError as pae:
            logger.exception('Player: PortAudioError during playback on default device')
            self._exception = pae
        except Exception as e:
            logger.exception(f'Player: Error during playback execution for {self.wav_path.name}')
            self._exception = e
        finally:
            self._is_playing = False  # Update state when task finishes or errors out
            logger.debug(f'Player: Playback task finished for {self.wav_path.name}')
