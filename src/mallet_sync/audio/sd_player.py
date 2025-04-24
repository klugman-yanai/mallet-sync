# audio/player.py
import threading
import time

from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from mallet_sync.config import get_logger

logger = get_logger(__name__)


class AudioPlayer:
    """
    Plays an audio file on the default output device using a streaming approach.

    Uses chunk-by-chunk streaming to avoid missing frames and efficiently handle
    large audio files without loading the entire file into memory.
    """

    def __init__(self, wav_path: Path, chunk_size: int = 1024):
        self.wav_path = wav_path
        self.chunk_size = chunk_size

        # Thread control
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._exception: Exception | None = None
        self._is_playing: bool = False

        # Audio properties
        self._duration: float = 0.0
        self._sample_rate: int = 0
        self._channels: int = 0
        self._can_play: bool = False

        # Initialize by checking file validity
        self._check_audio_file()

    def _check_audio_file(self) -> None:
        """Checks if the audio file exists and gets its basic properties."""
        try:
            if not self.wav_path.exists() or not self.wav_path.is_file():
                logger.error(f'Player: Audio file not found: {self.wav_path}')
                self._can_play = False
                return

            # Get audio file info without loading the entire file
            info = sf.info(str(self.wav_path))
            self._duration = info.duration
            self._sample_rate = info.samplerate
            self._channels = info.channels
            self._can_play = True

            logger.debug(
                f'Player: Audio file validated: {self.wav_path.name} '
                f'({self._duration:.2f}s, {self._sample_rate} Hz, {self._channels} channels)',
            )
        except Exception as e:
            logger.exception(f'Player: Error checking audio file {self.wav_path.name}')
            self._can_play = False
            self._exception = e

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

    def start(self) -> None:
        """
        Starts playback in a background thread using chunk-by-chunk streaming.

        Raises:
            RuntimeError: If playback is already in progress or audio failed to load
        """
        if not self._can_play:
            msg = f'Cannot start playback, audio not valid for {self.wav_path.name}'
            logger.error(f'Player: {msg}')
            raise RuntimeError(msg)

        if self.is_playing:
            msg = f'Playback already in progress for {self.wav_path.name}'
            logger.error(f'Player: {msg}')
            raise RuntimeError(msg)

        # Reset state
        self._stop_event.clear()
        self._exception = None
        self._is_playing = True

        # Start streaming thread
        self._thread = threading.Thread(
            target=self._streaming_playback_task,
            daemon=True,
            name=f'Play-{self.wav_path.stem}',
        )
        self._thread.start()
        logger.info(f'Player: Started streaming playback of {self.wav_path.name}')

    def stop(self) -> None:
        """
        Signals the playback thread to stop.

        Unlike join(), this method returns immediately without waiting for the thread.
        """
        if not self.is_playing:
            return

        logger.debug(f'Player: Signaling stop for {self.wav_path.name}')
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """
        Waits for the playback thread to complete.

        Args:
            timeout: Maximum time to wait for thread completion in seconds
        """
        if self._thread is None:
            logger.debug(f'Player: Join called but no active thread for {self.wav_path.name}')
            return

        logger.debug(f'Player: Joining playback thread for {self.wav_path.name}')
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.warning(
                f'Player: Playback thread join timed out for {self.wav_path.name}',
            )
        else:
            logger.debug(f'Player: Playback thread joined for {self.wav_path.name}')

        # Log exception if one occurred during playback
        if self._exception:
            logger.error(
                f'Player: Exception during playback for {self.wav_path.name}: {self._exception}',
            )

    def _streaming_playback_task(self) -> None:
        """
        Task executed by the playback thread. Streams audio chunk by chunk
        instead of loading the entire file into memory.
        """
        try:
            # Open the sound file for streaming
            with sf.SoundFile(self.wav_path) as sound_file, sd.OutputStream(
                samplerate=sound_file.samplerate,
                channels=sound_file.channels,
                blocksize=self.chunk_size,
                dtype='float32',
            ) as stream:
                logger.debug(f'Player: Opened output stream for {self.wav_path.name}')

                # Read and play chunks until end of file or stop requested
                while not self._stop_event.is_set():
                    # Read the next chunk (returns empty array at end of file)
                    data = sound_file.read(frames=self.chunk_size, dtype='float32')

                    # Check if we reached the end of the file
                    if not data.size:
                        logger.debug(f'Player: End of file reached for {self.wav_path.name}')
                        break

                    # Write this chunk to the output stream
                    stream.write(data)
                logger.debug(f'Player: Streaming playback completed for {self.wav_path.name}')

        except sf.SoundFileError as sfe:
            logger.exception(f'Player: SoundFile error for {self.wav_path.name}')
            self._exception = sfe
        except sd.PortAudioError as pae:
            logger.exception('Player: PortAudio error during playback')
            self._exception = pae
        except Exception as e:
            logger.exception(f'Player: Unexpected error during playback for {self.wav_path.name}')
            self._exception = e
        finally:
            self._is_playing = False
            logger.debug(f'Player: Streaming playback task exited for {self.wav_path.name}')
