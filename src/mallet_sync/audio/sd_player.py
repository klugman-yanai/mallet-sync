# src/mallet_sync/sd_player.py

import threading
import time

import sounddevice as sd
import soundfile as sf


class SoundDeviceAudioPlayer:
    """
    Simulates playing a WAV file for timing purposes.
    Does NOT actually play audio through speakers, just handles
    timing synchronization for recording workflows.
    """

    def __init__(self, wav_path: str, *, blocking: bool = True) -> None:
        self.wav_path: str = wav_path
        self.blocking: bool = blocking
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None
        self._is_playing: bool = False
        self._duration: float = 0.0

        # Load WAV info but don't open a stream
        try:
            info = sf.info(self.wav_path)
            self._duration = info.duration
            self._sample_rate = info.samplerate
            self._frames = info.frames
        except Exception as e:
            raise RuntimeError(f'Failed to read WAV file info: {e}') from e

    def start(self) -> None:
        if self._is_playing:
            raise RuntimeError('Playback already in progress.')
        self._stop_event.clear()
        self._exception = None
        self._thread = threading.Thread(target=self._simulate_playback, daemon=True)
        self._is_playing = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._is_playing = False
        if self._exception:
            raise self._exception

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
        if self._exception:
            raise self._exception

    def play(self) -> None:
        """Simulate playback synchronously (blocks until done)."""
        self._stop_event.clear()
        self._exception = None
        self._is_playing = True
        try:
            self._simulate_playback()
        finally:
            self._is_playing = False

    def _simulate_playback(self) -> None:
        """
        Reads a WAV file chunk by chunk, simulating real playback timing
        without sending to the audio device.
        """
        try:
            # Get file info
            info = sf.info(self.wav_path)
            sample_rate = info.samplerate
            _channels = info.channels
            frames = info.frames

            # Set up chunk parameters
            chunk_size = 1024  # frames per chunk
            _chunk_duration = chunk_size / sample_rate  # seconds per chunk

            # Process file in chunks
            with sf.SoundFile(self.wav_path, 'r') as sound_file:
                # Iterate through the file in chunks
                for block_frames in range(0, frames, chunk_size):
                    if self._stop_event.is_set():
                        break

                    # Read a chunk of data (we don't need the data, just simulating read)
                    frames_to_read = min(chunk_size, frames - block_frames)
                    data = sound_file.read(frames=frames_to_read)

                    # Calculate actual duration based on frames read
                    actual_chunk_duration = len(data) / sample_rate

                    # Sleep for the amount of time this chunk would take to play
                    time.sleep(actual_chunk_duration)

        except Exception as exc:
            self._exception = exc
            raise
        finally:
            self._is_playing = False
