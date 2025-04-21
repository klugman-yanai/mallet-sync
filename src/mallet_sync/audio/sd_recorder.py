from __future__ import annotations

import threading

from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from mallet_sync.config import DeviceConfig


class SoundDeviceStreamRecorder:
    """
    Records audio from a specified device to a WAV file.
    Threading is an implementation detail: start() launches background recording,
    stop() signals termination, join() waits for completion. Synchronous record() is also provided.

    Args:
        device_index: int - Index of the audio input device.
        channels: int - Number of channels to record.
        sample_rate: int - Sample rate in Hz.
        chunk_size: int - Frames per buffer.
        output_file: Path - Path to the WAV file to write.
        dtype: str - Numpy dtype string (default: 'int16').
        duration: float | None - Maximum duration to record (seconds). If None, records until stopped.
    """

    def __init__(
        self,
        *,
        device_index: int,
        channels: int,
        sample_rate: int,
        chunk_size: int,
        output_file: Path,
        dtype: str = 'int16',
        duration: float | None = None,
    ) -> None:
        """
        Initialize from explicit parameters. Prefer using from_device_config for clarity.
        """
        self.device_index: int = device_index
        self.channels: int = channels
        self.sample_rate: int = sample_rate
        self.chunk_size: int = chunk_size
        self.output_file: Path = output_file
        self.dtype: str = dtype
        self.duration: float | None = duration
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None
        self._exception: Exception | None = None
        self._is_recording: bool = False

    @classmethod
    def from_device_config(
        cls,
        device_config: DeviceConfig,
        output_file: Path,
        *,
        dtype: str = 'int16',
        duration: float | None = None,
    ) -> SoundDeviceStreamRecorder:
        """
        Construct a recorder from a DeviceConfig instance.
        """
        return cls(
            device_index=device_config.index,
            channels=device_config.max_input_channels,
            sample_rate=device_config.sample_rate,
            chunk_size=device_config.chunk_size,
            output_file=output_file,
            dtype=dtype,
            duration=duration,
        )

    def start(self) -> None:
        if self._is_recording:
            raise RuntimeError('Recording already in progress.')
        self._stop_event.clear()
        self._exception = None
        self._thread = threading.Thread(target=self._record, daemon=True)
        self._is_recording = True
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self._is_recording = False
        if self._exception:
            raise self._exception

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
        if self._exception:
            raise self._exception

    def is_recording(self) -> bool:
        return self._is_recording and self._thread is not None and self._thread.is_alive()

    def record(self) -> None:
        """
        Synchronous (blocking) recording. Returns when finished or stopped.
        """
        self._stop_event.clear()
        self._exception = None
        self._is_recording = True
        try:
            self._record()
        finally:
            self._is_recording = False
        if self._exception:
            raise self._exception

    def _record(self) -> None:
        frames: list[np.ndarray] = []
        total_frames: int = 0
        max_frames: int | None = (
            int(self.sample_rate * self.duration) if self.duration is not None else None
        )
        try:
            with sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
            ) as stream:
                while not self._stop_event.is_set():
                    data = stream.read(self.chunk_size)[0]
                    frames.append(data.copy())
                    total_frames += len(data)
                    if max_frames is not None and total_frames >= max_frames:
                        break
            self._write_audio(frames)
        except Exception as exc:
            self._exception = exc
            raise

    def _write_audio(self, frames: list[np.ndarray]) -> None:
        """
        Write recorded frames to an audio file using soundfile (modern, robust).
        Supports WAV, FLAC, OGG, etc. based on file extension.
        """
        if not frames:
            raise RuntimeError('No audio frames recorded.')
        arr = np.concatenate(frames, axis=0)
        sf.write(
            file=str(self.output_file),
            data=arr,
            samplerate=self.sample_rate,
            subtype=None,  # auto-select
            format=None,  # auto from extension
        )

    @staticmethod
    def record_audio(
        *,
        device_index: int,
        channels: int,
        sample_rate: int,
        chunk_size: int,
        output_file: Path,
        dtype: str = 'int16',
        duration: float | None = None,
    ) -> None:
        """
        Convenience static method to record audio synchronously.
        """
        recorder = SoundDeviceStreamRecorder(
            device_index=device_index,
            channels=channels,
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            output_file=output_file,
            dtype=dtype,
            duration=duration,
        )
        recorder.start()
        recorder.join()
        if recorder._exception:
            raise recorder._exception
