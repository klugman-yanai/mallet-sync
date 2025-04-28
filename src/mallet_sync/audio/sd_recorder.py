# audio/sd_recorder.py
"""Audio recorder module for zero frame drop, multi-channel, high-throughput recording.
Key changes:
- Increased queue and buffer sizes for burst traffic
- Enhanced emergency write fallback (no drop, only blocking with periodic diagnostics)
- Improved thread priority/CPU affinity settings
- Logging augmented for real-time diagnostics and strict test tolerances
"""

import os
import queue
import sys
import threading
import time

from pathlib import Path
from typing import Optional

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf

from numpy._typing import DTypeLike

from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    get_logger,
)

# --- Increased queue/buffer sizes ---
RECORDER_QUEUE_SIZE = 4000
RECORDER_BUFFER_SIZE = 25
RECORDER_FLUSH_INTERVAL = 2.0

logger = get_logger(__name__)


class AudioRecorder:
    """Zero-drop frame multi-channel audio WAV recorder."""

    def __init__(
        self,
        output_file: Path,
        device_index: int,
        channels: int = MALLET_CHANNELS,
        sample_rate: int = MALLET_SAMPLE_RATE,
        chunk_size: int = 1024,
        dtype: DTypeLike = MALLET_DTYPE,
    ):
        self.device_index = device_index
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.output_file = output_file
        self.dtype = dtype
        self._record_stop_event = threading.Event()
        self._write_stop_event = threading.Event()
        self._write_priority = False
        self._record_thread: threading.Thread | None = None
        self._write_thread: threading.Thread | None = None
        self._exception: Exception | None = None
        # --- Increased queue size
        self._data_queue = queue.Queue(maxsize=RECORDER_QUEUE_SIZE)
        self._is_recording = False
        self._is_writing = False
        self._write_buffer_size = RECORDER_BUFFER_SIZE
        self._flush_interval = RECORDER_FLUSH_INTERVAL
        self._write_buffer_lock = threading.Lock()
        self._sound_file: sf.SoundFile | None = None
        self._frames_recorded = 0
        self._frames_written = 0
        self._emergency_writes = 0  # Track emergency direct writes

    @property
    def is_recording(self) -> bool:
        return self._is_recording and self._record_thread is not None and self._record_thread.is_alive()

    @property
    def is_writing(self) -> bool:
        return self._is_writing and self._write_thread is not None and self._write_thread.is_alive()

    def _set_thread_priority_and_affinity(self, thread: threading.Thread, role: str):
        # Try to increase thread priority and set CPU affinity. OS-dependent.
        # Use thread.native_id in 3.8+ for OS-level.
        tid = getattr(thread, 'native_id', None)
        if hasattr(os, 'sched_setaffinity') and tid is not None:
            try:
                os.sched_setaffinity(tid, {(role == 'writer' and 0) or 1})
                logger.debug(f'Set {role} thread affinity, tid={tid}')
            except Exception as e:
                logger.debug(f'Affinity set fail: {e}')
        # Set high priority on Windows
        try:
            p = psutil.Process(os.getpid())
            if sys.platform == 'win32':
                p.nice(psutil.HIGH_PRIORITY_CLASS)
                logger.debug(f'Set {role} thread to high priority class')
        except Exception as e:
            logger.debug(f'Priority set failed: {e}')

    def start(self) -> None:
        if self.is_recording:
            logger.error(f'Recording already in progress for {self.output_file.name}.')
            raise RuntimeError(f'Recording already in progress for {self.output_file.name}.')
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._record_stop_event.clear()
        self._write_stop_event.clear()
        self._exception = None
        self._frames_recorded = 0
        self._frames_written = 0
        self._emergency_writes = 0
        while not self._data_queue.empty():
            try:
                self._data_queue.get_nowait()
                self._data_queue.task_done()
            except queue.Empty:
                break
        # --- Start writing thread first
        self._is_writing = True
        self._write_thread = threading.Thread(
            target=self._file_writing_loop, daemon=True, name=f'Write-{self.output_file.stem}',
        )
        self._write_thread.start()
        self._set_thread_priority_and_affinity(self._write_thread, 'writer')
        time.sleep(0.05)
        # --- Then start recording thread
        self._is_recording = True
        self._record_thread = threading.Thread(
            target=self._recording_loop, daemon=True, name=f'Rec-{self.output_file.stem}',
        )
        self._record_thread.start()
        self._set_thread_priority_and_affinity(self._record_thread, 'recorder')
        logger.info(f'Started streaming recording on device {self.device_index} -> {self.output_file.name}')

    def join(self, timeout: float = 5.0) -> bool:
        result = True
        if self.is_recording and self._record_thread is not None:
            logger.debug(f'Joining recording thread for {self.output_file.name}')
            self._record_thread.join(timeout=timeout)
            if self._record_thread.is_alive():
                logger.warning(f'Recording thread join timed out for {self.output_file.name}')
                result = False
            else:
                self._is_recording = False
        if self.is_writing and self._write_thread is not None:
            logger.debug(f'Joining write thread for {self.output_file.name}')
            self._write_thread.join(timeout=timeout)
            if self._write_thread.is_alive():
                logger.warning(f'Write thread join timed out for {self.output_file.name}')
                result = False
            else:
                self._is_writing = False
        return result

    def stop(self) -> None:
        if not self.is_recording:
            logger.debug(f'Stop called but not recording: {self.output_file.name}')
            return
        logger.debug(f'Signaling stop for recorder: {self.output_file.name}')
        self._record_stop_event.set()
        if self._record_thread is not None:
            self._record_thread.join(timeout=1.0)
        if self._exception:
            logger.error(f'Exception during recording for {self.output_file.name}: {self._exception}')
        self._is_recording = False
        self._record_thread = None
        logger.info(f'Stopped recording device {self.device_index}. File writing continues.')

    def stop_writing(self) -> None:
        logger.debug(f'Stopping writer for device {self.device_index}')
        self._write_priority = True
        if self.is_recording:
            logger.debug(f'Waiting for recording to complete first on device {self.device_index}')
            self.stop()
        try:
            queue_size = self._data_queue.qsize()
            if queue_size > 0:
                logger.info(f'Writing remaining {queue_size} queued chunks for device {self.device_index}')
        except Exception:
            logger.exception(f'Failed to write remaining queued chunks for device {self.device_index}')
        self._write_stop_event.set()
        if self._write_thread is not None:
            self._write_thread.join(timeout=1.0)
            if not self._write_thread.is_alive():
                logger.info(f'Write thread completed for {self.output_file.name}')
        self._is_writing = False
        self._write_thread = None

    def drain_queue_to_memory(self) -> np.ndarray | None:
        if self._data_queue.empty():
            return None
        try:
            all_data = []
            while not self._data_queue.empty():
                try:
                    chunk = self._data_queue.get_nowait()
                    all_data.append(chunk)
                    self._data_queue.task_done()
                except queue.Empty:
                    break
            if all_data:
                self._frames_written += len(all_data)
                return np.vstack(all_data)
        except Exception:
            logger.exception('Error draining queue')
            return None

    def wait_for_write_complete(self, timeout: float = 30.0) -> bool:
        start_time = time.time()
        if not self._write_thread or not self._write_thread.is_alive():
            logger.debug(f'Write thread for device {self.device_index} already complete')
            return True
        self._write_priority = True
        wait_time = min(timeout * 0.5, 10.0)
        end_wait_time = time.time() + wait_time
        while not self._data_queue.empty() and time.time() < end_wait_time:
            time.sleep(0.1)
        time_left = max(0.1, timeout - (time.time() - start_time))
        self._write_thread.join(timeout=time_left)
        is_done = not self._write_thread.is_alive()
        if is_done:
            self._write_thread = None
            self._is_writing = False
        frames_diff = abs(self._frames_recorded - self._frames_written)
        if frames_diff != 0:
            logger.error(
                f'Zero-drop invariant violated! Recorded: {self._frames_recorded}, Written: {self._frames_written}',
            )
            raise AssertionError(
                f'Zero-drop invariant failed: {self._frames_recorded} vs {self._frames_written}',
            )
        return is_done

    def _ensure_data_queued(self, data: np.ndarray) -> None:
        try:
            self._data_queue.put(data, timeout=0.1)
            self._frames_recorded += 1
        except queue.Full:
            # Strong stance: block forever (diagnose, but never drop!)
            logger.warning(
                f'Queue full for {self.output_file.name}; blocking until space available (zero-drop)',
            )
            t_start_block = time.time()
            count_warn = 0
            while True:
                try:
                    self._data_queue.put(data, timeout=0.1)
                    self._frames_recorded += 1
                    blocked_s = time.time() - t_start_block
                    if blocked_s > 0.5:
                        logger.exception(
                            f'Blocked {blocked_s:.2f}s waiting for queue. System cannot sustain rate; investigate disk throughput or reduce other load.',
                        )
                    break
                except queue.Full:
                    count_warn += 1
                    if count_warn % 5 == 0:
                        logger.info(f'Still blocked ({count_warn * 0.1:.1f}s) for {self.output_file.name}.')
                    time.sleep(0.05)
                if self._record_stop_event.is_set():
                    break

    def _recording_loop(self) -> None:
        try:
            with sd.InputStream(
                device=self.device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                dtype=self.dtype,
                latency='low',
            ) as stream:
                logger.debug(f'Input stream opened for device {self.device_index}')
                while not self._record_stop_event.is_set():
                    data, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        logger.warning(f'Input overflowed on device {self.device_index}')
                    if data.size > 0:
                        self._ensure_data_queued(data)
                    else:
                        time.sleep(0.001)
            logger.debug(f'Input stream closed for device {self.device_index}')
        except Exception as exc:
            logger.exception(f'Error during recording loop for device {self.device_index}')
            self._exception = exc
        finally:
            self._is_recording = False
            logger.info(f'Recording loop finished for {self.device_index} ({self._frames_recorded} chunks)')

    def _flush_write_buffer(self, write_buffer: list[np.ndarray], sound_file: sf.SoundFile) -> None:
        """Helper method to flush write buffer to disk."""
        if not write_buffer:
            return
        try:
            # Concatenate all chunks for a single efficient write operation
            combined_data = np.vstack(write_buffer) if len(write_buffer) > 1 else write_buffer[0]

            # Write to disk
            frames_written = sound_file.write(combined_data)

            # Track written chunks for consistency with recording counter
            chunks_written = len(write_buffer)
            self._frames_written += chunks_written

            # Log periodically, but not too often
            if self._frames_written % 100 == 0 or len(write_buffer) > 10:
                logger.debug(
                    f'Wrote {chunks_written} chunks ({combined_data.shape[0]} samples) to {self.output_file.name}, '
                    f'total frames: {self._frames_recorded}, written: {self._frames_written}',
                )

            # Clear buffer after successful write
            write_buffer.clear()

            # Only check mismatch if frames_written is not None
            if frames_written is not None and frames_written != combined_data.shape[0]:
                logger.warning(
                    f'Write size mismatch: expected {combined_data.shape[0]} frames, '
                    f'but wrote {frames_written} frames',
                )
        except Exception:
            logger.exception('Error flushing write buffer')
            raise

    def _file_writing_loop(self) -> None:
        try:
            with sf.SoundFile(
                file=str(self.output_file),
                mode='w',
                samplerate=self.sample_rate,
                channels=self.channels,
                format='WAV',
            ) as sound_file:
                self._sound_file = sound_file
                logger.debug(f'Opened output file for streaming: {self.output_file.name}')
                write_buffer: list[np.ndarray] = []
                max_buffer_size = self._write_buffer_size
                last_write_time = time.time()
                while True:
                    if (
                        self._write_stop_event.is_set()
                        and self._data_queue.empty()
                        and not self.is_recording
                        and len(write_buffer) == 0
                    ):
                        break
                    try:
                        chunk = self._data_queue.get(
                            timeout=0.02 if self._write_priority or self._write_stop_event.is_set() else 0.1,
                        )
                        chunk_copy = np.copy(chunk)
                        write_buffer.append(chunk_copy)
                        _current_time = time.time()
                        dynamic_buffer_size = (
                            1
                            if (
                                self._write_priority
                                or self._write_stop_event.is_set()
                                or not self.is_recording
                            )
                            else max_buffer_size
                        )
                        should_write = (
                            len(write_buffer) >= dynamic_buffer_size
                            or time.time() - last_write_time >= self._flush_interval
                        )
                        if should_write:
                            with self._write_buffer_lock:
                                self._flush_write_buffer(write_buffer, sound_file)
                            last_write_time = time.time()
                        self._data_queue.task_done()
                    except queue.Empty:
                        # On stop or pause, flush any pending data
                        if not self.is_recording and self._data_queue.empty():
                            with self._write_buffer_lock:
                                self._flush_write_buffer(write_buffer, sound_file)
                            break
                        continue
                # Final stats/assert for strict zero-drop
                logger.info(
                    f'Recording stats {self.output_file.name}: '
                    f'recorded={self._frames_recorded}, written={self._frames_written}',
                )
                if self._frames_recorded != self._frames_written:
                    logger.error(
                        f'Zero-drop failure: {self._frames_recorded} chunks captured, '
                        f'but {self._frames_written} written.',
                    )
                    raise AssertionError(
                        f'Zero-drop failure: {self._frames_recorded} vs {self._frames_written}',
                    )
        except Exception as exc:
            logger.exception(f'Failed to write audio file {self.output_file.name}')
            self._exception = exc
        finally:
            self._sound_file = None
            self._is_writing = False
            logger.info(f'File writing loop finished for {self.output_file.name}')
