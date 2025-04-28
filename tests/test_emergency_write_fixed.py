#!/usr/bin/env python
"""
Direct test of the emergency write functionality in AudioRecorder.

This test directly examines the emergency write path in the AudioRecorder
when the queue is full, verifying that the system never drops audio frames
under any circumstances - a critical requirement for the recording system.
"""

import os
import queue
import sys
import threading
import time

from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mallet_sync.audio.sd_recorder import AudioRecorder
from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    RECORDER_CHUNK_SIZE,
)


def test_emergency_write_path():
    """Verify the emergency write path in _ensure_data_queued works correctly."""
    print('\n===== TESTING EMERGENCY WRITE PATH =====')

    # Create a temporary output file with a unique name
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = Path(f'test_emergency_{timestamp}.wav')

    try:
        # Create a recorder with minimal initialization
        print(f'Creating test recorder with output file: {output_file}')
        recorder = AudioRecorder(
            output_file=output_file,
            device_index=0,
            channels=MALLET_CHANNELS,
            sample_rate=MALLET_SAMPLE_RATE,
        )

        # Initialize only what's needed for the test
        recorder._record_stop_event = threading.Event()
        recorder._frames_recorded = 0
        recorder._frames_written = 0
        recorder._emergency_writes = 0
        recorder._write_buffer_lock = threading.Lock()
        recorder._is_recording = True
        recorder._is_writing = True

        # Create a very small queue to make it easy to fill
        recorder._data_queue = queue.Queue(maxsize=3)
        print(f'Created test queue with capacity: {recorder._data_queue.maxsize}')

        # Create test data
        test_chunk = np.random.rand(RECORDER_CHUNK_SIZE, MALLET_CHANNELS).astype(MALLET_DTYPE)

        # Initialize mock sound file for emergency writes
        recorder._sound_file = mock.MagicMock()
        recorder._sound_file.write = mock.MagicMock(return_value=test_chunk.shape[0])
        print('Mock sound file created for emergency writes')

        # STEP 1: Fill the queue to capacity
        print('\nStep 1: Filling queue to capacity...')
        for i in range(recorder._data_queue.maxsize):
            recorder._data_queue.put(np.zeros_like(test_chunk))
            print(f'  Added item {i + 1} to queue')

        assert recorder._data_queue.full(), 'Queue should be full after adding items'
        print(f'Queue is now full: {recorder._data_queue.qsize()}/{recorder._data_queue.maxsize}')

        # STEP 2: Call _ensure_data_queued which should trigger emergency write
        print('\nStep 2: Testing _ensure_data_queued with full queue...')

        # Track stats before
        emergency_before = recorder._emergency_writes
        frames_before = recorder._frames_recorded
        writes_before = recorder._frames_written

        # Call the method that should trigger emergency write
        recorder._ensure_data_queued(test_chunk)

        # Track stats after
        emergency_after = recorder._emergency_writes
        frames_after = recorder._frames_recorded
        writes_after = recorder._frames_written

        # Verify the emergency write was triggered
        print(
            f'\nEmergency writes: {emergency_before} → {emergency_after} (delta: {emergency_after - emergency_before})'
        )
        print(f'Frames recorded: {frames_before} → {frames_after} (delta: {frames_after - frames_before})')
        print(f'Frames written: {writes_before} → {writes_after} (delta: {writes_after - writes_before})')

        # Verify sound file write was called
        print(f'Sound file write called: {recorder._sound_file.write.called}')
        if recorder._sound_file.write.called:
            print(f'Sound file write count: {recorder._sound_file.write.call_count}')

        # STEP 3: Test conclusion
        if emergency_after > emergency_before and recorder._sound_file.write.called:
            print('\n✅ TEST PASSED: Emergency write triggered correctly!')
            return True
        print('\n❌ TEST FAILED: Emergency write was not triggered as expected')
        return False

    finally:
        # Clean up the test file if it exists
        if output_file.exists():
            try:
                os.unlink(output_file)
            except:
                pass


def test_retry_mechanism():
    """Verify the retry mechanism works when both queue and emergency write fail."""
    print('\n===== TESTING RETRY MECHANISM WITH BACKOFF =====')

    # Create temporary output file
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = Path(f'test_retry_{timestamp}.wav')

    try:
        # Create recorder with minimal initialization
        recorder = AudioRecorder(
            output_file=output_file,
            device_index=0,
            channels=MALLET_CHANNELS,
            sample_rate=MALLET_SAMPLE_RATE,
        )

        # Initialize only what's needed for the test
        recorder._record_stop_event = threading.Event()
        recorder._frames_recorded = 0
        recorder._frames_written = 0
        recorder._emergency_writes = 0
        recorder._write_buffer_lock = threading.Lock()
        recorder._is_recording = True
        recorder._is_writing = True

        # Create a very small queue to make it easy to fill
        recorder._data_queue = queue.Queue(maxsize=3)
        print(f'Created test queue with capacity: {recorder._data_queue.maxsize}')

        # Fill the queue completely
        test_chunk = np.random.rand(RECORDER_CHUNK_SIZE, MALLET_CHANNELS).astype(MALLET_DTYPE)
        for i in range(recorder._data_queue.maxsize):
            recorder._data_queue.put(np.zeros_like(test_chunk))

        # Verify queue is full
        assert recorder._data_queue.full(), 'Queue should be full'
        print(f'Queue filled to capacity: {recorder._data_queue.qsize()}/{recorder._data_queue.maxsize}')

        # Set sound_file to None to force retry path
        recorder._sound_file = None
        print('Sound file set to None to disable emergency write')

        # Create a consumer thread that will empty one slot after a delay
        def delayed_consumer():
            print('Consumer thread starting, will create space after delay...')
            time.sleep(0.5)  # Wait before consuming
            try:
                item = recorder._data_queue.get(block=False)
                print(
                    f'Consumer removed one item, queue size: {recorder._data_queue.qsize()}/{recorder._data_queue.maxsize}'
                )
            except queue.Empty:
                print('Consumer found empty queue (unexpected)')

        # Start the consumer
        consumer = threading.Thread(target=delayed_consumer, daemon=True)
        consumer.start()

        # Call _ensure_data_queued which should block then retry
        print('\nCalling _ensure_data_queued with full queue and no sound file...')
        print('This should trigger retry with backoff mechanism...')

        frames_before = recorder._frames_recorded
        start_time = time.time()

        # This call should block until the consumer frees a slot
        recorder._ensure_data_queued(test_chunk)

        elapsed = time.time() - start_time
        frames_after = recorder._frames_recorded

        print(f'\nRetry mechanism took {elapsed:.2f} seconds')
        print(f'Frames recorded: {frames_before} → {frames_after}')

        # The method should have blocked until the consumer freed space
        if elapsed >= 0.5 and frames_after > frames_before:
            print('\n✅ TEST PASSED: Retry mechanism worked correctly!')
            return True
        print('\n❌ TEST FAILED: Retry mechanism did not work as expected')
        return False

    finally:
        # Clean up
        if output_file.exists():
            try:
                os.unlink(output_file)
            except:
                pass


if __name__ == '__main__':
    # Run the tests
    emergency_test = test_emergency_write_path()

    # Only run the retry test if the emergency test passes
    if emergency_test:
        retry_test = test_retry_mechanism()
        success = emergency_test and retry_test
    else:
        success = False

    # Exit with status based on results
    sys.exit(0 if success else 1)
