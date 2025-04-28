#!/usr/bin/env python
"""
Direct test for AudioRecorder's _ensure_data_queued method.

This targeted test directly examines the behavior of the critical method that
handles queue full conditions and emergency writes, focusing specifically on:

1. Standard queue operation (baseline)
2. Queue full -> emergency write path
3. Queue full + emergency write fail -> retry with backoff

This test is deliberately minimal and focused on this core functionality.
"""

import queue
import sys
import threading
import time

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
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


class TestEnsureDataQueued:
    """
    Test suite focused on the core _ensure_data_queued method to verify
    queue backpressure, emergency writing, and retry mechanisms.
    """

    def setup_method(self):
        """Set up test environment before each test method."""
        # Create a temporary output file
        self.output_file = Path('test_output.wav')

        # Create test recorder but don't start it
        self.recorder = AudioRecorder(
            output_file=self.output_file,
            device_index=0,
            channels=MALLET_CHANNELS,
        )

        # Mock the _recording_loop and _file_writing_loop methods
        # so we can directly test just the _ensure_data_queued method
        self.recorder._recording_loop = mock.MagicMock()
        self.recorder._file_writing_loop = mock.MagicMock()

        # Create a controlled queue for testing
        self.recorder._data_queue = queue.Queue(maxsize=10)  # Small size for easier testing

        # Create test data
        self.test_chunk = np.random.rand(RECORDER_CHUNK_SIZE, MALLET_CHANNELS).astype(MALLET_DTYPE)

        # Track statistics
        self.recorder._frames_recorded = 0
        self.recorder._frames_written = 0
        self.recorder._emergency_writes = 0

    def teardown_method(self):
        """Clean up after each test method."""
        if self.output_file.exists():
            self.output_file.unlink()

    def test_normal_queue_operation(self):
        """Verify normal queue operation when there's space available."""
        # Queue has space (it's empty)
        self.recorder._ensure_data_queued(self.test_chunk)

        # Verify it was added to queue
        assert self.recorder._data_queue.qsize() == 1
        assert self.recorder._frames_recorded == 1
        assert self.recorder._emergency_writes == 0

        # Verify the chunk in the queue matches our test chunk
        queued_chunk = self.recorder._data_queue.get()
        assert queued_chunk.shape == self.test_chunk.shape

    def test_emergency_write_when_queue_full(self):
        """Verify emergency write happens when queue is full and sound_file is available."""
        # Create a mock sound_file
        mock_sound_file = mock.MagicMock()
        self.recorder._sound_file = mock_sound_file

        # Fill the queue completely
        for _ in range(self.recorder._data_queue.maxsize):
            self.recorder._data_queue.put(np.zeros_like(self.test_chunk))

        # Verify queue is full
        assert self.recorder._data_queue.full()

        # Now try to queue more data, which should trigger emergency write
        self.recorder._ensure_data_queued(self.test_chunk)

        # Verify emergency write was triggered
        assert mock_sound_file.write.called
        assert self.recorder._emergency_writes == 1
        assert self.recorder._frames_recorded == 1  # Should still count the frame

    def test_retry_when_queue_full_and_no_sound_file(self):
        """Verify retry mechanism when queue is full and emergency write is not possible."""
        # Ensure no sound_file is available
        self.recorder._sound_file = None

        # Fill the queue
        for _ in range(self.recorder._data_queue.maxsize):
            self.recorder._data_queue.put(np.zeros_like(self.test_chunk))

        # Verify queue is full
        assert self.recorder._data_queue.full()

        # Set up a consumer to empty the queue after a delay
        def delayed_consumer():
            time.sleep(0.5)  # Wait before consuming
            try:
                # Empty one slot
                self.recorder._data_queue.get(block=False)
            except queue.Empty:
                pass

        # Start the consumer thread
        consumer_thread = threading.Thread(target=delayed_consumer)
        consumer_thread.daemon = True
        consumer_thread.start()

        # Start measuring time
        start_time = time.time()

        # Try to queue data - this should block until the consumer creates space
        self.recorder._ensure_data_queued(self.test_chunk)

        # Calculate how long it took
        elapsed = time.time() - start_time

        # Verify timing indicates retry with backoff (should be >= 0.5s from consumer delay)
        assert elapsed >= 0.5, f'Retry completed too quickly: {elapsed:.2f}s'

        # Verify the data was eventually queued
        assert self.recorder._frames_recorded == 1
        assert self.recorder._emergency_writes == 0

    def test_hybrid_direct_queue_implementation(self):
        """
        Test the actual implementation of _ensure_data_queued more directly.

        This test bypasses mocking and examines the code's ability to:
        1. Try queue first
        2. Fall back to emergency write
        3. Use retry as last resort
        """
        # Create a recorder instance
        recorder = AudioRecorder(
            output_file=self.output_file,
            device_index=0,
            channels=MALLET_CHANNELS,
        )

        # Initialize the critical fields but don't start threads
        recorder._record_stop_event = threading.Event()
        recorder._write_buffer_lock = threading.Lock()
        recorder._is_writing = True
        recorder._data_queue = queue.Queue(maxsize=5)  # Very small queue for testing

        # Patch the sound file directly
        recorder._sound_file = mock.MagicMock()
        recorder._sound_file.write = mock.MagicMock(return_value=RECORDER_CHUNK_SIZE)

        # Test data
        test_chunk = np.random.rand(RECORDER_CHUNK_SIZE, MALLET_CHANNELS).astype(MALLET_DTYPE)

        # Phase 1: Normal queue operation
        for i in range(3):
            recorder._ensure_data_queued(test_chunk.copy())

        assert recorder._frames_recorded == 3
        assert recorder._emergency_writes == 0
        assert recorder._data_queue.qsize() == 3

        # Phase 2: Fill queue to trigger emergency write
        for i in range(2):  # Fill remaining 2 slots
            recorder._ensure_data_queued(test_chunk.copy())

        assert recorder._data_queue.full()
        assert recorder._frames_recorded == 5
        assert recorder._emergency_writes == 0

        # Phase 3: Queue full, should trigger emergency write
        recorder._ensure_data_queued(test_chunk.copy())

        assert recorder._frames_recorded == 6
        assert recorder._emergency_writes == 1
        assert recorder._sound_file.write.call_count == 1

        # Phase 4: Block emergency write and test retry
        # Replace sound_file.write with one that fails first then succeeds
        fail_count = [2]  # Fail twice then succeed

        original_write = recorder._sound_file.write

        def failing_write(*args, **kwargs):
            if fail_count[0] > 0:
                fail_count[0] -= 1
                raise RuntimeError('Simulated write failure')
            return original_write(*args, **kwargs)

        recorder._sound_file.write = failing_write

        # Start a thread to consume from queue after delay
        def delayed_consumer():
            time.sleep(0.7)  # Wait before consuming
            try:
                recorder._data_queue.get(block=False)  # Make space in queue
            except queue.Empty:
                pass

        consumer = threading.Thread(target=delayed_consumer)
        consumer.daemon = True
        consumer.start()

        # Attempt to queue data - should try emergency write, fail, then retry queue
        start_time = time.time()
        recorder._ensure_data_queued(test_chunk.copy())
        elapsed = time.time() - start_time

        # Verify behavior
        assert elapsed >= 0.7, f'Retry completed too quickly: {elapsed:.2f}s'
        assert recorder._frames_recorded == 7
        assert fail_count[0] == 0  # Should have attempted emergency write twice

        # Clean up
        if Path('test_output.wav').exists():
            Path('test_output.wav').unlink()


if __name__ == '__main__':
    # Run the tests
    test = TestEnsureDataQueued()

    print('Testing _ensure_data_queued method directly')

    # Setup
    test.setup_method()

    try:
        # Run tests
        print('\nTest 1: Normal Queue Operation')
        test.test_normal_queue_operation()
        print('✓ Normal queue operation works correctly')

        # Reset for next test
        test.setup_method()
        print('\nTest 2: Emergency Write When Queue Full')
        test.test_emergency_write_when_queue_full()
        print('✓ Emergency write triggered correctly when queue full')

        # Reset for next test
        test.setup_method()
        print('\nTest 3: Retry When Queue Full And No Sound File')
        test.test_retry_when_queue_full_and_no_sound_file()
        print('✓ Retry mechanism works correctly')

        # This test uses its own instance
        print('\nTest 4: Hybrid Direct Implementation Test')
        test.test_hybrid_direct_queue_implementation()
        print('✓ Full implementation handles all cases correctly')

        print('\nAll tests PASSED!')
        sys.exit(0)

    except AssertionError as e:
        print(f'❌ Test failed: {e}')
        sys.exit(1)
    finally:
        # Clean up
        test.teardown_method()
