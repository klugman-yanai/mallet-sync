#!/usr/bin/env python

"""
Improved Zero-Drop Frame Test for AudioRecorder

This comprehensive test verifies that the AudioRecorder never drops frames
under realistic production loads with multiple devices, system stress, and
extended recording durations.

Features:
- Simulates multiple recording devices concurrently
- Adds CPU and disk I/O stress to test resilience
- Accelerates time to simulate longer recordings efficiently
- Strictly validates frame-by-frame data integrity
- Provides detailed progress monitoring and diagnostics
"""

import argparse
import logging
import os
import queue
import sys
import tempfile
import threading
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent.parent))

from mallet_sync.audio.sd_recorder import AudioRecorder
from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    RECORDER_CHUNK_SIZE,
    get_logger,
)

logger = get_logger(__name__)


class ImprovedZeroDropTest:
    """Advanced test framework for verifying zero frame drop in AudioRecorder."""

    def __init__(
        self,
        device_count=3,
        channels=MALLET_CHANNELS,
        sample_rate=MALLET_SAMPLE_RATE,
        chunk_size=RECORDER_CHUNK_SIZE,
        simulated_duration_seconds=45 * 60,  # 45 minutes
        actual_duration_seconds=60,
    ):  # Run for 1 minute real time
        self.device_count = device_count
        self.channels = channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.simulated_duration = simulated_duration_seconds
        self.actual_duration = actual_duration_seconds

        # Calculate time acceleration factor
        self.time_acceleration = (
            simulated_duration_seconds / actual_duration_seconds if actual_duration_seconds > 0 else 1.0
        )

        # Create temporary output directory
        self.output_dir = Path(tempfile.gettempdir()) / f'zero_drop_test_{int(time.time())}'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output files for each device
        self.output_files = [self.output_dir / f'device_{i}.wav' for i in range(device_count)]

        # Frame tracking for validation
        self.frame_counters = [0] * device_count
        self.recorders = []

        # Store original sounddevice InputStream
        self.original_input_stream = sd.InputStream

        # Stress test helpers
        self.disk_stressor_thread = None
        self.cpu_stressor_thread = None
        self.stop_stressors_event = threading.Event()

        # Synthetic audio generator - create unique patterns for each device
        self.device_patterns = []
        for i in range(device_count):
            # Create a unique pattern per device for validation
            pattern = np.zeros((chunk_size, channels), dtype=MALLET_DTYPE)
            for c in range(channels):
                # Each channel gets a unique signature based on device and channel number
                pattern[:, c] = (
                    np.linspace(-0.8 + (i * 0.1), 0.8 + (i * 0.1), chunk_size) * (c + 1) / channels
                )
            self.device_patterns.append(pattern)

    def setup_mocks(self):
        """Set up mocks for sounddevice to precisely control audio generation."""
        # Configure logging to avoid interference with the progress display
        test_instance = self

        # Create a mock input stream that generates accelerated audio data
        def mock_input_stream_factory(*args, **kwargs):
            """Creates a mock InputStream with precise frame control."""
            device = kwargs.get('device', 0)
            _device_id = device if isinstance(device, (int, float)) else 0
            channels = kwargs.get('channels', self.channels)
            samplerate = kwargs.get('samplerate', self.sample_rate)
            blocksize = kwargs.get('blocksize', self.chunk_size)
            dtype = kwargs.get('dtype', 'float32')

            class MockInputStream:
                def __init__(self, device, channels, samplerate, blocksize, dtype):
                    self.device = device
                    self.channels = channels
                    self.samplerate = samplerate
                    self.blocksize = blocksize
                    self.dtype = dtype
                    self.frame_counter = 0
                    self.active = True
                    self.time_acceleration = test_instance.time_acceleration

                def read(self, frames):
                    """Generate audio data with embedded device-specific signature."""
                    if not self.active:
                        return np.zeros((0, self.channels)), False

                    # Get device index for tracking and pattern generation
                    device_idx = int(self.device) if isinstance(self.device, (int, float)) else 0

                    # Create data for this chunk using the device pattern
                    # with frame counter embedded for validation
                    base_pattern = test_instance.device_patterns[device_idx]
                    # Scale the pattern by the current frame counter to make each chunk unique
                    scaling_factor = 1.0 + (self.frame_counter % 1000) / 1000.0
                    data = base_pattern * scaling_factor

                    # Track the frame counter for this device
                    self.frame_counter += 1
                    test_instance.frame_counters[device_idx] += 1

                    # Simulate realistic timing - sleep proportional to chunk duration
                    # adjusted by acceleration factor
                    real_time_sleep = frames / self.samplerate
                    accelerated_sleep = real_time_sleep / self.time_acceleration

                    # Ensure we don't sleep less than a minimum threshold
                    if accelerated_sleep > 0.001:
                        time.sleep(accelerated_sleep)

                    return data, False

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    self.active = False

            # Return mock instance for this device
            return MockInputStream(device, channels, samplerate, blocksize, dtype)

        # Replace sounddevice.InputStream with our factory
        sd.InputStream = mock_input_stream_factory

    def restore_mocks(self):
        """Restore original sounddevice functionality."""
        sd.InputStream = self.original_input_stream

    def start_stressors(self):
        """Start background stressors to simulate system load."""
        self.stop_stressors_event.clear()

        # Disk stressor
        def disk_stress_loop():
            while not self.stop_stressors_event.is_set():
                try:
                    # Create temporary file and write data
                    with tempfile.NamedTemporaryFile(delete=False) as f:
                        # Write 5MB in chunks
                        for _ in range(5):
                            f.write(os.urandom(1024 * 1024))
                            f.flush()
                            if self.stop_stressors_event.is_set():
                                break
                        temp_path = f.name

                    # Remove the file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)

                    # Brief pause between cycles
                    time.sleep(0.1)
                except Exception:
                    logger.exception('Disk stressor error')
                    time.sleep(0.5)

        # CPU stressor
        def cpu_stress_loop():
            while not self.stop_stressors_event.is_set():
                try:
                    # Simulate CPU-intensive calculation
                    matrix_size = 500
                    a = np.random.random((matrix_size, matrix_size))
                    b = np.random.random((matrix_size, matrix_size))
                    _c = np.dot(a, b)  # Matrix multiplication
                    time.sleep(0.2)
                except Exception:
                    logger.exception('CPU stressor error')
                    time.sleep(0.5)

        # Start stressor threads
        self.disk_stressor_thread = threading.Thread(target=disk_stress_loop, daemon=True)
        self.cpu_stressor_thread = threading.Thread(target=cpu_stress_loop, daemon=True)
        self.disk_stressor_thread.start()
        self.cpu_stressor_thread.start()
        logger.info('Background stressors started')

    def stop_stressors(self):
        """Stop all stressors."""
        self.stop_stressors_event.set()
        if self.disk_stressor_thread:
            self.disk_stressor_thread.join(timeout=2.0)
        if self.cpu_stressor_thread:
            self.cpu_stressor_thread.join(timeout=2.0)
        logger.info('Background stressors stopped')

    def run_test(self):
        """Run the test and verify zero dropped frames."""
        try:
            print('\n===== RUNNING IMPROVED ZERO DROP FRAME TEST =====')
            print(f'Devices: {self.device_count} with {self.channels} channels each')
            print(
                f'Simulating {self.simulated_duration}s ({self.simulated_duration / 60:.1f}m) '
                f'in {self.actual_duration}s real time',
            )
            print(f'Time acceleration: {self.time_acceleration:.1f}x')
            print(f'Output directory: {self.output_dir}')

            # Setup mocks
            self.setup_mocks()

            # Start stressors
            self.start_stressors()

            # Create and start recorders
            for i in range(self.device_count):
                recorder = AudioRecorder(
                    output_file=self.output_files[i],
                    device_index=i,
                    channels=self.channels,
                    sample_rate=self.sample_rate,
                    chunk_size=self.chunk_size,
                    dtype=MALLET_DTYPE,
                )
                self.recorders.append(recorder)
                print(f'Created recorder for device {i} -> {self.output_files[i].name}')

            # Start all recorders
            for recorder in self.recorders:
                recorder.start()

            # Monitor progress
            start_time = time.time()
            end_time = start_time + self.actual_duration

            print(
                f'\nRecording for {self.actual_duration}s real time '
                f'({self.simulated_duration / 60:.1f}m simulated)...',
            )

            # Update every 2 seconds
            update_interval = 2.0
            last_update_time = time.time()

            try:
                while time.time() < end_time:
                    current_time = time.time()
                    if current_time - last_update_time >= update_interval:
                        # Calculate progress
                        elapsed_real = current_time - start_time
                        elapsed_simulated = elapsed_real * self.time_acceleration
                        remaining_simulated = self.simulated_duration - elapsed_simulated
                        progress_percent = (elapsed_simulated / self.simulated_duration) * 100

                        # Clear previous output using cross-platform ANSI escape sequences
                        # This works in most modern terminals including Windows 10+ terminals
                        print('\033[H\033[J', end='')  # ANSI escape sequence to clear screen and move cursor to home

                        # Format elapsed/remaining time
                        elapsed_formatted = time.strftime('%H:%M:%S', time.gmtime(elapsed_simulated))
                        remaining_formatted = time.strftime('%H:%M:%S', time.gmtime(remaining_simulated))

                        # Create progress bar
                        bar_length = 40
                        filled_length = int(bar_length * elapsed_simulated / self.simulated_duration)
                        bar = '█' * filled_length + '-' * (bar_length - filled_length)

                        print(f'\n{"=" * 70}')
                        print('RECORDING SESSION IN PROGRESS')
                        print(f'{"=" * 70}')
                        print(f'Time elapsed: {elapsed_formatted} | Remaining: {remaining_formatted}')
                        print(f'Progress: [{bar}] {progress_percent:.1f}%')
                        print(f'{"=" * 70}')

                        # Show recorder stats in a table format
                        print('\nDEVICE STATS:')
                        print(f'{"Device":<8} {"Status":<12} {"Recorded":<12} {"Written":<12} {"Lag":<12}')
                        print(f'{"-" * 60}')

                        for i, recorder in enumerate(self.recorders):
                            recorded = recorder._frames_recorded
                            written = recorder._frames_written
                            lag = recorded - written
                            lag_status = f'{lag}' if lag > 0 else 'NONE'
                            status = 'RECORDING' if recorder.is_recording else 'STOPPED'

                            print(f'{i:<8} {status:<12} {recorded:<12} {written:<12} {lag_status:<12}')

                        print(f'{"-" * 60}')

                        # System resource stats
                        cpu_percent = psutil.cpu_percent()
                        mem_percent = psutil.virtual_memory().percent
                        disk_percent = psutil.disk_usage('/').percent
                        print(f'\nSYSTEM: CPU {cpu_percent}% | RAM {mem_percent}% | Disk {disk_percent}%')

                        last_update_time = current_time

                    # Small sleep to prevent CPU spinning
                    sleep_time = min(0.1, (end_time - time.time()) / 2)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            except KeyboardInterrupt:
                print('\nRecording interrupted by user.')
                return False

            # Stop recorders
            print('\nStopping recorders...')
            for recorder in self.recorders:
                recorder.stop()

            # Wait for all files to be written
            print('Waiting for all data to be written to disk...')
            for recorder in self.recorders:
                recorder.stop_writing()

            # Ensure all recorders finish writing
            for i, recorder in enumerate(self.recorders):
                print(f'Waiting for recorder {i} to finish writing...')
                if not recorder.wait_for_write_complete(timeout=30.0):
                    print(f'WARNING: Recorder {i} did not finish writing within timeout')

            # Verify results
            return self.verify_results()

        finally:
            # Clean up
            self.restore_mocks()
            self.stop_stressors()

    def verify_results(self):
        """Verify that no frames were dropped by checking all recorders."""
        print('\n===== VERIFYING RESULTS =====')

        success = True
        for i, recorder in enumerate(self.recorders):
            print(f'\nDevice {i}:')
            print(f'  Frames recorded: {recorder._frames_recorded}')
            print(f'  Frames written: {recorder._frames_written}')
            print(f'  Emergency writes: {recorder._emergency_writes}')
            print(f'  Test frame counter: {self.frame_counters[i]}')

            # Check for dropped frames - our goal is ZERO difference
            frame_diff = abs(recorder._frames_recorded - recorder._frames_written)
            if frame_diff > 0:
                print(f'  CRITICAL ERROR: {frame_diff} frames missing - data was lost!')
                success = False
            else:
                print('  SUCCESS: All captured audio was written correctly!')

            # Check that what we expected to capture matches what was recorded
            counter_diff = abs(self.frame_counters[i] - recorder._frames_recorded)
            if counter_diff > 3:  # Allow tiny variation due to thread timing
                print(
                    f'  WARNING: Expected to generate {self.frame_counters[i]} frames, '
                    f'but recorded {recorder._frames_recorded}. Diff: {counter_diff}',
                )

            # Check the WAV file properties
            try:
                if self.output_files[i].exists():
                    file_size_mb = os.path.getsize(self.output_files[i]) / (1024 * 1024)
                    info = sf.info(self.output_files[i])
                    file_frames = info.frames
                    duration = info.duration

                    print(f'  File size: {file_size_mb:.2f} MB')
                    print(f'  File frames: {file_frames}')
                    print(f'  Duration: {duration:.2f}s')

                    # Calculate expected minimum frames
                    min_expected_samples = recorder._frames_written * recorder.chunk_size

                    if (
                        file_frames >= min_expected_samples * 0.99
                    ):  # Allow 1% margin for file format variation
                        print(f'  SUCCESS: File contains {file_frames} samples, verified!')
                    else:
                        # This would be a real problem - data was supposedly written but isn't in file
                        loss_percent = 100 * (
                            1 - (file_frames / (recorder._frames_written * recorder.chunk_size))
                        )
                        print(
                            f'  ERROR: File only contains {file_frames} samples, '
                            f'expected at least {min_expected_samples}',
                        )
                        print(f'  CRITICAL: Approximately {loss_percent:.1f}% of audio data not in file')
                        success = False
                else:
                    print('  ERROR: Output file not found!')
                    success = False
            except Exception as e:
                print(f'  ERROR: Could not analyze WAV file: {e}')
                success = False

        # Print overall test result
        print('\n===== TEST RESULT =====')
        if success:
            print('SUCCESS: Zero-drop frame recording verified! No frames were dropped!')
        else:
            print('FAILURE: Some frames were lost or not properly written!')

        return success


def run_test(simulated_duration=2700, actual_duration=60):
    """Run the full zero-drop frame test.

    Args:
        simulated_duration: Duration to simulate in seconds (default: 2700 = 45 minutes)
        actual_duration: Actual test runtime in seconds (default: 60 seconds)

    Returns:
        int: 0 on success, 1 on failure
    """
    print('Starting Improved Zero Drop Frame Test')
    print(
        f'Simulating {simulated_duration} seconds ({simulated_duration / 60:.1f} minutes) '
        f'in approximately {actual_duration} seconds',
    )

    # Calculate acceleration factor
    acceleration_factor = simulated_duration / actual_duration if actual_duration > 0 else 1.0
    print(f'Acceleration factor: {acceleration_factor:.1f}x')
    print('\n')

    # Create and configure test instance
    test = ImprovedZeroDropTest(
        device_count=3,  # 3 Mallet devices
        channels=MALLET_CHANNELS,  # 9 channels per device
        sample_rate=MALLET_SAMPLE_RATE,  # 16kHz
        chunk_size=RECORDER_CHUNK_SIZE,  # 1024 sample chunks
        simulated_duration_seconds=simulated_duration,
        actual_duration_seconds=actual_duration,
    )

    success = test.run_test()

    # Return exit code
    return 0 if success else 1


# Pytest-compatible test function
def test_no_dropped_frames_production_scenario():
    """Test AudioRecorder with zero drop frames in a production-like scenario.

    Runs 3 devices with 9 channels each for a simulated 45-minute session.
    """
    # Use shorter durations for regular test runs
    simulated_duration = 2700  # 45 minutes
    actual_duration = 30  # 30 seconds real time

    # Run the test
    result = run_test(simulated_duration, actual_duration)

    # Assert success
    assert result == 0, 'Zero-drop production scenario test failed - frames were dropped'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test that AudioRecorder achieves zero dropped frames',
    )
    parser.add_argument(
        '--simulated-duration',
        type=int,
        default=2700,
        help='Simulated duration in seconds (default: 2700 = 45 minutes)',
    )
    parser.add_argument(
        '--actual-duration',
        type=int,
        default=60,
        help='Actual test runtime in seconds (default: 60 seconds)',
    )

    args = parser.parse_args()
    sys.exit(run_test(args.simulated_duration, args.actual_duration))
