#!/usr/bin/env python
"""
Multi-device recorder resilience test specifically designed to validate
the system under the user's actual recording conditions:
- 3 USB microphones with multiple channels each
- 45-minute recording sessions
- Simulated disk slowdowns that match real-world conditions

This test includes both fast simulation mode and optional full-duration testing.
"""

import gc
import os
import queue
import random
import sys
import tempfile
import threading
import time

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import the recorder class before mocking sounddevice
# We'll need to mock sounddevice to avoid actual device access
import sounddevice as sd

from mallet_sync.audio.sd_recorder import AudioRecorder
from mallet_sync.config import (
    MALLET_CHANNELS,
    MALLET_DTYPE,
    MALLET_SAMPLE_RATE,
    RECORDER_CHUNK_SIZE,
)


# Create mock audio generator classes for our tests
class MockInputStream:
    """
    Mock implementation of sounddevice.InputStream that generates
    synthetic audio data with embedded sequence numbers.
    """

    def __init__(
        self,
        device=None,
        channels=None,
        samplerate=None,
        callback=None,
        blocksize=None,
        dtype=None,
        **kwargs,
    ):
        self.device = device
        self.channels = channels or MALLET_CHANNELS
        self.samplerate = samplerate or MALLET_SAMPLE_RATE
        self.callback = callback
        self.blocksize = blocksize or RECORDER_CHUNK_SIZE
        self.dtype = dtype or MALLET_DTYPE
        self.active = False
        self.closed = False
        self.thread = None
        self.sequence_number = 0
        self.stop_event = threading.Event()
        self.frequency = 440  # Hz (A4)
        self.time_index = 0

        # Used to simulate device issues
        self.slowdown_factor = 1.0
        self.error_mode = None

    def start(self):
        """Start the stream"""
        if self.active:
            return

        self.active = True
        self.thread = threading.Thread(target=self._generate_audio, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the stream"""
        self.active = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def close(self):
        """Close the stream"""
        self.stop()
        self.closed = True

    def read(self, frames):
        """Read from the stream (only used in blocking mode)"""
        if not self.active:
            return np.zeros((0, self.channels)), False

        # Create synthetic audio
        data = self._create_synthetic_audio(frames)
        return data, False  # No overflow

    def _generate_audio(self):
        """Generate audio in the background thread"""
        while self.active and not self.stop_event.is_set():
            # Simulate device slowdowns if configured
            if self.slowdown_factor > 1.0:
                time.sleep((1.0 / self.samplerate) * self.blocksize * self.slowdown_factor)

            # Simulate errors if configured
            overflowed = self.error_mode == 'overflow'

            # Generate synthetic audio chunk
            data = self._create_synthetic_audio(self.blocksize)

            # Call the callback with the generated data
            if self.callback:
                self.callback(data, overflowed, None, None)

            # Normal delay between chunks
            time.sleep((1.0 / self.samplerate) * self.blocksize * 0.90)  # Slightly faster than real-time

    def _create_synthetic_audio(self, frames):
        """Create synthetic audio with embedded sequence markers"""
        # Create synthetic audio with embedded sequence marker
        chunk = np.zeros((frames, self.channels), dtype=np.float32)

        # Generate time points for this chunk
        t = np.arange(self.time_index, self.time_index + frames) / self.samplerate
        self.time_index += frames

        # First channel: sine wave
        chunk[:, 0] = 0.5 * np.sin(2 * np.pi * self.frequency * t)

        # Second channel: embed sequence number (scaled to audio range)
        sequence_value = (self.sequence_number % 100) / 100.0  # Scale to 0-1 range
        chunk[:, 1 % self.channels] = sequence_value

        # Other channels: shaped noise
        for ch in range(2, self.channels):
            # Random noise with slight sequence-based amplitude modulation
            noise = np.random.randn(frames) * 0.1
            noise *= 0.5 + 0.5 * (self.sequence_number % 10) / 10
            chunk[:, ch] = noise

        # Convert to target dtype if not float32
        if self.dtype != np.float32:
            if self.dtype == np.int16:
                # Scale to int16 range
                chunk = (chunk * 32767).astype(np.int16)
            elif self.dtype == np.int32:
                chunk = (chunk * 2147483647).astype(np.int32)

        self.sequence_number += 1
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass
class DeviceConfig:
    """Configuration for a simulated recording device."""

    device_id: int
    name: str
    channels: int
    sample_rate: int

    def __str__(self) -> str:
        return f'Device {self.device_id}: {self.name} ({self.channels} channels @ {self.sample_rate}Hz)'


class SimulatedDiskSlowdown:
    """
    Realistic disk slowdown simulator based on observed behavior during recording sessions.

    This creates periodic slowdowns that match typical patterns seen during long recordings:
    - Random brief slowdowns (100-500ms) that occur frequently
    - Occasional longer slowdowns (1-3s) that occur less frequently
    - Rare severe slowdowns (5-10s) that might occur once or twice per session
    """

    def __init__(self):
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.active = False

        # Slowdown statistics
        self.brief_slowdowns = 0
        self.medium_slowdowns = 0
        self.severe_slowdowns = 0

        # Control speed of the test (1.0 = real time, higher = faster)
        self.time_acceleration = 1.0

    def start(self, time_acceleration: float = 1.0):
        """Start the disk slowdown simulation thread."""
        self.time_acceleration = time_acceleration
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._slowdown_loop,
            daemon=True,
            name='DiskSlowdown',
        )
        self.thread.start()

    def stop(self):
        """Stop the disk slowdown simulation thread."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=5.0)

    def _slowdown_loop(self):
        """Main slowdown simulation loop."""
        # Realistic slowdown pattern based on 45-minute session observations
        # Brief slowdowns (100-500ms): Every 20-40 seconds
        # Medium slowdowns (1-3s): Every 3-5 minutes
        # Severe slowdowns (5-10s): 1-2 times per 45-minute session

        brief_interval = 30.0 / self.time_acceleration  # Average seconds between brief slowdowns
        medium_interval = 240.0 / self.time_acceleration  # Average seconds between medium slowdowns
        severe_interval = 1350.0 / self.time_acceleration  # Average seconds between severe slowdowns

        brief_timer = 0
        medium_timer = 0
        severe_timer = 0

        while not self.stop_event.is_set():
            # Update timers
            sleep_time = 0.1 / self.time_acceleration
            time.sleep(sleep_time)

            brief_timer += sleep_time
            medium_timer += sleep_time
            severe_timer += sleep_time

            # Check for brief slowdown (100-500ms)
            if brief_timer >= brief_interval * random.uniform(0.8, 1.2):
                duration = random.uniform(0.1, 0.5) / self.time_acceleration
                self._simulate_slowdown(duration, 'brief')
                brief_timer = 0

            # Check for medium slowdown (1-3s)
            if medium_timer >= medium_interval * random.uniform(0.8, 1.2):
                duration = random.uniform(1.0, 3.0) / self.time_acceleration
                self._simulate_slowdown(duration, 'medium')
                medium_timer = 0

            # Check for severe slowdown (5-10s)
            if severe_timer >= severe_interval * random.uniform(0.8, 1.2):
                duration = random.uniform(5.0, 10.0) / self.time_acceleration
                self._simulate_slowdown(duration, 'severe')
                severe_timer = 0

    def _simulate_slowdown(self, duration: float, severity: str):
        """
        Simulate a disk slowdown of specified duration and severity.

        This works by temporarily blocking file operations across all threads.
        """
        if self.stop_event.is_set():
            return

        # Set active flag to indicate slowdown in progress
        self.active = True

        # Track statistics
        if severity == 'brief':
            self.brief_slowdowns += 1
        elif severity == 'medium':
            self.medium_slowdowns += 1
        elif severity == 'severe':
            self.severe_slowdowns += 1

        # Log slowdown for test visibility
        print(f'Simulating {severity} disk slowdown ({duration:.2f}s)')

        # Replace all filesystem calls with slower versions during the slowdown
        original_open = open
        original_write = os.write
        original_close = os.close

        # Create delayed versions of file operations
        def delayed_open(*args, **kwargs):
            time.sleep(min(0.1, duration * 0.1))  # Small delay on open
            return original_open(*args, **kwargs)

        def delayed_write(fd, data, *args, **kwargs):
            # Severe delays on write operations
            time.sleep(min(0.5, duration * 0.2))
            return original_write(fd, data, *args, **kwargs)

        def delayed_close(fd, *args, **kwargs):
            time.sleep(min(0.1, duration * 0.1))  # Small delay on close
            return original_close(fd, *args, **kwargs)

        try:
            # Apply the monkey patches to slow down file operations
            # We need to do this very carefully to not affect the test framework itself
            import builtins

            builtins.open = delayed_open
            os.write = delayed_write
            os.close = delayed_close

            # Wait for the duration of the slowdown
            slow_start = time.time()
            while (time.time() - slow_start < duration) and not self.stop_event.is_set():
                time.sleep(0.01)

        finally:
            # Restore original file operations
            builtins.open = original_open
            os.write = original_write
            os.close = original_close
            self.active = False


class MultiDeviceRecorderTest:
    """
    Test harness for evaluating the recorder's performance with multiple devices
    matching the user's actual recording environment.
    """

    def __init__(
        self,
        device_count: int = 3,
        test_duration_seconds: int = 2700,  # 45 minutes default
        output_dir: str | None = None,
        fast_mode: bool = True,
    ):
        self.device_count = device_count
        self.test_duration = test_duration_seconds
        self.fast_mode = fast_mode
        self.time_acceleration = 60.0 if fast_mode else 1.0  # 60x faster in fast mode

        # Setup output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(tempfile.gettempdir()) / 'multi_recorder_test'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this test run
        self.timestamp = time.strftime('%Y%m%d_%H%M%S')

        # Setup simulated devices with different channel configurations
        self.devices = [
            DeviceConfig(
                device_id=i,
                name=f'Simulated USB Mic {i + 1}',
                channels=9 if i == 0 else (6 if i == 1 else 4),  # Realistic channel configs
                sample_rate=MALLET_SAMPLE_RATE,
            )
            for i in range(device_count)
        ]

        # Create output files
        self.output_files = [
            self.output_dir / f'device_{i}_{self.timestamp}.wav' for i in range(device_count)
        ]

        # Recorders for each device
        self.recorders: list[AudioRecorder] = []

        # Disk slowdown simulator
        self.disk_slowdown = SimulatedDiskSlowdown()

        # Test statistics
        self.stats = {
            'success': False,
            'devices': [],
            'device_success_count': 0,
            'total_frames_recorded': 0,
            'total_frames_written': 0,
            'total_emergency_writes': 0,
            'slowdowns': {
                'brief': 0,
                'medium': 0,
                'severe': 0,
            },
            'errors': [],
        }

    def setup_mocks(self):
        """Set up mocks for sounddevice"""
        # Create a map of device_id to mock stream instances
        self.mock_streams = {}

        # Store original classes
        self.original_input_stream = sd.InputStream

        # Create a function to return our mock stream instead
        def mock_input_stream_factory(**kwargs):
            device_id = kwargs.get('device', 0)
            device_channels = next(
                (d.channels for d in self.devices if d.device_id == device_id),
                MALLET_CHANNELS,
            )

            # Set the correct channel count
            kwargs['channels'] = device_channels

            # Create a mock stream
            mock_stream = MockInputStream(**kwargs)
            self.mock_streams[device_id] = mock_stream
            return mock_stream

        # Apply the patch to replace sd.InputStream with our factory
        sd.InputStream = mock_input_stream_factory

    def restore_mocks(self):
        """Restore original implementations"""
        # Restore original implementations
        sd.InputStream = self.original_input_stream

    def run_test(self) -> dict[str, Any]:
        """
        Run a multi-device recording test that simulates the user's actual environment.

        In fast mode, this completes in approximately 45-75 seconds (simulating 45 minutes).
        In normal mode, this runs for the actual 45-minute duration.
        """
        try:
            # Set up mocks before starting test
            self.setup_mocks()
            # Print test configuration
            print(f'Starting multi-device recorder test ({self.device_count} devices)')
            print(f'Mode: {"FAST (simulated)" if self.fast_mode else "FULL DURATION"}')
            print(f'Simulated duration: {self.test_duration} seconds')
            print(f'Actual test time: ~{self.test_duration / self.time_acceleration:.1f} seconds')
            print(f'Output directory: {self.output_dir}')
            print('\nDevice configurations:')
            for device in self.devices:
                print(f'  {device}')

            # Initialize recorders for each device
            for i, device in enumerate(self.devices):
                recorder = AudioRecorder(
                    output_file=self.output_files[i],
                    device_index=device.device_id,
                    channels=device.channels,
                    sample_rate=device.sample_rate,
                )
                self.recorders.append(recorder)

                # Initialize device stats
                self.stats['devices'].append(
                    {
                        'device_id': device.device_id,
                        'channels': device.channels,
                        'frames_recorded': 0,
                        'frames_written': 0,
                        'emergency_writes': 0,
                        'success': False,
                        'error': None,
                    },
                )

            # Start disk slowdown simulator
            self.disk_slowdown.start(time_acceleration=self.time_acceleration)

            # Start all recorders
            print('\nStarting recorders...')
            for recorder in self.recorders:
                recorder.start()

            # Simulate recording for the specified duration
            start_time = time.time()
            expected_end_time = start_time + (self.test_duration / self.time_acceleration)

            # Monitor progress and periodically report status
            while time.time() < expected_end_time:
                # Check for any recorder exceptions
                for i, recorder in enumerate(self.recorders):
                    if recorder._exception:
                        self.stats['errors'].append(
                            f'Device {i} error: {recorder._exception}',
                        )

                # Calculate progress
                elapsed = time.time() - start_time
                simulated_elapsed = elapsed * self.time_acceleration
                remaining = expected_end_time - time.time()

                # Print status update
                print(
                    f'\nProgress: {simulated_elapsed:.1f}s / {self.test_duration}s '
                    f'(Actual: {elapsed:.1f}s, Remaining: {remaining:.1f}s)',
                )

                for i, recorder in enumerate(self.recorders):
                    queue_size = (
                        recorder._data_queue.qsize()
                        if hasattr(recorder, '_data_queue') and not recorder._data_queue.empty()
                        else 0
                    )

                    print(
                        f'  Device {i}: Queue={queue_size}, '
                        f'Recorded={recorder._frames_recorded}, '
                        f'Written={recorder._frames_written}, '
                        f'Emergency={recorder._emergency_writes}',
                    )

                # Simulate additional device stress during the recording
                if random.random() < 0.2:  # 20% chance each update
                    self._simulate_random_device_stress()

                # Sleep until next update
                time.sleep(min(5.0, remaining / 2))

            # Test completed, stop all recorders
            print('\nTest duration completed, stopping recorders...')
            for recorder in self.recorders:
                recorder.stop()
                recorder.stop_writing()

            # Stop disk slowdown simulator
            self.disk_slowdown.stop()

            # Wait for all recorders to finish writing
            print('Waiting for file writing to complete...')
            for recorder in self.recorders:
                recorder.wait_for_write_complete(timeout=10.0)

            # Collect final statistics and verify results
            return self._verify_results()

        except Exception as e:
            print(f'Test encountered an error: {e}')
            self.stats['errors'].append(str(e))
            self.stats['success'] = False
            return self.stats
        finally:
            # Cleanup
            self.disk_slowdown.stop()
            for recorder in self.recorders:
                if recorder.is_recording or recorder.is_writing:
                    try:
                        recorder.stop()
                        recorder.stop_writing()
                    except:
                        pass

            # Restore original implementations
            self.restore_mocks()

    def _simulate_random_device_stress(self):
        """
        Simulates random stressful conditions that might affect recording:
        - USB bus congestion
        - Occasional device stalls
        - High CPU usage spikes
        - Device buffer overflows
        """
        # Choose a random stress scenario
        scenario = random.choice(
            [
                'cpu_spike',
                'device_stall',
                'usb_congestion',
                'buffer_overflow',
            ],
        )

        if scenario == 'cpu_spike':
            # Simulate CPU spike by running intensive calculations
            print('Simulating CPU spike...')
            size = 1000
            for _ in range(3):
                a = np.random.random((size, size))
                b = np.random.random((size, size))
                c = np.dot(a, b)  # Computationally intensive

        elif scenario == 'device_stall':
            # Simulate a device briefly stalling by slowing down its mock stream
            device_idx = random.randint(0, len(self.recorders) - 1)
            device_id = self.devices[device_idx].device_id
            print(f'Simulating stall on device {device_idx}...')

            if device_id in self.mock_streams:
                # Slow down the stream temporarily
                self.mock_streams[device_id].slowdown_factor = 5.0

                # Reset after a brief period
                def restore_stream_speed():
                    time.sleep(0.5)
                    if device_id in self.mock_streams:
                        self.mock_streams[device_id].slowdown_factor = 1.0

                # Start restore thread
                threading.Thread(target=restore_stream_speed, daemon=True).start()

        elif scenario == 'buffer_overflow':
            # Simulate audio buffer overflow
            device_idx = random.randint(0, len(self.recorders) - 1)
            device_id = self.devices[device_idx].device_id
            print(f'Simulating buffer overflow on device {device_idx}...')

            if device_id in self.mock_streams:
                # Set error mode to overflow temporarily
                self.mock_streams[device_id].error_mode = 'overflow'

                # Reset after a brief period
                def restore_stream_normal():
                    time.sleep(0.3)
                    if device_id in self.mock_streams:
                        self.mock_streams[device_id].error_mode = None

                # Start restore thread
                threading.Thread(target=restore_stream_normal, daemon=True).start()

        elif scenario == 'usb_congestion':
            # Simulate USB bus congestion by creating large data transfers
            print('Simulating USB bus congestion...')
            large_data = [np.random.bytes(10 * 1024 * 1024) for _ in range(5)]
            time.sleep(0.3)  # Hold the data briefly
            del large_data
            gc.collect()

    def _verify_results(self) -> dict[str, Any]:
        """Verify that all recordings completed successfully without data loss."""
        # Update stats from disk slowdown simulator
        self.stats['slowdowns']['brief'] = self.disk_slowdown.brief_slowdowns
        self.stats['slowdowns']['medium'] = self.disk_slowdown.medium_slowdowns
        self.stats['slowdowns']['severe'] = self.disk_slowdown.severe_slowdowns

        # Check each recorder's final state
        for i, recorder in enumerate(self.recorders):
            device_stats = self.stats['devices'][i]

            # Update stats
            device_stats['frames_recorded'] = recorder._frames_recorded
            device_stats['frames_written'] = recorder._frames_written
            device_stats['emergency_writes'] = recorder._emergency_writes

            # Update totals
            self.stats['total_frames_recorded'] += recorder._frames_recorded
            self.stats['total_frames_written'] += recorder._frames_written
            self.stats['total_emergency_writes'] += recorder._emergency_writes

            # Check for successful recording (no data loss)
            if recorder._exception:
                device_stats['error'] = str(recorder._exception)
                device_stats['success'] = False
            else:
                # Check that frames_recorded ~= frames_written + emergency_writes
                # (Allowing small differences due to timing of stop events)
                diff = abs(
                    recorder._frames_recorded - (recorder._frames_written + recorder._emergency_writes),
                )
                if diff <= 5:  # Allow small differences (typically 1-2 frames at shutdown)
                    device_stats['success'] = True
                    self.stats['device_success_count'] += 1
                else:
                    device_stats['success'] = False
                    device_stats['error'] = f'Frame count mismatch: {diff} frames missing'

        # Overall success if all devices succeeded
        self.stats['success'] = self.stats['device_success_count'] == len(self.recorders)

        # Verify output files exist and have reasonable sizes
        for i, output_file in enumerate(self.output_files):
            if output_file.exists():
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                self.stats['devices'][i]['file_size_mb'] = file_size_mb

                # Verify file with soundfile if possible
                try:
                    with sf.SoundFile(output_file, 'r') as f:
                        self.stats['devices'][i]['duration_seconds'] = len(f) / f.samplerate
                        self.stats['devices'][i]['file_valid'] = True
                except Exception as e:
                    self.stats['devices'][i]['file_valid'] = False
                    self.stats['devices'][i]['file_error'] = str(e)
            else:
                self.stats['devices'][i]['file_exists'] = False
                self.stats['success'] = False

        # Print summary report
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print a summary of test results."""
        print('\n' + '=' * 50)
        print(f'TEST RESULTS: {"PASSED" if self.stats["success"] else "FAILED"}')
        print('=' * 50)

        print(f'\nSimulated duration: {self.test_duration}s ({self.test_duration / 60:.1f} minutes)')
        print(f'Devices tested: {len(self.recorders)}')
        print(f'Devices successful: {self.stats["device_success_count"]}/{len(self.recorders)}')

        print('\nRecording statistics:')
        print(f'  Total frames recorded: {self.stats["total_frames_recorded"]}')
        print(f'  Total frames written: {self.stats["total_frames_written"]}')
        print(f'  Total emergency writes: {self.stats["total_emergency_writes"]}')

        print('\nSimulated disk slowdowns:')
        print(f'  Brief (100-500ms): {self.stats["slowdowns"]["brief"]}')
        print(f'  Medium (1-3s): {self.stats["slowdowns"]["medium"]}')
        print(f'  Severe (5-10s): {self.stats["slowdowns"]["severe"]}')

        print('\nDevice Details:')
        for i, device_stats in enumerate(self.stats['devices']):
            print(f'\nDevice {i} ({self.devices[i].name}, {self.devices[i].channels} channels):')
            print(f'  Success: {device_stats["success"]}')
            print(f'  Frames recorded: {device_stats["frames_recorded"]}')
            print(f'  Frames written: {device_stats["frames_written"]}')
            print(f'  Emergency writes: {device_stats["emergency_writes"]}')

            if 'file_size_mb' in device_stats:
                print(f'  Output file size: {device_stats["file_size_mb"]:.2f} MB')

            if 'duration_seconds' in device_stats:
                print(f'  Output duration: {device_stats["duration_seconds"]:.2f}s')

            if not device_stats['success'] and 'error' in device_stats and device_stats['error']:
                print(f'  Error: {device_stats["error"]}')

        if self.stats['errors']:
            print('\nTest Errors:')
            for error in self.stats['errors']:
                print(f'  - {error}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test multi-device recorder resilience')
    parser.add_argument(
        '--mode',
        choices=['fast', 'full'],
        default='fast',
        help='Test mode: fast (simulated acceleration) or full (real-time duration)',
    )
    parser.add_argument(
        '--devices',
        type=int,
        default=3,
        help='Number of recording devices to simulate (default: 3)',
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=2700,
        help='Simulated recording duration in seconds (default: 2700 = 45 minutes)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save test output (default: temp dir)',
    )

    args = parser.parse_args()

    fast_mode = args.mode == 'fast'

    print(f'Running {"FAST" if fast_mode else "FULL DURATION"} multi-device recorder test')
    print(f'Simulating {args.devices} devices for {args.duration}s ({args.duration / 60:.1f} minutes)')

    test = MultiDeviceRecorderTest(
        device_count=args.devices,
        test_duration_seconds=args.duration,
        output_dir=args.output_dir,
        fast_mode=fast_mode,
    )

    result = test.run_test()

    if result['success']:
        print('\nTEST PASSED - All recorders handled audio data without drops')
        sys.exit(0)
    else:
        print('\nTEST FAILED - One or more recorders failed to handle audio properly')
        sys.exit(1)
