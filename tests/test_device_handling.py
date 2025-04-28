#!/usr/bin/env python
"""
Test for verifying the improved device detection and validation functionality.

This test focuses on ensuring our improvements to device handling, error reporting,
and multi-device orchestration work correctly.
"""

import os
import sys
import tempfile

from pathlib import Path

import numpy as np
import pytest
import sounddevice as sd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mallet_sync.config import MALLET_CHANNELS, DeviceInfo
from mallet_sync.utils import (
    _get_device_info,
    find_mallet_devices,
    validate_device_availability,
)

# Mock production-like devices for testing - matching real-world mallet setup
MOCK_DEVICES = [
    # Standard Kardome Mallet device with 9 channels
    DeviceInfo(
        name='Kardome Mallet 9-Channel (USB Audio)',
        index=0,
        hostapi=0,
        max_input_channels=9,
        max_output_channels=0,
        default_samplerate=16000.0,
        supported_samplerates=[8000, 16000, 44100],
    ),
    # Another Mallet device with different naming pattern
    DeviceInfo(
        name='KT Array Mic (9-ch)',
        index=1,
        hostapi=0,
        max_input_channels=9,
        max_output_channels=0,
        default_samplerate=16000.0,
        supported_samplerates=[8000, 16000, 44100],
    ),
    # Third Mallet device with another variant of naming
    DeviceInfo(
        name='Mallet Array USB Audio Device',
        index=2,
        hostapi=0,
        max_input_channels=9,
        max_output_channels=0,
        default_samplerate=16000.0,
        supported_samplerates=[8000, 16000, 44100],
    ),
    # Non-Mallet device that might confuse matching (has "mallet" in name but wrong channels)
    DeviceInfo(
        name='Some Mallet-like Device',
        index=3,
        hostapi=0,
        max_input_channels=2,  # Not enough channels to be a real Mallet
        max_output_channels=2,
        default_samplerate=44100.0,
        supported_samplerates=[44100],
    ),
    # Output device (MAYA44)
    DeviceInfo(
        name='MAYA44 USB Audio Device',
        index=4,
        hostapi=0,
        max_input_channels=0,
        max_output_channels=4,
        default_samplerate=44100.0,
        supported_samplerates=[44100, 48000],
    ),
    # Generic other device that shouldn't be detected
    DeviceInfo(
        name='Built-in Audio',
        index=5,
        hostapi=0,
        max_input_channels=2,
        max_output_channels=2,
        default_samplerate=44100.0,
        supported_samplerates=[44100, 48000],
    ),
]


class MockQuery:
    """Mock for sounddevice.query_devices that returns our test devices."""

    def __init__(self, devices):
        self.devices = devices

    def __call__(self, *args, **kwargs):
        if 'index' in kwargs or args:
            idx = kwargs.get('index', args[0] if args else 0)
            for dev in self.devices:
                if dev.index == idx:
                    return {
                        'name': dev.name,
                        'index': dev.index,
                        'hostapi': dev.hostapi,
                        'max_input_channels': dev.max_input_channels,
                        'max_output_channels': dev.max_output_channels,
                        'default_samplerate': dev.default_samplerate,
                        'supported_samplerates': dev.supported_samplerates,
                    }
            raise ValueError(f'No device with index {idx}')
        return [
            {
                'name': dev.name,
                'index': dev.index,
                'hostapi': dev.hostapi,
                'max_input_channels': dev.max_input_channels,
                'max_output_channels': dev.max_output_channels,
                'default_samplerate': dev.default_samplerate,
                'supported_samplerates': dev.supported_samplerates,
            }
            for dev in self.devices
        ]


@pytest.fixture
def mock_sounddevice():
    """Set up and tear down the sounddevice mocks."""
    original_query = sd.query_devices
    sd.query_devices = MockQuery(MOCK_DEVICES)

    yield

    sd.query_devices = original_query


def test_device_info_extraction(mock_sounddevice):
    """Test extracting device info with different device structures."""
    # Test Kardome device
    device_info = _get_device_info(0)
    assert device_info is not None
    assert device_info.name == 'Kardome Mallet 9-Channel (USB Audio)'
    assert device_info.max_input_channels == 9

    # Test KT device
    device_info = _get_device_info(1)
    assert device_info is not None
    assert device_info.name == 'KT Array Mic (9-ch)'

    # Test output device (MAYA44)
    device_info = _get_device_info(4)
    assert device_info is not None
    assert device_info.name == 'MAYA44 USB Audio Device'
    assert device_info.max_output_channels == 4


def test_find_mallet_devices(mock_sounddevice):
    """Test finding Mallet devices with the improved logic."""
    mallet_devices = find_mallet_devices()

    # Should find exactly 3 mallet devices (all valid mallet devices)
    assert len(mallet_devices) == 3

    # Devices should be sorted by index
    assert mallet_devices[0][0].index == 0
    assert mallet_devices[1][0].index == 1
    assert mallet_devices[2][0].index == 2

    # Check that devices were properly assigned roles according to the actual MALLET_ROLES order
    # MALLET_ROLES is defined as ['hmtc', 'wired', 'main'] in config.py
    assert mallet_devices[0][1] == "hmtc"   # First role from MALLET_ROLES
    assert mallet_devices[1][1] == "wired"  # Second role
    assert mallet_devices[2][1] == "main"   # Third role

    # Verify the device with "mallet" in name but wrong channels was NOT included
    for device, _ in mallet_devices:
        assert device.index != 3  # The "Some Mallet-like Device" should be excluded


def test_device_validation(mock_sounddevice):
    """Test device availability validation with realistic production device setup."""
    # Create a list of device tuples similar to what find_mallet_devices returns
    # Using the three valid Mallet devices
    devices = [(device, f'role_{i}') for i, device in enumerate(MOCK_DEVICES[:3])]

    # Validation should pass since these devices are available
    assert validate_device_availability(devices) is True

    # Test with an empty device list
    assert validate_device_availability([]) is False

    # Test with a non-existent device index
    bad_device = DeviceInfo(
        name='Non-existent Mallet Device',
        index=99,  # This index doesn't exist in our mock
        hostapi=0,
        max_input_channels=9,
        max_output_channels=0,
        default_samplerate=16000.0,
        supported_samplerates=[16000],
    )

    # Test with a mix of valid and invalid devices
    mixed_devices = [(MOCK_DEVICES[0], 'main'), (bad_device, 'bad')]
    assert validate_device_availability(mixed_devices) is False

    # Test with a device that has insufficient channels
    insufficient_device = MOCK_DEVICES[3]  # The "Some Mallet-like Device" with only 2 channels
    assert insufficient_device.max_input_channels < MALLET_CHANNELS  # Verify our test setup
    assert validate_device_availability([(insufficient_device, 'insufficient')]) is False


if __name__ == '__main__':
    # Run the tests manually
    pytest.main(['-xvs', __file__])
