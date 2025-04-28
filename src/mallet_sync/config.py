# config.py
import logging

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from standard_logger import StandardLogger, setup_logging
from standard_logger.logger import LoggerConfig

# --- Constants ---
INPUT_AUDIO_DIR = Path('./recordings')  # Define relative to project root
OUTPUT_BASE_DIR = Path('./output')  # Also make output relative to root
MALLET_KEYWORDS = ('kardome', 'mallet', 'kt')
MALLET_SAMPLE_RATE = 16000
MALLET_CHANNELS = 9
MALLET_DTYPE = 'int16'
RECORDER_CHUNK_SIZE = 1024
FILENAME_TEMPLATE = 'mallet_{role}_{context}.wav'
SLEEP_TIME_SEC = 3
NUM_MALLETS = 3
MALLET_ROLES = ['hmtc', 'wired', 'main']

# --- Audio Recorder Configuration ---
# Queue size for audio data (frames) - increased for 9-channel audio
RECORDER_QUEUE_SIZE = 2000
# Number of frames to accumulate before writing to disk
RECORDER_BUFFER_SIZE = 10
# Seconds between forced buffer flushes
RECORDER_FLUSH_INTERVAL = 3.0


# --- Data Models ---
@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """Minimal device info storage."""

    name: str
    index: int
    hostapi: int
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    supported_samplerates: list[tuple[float, float]] = field(default_factory=list)


# --- Basic Logging Setup ---
log_config = LoggerConfig(
    app_name="Kardome Mallet Recorder",
    log_file_path=False,
)

setup_logging(log_config)


def get_logger(name: str) -> StandardLogger:
    """Gets a logger instance and casts it for type checking custom methods."""
    # Casting is necessary for static type checkers (like Pyright/Mypy)
    # to recognize the custom methods (panel, rule, progress) on the
    # logger instance returned by logging.getLogger().
    return cast(StandardLogger, logging.getLogger(name))


logger = get_logger(__name__)
