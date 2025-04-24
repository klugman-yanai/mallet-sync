"""Logging utilities for the mallet-sync application."""

import time

from pathlib import Path
from typing import Any

from mallet_sync.config import get_logger

logger = get_logger(__name__)


def log_startup_info(input_dir: Path, output_dir: Path, sleep_time: float) -> None:
    """Log startup information at appropriate levels."""
    logger.info('Starting Mallet Sync Recorder...')
    logger.debug(
        f'Configuration: input={input_dir.resolve()}, '
        f'output={output_dir.resolve()}, '
        f'sleep_time={sleep_time}s',
    )


def log_completion_summary(
    start_time: float,
    files_processed: int,
    devices_used: int,
    output_dir: Path,
) -> None:
    """
    Log a summary of the processing run with human-readable metrics.

    Args:
        start_time: The timestamp when processing started
        files_processed: Number of files processed
        devices_used: Number of recording devices used
        output_dir: Directory where recordings were saved
    """
    total_run_time = time.time() - start_time
    run_time_str = (
        f'{int(total_run_time // 60)} minutes and {total_run_time % 60:.1f} seconds'
        if total_run_time >= 60 else
        f'{total_run_time:.1f} seconds'
    )

    logger.info(
        f'\n{"=" * 60}\n'
        f'Mallet Sync Recording completed in {run_time_str}\n'
        f'Processed {files_processed} files with {devices_used} devices\n'
        f'Output saved to: {output_dir.resolve()}\n'
        f'{"=" * 60}',
    )


def format_time_str(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f'{minutes}m {seconds:.1f}s'
    return f'{seconds:.1f} seconds'


def format_size_mb(bytes_value: int) -> float:
    """Convert bytes to MB with 2 decimal precision."""
    return bytes_value / (1024 * 1024)
