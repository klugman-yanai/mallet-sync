# main.py
"""
Main orchestration module for the Mallet Sync Recorder.

This module provides the primary workflow for the Mallet recording process,
with a clean focus on orchestration only. Implementation details are delegated
to specialized utility modules.
"""

import argparse
import signal
import sys
import time

from pathlib import Path
from typing import NoReturn

# Use absolute imports for stability and clarity
from mallet_sync.audio.core import check_ambient_files, process_audio_batch
from mallet_sync.config import (
    INPUT_AUDIO_DIR,
    MALLET_ROLES,
    NUM_MALLETS,
    OUTPUT_BASE_DIR,
    SLEEP_TIME_SEC,
    get_logger,
)
from mallet_sync.logging_utils import log_completion_summary, log_startup_info
from mallet_sync.utils import (
    check_maya44_output_device,
    create_output_dir,
    find_mallet_devices,
    scan_audio_files,
)

logger = get_logger(__name__)


def handle_keyboard_interrupt(signum: int, frame: object) -> NoReturn:
    """Handle keyboard interrupt (Ctrl+C) to exit gracefully."""
    logger.warning('\nOperation cancelled by user (Ctrl+C)')
    logger.info('Cleaning up and exiting...')
    sys.exit(0)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Mallet Sync Recording System')
    parser.add_argument(
        '--exclude-calibration',
        action='store_true',
        help='Process only test files, exclude calibration files',
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the mallet-sync recording script.

    This function orchestrates the entire recording workflow:
    1. Environment checks (devices, files)
    2. Resource preparation (output directory)
    3. Processing execution
    4. Results summarization

    The function handles graceful exits for missing devices or files.
    """
    # Set up keyboard interrupt handler for graceful exit on Ctrl+C
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)

    # Initialize and parse arguments
    start_time = time.time()
    args = parse_arguments()

    # Log startup information (configuration details at debug level)
    log_startup_info(INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, SLEEP_TIME_SEC)

    # --- Environment/Prerequisite Checks ---

    # 1. Verify MAYA44 output device is selected
    if not check_maya44_output_device():
        return  # Error already logged by the check function

    # 2. Find and validate recording devices
    selected_mallets = find_mallet_devices()
    if not selected_mallets:
        return  # Error already logged by the device detection function

    # 3. Check for ambient calibration files
    has_ambient_files = check_ambient_files(INPUT_AUDIO_DIR)

    # 4. Scan for audio files to process
    files_to_process = scan_audio_files(
        INPUT_AUDIO_DIR,
        exclude_calibration=args.exclude_calibration,
    )
    if not files_to_process:
        return  # Error already logged by the scan function

    # --- Setup & Execution ---

    # 5. Prepare output location
    output_dir = create_output_dir(OUTPUT_BASE_DIR)

    # 6. Execute core audio processing pipeline
    process_audio_batch(
        files=files_to_process,
        mallet_devices=selected_mallets,
        output_dir=output_dir,
        sleep_time_sec=SLEEP_TIME_SEC,
        needs_silence=not has_ambient_files,
    )

    # --- Completion ---

    # 7. Log completion summary
    log_completion_summary(
        start_time,
        len(files_to_process),
        len(selected_mallets),
        output_dir,
    )


if __name__ == '__main__':
    main()
