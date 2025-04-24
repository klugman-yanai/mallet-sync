# main.py
import argparse
import sys
import time

from pathlib import Path

# Use absolute imports
from mallet_sync.audio.core import check_ambient_files, process_audio_batch
from mallet_sync.config import INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, SLEEP_TIME_SEC, get_logger
from mallet_sync.utils import create_output_dir, find_mallet_devices, scan_audio_files

logger = get_logger(__name__)  # Get logger for main module


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Mallet Sync Recording System')
    parser.add_argument(
        '--exclude-calibration',
        action='store_true',
        help='Process only test files, exclude calibration files',
    )
    return parser.parse_args()


def main():
    """Main entry point for the mallet-sync recording script."""
    # Parse command line arguments
    args = parse_arguments()

    logger.info('Starting Mallet Sync Recorder (using default output)...')

    # 1. Find Mallet Devices
    selected_mallets = find_mallet_devices()

    # 2. Check for ambient files and determine if silence is needed
    has_ambient_files = check_ambient_files(INPUT_AUDIO_DIR)

    # 3. Scan for Audio Files
    files_to_process = scan_audio_files(INPUT_AUDIO_DIR, exclude_calibration=args.exclude_calibration)

    # 4. Create Output Directory
    output_dir = create_output_dir(OUTPUT_BASE_DIR)

    # 5. Process all audio files (with silence streaming if needed)
    process_audio_batch(
        files=files_to_process,
        mallet_devices=selected_mallets,
        output_dir=output_dir,
        sleep_time_sec=SLEEP_TIME_SEC,
        needs_silence=not has_ambient_files,
    )


if __name__ == '__main__':
    main()
