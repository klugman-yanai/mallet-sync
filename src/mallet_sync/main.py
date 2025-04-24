# main.py
import argparse
import sys
import time

from pathlib import Path

# Use absolute imports
from mallet_sync.audio.core import play_and_record_cycle
from mallet_sync.config import INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, get_logger
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

    # 2. Scan for Audio Files
    files_to_process = scan_audio_files(INPUT_AUDIO_DIR, exclude_calibration=args.exclude_calibration)

    # 3. Create Output Directory
    output_dir = create_output_dir(OUTPUT_BASE_DIR)

    # 4. Run Recording Cycles
    logger.info(f'Processing {len(files_to_process)} audio files...')
    start_time = time.time()
    for wav_file in files_to_process:
        play_and_record_cycle(selected_mallets, wav_file, output_dir)
        time.sleep(0.5)

    end_time = time.time()
    logger.info(f'All processing finished in {end_time - start_time:.2f} seconds.')
    logger.info(f'Recordings saved in: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
