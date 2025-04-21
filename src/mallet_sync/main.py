# main.py
import sys
import time

# Use absolute imports
from mallet_sync.audio.core import play_and_record_cycle
from mallet_sync.config.config import INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, get_logger
from mallet_sync.utils.device_utils import find_mallet_devices
from mallet_sync.utils.file_utils import create_output_dir, scan_audio_files

logger = get_logger(__name__)  # Get logger for main module


def main():
    """Main entry point for the mallet-sync recording script."""
    logger.info('Starting Mallet Sync Recorder (using default output)...')

    # 1. Find Mallet Devices
    selected_mallets = find_mallet_devices()
    if not selected_mallets:
        logger.critical('Could not select exactly two required Mallet devices. Exiting.')
        sys.exit(1)

    # 2. Scan for Audio Files
    files_to_process = scan_audio_files(INPUT_AUDIO_DIR)
    if not files_to_process:
        logger.warning(f'No input audio files found in {INPUT_AUDIO_DIR}. Nothing to process.')
        sys.exit(0)

    # 3. Create Output Directory
    try:
        output_dir = create_output_dir(OUTPUT_BASE_DIR)
    except Exception:
        logger.critical('Failed to prepare output directory. Exiting.')
        sys.exit(1)

    # 4. Run Recording Cycles
    logger.info(f'Processing {len(files_to_process)} audio files...')
    start_time = time.time()
    for wav_file in files_to_process:
        play_and_record_cycle(selected_mallets, wav_file, output_dir)
        # Add a small pause between cycles if desired
        time.sleep(0.5)

    end_time = time.time()
    logger.info(f'All processing finished in {end_time - start_time:.2f} seconds.')
    logger.info(f'Recordings saved in: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
