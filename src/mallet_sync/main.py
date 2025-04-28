import argparse
import signal
import sys
import time

from datetime import datetime
from pathlib import Path
from typing import NoReturn

from mallet_sync.audio.core import check_ambient_files, process_audio_batch
from mallet_sync.config import (
    FILENAME_TEMPLATE,
    INPUT_AUDIO_DIR,
    MALLET_ROLES,
    OUTPUT_BASE_DIR,
    SLEEP_TIME_SEC,
    get_logger,
)
from mallet_sync.logging_utils import log_completion_summary, log_startup_info
from mallet_sync.utils import (
    check_maya44_output_device,
    create_role_dirs,
    create_session_dir,
    find_mallet_devices,
    scan_audio_files,
)

logger = get_logger(__name__)


def handle_keyboard_interrupt(signum: int, frame: object) -> NoReturn:
    logger.warning('\nOperation cancelled by user (Ctrl+C)')
    logger.info('Cleaning up and exiting...')
    sys.exit(0)


def main() -> None:
    signal.signal(signal.SIGINT, handle_keyboard_interrupt)
    time.sleep(5)
    start_time = time.time()
    log_startup_info(INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, SLEEP_TIME_SEC)

    if not check_maya44_output_device():
        return

    mallet_devices = find_mallet_devices()
    if not mallet_devices:
        return

    # NOTE: Skip calibration if needed here
    files_to_process = scan_audio_files(
        INPUT_AUDIO_DIR,
        exclude_calibration=False,
    )
    if not files_to_process:
        return

    has_ambient_files = check_ambient_files(INPUT_AUDIO_DIR)

    session_dir = create_session_dir(OUTPUT_BASE_DIR)
    role_dirs = create_role_dirs(session_dir, MALLET_ROLES)

    process_audio_batch(
        files=files_to_process,
        mallet_devices=mallet_devices,
        role_dirs=role_dirs,
        sleep_time_sec=SLEEP_TIME_SEC,
        needs_silence=not has_ambient_files,
    )

    log_completion_summary(
        start_time,
        len(files_to_process),
        len(mallet_devices),
        session_dir,
    )


if __name__ == '__main__':
    main()


# def generate_txt_marker(filepath: Path, orig_name: str):
#     timestamp = datetime.now().isoformat()
#     content = f'Marker for {orig_name} - created at {timestamp}\n'
#     filepath.write_text(content)
#
# def dry_run_main() -> None:
#     time.sleep(1)
#     log_startup_info(INPUT_AUDIO_DIR, OUTPUT_BASE_DIR, SLEEP_TIME_SEC)
#     files_to_process = scan_audio_files(
#         INPUT_AUDIO_DIR,
#         exclude_calibration=False,
#     )
#     if not files_to_process:
#         logger.error("No files to process. Exiting dry-run.")
#         return

#     session_dir = create_session_dir(OUTPUT_BASE_DIR)
#     role_dirs = create_role_dirs(session_dir, MALLET_ROLES)

#     logger.info(f"[DRY-RUN] Simulating recordings for {len(files_to_process)} files into session dir: {session_dir}")
#     for idx, wav_file in enumerate(files_to_process, 1):
#         context_name = wav_file.stem
#         logger.info(f"[DRY-RUN] Processing file {idx}: {wav_file.name}")
#         for role in MALLET_ROLES:
#             marker_filename = FILENAME_TEMPLATE.format(role=role, context=context_name).replace('.wav', '.txt')
#             marker_path = role_dirs[role] / marker_filename
#             logger.info(f"[DRY-RUN] Creating marker file: {marker_path}")
#             generate_txt_marker(marker_path, wav_file.name)
#         time.sleep(0.05)  # Simulate small delay
#     logger.info("[DRY-RUN] All marker files written")
#     logger.info(f"[DRY-RUN] Output directory tree: {session_dir.resolve()}")


# if __name__ == "__main__":
#     dry_run_main()
