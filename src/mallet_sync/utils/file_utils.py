# utils/file_utils.py
from datetime import datetime
from pathlib import Path

# Use absolute imports
from mallet_sync.config.config import FILENAME_TEMPLATE, get_logger

logger = get_logger(__name__)


def scan_audio_files(input_dir: Path) -> list[Path]:
    """Scans the input directory for WAV files to process."""
    # (Implementation remains the same as before)
    files_to_process = []
    patterns = ['calibrate_*.wav', 'test_*.wav']
    logger.info(f"Scanning for {patterns} in '{input_dir.resolve()}'...")
    for pattern in patterns:
        try:
            found = sorted(list(input_dir.glob(pattern)))
            files_to_process.extend(found)
            logger.info(f"  Found {len(found)} files matching '{pattern}'")
        except Exception:
            logger.exception(f"  Error scanning for pattern '{pattern}'")
    if not files_to_process:
        logger.warning(f'No WAV files matching {patterns} found in {input_dir}')
    return files_to_process


def create_output_dir(base_dir: Path) -> Path:
    """Creates a timestamped output directory."""
    # (Implementation remains the same as before)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / timestamp
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'Created output directory: {output_dir.resolve()}')
    except OSError:
        logger.critical(f'Failed to create output directory {output_dir}')
        raise
    else:
        return output_dir


def generate_output_path(output_dir: Path, role: str, context: str) -> Path:
    """Generates the full path for an output recording file."""
    # (Implementation remains the same as before)
    filename = FILENAME_TEMPLATE.format(role=role, context=context)
    return output_dir / filename
