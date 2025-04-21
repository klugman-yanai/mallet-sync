"""mallet_sync

Main module for mallet-sync audio recording system. This provides the entry point
for running the application with a clean, orchestration-focused structure.

Assumptions:
- Python 3.13+ is required
- Custom logger is used
- Audio I/O will use sounddevice (not pyaudio)
"""

from mallet_sync.config import get_logger
from mallet_sync.config.config import build_session
from mallet_sync.utils.audio_utils import record_ambient_noise, record_tests, record_zones
from mallet_sync.utils.device_utils import create_recorders, detect_devices
from mallet_sync.utils.file_utils import save_session_safely, setup_output_dir

logger = get_logger(__name__)


def main() -> None:
    """Run the main recording workflow for mallet synchronization.

    This function orchestrates the recording process but delegates all implementation
    details to specialized utility modules. It follows a clean, linear workflow:
    1. Setup recording environment and devices
    2. Record ambient noise, zone calibrations, and test files
    3. Save comprehensive session metadata

    Error handling is implemented within each specialized function, keeping this
    main workflow clean and focused on orchestration rather than implementation.
    """
    # Setup phase
    output_dir = setup_output_dir()
    devices = detect_devices()
    recorders = create_recorders(devices, output_dir)
    session = build_session(devices, recorders, output_dir)

    # Execute recordings in sequence with built-in error handling
    # The auto-detection of ambient_noise.wav happens within the function
    record_ambient_noise(session)
    record_zones(session)
    record_tests(session)

    # Finalization phase - always attempt to save metadata
    save_session_safely(output_dir, session)


# ---- Main Entry Point --------------------------------------------------------


if __name__ == '__main__':
    main()
