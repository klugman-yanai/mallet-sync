# =============================================================================
# Imports...
# =============================================================================
from os import (path, mkdir, getcwd)
from datetime import datetime
from time import sleep
# from .audio_multi_player import AudioMultiPlayer
from audio_player import AudioPlayer
from pyaudio_stream_recorder import PyAudioStreamUSBRecorder

DATE_TIME_FORMAT_STR = '%Y_%m_%d_%H_%M_%S'

# =============================================================================
# This function find a specific (e.g. Kardome) and returns its index
# =============================================================================
def get_audio_device_by_name(device_names: list) -> tuple:
    """
    Searches for an audio device with a given name and returns its index and driver name.

    Parameters
    ----------
    * device_name: [str]
        A string representing the name of the audio device to search for.

    Returns:
    * (audio_device_index, audio_device_driver_name): (int, str)
        A tuple containing the index of the audio device and its driver name.
    """

    import pyaudio
    p = pyaudio.PyAudio()

    audio_device_index = -1
    audio_device_driver_name = None

    info = p.get_host_api_info_by_index(0)

    # Get the number of devices available.
    num_devices = info.get('deviceCount')

    found_device = False  # Initialize a flag variable

    for device_name in device_names:
        if found_device:
            break # If the device is already found, exit the outer loop

        # Search for the specified device name.
        for idx in range(0, num_devices):

            audio_device_driver_name = p.get_device_info_by_host_api_device_index(0, idx).get('name')
            # print(f'USB: {audio_device_driver_name.lower()}')
            if device_name.lower() in audio_device_driver_name.lower():
                audio_device_index = idx
                found_device = True
                break # Exit the inner loop when the device is found

    p.terminate()

    return audio_device_index, audio_device_driver_name
# =============================================================================

# =============================================================================
# Starting point
# =============================================================================
if __name__ == "__main__":
    is_calibrate = True
    sleep(10)
    current_dir = getcwd()
    now = datetime.now().strftime(DATE_TIME_FORMAT_STR)
    experiment_dir = path.join(current_dir, now)
    if not path.exists(experiment_dir):
        mkdir(experiment_dir)
        print(f"Created directory: {experiment_dir}")

    win_device_names = ['kardome', 'kt']
    channels   = 9
    fs         = 16000
    chunk_size = 1024

    # =====================================================================
    # Mallet Connection
    # =====================================================================
    audio_device_index = -1
    audio_device_index, device_name = get_audio_device_by_name(win_device_names)
    if audio_device_index == -1:
        print("No Mallet device found")
        exit()
    else:
        print(f"Mallet device found: {device_name}")

    if is_calibrate:
        # =====================================================================
        # Ambient Noise
        # =====================================================================
        print("Calibrating Ambient Noise...")
        ambient_noise_file_path = path.join(current_dir, f"ambient_noise.wav")

        recorder = PyAudioStreamUSBRecorder(input_device_index=audio_device_index,
                                            channels=channels,
                                            recorded_file=ambient_noise_file_path,
                                            fs=fs,
                                            chunk_size=chunk_size)
        recorder.start()
        sleep(7)
        recorder.stop()
        recorder.join()
        print("Done calibrating Ambient Noise")

    #     # =====================================================================
    #     # Zone calibration
    #     # =====================================================================
    #     sleep(3)
    #     print("Calibrating Zone...")
    #     zone_1_calib_file_path_to_play = r"C:\Users\DimaTepliakov\Downloads\calibrate_S1.wav"
    #     zone_1_recorder_file_path = path.join(current_dir, "zone_1.wav")
    #     player = AudioPlayer(audio_file_path=zone_1_calib_file_path_to_play)
    #     recorder = PyAudioStreamUSBRecorder(input_device_index=audio_device_index,
    #                                         channels=channels,
    #                                         recorded_file=zone_1_recorder_file_path,
    #                                         fs=fs,
    #                                         chunk_size=chunk_size)
    #     recorder.start()
    #     player.start()
    #     player.join()
    #     recorder.stop()
    #     recorder.join()
    #     print("Done calibrating Zone.")

    # # TODO: get all the test file paths
    # from pathlib import Path

    # root = Path(r'')
    # files_paths = [f for f in root.glob('*.wav')]
    # # =====================================================================
    # # Test
    # # =====================================================================
    # sleep(3)
    # print("Recording test...")
    # now = datetime.now().strftime(DATE_TIME_FORMAT_STR)
    # recorder_file_path = path.join(experiment_dir, f"test_{now}.wav")

    # player = AudioPlayer(audio_file_path=zone_1_calib_file_path_to_play)
    # recorder = PyAudioStreamUSBRecorder(input_device_index=audio_device_index,
    #                                     channels=channels,
    #                                     recorded_file=recorder_file_path,
    #                                     fs=fs,
    #                                     chunk_size=chunk_size)
    # recorder.start()
    # player.start()
    # player.join()
    # recorder.stop()
    # recorder.join()
    # print("Done recording test.")
