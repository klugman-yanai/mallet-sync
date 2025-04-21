# =============================================================================
# Imports...
# =============================================================================
from threading import (Thread, Event, Lock)

import wave
import pyaudio
import numpy as np

# =============================================================================
# ...
# =============================================================================
class PyAudioStreamUSBRecorder(Thread):
    # =========================================================================
    # This class records from a device in a separate thread
    # input_device_index  = with which device to record
    # =========================================================================
    def __init__(self, input_device_index: int, channels: int, recorded_file: str, 
                 fs: int, chunk_size: int,  
                 save_file: bool = True, data_type = np.int16, format: int = pyaudio.paInt16):
        """
        Initialize the PyAudioStreamUSBRecorder object.
        """

        Thread.__init__(self)

        self.input_device_index = input_device_index
        self.channels           = channels
        self.recorded_file      = recorded_file
        self.fs                 = fs
        self.chunk_size         = chunk_size
        self.save_file          = save_file
        self.data_type          = data_type
        self.format             = format

        self.__stop_streaming   = Event()
        self.__pause            = Event()

        self.initialize()
    # =========================================================================
        
    # =========================================================================
    # ...
    # =========================================================================
    def initialize(self):
        self.p = pyaudio.PyAudio()

        if self.save_file:
            # write recording to a WAV file
            self.wf = wave.open(self.recorded_file, 'wb')
            self.wf.setnchannels(self.channels)
            self.wf.setsampwidth(self.p.get_sample_size(self.format))
            self.wf.setframerate(self.fs)

        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.fs,
                                  input=True,
                                  input_device_index=self.input_device_index,
                                  frames_per_buffer=self.chunk_size)
    # =========================================================================
        
    # =========================================================================
    # The recording process itself, run in a separate thread.
    # =========================================================================
    def run(self):
        """
        The recording process itself, run in a separate thread.
        """
        for _ in range(int(self.fs / self.chunk_size * 0.7)):
            self.stream.read(self.chunk_size, exception_on_overflow=False)

        while not self.__stop_streaming.is_set():
            if self.__pause.is_set():
                continue

            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)

                if self.save_file:
                    self.wf.writeframes(data)

            except Exception as e:
                print(f'Error: {e}')
                pass
            
        self.cleanup()
    # =========================================================================

    # =========================================================================
    # Cleanup
    # =========================================================================
    def cleanup(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

        if self.save_file:
            self.wf.close()
    # =========================================================================
    
    # =========================================================================
    # Set the event in order to stop the streaming.
    # =========================================================================
    def stop(self):
        """
        Set the event in order to stop the streaming.
        """
        self.__stop_streaming.set()
    # =========================================================================

    # =========================================================================
    # Set the event in order to pause the streaming.
    # =========================================================================
    def pause(self):
        """
        Set the event in order to pause the streaming.
        """
        self.__pause.set()
    # =========================================================================

    # =========================================================================
    # Clear the event in order to resume the streaming.
    # =========================================================================
    def resume(self):
        """
        Clear the event in order to resume the streaming.
        """
        self.__pause.clear()
    # =========================================================================

# =============================================================================
