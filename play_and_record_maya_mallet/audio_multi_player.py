# =============================================================================
# Imports...
# =============================================================================
import pyaudio
import wave
from threading import Thread, Event

# https://pypi.org/project/PyAudio/

# =============================================================================
# Play using selected output device
# =============================================================================
class AudioMultiPlayer(Thread):
    # =========================================================================
    # Plays a WAV file through the speakers...
    # =========================================================================
    def __init__(self, audio_file_path: str, output_device_index: int, 
                 stop_playing_event: Event = None, repeat: bool = False,
                 chunk_size: int = 1024):

        Thread.__init__(self)

        self.audio_file_path = audio_file_path
        self.chunk_size = chunk_size
        self.stop_playing_event = stop_playing_event
        self.output_device_index = output_device_index
        self.repeat = repeat

        self.initialize()
    # =========================================================================

    # =========================================================================
    # ...
    # =========================================================================
    def initialize(self):
        self.wf = wave.open(self.audio_file_path, 'rb')

        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(format=self.p.get_format_from_width(self.wf.getsampwidth()),
                                  channels=self.wf.getnchannels(),
                                  rate=self.wf.getframerate(),
                                  output_device_index=self.output_device_index,
                                  output=True)
    # =========================================================================

    # =========================================================================
    # ...
    # =========================================================================
    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

        self.p.terminate()
    # =========================================================================

    # =========================================================================
    # ...
    # =========================================================================
    def resume(self):
        self.initialize()
        self.run()
    # =========================================================================
    
    # =========================================================================
    # ...
    # =========================================================================
    def run(self):

        data = self.wf.readframes(self.chunk_size)

        while data != b'' and not (self.stop_playing_event is not None and self.stop_playing_event.is_set()):
            self.stream.write(data)
            data = self.wf.readframes(self.chunk_size)
        if self.repeat and not (self.stop_playing_event is not None and self.stop_playing_event.is_set()):
            self.resume()
        else:
            self.stop()
    # =========================================================================
# =============================================================================