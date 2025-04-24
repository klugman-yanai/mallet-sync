# =============================================================================
# Imports...
# =============================================================================
import pyaudio
import wave
from threading import Thread, Event

# https://pypi.org/project/PyAudio/

# =============================================================================
# Play to 2 speaker class
# =============================================================================
class AudioPlayer(Thread):
    # =========================================================================
    # Plays a WAV file through the speakers...
    # =========================================================================
    def __init__(self, audio_file_path: str, stop_playing_event: Event = None,
                 chunk_size: int = 1024):

        Thread.__init__(self)

        self.audio_file_path = audio_file_path
        self.chunk_size = chunk_size
        self.stop_playing_event = stop_playing_event

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
    # TODO: DOCS...
    # =========================================================================
    def run(self):

        data = self.wf.readframes(self.chunk_size)

        while data != b'' and not (self.stop_playing_event is not None and self.stop_playing_event.is_set()):
            self.stream.write(data)
            data = self.wf.readframes(self.chunk_size)

        self.stop()
    # =========================================================================
# =============================================================================
