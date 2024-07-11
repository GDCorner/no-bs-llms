# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

# in-built wave library for wave file parsing
import wave
import pyaudio

CHUNK_SIZE = 1024

filename = "recording.wav"

# Instantiate PyAudio
pyaudio_instance = pyaudio.PyAudio()


with wave.open(filename, 'rb') as wf:
    # Waves store bit depth as number of bytes, so we convert this to PyAudio format
    bit_depth = pyaudio_instance.get_format_from_width(wf.getsampwidth())

    # Open stream
    stream = pyaudio_instance.open(format=bit_depth,
                                    channels=wf.getnchannels(),
                                    rate=wf.getframerate(),
                                    output=True)

    # Play samples from the wave file in the same chunksize
    while len(data := wf.readframes(CHUNK_SIZE)):
        stream.write(data)

    # Close stream
    stream.close()

    # Release PyAudio system resources
    pyaudio_instance.terminate()
