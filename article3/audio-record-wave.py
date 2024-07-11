# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

# inbuilt wave library to save the recorded audio
import wave
import pyaudio

CHUNK_SIZE = 1024  # Record in chunks of 1024 samples
SAMPLE_BIT_DEPTH = pyaudio.paInt16  # 16 bits per sample, 2 bytes
NUM_CHANNELS = 1
SAMPLE_RATE = 16000  # Record at 16khz which Whisper is designed for https://github.com/openai/whisper/discussions/870
RECORDING_LENGTH = 10
filename = "recording.wav"

pyaudio_instance = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Please Speak Now...')

stream = pyaudio_instance.open(format=SAMPLE_BIT_DEPTH,
                channels=NUM_CHANNELS,
                rate=SAMPLE_RATE,
                frames_per_buffer=CHUNK_SIZE,
                input=True)

chunks = []  # Initialize list to store chunks of samples as we received them

# Store data in chunks for the required number of seconds
TOTAL_SAMPLES = SAMPLE_RATE * RECORDING_LENGTH

for i in range(int(TOTAL_SAMPLES / CHUNK_SIZE) + 1):
    data = stream.read(CHUNK_SIZE)
    chunks.append(data)

# Join the chunks together to get the full recording in bytes
frames = b''.join(chunks)

print(f"Total frame bytes received: {len(frames)}")
print(f"Total frames received: {int(len(frames) / 2)}")

# Trim the recorded frames to the desired recording length
# The frames were received as bytes, so we need to account for 2 bytes per sample
frames = frames[:TOTAL_SAMPLES * 2]

print(f"Total desired sample length: {TOTAL_SAMPLES}")
print(f"Total frame bytes trimmed: {len(frames)}")
print(f"Total frames trimmed: {int(len(frames) / 2)}")

# Stop and close the stream 
stream.stop_stream()
stream.close()

print('Saving Recording')

# Python inbuilt library for dealing with wave files
import wave
# Open our wave file as write binary
wf = wave.open(filename, 'wb')
# Set the number of channels to match our recording
wf.setnchannels(NUM_CHANNELS)
# Set the bit depth of our samples. Waves store the sample
# size as bytes, not bits, so we need to do a conversion
wf.setsampwidth(pyaudio_instance.get_sample_size(SAMPLE_BIT_DEPTH))
# Set the sampling rate
wf.setframerate(SAMPLE_RATE)
# Then write out the samples we recorded earlier
wf.writeframes(frames)
# Close the file
wf.close()

print('Recording Saved')

# Release PyAudio
pyaudio_instance.terminate()
