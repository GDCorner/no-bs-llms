# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

import numpy as np
import pyaudio
import whisper
from pprint import pprint

CHUNK_SIZE = 1024  # Record in chunks of 1024 samples
SAMPLE_BIT_DEPTH = pyaudio.paInt16  # 16 bits per sample
NUM_CHANNELS = 1
SAMPLE_RATE = 16000  # Record at 16khz which Whisper is designed for
RECORDING_LENGTH = 10.0 # Recording time in seconds

pyaudio_instance = pyaudio.PyAudio()

def record_voice(record_time=10.0):
    print('Please Speak Now...')

    stream = pyaudio_instance.open(format=SAMPLE_BIT_DEPTH,
                    channels=NUM_CHANNELS,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=CHUNK_SIZE,
                    input=True)

    chunks = []  # Initialize list to store chunks of samples as we received them

    TOTAL_SAMPLES = int(SAMPLE_RATE * record_time)

    for i in range(int(TOTAL_SAMPLES / CHUNK_SIZE) + 1):
        data = stream.read(CHUNK_SIZE)
        chunks.append(data)

    # Join the chunks together to get the full recording in bytes
    frames = b''.join(chunks)

    # Trim extra samples we didn't want to record due to chunk size
    frames = frames[:TOTAL_SAMPLES * 2]

    # Stop and close the stream 
    stream.stop_stream()
    stream.close()

    print("Recording complete")
    return frames

def play_recorded_voice(samples):
    stream = pyaudio_instance.open(format=SAMPLE_BIT_DEPTH,
                                    channels=NUM_CHANNELS,
                                    rate=SAMPLE_RATE,
                                    output=True)

    # Play samples from the wave file in the same chunksize
    while len(samples) > 0:
        data = samples[:CHUNK_SIZE]
        samples = samples[CHUNK_SIZE:]
        stream.write(data)

def convert_audio_for_whisper(samples):
    #samples is an ndarray and must be float32
    whisper_samples = np.array([], dtype=np.float32)

    # Make a numpy array from the samples buffer
    new_samples_int = np.frombuffer(samples, dtype=np.int16)
    # Convert to float and normalize into the range of -1.0 to 1.0
    new_samples = new_samples_int.astype(np.float32) / 32768.0

    whisper_samples = np.append(whisper_samples, new_samples)

    return new_samples

def find_min_max_values(samples):
    max = 0.0
    min = 1.0
    for sample in samples:
        if sample > max:
            max = sample
        if sample < min:
            min = sample

    print("Max value in samples is: ", max)
    print("Min value in samples is: ", min)

def pad_or_trim_audio_for_whisper(samples):
    WHISPER_EXPECTED_SAMPLES = 30 * 16000 # Whisper expects 30 seconds of audio at 16khz
    if len(samples) < WHISPER_EXPECTED_SAMPLES:
        # pad the samples with zeros, remembering to keep at float32
        padding = np.zeros(WHISPER_EXPECTED_SAMPLES - len(samples), dtype=np.float32)
        samples = np.append(samples, padding)
    else:
        # trim it. This is likely impossible since we record 10 seconds but just incase the value ever changes
        samples = samples[:WHISPER_EXPECTED_SAMPLES]
    
    return samples


samples = record_voice(RECORDING_LENGTH)
print(len(samples))
whisper_samples = convert_audio_for_whisper(samples)
print(len(whisper_samples))

#play_recorded_voice(samples)

# Release PortAudio
pyaudio_instance.terminate()

# Load the whisper model
model = whisper.load_model("tiny")

transcription_result = model.transcribe(whisper_samples)

pprint(transcription_result)

print("The transcribed text is:")
print(transcription_result['text'])
