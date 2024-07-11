# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

import pyaudio
from piper import PiperVoice


#Make sure the json file is next to this model
piper_model = "en_US-hfc_female-medium.onnx"
# We can't use CUDA on RPi, so forced off. Turn on if you'd like
voice = PiperVoice.load(piper_model, config_path=f"{piper_model}.json")

# Need to init audio after loading model to get sample rate
SAMPLE_BIT_DEPTH = pyaudio.paInt16  # 16 bits per sample
NUM_CHANNELS = 1 # mono
pyaudio_instance = pyaudio.PyAudio()


def speak_answer(text):
    print("Generating Message Audio")

    # Open a stream to output the audio. Notice we get the sample rate from the settings of the loaded model.
    output_stream = pyaudio_instance.open(format=SAMPLE_BIT_DEPTH,
                                            channels=NUM_CHANNELS,
                                            rate=voice.config.sample_rate,
                                            output=True)

    synthesize_args = {
            "sentence_silence": 0.0,
        }

    # Synthesize the audio to a raw stream
    message_audio_stream = voice.synthesize_stream_raw(text, **synthesize_args)

    # Larger chunk sizes tend to help stuttering here
    CHUNK_SIZE = 4096

    # message audio is in bytes
    message_audio = bytearray()

    for audiobytes in message_audio_stream:
        # We keep acruing audio until we have enough for a chunk
        message_audio += audiobytes
        while len(message_audio) > CHUNK_SIZE:
            # Once we have enough for a chunk (potentially multiple chunks)
            # we extract it from the buffer
            latest_chunk = bytes(message_audio[:CHUNK_SIZE])
            message_audio = message_audio[CHUNK_SIZE:]
            # Output the latest chunk to the audio stream
            output_stream.write(latest_chunk)

    # Once we reach he there is no more incoming audio from the TTS Engine
    # Write whatever is left in message audio buffer that wasn't large enough for a final chunk
    output_stream.write(bytes(message_audio))

    # Close stream
    output_stream.close()

#text = "Hello, Welcome to the No BS Intro to LLMs. I hope you enjoy this series! We've covered a lot of ground here!"
text = "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way—in short, the period was so far like the present period, that some of its noisiest authorities insisted on its being received, for good or for evil, in the superlative degree of comparison only."
speak_answer(str(text))

# Release PortAudio system resources
pyaudio_instance.terminate()
