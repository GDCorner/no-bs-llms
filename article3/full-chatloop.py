# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

import numpy as np
from openai import OpenAI, ChatCompletion
import whisper
import pyaudio
from piper import PiperVoice

# Connect to the LLM server
client = OpenAI(base_url="http://localhost:8080/",
                api_key="local-no-key-required")

# Open the whisper tiny model
model = whisper.load_model("tiny")

#Load the piper model
piper_model = "en_US-hfc_female-medium.onnx"
# We can't use CUDA on RPi, so forced off. Turn on if you'd like
voice = PiperVoice.load(piper_model, config_path=f"{piper_model}.json")

# Open a PyAudio instance
pyaudio_instance = pyaudio.PyAudio()

# Begin the message history array, we'll start with only the system prompt, as
# we'll now prompt the user for input
messages_history=[
        {
            "role": "system",
            "content": "You are a helpful teacher who is teaching students about astronomy and the Solar System."
        },
    ]

# A little util function to pull out the interesting response from the OpenAI
# chat completion response.
# This just pulls out the dict with the content and role fields.
def get_message_from_openai_response(response: ChatCompletion):
    chat_message = response.choices[0].message
    response = {
        "role": chat_message.role,
        "content": chat_message.content
    }
    return response
    
def add_user_message(message):
    """Add a user message to the message history."""
    messages_history.append(
        {
            "role": "user",
            "content": message
        }
    )

def add_assistant_message(message):
    """Add a user message to the message history."""
    messages_history.append(
        {
            "role": "assistant",
            "content": message
        }
    )
    
# A function to print the entire message history
def print_message_history():
    """Print the full message history."""
    print("===============================")
    for message in messages_history:
        print(message["role"], ":", message["content"])

## Voice Record Settings
CHUNK_SIZE = 1024  # Record in chunks of 1024 samples
SAMPLE_BIT_DEPTH = pyaudio.paInt16  # 16 bits per sample
NUM_CHANNELS = 1
SAMPLE_RATE = 16000  # Record at 16khz which Whisper is designed for
RECORDING_LENGTH = 10.0 # Recording time in seconds

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

def convert_audio_for_whisper(samples):
    #samples is an ndarray and must be float32
    whisper_samples = np.array([], dtype=np.float32)

    # Make a numpy array from the samples buffer
    new_samples_int = np.frombuffer(samples, dtype=np.int16)
    # Convert to float and normalize into the range of -1.0 to 1.0
    new_samples = new_samples_int.astype(np.float32) / 32768.0

    whisper_samples = np.append(whisper_samples, new_samples)

    return new_samples

def get_user_message_from_voice():
    samples = record_voice(RECORDING_LENGTH)
    whisper_samples = convert_audio_for_whisper(samples)
    transcription = model.transcribe(whisper_samples)
    return transcription['text']

def speak_answer(text):
    print("Generating Message Audio")

    # Open stream
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

    # message audio is bytes
    message_audio = bytearray()
    message_chunks = []

    for audiobytes in message_audio_stream:
        message_audio += audiobytes
        while len(message_audio) > CHUNK_SIZE:
            latest_chunk = bytes(message_audio[:CHUNK_SIZE])
            message_chunks.append(latest_chunk)
            output_stream.write(latest_chunk)
            message_audio = message_audio[CHUNK_SIZE:]

    # Write whatever is left in message audio that wasn't large enough for a final chunk
    output_stream.write(bytes(message_audio))

    # Close stream
    output_stream.close()

def main():
    # Chat loop
    while True:
        # Get a new user input
        text_input = input("Press enter to ask a question, or type 'exit' to quit: ")

        # Check if the user has tried to quit
        if text_input.lower().strip() == "exit":
            break
        
        user_message = get_user_message_from_voice()
        print("The user asked: ", user_message)
        
        # Add the user message to the message history
        add_user_message(user_message)

        # Generate a response from the history of the conversation
        llm_output = client.chat.completions.create(
            model="Meta-Llama-3-8B-Instruct-q5_k_m",
            messages=messages_history,
            stream=True
        )

        full_response = ""
        for chunk in llm_output:
            latest_chunk_str = chunk.choices[0].delta.content
            if latest_chunk_str is None:
                continue
            full_response += latest_chunk_str
            print(latest_chunk_str, end='', flush=True)

        # Force a new line to be printed after the response is completed
        print()
        speak_answer(full_response)

        add_assistant_message(full_response)

        # Print the full conversation
        print_message_history()

    # Release PortAudio
    pyaudio_instance.terminate()

    print("Chatbot finished - Goodbye!")

if __name__ == "__main__":
    main()
