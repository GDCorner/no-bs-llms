# Copyright 2024 GDCorner
# No BS Intro To Developing With LLMs

import whisper
import time

model = whisper.load_model("tiny")
#get the start time
start_time = time.time()
result = model.transcribe("recording.wav")
print(result["text"])
#calculate the time taken
end_time = time.time()
print(f"Total time: {end_time - start_time}")
