import whisper
import json

model = whisper.load_model("base")

result = model.transcribe(audio = r"audios\2_Your First HTML Website ｜ Sigma Web Development Course - Tutorial #2 [kJEsTjH5mVg].webm.mp3", 
                          language="hi",
                          task="translate",
                           word_timestamps=False )

 
chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

print(chunks)

with open("output.json", "w") as f:
    json.dump(chunks,f)

