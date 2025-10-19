import whisper

model = whisper.load_model("large")
result = model.transcribe("audios/MENTORIABULLYING1-1.mp3")
print(result["text"])
