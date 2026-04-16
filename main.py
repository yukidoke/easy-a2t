import psutil
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

model_size = "small"

model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=psutil.cpu_count(logical=False))

app = FastAPI()

def get_transcribed_text(file: UploadFile):
    segments, info = model.transcribe(
        file.file,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        beam_size=5,
        without_timestamps=False
    )
    with open(file.filename + ".txt", "w") as f:
        for segment in segments:
            print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}", file=f)

@app.post("/uploadvoices")
async def create_upload_files(files: list[UploadFile], background_tasks: BackgroundTasks):
    for file in files:
        background_tasks.add_task(get_transcribed_text, file=file)
        print("Task event")
    return {"filenames": [file.filename for file in files]}

app.mount("/", StaticFiles(directory="static", html=True), name="static")


