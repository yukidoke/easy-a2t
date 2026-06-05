import io
import psutil
import qrcode
import socket
from fastapi import BackgroundTasks, FastAPI, UploadFile, Response
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel, BatchedInferencePipeline

# Config
use_gpu = True
model_size = "turbo"

if use_gpu:
    model = WhisperModel(
        model_size,
        device="cuda",
        compute_type="float16",
    )
else:
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=psutil.cpu_count(logical=False),
    )


batched_model = BatchedInferencePipeline(model=model)


# generate qr-code
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


ip_address = get_local_ip()
qr = qrcode.QRCode(
    version=1, error_correction=qrcode.ERROR_CORRECT_L, box_size=10, border=4
)
qr.add_data("http://" + ip_address + ":8000")
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")

app = FastAPI()


# fastapi
def get_transcribed_text(file: UploadFile):
    segments, info = batched_model.transcribe(
        file.file,
        language="ja",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        batch_size=16,
        beam_size=5,
        without_timestamps=False,
    )
    with open(file.filename + ".txt", "w", encoding="utf-8") as f:
        for segment in segments:
            print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}", file=f)
    print("Transcribe End.")


@app.post("/uploadvoices")
async def create_upload_files(
    files: list[UploadFile], background_tasks: BackgroundTasks
):
    for file in files:
        background_tasks.add_task(get_transcribed_text, file=file)
        print("Task event")
    return {"filenames": [file.filename for file in files]}


@app.get("/qr")
def image():
    img_bytes = io.BytesIO()
    img.get_image().save(img_bytes, format="PNG")
    return Response(content=img_bytes.getvalue(), media_type="image/png")


app.mount("/", StaticFiles(directory="static", html=True), name="static")
