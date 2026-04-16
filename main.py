import io
import psutil
import qrcode
import socket
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from starlette.responses import StreamingResponse

model_size = "small"

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

ip_address = get_local_ip()
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_L,
    box_size=10,
    border=4
)
qr.add_data("http://" + ip_address + ":8000")
qr.make(fit=True)
img = qr.make_image(fill="black", back_color="white")

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

@app.get("/qr")
def image():
    img_bytes = io.BytesIO()
    img.get_image().save(img_bytes, format="PNG")
    return StreamingResponse(img_bytes.getvalue(), media_type="image/png")

app.mount("/", StaticFiles(directory="static", html=True), name="static")


