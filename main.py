from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles

app = FastAPI()

@app.post("/uploadvoices/")
async def create_upload_files(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}

app.mount("/", StaticFiles(directory="static", html=True), name="static")


