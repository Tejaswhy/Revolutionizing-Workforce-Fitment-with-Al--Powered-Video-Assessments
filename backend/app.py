# backend/app.py

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from pydantic import BaseModel

from interviewer import process_interview
from interviewer import process_audio_interview

# =====================================
# FASTAPI
# =====================================
app = FastAPI()

# =====================================
# REQUEST MODEL
# =====================================
class InterviewRequest(BaseModel):

    text: str

    lang: str = "en"

# =====================================
# HOME
# =====================================
@app.get("/")
def home():

    return {

        "message":
        "SkillFit AI Interview Backend Running"
    }

# =====================================
# TEXT INTERVIEW
# =====================================
@app.post("/interview")
def interview(req: InterviewRequest):

    result = process_interview(

        req.text,

        req.lang
    )

    return result

# =====================================
# AUDIO INTERVIEW
# =====================================
@app.post("/interview_audio")
async def interview_audio(

    file: UploadFile = File(...),

    lang: str = Form(...)
):

    try:

        # =================================
        # SAVE AUDIO
        # =================================
        audio_path = \
            "/Users/tejasy/Documents/documents/govt/project/input.wav"

        with open(audio_path, "wb") as f:

            f.write(await file.read())

        # =================================
        # PROCESS AUDIO
        # =================================
        result = process_audio_interview(

            audio_path,

            lang
        )

        return result

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }