import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import numpy as np
from gtts import gTTS
import os
from googletrans import Translator
from llama_cpp import Llama
import json


import re

# ==============================
# CONFIG
# ==============================
SAMPLE_RATE = 16000
DURATION = 10
AUDIO_FILE = "input.wav"
face_path = "/Users/tejasy/Desktop/small.mp4"

checkpoint = "checkpoints/wav2lip.pth"

output_path = "results/output.mp4"

os.makedirs("results", exist_ok=True)

# ==============================
# LOAD WHISPER MODEL
# ==============================
print("Loading Whisper model... ⏳")

whisper_model = whisper.load_model("small")

print("✅ Whisper loaded")

# ==============================
# LOAD TRANSLATOR
# ==============================
translator = Translator()

# ==============================
# LOAD PHI MODEL
# ==============================
print("Loading Phi model... ⏳")

llm = Llama(
    model_path="phi-2.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4
)

print("✅ Phi model loaded")

# ==============================
# LOAD INTERVIEW QUESTIONS
# ==============================
with open("skillfit_interview_questions.json", "r") as f:

    interview_data = json.load(f)

# ==============================
# RECORD AUDIO
# ==============================
def record_audio(filename=AUDIO_FILE,
                 duration=DURATION,
                 fs=SAMPLE_RATE):

    print("\n🎤 Recording... Speak now")

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype=np.float32
    )

    sd.wait()

    audio_int16 = np.int16(recording * 32767)

    write(filename, fs, audio_int16)

    print("✅ Recording complete")

    os.system(f"afplay {filename}")

# ==============================
# SPEECH TO TEXT
# ==============================
def speech_to_text(audio_path):

    print("\n🧠 Transcribing with Whisper...")

    result = whisper_model.transcribe(
        audio_path,
        task="translate",
        temperature=0,
        best_of=5,
        fp16=False
    )

    original_text = result["text"]

    detected_lang = result["language"]

    return original_text, detected_lang

# ==============================
# TRANSLATE TO ENGLISH
# ==============================
def translate_to_english(text):

    try:

        translated = translator.translate(
            text,
            dest='en'
        )

        return translated.text

    except Exception as e:

        print("❌ Translation Error:", e)

        return text

# ==============================
# GENERAL TRANSLATION
# ==============================
def translate_text(text, target_lang):

    try:

        translated = translator.translate(
            text,
            dest=target_lang
        )

        return translated.text

    except Exception as e:

        print("❌ Translation Error:", e)

        return text

# ==============================
# PHI EXTRACTION
# ==============================
def extract_info(sentence):

    prompt = f"""
You are an information extractor.

Extract structured data from the sentence.

Return ONLY JSON in ONE LINE:
{{"skills":[],"confidence":"","experience":"","explanations":{{}}}}

Rules:
- skills: short phrases
- confidence: low/medium/high
- experience: none/informal/formal
- explanations: short meanings
- no extra text

Sentence: {sentence}

Output:
"""

    output = llm(
        prompt,
        max_tokens=120,
        temperature=0.2,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["\n\n"]
    )

    response = output["choices"][0]["text"].strip()

    return response

# ==============================
# SAFE JSON PARSER
# ==============================
def parse_json(text):

    try:

        match = re.search(r'\{.*\}', text, re.DOTALL)

        if match:

            clean_json = match.group()

            parsed = json.loads(clean_json)

            parsed.setdefault("skills", [])
            parsed.setdefault("confidence", "low")
            parsed.setdefault("experience", "none")
            parsed.setdefault("explanations", {})
            parsed.setdefault("score", 0)

            return parsed

    except Exception as e:

        print("\n❌ Invalid JSON")
        print(text)

    return {
        "skills": [],
        "confidence": "low",
        "experience": "none",
        "explanations": {},
        "score": 0
    }

# ==============================
# DYNAMIC TTS
# ==============================
def text_to_speech(
        text,
        lang="en",
        filename="q1.mp3"):

    try:

        tts = gTTS(
            text=text,
            lang=lang
        )

        tts.save(filename)

        print(f"🔊 Audio saved as {filename}")

    except Exception as e:

        print("❌ TTS Error:", e)

# ==============================
# ASK QUESTION
# ==============================
def ask_question(question, lang):

    translated_question = translate_text(
        question,
        lang
    )

    print("\n🤖 AI Question:")
    print(translated_question)

    # ==============================
    # CREATE QUESTION AUDIO
    # ==============================
    text_to_speech(
        translated_question,
        lang
    )

    # ==============================
    # WAV2LIP COMMAND
    # ==============================
    command = f"""
    python inference.py \
    --checkpoint_path "{checkpoint}" \
    --face "{face_path}" \
    --audio "q1.mp3" \
    --outfile "{output_path}" \
    --resize_factor 2 \
    --wav2lip_batch_size 32
    """

    print("\n🎬 Running Wav2Lip...\n")

    os.system(command)

    # ==============================
    # PLAY VIDEO
    # ==============================
    os.system(f"open {output_path}")

# ==============================
# DETECT JOB ROLE
# ==============================
def detect_role(skills):

    skills_text = " ".join(skills).lower()

    role_keywords = {

        "Electronics Technician": [
            "electronics",
            "arduino",
            "circuit",
            "repair",
            "pcb"
        ],

        "Wireman": [
            "wiring",
            "electric",
            "cable",
            "mcb"
        ],

        "Painter (Industrial)": [
            "painting",
            "spray",
            "epoxy"
        ],

        "Mason / Construction Worker": [
            "construction",
            "brick",
            "cement",
            "plaster"
        ],

        "Security Guard": [
            "security",
            "guard",
            "cctv"
        ],

        "Domestic Helper / Caretaker": [
            "caretaker",
            "cooking",
            "cleaning"
        ],

        "Factory Floor Worker": [
            "factory",
            "assembly",
            "machine",
            "production"
        ],

        "Garment / Textile Worker": [
            "sewing",
            "garment",
            "textile",
            "stitching"
        ]
    }

    best_role = None
    best_score = 0

    for role, keywords in role_keywords.items():

        score = 0

        for keyword in keywords:

            if keyword in skills_text:
                score += 1

        if score > best_score:

            best_score = score
            best_role = role

    return best_role

# ==============================
# GET ROLE DATA
# ==============================
def get_role_data(role_name):

    for role in interview_data["roles"]:

        if role["role"] == role_name:

            return role

    return None

# ==============================
# GENERATE NEXT QUESTION
# ==============================
def generate_next_question(role_data, asked_questions):

    for q in role_data["questions"]:

        if q["q_id"] not in asked_questions:

            asked_questions.add(q["q_id"])

            return q

    return None

# ==============================
# SCORE ANSWER
# ==============================
def score_answer(question, answer):

    prompt = f"""
You are an interview evaluator.

Question:
{question['question_en']}

Rubric:
{question['rubric']}

Candidate Answer:
{answer}

Rules:
- Score from 1 to 5 only
- Return ONLY number
"""

    output = llm(
        prompt,
        max_tokens=5,
        temperature=0.1
    )

    response = output["choices"][0]["text"].strip()

    try:

        score = int(re.search(r'\d+', response).group())

        if score < 1:
            score = 1

        if score > 5:
            score = 5

        return score

    except:

        return 1

# ==============================
# MEMORY
# ==============================
conversation_history = []

total_score = 0

asked_questions = set()

current_question = "Tell me about yourself."

selected_role_data = None
# ==============================
# MAIN FLOW
# ==============================
if __name__ == "__main__":

    # ==============================
    # ASK FIRST QUESTION
    # ==============================
    ask_question(
        current_question,
        "en"
    )

    # ==============================
    # RECORD AUDIO
    # ==============================
    record_audio()

    # ==============================
    # SPEECH TO TEXT
    # ==============================
    original_text, detected_lang = speech_to_text(AUDIO_FILE)

    print("\n🌍 Detected Language:")
    print(detected_lang)

    print("\n📝 Original Text:")
    print(original_text)

    # ==============================
    # CHECK EMPTY
    # ==============================
    if original_text.strip():

        # ==============================
        # TRANSLATE TO ENGLISH
        # ==============================
        text_en = translate_to_english(original_text)

        print("\n📝 English Translation:")
        print(text_en)

        # ==============================
        # PHI ANALYSIS
        # ==============================
        raw_output = extract_info(text_en)

        print("\n🤖 Raw Phi Output:")
        print(raw_output)

        # ==============================
        # JSON PARSE
        # ==============================
        parsed = parse_json(raw_output)

        # ==============================
        # FALLBACK SKILL DETECTION
        # ==============================
        if len(parsed["skills"]) == 0:

            text_lower = text_en.lower()

            if "electrician" in text_lower:
                parsed["skills"].append("electrician")

            if "wiring" in text_lower:
                parsed["skills"].append("electrical wiring")

            if "electronics" in text_lower:
                parsed["skills"].append("electronics")

            if "sewing" in text_lower:
                parsed["skills"].append("sewing")

            if "construction" in text_lower:
                parsed["skills"].append("construction")

            if "security" in text_lower:
                parsed["skills"].append("security")

        print("\n✅ Parsed JSON:")
        print(json.dumps(parsed, indent=4))

        # ==============================
        # DETECT ROLE
        # ==============================
        detected_role = detect_role(parsed["skills"])

        print("\n🧠 Detected Role:")
        print(detected_role)

        # ==============================
        # ROLE NOT DETECTED
        # ==============================
        if not detected_role:

            print("\n❌ Could not detect suitable role")
            exit()

        # ==============================
        # GET ROLE QUESTIONS
        # ==============================
        selected_role_data = get_role_data(detected_role)

        # ==============================
        # CREATE AI RESPONSE
        # ==============================
        ai_reply = (
            f"Your detected skills are "
            f"{', '.join(parsed['skills'])}. "
            f"Detected role is "
            f"{detected_role}."
        )

        print("\n🤖 AI English Reply:")
        print(ai_reply)

        # ==============================
        # TRANSLATE BACK
        # ==============================
        translated_reply = translate_text(
            ai_reply,
            detected_lang
        )

        print("\n🌍 Final Translated Reply:")
        print(translated_reply)

        # ==============================
        # TEXT TO SPEECH
        # ==============================
        text_to_speech(
            translated_reply,
            detected_lang
        )

        # ==============================
        # PLAY AUDIO
        # ==============================
        os.system("afplay q1.mp3")

        # ==============================
        # SAVE CONVERSATION
        # ==============================
        conversation_history.append({
            "question": current_question,
            "answer": text_en,
            "analysis": parsed
        })

        # ==============================
        # INTERVIEW LOOP
        # ==============================
        for i in range(5):

            # ==============================
            # GENERATE NEXT QUESTION
            # ==============================
            next_question = generate_next_question(
                selected_role_data,
                asked_questions
            )

            # ==============================
            # CHECK END
            # ==============================
            if not next_question:

                print("\n✅ Interview Completed")
                break

            question_text = next_question["question_en"]

            print("\n🧠 Next AI Question:")
            print(question_text)

            # ==============================
            # ASK NEXT QUESTION
            # ==============================
            ask_question(
                question_text,
                detected_lang
            )

            # ==============================
            # RECORD ANSWER
            # ==============================
            record_audio()

            answer_text, _ = speech_to_text(AUDIO_FILE)

            print("\n📝 Candidate Answer:")
            print(answer_text)

            # ==============================
            # END INTERVIEW COMMAND
            # ==============================
            if "end interview" in answer_text.lower():

                print("\n🛑 Interview ended by candidate")
                break

            # ==============================
            # SCORE ANSWER
            # ==============================
            question_score = score_answer(
                next_question,
                answer_text
            )

            print(f"\n⭐ Question Score: {question_score}/5")

            total_score += question_score

            # ==============================
            # SAVE HISTORY
            # ==============================
            conversation_history.append({
                "question": question_text,
                "answer": answer_text,
                "score": question_score
            })

        # ==============================
        # FINAL RESULT
        # ==============================
        print("\n======================")
        print("FINAL INTERVIEW RESULT")
        print("======================")

        print(f"Total Score: {total_score}/25")

        if total_score >= 20:

            fitment = "JOB READY"

        elif total_score >= 15:

            fitment = "REQUIRES UPSKILLING"

        elif total_score >= 10:

            fitment = "MANUAL REVIEW"

        else:

            fitment = "LOW CONFIDENCE"

        print(f"Fitment Status: {fitment}")

        # ==============================
        # FUTURE WAV2LIP
        # ==============================
        print("\n🎬 Ready for Wav2Lip integration")

    else:

        print("⚠️ No speech detected")