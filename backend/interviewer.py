# ==============================
# IMPORTS
# ==============================
import time
import os
import json
import re

from gtts import gTTS
from googletrans import Translator
from llama_cpp import Llama

# ==============================
# CONFIG
# ==============================
face_path = "../small.mp4"

checkpoint = "../wav2lip.pth"

output_path = "results/output.mp4"

os.makedirs(
    "results",
    exist_ok=True
)

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
# LOAD QUESTIONS
# ==============================
with open(
    "skillfit_interview_questions.json",
    "r"
) as f:

    interview_data = json.load(f)

# ==============================
# MEMORY
# ==============================
conversation_history = []

total_score = 0

asked_questions = set()

current_question = \
    "Tell me about yourself."

selected_role_data = None

# ==============================
# TRANSLATE
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
# ASK QUESTION
# ==============================
def ask_question(question, lang):

    translated_question = translate_text(
        question,
        lang
    )

    print("\n🤖 AI Question:")
    print(translated_question)

    text_to_speech(
        translated_question,
        lang,
        "q1.mp3"
    )

    # ==============================
    # DELETE OLD VIDEO
    # ==============================
    if os.path.exists(output_path):

        os.remove(output_path)

        print("🗑 Old video deleted")

    # ==============================
    # WAV2LIP COMMAND
    # ==============================
    command = f"""
    python3 inference.py \
    --checkpoint_path "{checkpoint}" \
    --face "{face_path}" \
    --audio "../backend/q1.mp3" \
    --outfile "../backend/{output_path}" \
    --resize_factor 2 \
    --wav2lip_batch_size 32
    """

    print("\n🎬 Running Wav2Lip...\n")

    result = os.system(
        f'cd "../Wav2Lip" && {command}'
    )

    print("\nCOMMAND RESULT:", result)

    if os.path.exists(output_path):

        print(f"\n✅ Video saved at:\n{output_path}")

    else:

        print("\n❌ Video generation failed")

    return output_path

# ==============================
# TEXT TO SPEECH
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
# EXTRACT INFO
# ==============================
def extract_info(sentence):

    prompt = f"""
You are an information extractor.

Extract structured data from the sentence.

Return ONLY JSON:
{{"skills":[]}}

Sentence:
{sentence}
"""

    output = llm(
        prompt,
        max_tokens=120,
        temperature=0.2,
        top_p=0.9
    )

    response = output["choices"][0]["text"].strip()

    return response

# ==============================
# SAFE JSON PARSER
# ==============================
def parse_json(text):

    try:

        match = re.search(
            r'\{.*\}',
            text,
            re.DOTALL
        )

        if match:

            clean_json = match.group()

            parsed = json.loads(
                clean_json
            )

            parsed.setdefault(
                "skills",
                []
            )

            return parsed

    except Exception as e:

        print("\n❌ Invalid JSON")
        print(text)

    return {
        "skills": []
    }

# ==============================
# DETECT JOB ROLE
# ==============================
def detect_role(skills):

    skills_text = \
        " ".join(skills).lower()

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
            "electrician",
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
def generate_next_question(
        role_data,
        asked_questions):

    for q in role_data["questions"]:

        if q["q_id"] not in asked_questions:

            asked_questions.add(
                q["q_id"]
            )

            return q

    return None

# ==============================
# SCORE ANSWER
# ==============================
def score_answer(question, answer):

    global total_score

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

    response = \
        output["choices"][0]["text"].strip()

    try:

        score = int(
            re.search(
                r'\d+',
                response
            ).group()
        )

        if score < 1:
            score = 1

        if score > 5:
            score = 5

        total_score += score

        return score

    except:

        total_score += 1

        return 1
