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
face_path = "/Users/tejasy/Desktop/small.mp4"

checkpoint ="/Users/tejasy/Documents/documents/govt/Wav2Lip/checkpoints/wav2lip.pth"

output_path = "/Users/tejasy/Documents/documents/govt/project/backend/results/output.mp4"

os.makedirs(
    "/Users/tejasy/Documents/documents/govt/project/backend/results",
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
    model_path="/Users/tejasy/Documents/documents/govt/project/phi-2.Q4_K_M.gguf",
    n_ctx=512,
    n_threads=4
)

print("✅ Phi model loaded")

# ==============================
# LOAD QUESTIONS
# ==============================
with open(
    "/Users/tejasy/Documents/documents/govt/project/backend/skillfit_interview_questions.json",
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

# ==============================
# TEXT TO SPEECH
# ==============================
def text_to_speech(
        text,
        lang="en",
        filename="/Users/tejasy/Documents/documents/govt/project/q1.mp3"):

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
# DELETE OLD VIDEO
# ==============================
if os.path.exists(output_path):

    os.remove(output_path)

    print("🗑 Old video deleted")

# ==============================
# WAV2LIP COMMAND
# ==============================
command = f"""
python3 "/Users/tejasy/Documents/documents/govt/Wav2Lip/inference.py" \
--checkpoint_path "{checkpoint}" \
--face "{face_path}" \
--audio "/Users/tejasy/Documents/documents/govt/project/q1.mp3" \
--outfile "{output_path}" \
--resize_factor 2 \
--wav2lip_batch_size 32
"""

print("\n🎬 Running Wav2Lip...\n")

print(command)

result = os.system(
    f'cd "/Users/tejasy/Documents/documents/govt/Wav2Lip" && {command}'
)

print("\nCOMMAND RESULT:", result)

if os.path.exists(output_path):

    print(f"\n✅ Video saved at:\n{output_path}")

else:

    print("\n❌ Video generation failed")

time.sleep(2)


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

# ==============================
# MAIN FASTAPI FUNCTION
# ==============================
def process_interview(
        user_text,
        lang="en"):

    global selected_role_data
    global current_question

    # ==============================
    # START INTERVIEW
    # ==============================
    if user_text == "start interview":

        current_question = \
            "Tell me about yourself."

        video_path = ask_question(
            current_question,
            lang
        )

        return {

            "status":
            "success",

            "question":
            current_question,

            "video":
            video_path,

            "role":
            "Unknown",

            "skills":
            [],

            "score":
            total_score
        }

    # ==============================
    # EXTRACT INFO
    # ==============================
    raw_output = extract_info(
        user_text
    )

    parsed = parse_json(
        raw_output
    )

    # ==============================
    # FALLBACK SKILLS
    # ==============================
    if len(parsed["skills"]) == 0:

        text_lower = user_text.lower()

        if "electrician" in text_lower:

            parsed["skills"].append(
                "electrician"
            )

        if "wiring" in text_lower:

            parsed["skills"].append(
                "electrical wiring"
            )

        if "electronics" in text_lower:

            parsed["skills"].append(
                "electronics"
            )

        if "construction" in text_lower:

            parsed["skills"].append(
                "construction"
            )

        if "security" in text_lower:

            parsed["skills"].append(
                "security"
            )

    # ==============================
    # DETECT ROLE
    # ==============================
    if selected_role_data is None:

        detected_role = detect_role(
            parsed["skills"]
        )

        if not detected_role:

            return {

                "status":
                "error",

                "message":
                "Could not detect role"
            }

        selected_role_data = \
            get_role_data(
                detected_role
            )

    # ==============================
    # GENERATE NEXT QUESTION
    # ==============================
    next_question = \
        generate_next_question(
            selected_role_data,
            asked_questions
        )

    # ==============================
    # END INTERVIEW
    # ==============================
    if not next_question:

        return {

            "status":
            "completed",

            "score":
            total_score
        }

    # ==============================
    # END COMMAND
    # ==============================
    if "end interview" in user_text.lower():

        return {

            "status":
            "completed",

            "score":
            total_score
        }

    # ==============================
    # SCORE ANSWER
    # ==============================
    question_score = score_answer(
        next_question,
        user_text
    )

    print(
        f"\n⭐ Question Score: "
        f"{question_score}/5"
    )

    # ==============================
    # SAVE HISTORY
    # ==============================
    conversation_history.append({

        "question":
        next_question["question_en"],

        "answer":
        user_text,

        "score":
        question_score
    })

    # ==============================
    # ASK NEXT QUESTION
    # ==============================
    current_question = \
        next_question["question_en"]

    video_path = ask_question(
        current_question,
        lang
    )

    # ==============================
    # RETURN
    # ==============================
    return {

        "status":
        "success",

        "question":
        current_question,

        "video":
        video_path,

        "role":
        selected_role_data["role"],

        "skills":
        parsed["skills"],

        "score":
        total_score
    }
# =====================================
# AUDIO INTERVIEW
# =====================================
def process_audio_interview(

    audio_path,

    lang
):

    try:

        # =============================
        # SPEECH TO TEXT
        # =============================
        answer_text, _ = speech_to_text(
            audio_path
        )

        print("\n📝 User Answer:")
        print(answer_text)

        # =============================
        # PROCESS INTERVIEW
        # =============================
        result = process_interview(

            answer_text,

            lang
        )

        return result

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }
    
# =====================================
# WHISPER SPEECH TO TEXT
# =====================================
def speech_to_text(audio_path):

    try:

        import whisper

        model = whisper.load_model("base")

        result = model.transcribe(audio_path)

        detected_text = result["text"]

        detected_lang = result["language"]

        print("\n🌍 Detected Language:")
        print(detected_lang)

        print("\n📝 Candidate Answer:")
        print(detected_text)

        return detected_text, detected_lang

    except Exception as e:

        print("❌ Speech To Text Error:", e)

        return "", "en"