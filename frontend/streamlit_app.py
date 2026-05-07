# frontend/streamlit_app.py

import streamlit as st
import requests
import os
import time
from audio_recorder_streamlit import audio_recorder

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="SkillFit AI",
    page_icon="🎤",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.title {
    font-size: 3rem;
    font-weight: bold;
    color: white;
}

.subtitle {
    font-size: 1.2rem;
    color: #BBBBBB;
}

.card {
    background-color: #161B22;
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #30363D;
}

.question-box {

    background: linear-gradient(
        135deg,
        #1F6FEB,
        #238636
    );

    padding: 25px;

    border-radius: 20px;

    color: white;

    font-size: 24px;

    font-weight: bold;
}

.metric-card {

    background-color: #161B22;

    padding: 15px;

    border-radius: 20px;

    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# SESSION STATE
# =========================================
if "started" not in st.session_state:

    st.session_state.started = False

if "question" not in st.session_state:

    st.session_state.question = \
        "Tell me about yourself."

if "role" not in st.session_state:

    st.session_state.role = "Unknown"

if "skills" not in st.session_state:

    st.session_state.skills = []

if "score" not in st.session_state:

    st.session_state.score = 0

# =========================================
# HEADER
# =========================================
st.markdown(
    '<div class="title">🎤 SkillFit AI Interviewer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI Powered Multilingual Interview Platform'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

# =========================================
# LANGUAGE
# =========================================
language = st.selectbox(
    "🌍 Select Language",
    [
        "en",
        "kn",
        "hi",
        "ta",
        "te"
    ]
)

# =========================================
# START SCREEN
# =========================================
if not st.session_state.started:

    st.markdown("""
    <div class="card">

    <h2>🚀 Start Your AI Interview</h2>

    <p>
    Click below to begin your AI interview.
    Lip synced interviewer video will be generated automatically.
    </p>

    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "▶ Start Interview",
        use_container_width=True
    ):

        with st.spinner(
            "Generating AI interviewer..."
        ):

            response = requests.post(

                "http://127.0.0.1:8000/interview",

                json={

                    "text":
                    "start interview",

                    "lang":
                    language
                }
            )

            data = response.json()

            if data["status"] == "success":

                st.session_state.started = True

                st.session_state.question = \
                    data["question"]

                st.session_state.role = \
                    data["role"]

                st.session_state.skills = \
                    data["skills"]

                st.session_state.score = \
                    data["score"]

                time.sleep(1)

                st.rerun()

# =========================================
# MAIN INTERVIEW UI
# =========================================
if st.session_state.started:

    col1, col2 = st.columns([2, 1])

    # =====================================
    # LEFT SIDE
    # =====================================
    with col1:

        # =================================
        # QUESTION
        # =================================
        st.markdown(
            f"""
            <div class="question-box">

            🧠 {st.session_state.question}

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # =================================
        # VIDEO
        # =================================
        st.markdown("""
        <div class="card">

        <h3>🎬 AI Interviewer</h3>

        </div>
        """, unsafe_allow_html=True)

        video_path = \
            "/Users/tejasy/Documents/documents/govt/project/backend/results/output.mp4"

        if os.path.exists(video_path):

            video_file = open(
                video_path,
                "rb"
            )

            video_bytes = video_file.read()

            st.video(
                video_bytes,
                start_time=0
            )

        else:

            st.warning(
                "AI interviewer video not generated yet."
            )

        # =================================
        # AUDIO RECORD
        # =================================
        st.markdown("""
        <div class="card">

        <h3>🎤 Speak Your Answer</h3>

        <p>
        Click microphone and answer the question.
        </p>

        </div>
        """, unsafe_allow_html=True)

        audio_bytes = audio_recorder(

            pause_threshold=3.0,

            sample_rate=41000,

            text="🎤 Click To Record",

            recording_color="#ff4b4b",

            neutral_color="#6c757d",

            icon_name="microphone",

            icon_size="2x"
        )

        # =================================
        # AUDIO RECEIVED
        # =================================
        if audio_bytes:

            st.success(
                "✅ Answer Recorded"
            )

            st.audio(audio_bytes)

            audio_path = \
                "/Users/tejasy/Documents/documents/govt/project/input.wav"

            with open(audio_path, "wb") as f:

                f.write(audio_bytes)

            # =================================
            # SEND TO BACKEND
            # =================================
            with st.spinner(
                "🧠 Processing Interview..."
            ):

                files = {

                    "file": open(audio_path, "rb")
                }

                data = {

                    "lang": language
                }

                response = requests.post(

                    "http://127.0.0.1:8000/interview_audio",

                    files=files,

                    data=data
                )

                data = response.json()

                # =================================
                # SUCCESS
                # =================================
                if data["status"] == "success":

                    st.session_state.question = \
                        data["question"]

                    st.session_state.role = \
                        data["role"]

                    st.session_state.skills = \
                        data["skills"]

                    st.session_state.score = \
                        data["score"]

                    time.sleep(1)

                    st.rerun()

                # =================================
                # COMPLETED
                # =================================
                elif data["status"] == "completed":

                    st.balloons()

                    st.success(
                        "✅ Interview Completed"
                    )

                    st.metric(
                        "Final Score",
                        data["score"]
                    )

                # =================================
                # ERROR
                # =================================
                else:

                    st.error(
                        data["message"]
                    )

    # =====================================
    # RIGHT SIDE
    # =====================================
    with col2:

        st.markdown("""
        <div class="card">

        <h3>📊 Interview Analytics</h3>

        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Current Score",
            st.session_state.score
        )

        st.write("")

        st.markdown("""
        <div class="card">

        <h3>🛠 Detected Skills</h3>

        </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.skills) > 0:

            for skill in st.session_state.skills:

                st.success(skill)

        else:

            st.info("No skills detected yet.")

        st.write("")

        st.markdown("""
        <div class="card">

        <h3>👤 Detected Role</h3>

        </div>
        """, unsafe_allow_html=True)

        st.info(st.session_state.role)