import streamlit as st
import sounddevice as sd
import numpy as np
import noisereduce as nr
import time
import os
import queue
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import pyttsx3

load_dotenv(override=True)

# PAGE CONFIG
st.set_page_config(
    page_title="Voice Assistant",
    page_icon="🎙️",
    layout="wide"
)

# CONFIG
RESPONSE_AUDIO_PATH = "ai_response.wav"
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(SAMPLE_RATE * (CHUNK_DURATION_MS / 1000))
API_KEY = os.getenv("GROK_API_KEY")

# AUDIO QUEUE
def get_shared_queue():
    if "audio_q" not in st.session_state:
        st.session_state.audio_q = queue.Queue()
    return st.session_state.audio_q

audio_q = get_shared_queue()

# LOAD WHISPER
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

whisper_model = load_whisper_model()

# AUDIO CALLBACK
def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())

# RECORD AUDIO
def handle_recording():

    st.session_state.is_recording = True
    st.session_state.recorded_data = []

    while not audio_q.empty():
        audio_q.get()

    stream = sd.InputStream(
        channels=1,
        samplerate=16000,
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    )

    start_time = time.time()

    with stream:
        while st.session_state.is_recording:

            try:
                chunk = audio_q.get(timeout=0.5)
                st.session_state.recorded_data.append(chunk)

                if time.time() - start_time >= 12:
                    st.session_state.is_recording = False

            except queue.Empty:
                continue

    if len(st.session_state.recorded_data) == 0:
        return np.zeros((1,1))

    return np.concatenate(st.session_state.recorded_data, axis=0)

# AI PIPELINE
def run_pipeline(raw_audio):

    try:

        audio_flat = raw_audio.flatten()

        clean_audio = nr.reduce_noise(
            y=audio_flat,
            sr=SAMPLE_RATE,
            prop_decrease=0.8
        )

        with st.status("Thinking...", expanded=False):

            audio_for_whisper = clean_audio.astype(np.float32) / (np.max(np.abs(clean_audio)) + 1e-6)

            result = whisper_model.transcribe(
                audio_for_whisper,
                fp16=False,
                language="en"
            )

            transcript = result.get("text", "").strip()

            if not transcript:
                return

            st.session_state.sessions[st.session_state.current_session].append(
                {"role": "user", "content": transcript}
            )

            client = OpenAI(
                api_key=API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )

            limit_prompt = (
                "You are a storytelling assistant for kids aged 6–18. "
                "Always give complete paragraphs and never stop mid sentence. "
                "If the user interrupts with a question, answer it clearly and then ask "
                "'Do you want me to continue the story?'. "
                "Use simple language and keep the story engaging."
            )

            messages = [{"role": "system", "content": limit_prompt}] + st.session_state.sessions[st.session_state.current_session]

            answer = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=1024,
                temperature=0.7
            ).choices[0].message.content.strip()

            if answer and answer[-1] not in ".!?":
                answer = answer.rstrip() + "."

            st.session_state.sessions[st.session_state.current_session].append(
                {"role": "assistant", "content": answer}
            )

            st.session_state.stop_speaking = False

            engine = pyttsx3.init()
            engine.save_to_file(answer, RESPONSE_AUDIO_PATH)
            engine.runAndWait()

            st.session_state.play_audio = True

    except Exception as e:
        st.error(f"Error: {e}")

# MAIN UI
def main():

    if "sessions" not in st.session_state:
        st.session_state.sessions = {"Chat 1": []}

    if "current_session" not in st.session_state:
        st.session_state.current_session = "Chat 1"

    if "stop_speaking" not in st.session_state:
        st.session_state.stop_speaking = False

    if "play_audio" not in st.session_state:
        st.session_state.play_audio = False

    session_col, chat_col, control_col = st.columns([1.2,4,1.2])

    # SESSION PANEL
    with session_col:

        st.subheader("Session")

        if st.button("🗑 New Chat"):

            i = 1
            while f"Chat {i}" in st.session_state.sessions:
                i += 1

            new_id = f"Chat {i}"

            st.session_state.sessions[new_id] = []
            st.session_state.current_session = new_id
            st.session_state.play_audio = False
            st.session_state.stop_speaking = True
            st.rerun()

        if st.button("🧹 Clear Chat"):
            st.session_state.sessions[st.session_state.current_session] = []
            st.session_state.stop_speaking = True
            st.rerun()

        st.markdown("---")

        chats = list(st.session_state.sessions.keys())

        for chat in chats:

            col1, col2 = st.columns([4,1])

            with col1:

                label = f"💬 {chat}" if chat == st.session_state.current_session else chat

                if st.button(label, key=f"btn_{chat}", use_container_width=True):
                    st.session_state.current_session = chat
                    st.session_state.stop_speaking = True
                    st.rerun()

            with col2:

                if st.button("🗑", key=f"del_{chat}"):

                    del st.session_state.sessions[chat]

                    remaining = list(st.session_state.sessions.keys())

                    if remaining:
                        st.session_state.current_session = remaining[0]
                    else:
                        st.session_state.sessions["Chat 1"] = []
                        st.session_state.current_session = "Chat 1"

                    st.session_state.stop_speaking = True
                    st.rerun()

    # CHAT PANEL
    with chat_col:

        st.title(f"🎙️ {st.session_state.current_session}")

        messages = st.session_state.sessions.get(st.session_state.current_session, [])

        with st.container(height=500):

            for msg in messages:
                st.chat_message(msg["role"]).write(msg["content"])

        if st.session_state.play_audio and not st.session_state.stop_speaking:

            st.audio(
                RESPONSE_AUDIO_PATH,
                format="audio/wav",
                autoplay=True
            )

    # CONTROL PANEL
    with control_col:

        st.subheader("Voice On")

        if not st.session_state.get("is_recording", False):

            if st.button("🎙️ Start Talking"):

                st.session_state.stop_speaking = False

                raw_audio = handle_recording()

                run_pipeline(raw_audio)

                st.rerun()

        else:

            if st.button("🎙️ Recording... Click to Stop"):

                st.session_state.is_recording = False
                st.rerun()

        st.subheader("Stop AI")

        if st.button("⏹ Stop AI"):

            st.session_state.stop_speaking = True
            st.session_state.play_audio = False
            st.rerun()


if __name__ == "__main__":
    main()