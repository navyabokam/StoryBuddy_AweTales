import base64
import importlib
import io
import os
import re
import uuid
from time import perf_counter

import librosa
import noisereduce as nr
import numpy as np
import pyttsx3
import whisper
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample_poly
import torch
from pathlib import Path
import sys

load_dotenv(override=True)

APP_SECRET = os.getenv("APP_SECRET", "dev-secret-change-me")
API_KEY = os.getenv("GROK_API_KEY")
SAMPLE_RATE = 16000
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "tiny")
WHISPER_FALLBACK_MODEL = os.getenv("WHISPER_FALLBACK_MODEL", "")
USE_FASTER_WHISPER = os.getenv("USE_FASTER_WHISPER", "0") == "1"
FASTER_WHISPER_COMPUTE_TYPE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
ENABLE_NOISE_REDUCTION = os.getenv("ENABLE_NOISE_REDUCTION", "0") == "1"
ENABLE_SERVER_TTS = os.getenv("ENABLE_SERVER_TTS", "0") == "1"
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MAX_CHAT_MESSAGES = int(os.getenv("MAX_CHAT_MESSAGES", "10"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"
MIN_AUDIO_SECONDS = float(os.getenv("MIN_AUDIO_SECONDS", "0.35"))
ENABLE_ASR_WARMUP = os.getenv("ENABLE_ASR_WARMUP", "0") == "1"

SELF_HARM_PATTERN = re.compile(
    r"\b(suicide|kill\s*myself|self\s*harm|self-harm|end\s*my\s*life|how\s*to\s*die|commit\s*suicide)\b",
    re.IGNORECASE,
)

app = Flask(__name__)
app.secret_key = APP_SECRET

OPENAI_CLIENT = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1") if API_KEY else None


@app.before_request
def ensure_client_state() -> None:
    if "sessions" not in session:
        session["sessions"] = {"Chat 1": []}
        session["current_session"] = "Chat 1"


def load_whisper_model(model_name: str | None = None):
    name = model_name or WHISPER_MODEL_NAME
    if not hasattr(load_whisper_model, "_models"):
        load_whisper_model._models = {}
    if name not in load_whisper_model._models:
        load_whisper_model._models[name] = whisper.load_model(name)
    return load_whisper_model._models[name]


def load_faster_whisper_model(model_name: str | None = None):
    name = model_name or WHISPER_MODEL_NAME
    if not hasattr(load_faster_whisper_model, "_models"):
        load_faster_whisper_model._models = {}
    if name not in load_faster_whisper_model._models:
        module = importlib.import_module("faster_whisper")
        WhisperModel = getattr(module, "WhisperModel")
        load_faster_whisper_model._models[name] = WhisperModel(name, compute_type=FASTER_WHISPER_COMPUTE_TYPE)
    return load_faster_whisper_model._models[name]


def transcribe_with_faster_whisper(audio_for_whisper: np.ndarray) -> str:
    model = load_faster_whisper_model()
    segments, _ = model.transcribe(
        audio_for_whisper,
        language="en",
        condition_on_previous_text=False,
        temperature=0,
        beam_size=1,
    )
    return " ".join(seg.text.strip() for seg in segments if seg.text).strip()


def transcribe_audio(audio_for_whisper: np.ndarray) -> str:
    if USE_FASTER_WHISPER:
        return transcribe_with_faster_whisper(audio_for_whisper)

    base_options = {
        "fp16": False,
        "language": "en",
        "temperature": 0,
        "condition_on_previous_text": False,
    }
    transcript = load_whisper_model().transcribe(audio_for_whisper, **base_options)["text"].strip()

    # If transcript is too short/empty, retry once with a stronger model.
    if (
        len(transcript.split()) < 2
        and WHISPER_FALLBACK_MODEL
        and WHISPER_FALLBACK_MODEL != WHISPER_MODEL_NAME
    ):
        try:
            fallback = load_whisper_model(WHISPER_FALLBACK_MODEL)
            transcript = fallback.transcribe(audio_for_whisper, **base_options)["text"].strip()
        except RuntimeError:
            # Keep primary transcript if fallback model cannot be loaded on low-memory systems.
            pass

    return transcript


def retrieve_context(query: str) -> str:
    if not _VECTOR_DB or not query.strip():
        return ""
    try:
        docs = _VECTOR_DB.similarity_search(query, k=RAG_TOP_K)
        return "\n\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", "")).strip()
    except Exception:
        return ""


def clean_transcript(text: str) -> str:
    # Collapse consecutive duplicate words like "hello hello hello" -> "hello".
    cleaned = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return cleaned.strip()


def remove_repeated_sentences(text: str) -> str:
    seen: set[str] = set()
    result: list[str] = []
    for sentence in re.split(r"[.!?]", text):
        normalized = " ".join(sentence.strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    # Join with '. ' and ensure each sentence is capitalized and ends with a period
    return ". ".join(s.capitalize() for s in result if s) + "."


def is_self_harm_request(text: str) -> bool:
    return bool(SELF_HARM_PATTERN.search(text or ""))


def get_india_crisis_response() -> str:
    return (
        "I cannot help with instructions or stories about suicide or self-harm. "
        "If you are feeling overwhelmed, please talk to someone you trust right now. "
        "In India, you can contact Tele-MANAS at 14416 or 1-800-891-4416 (24x7), "
        "or call emergency services at 112 if there is immediate danger. "
        "If you are outside India, contact your local emergency number or a local crisis helpline."
    )


def setup_vector_db():
    try:
        from langchain_community.document_loaders import TextLoader
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import CharacterTextSplitter

        if not os.path.exists("knowledge.txt"):
            with open("knowledge.txt", "w", encoding="utf-8") as f:
                f.write("Artificial intelligence is machines simulating human intelligence.\n")

        loader = TextLoader("knowledge.txt")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(text_splitter.split_documents(docs), embeddings)
    except Exception:
        return None


_VECTOR_DB = setup_vector_db()


def get_state() -> tuple[dict, str]:
    sessions = session.get("sessions", {"Chat 1": []})
    current = session.get("current_session", "Chat 1")
    if current not in sessions:
        sessions[current] = []
    return sessions, current


def decode_audio_upload(file_storage) -> np.ndarray:
    audio_bytes = file_storage.read()
    sample_rate, audio = wav_read(io.BytesIO(audio_bytes))

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / float(max_val)
    else:
        audio = audio.astype(np.float32)

    if sample_rate != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sample_rate).astype(np.float32)

    return audio


def synthesize_to_base64_wav(text: str) -> str:
    tmp_path = f"ai_response_{uuid.uuid4().hex}.wav"
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# --- Silero VAD via torch.hub ---
def filter_speech_segments(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    # TEMPORARY: Bypass Silero VAD and just return the input audio for 6 seconds
    # This avoids get_speech_timestamps error and ensures fast response
    # You can later restore VAD when the function is fixed
    return audio


# --- Advanced Noise Suppression ---
def advanced_noise_suppression(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    # Placeholder for RNNoise or deep-learning denoising
    # If RNNoise or deep model is available, use it here
    # Otherwise fallback to noisereduce
    try:
        import rnnoise
        # Example: audio = rnnoise.process(audio, sample_rate)
        # (You must install and integrate RNNoise Python bindings)
        pass  # Replace with RNNoise call
    except ImportError:
        audio = nr.reduce_noise(y=audio, sr=sample_rate, prop_decrease=0.8)
    return audio

# --- Advanced TTS fallback only ---
def advanced_tts(text: str) -> str:
    return synthesize_to_base64_wav(text)

# --- Real-Time Streaming Endpoint ---
@app.post("/api/voice/stream")
def api_voice_stream():
    if not API_KEY:
        return jsonify({"ok": False, "error": "Missing GROK_API_KEY in environment."}), 400
    if "audio_chunk" not in request.files:
        return jsonify({"ok": False, "error": "No audio chunk uploaded."}), 400
    try:
        chunk_audio = decode_audio_upload(request.files["audio_chunk"])
        sessions_data, current = get_state()
        # Buffering: store chunks in session
        if "audio_buffer" not in session:
            session["audio_buffer"] = np.array([], dtype=np.float32)
        session["audio_buffer"] = np.concatenate([session["audio_buffer"], chunk_audio])
        session.modified = True
        # If chunk is last, process pipeline
        is_last = request.form.get("is_last", "0") == "1"
        result = {}
        if is_last:
            result = run_pipeline(session["audio_buffer"], sessions_data, current)
            session["audio_buffer"] = np.array([], dtype=np.float32)
            session["sessions"] = sessions_data
            session.modified = True
        return jsonify({
            "ok": True,
            "result": result,
            "buffer_len": len(session["audio_buffer"]),
            "current_session": current,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# --- ASR with Segmentation & Timestamps ---
def transcribe_audio_segments(audio_for_whisper: np.ndarray) -> list:
    segments = []
    if USE_FASTER_WHISPER:
        model = load_faster_whisper_model()
        segs, _ = model.transcribe(
            audio_for_whisper,
            language="en",
            condition_on_previous_text=False,
            temperature=0,
            beam_size=1,
        )
        for seg in segs:
            segments.append({
                "text": seg.text.strip(),
                "start": seg.start,
                "end": seg.end,
            })
    else:
        model = load_whisper_model()
        try:
            result = model.transcribe(
                audio_for_whisper,
                fp16=False,
                language="en",
                temperature=0,
                condition_on_previous_text=False,
            )
        except Exception:
            # Whisper can fail on extremely short/near-silent clips.
            return []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start": seg["start"],
                "end": seg["end"],
            })
    return segments

# --- Pipeline Optimization ---
def run_pipeline(raw_audio: np.ndarray, sessions: dict, current_session: str) -> dict:
    pipeline_t0 = perf_counter()
    clean_audio = raw_audio.flatten()
    if clean_audio.size == 0:
        return {"transcript": "", "answer": "", "audio": "", "segments": []}

    if ENABLE_NOISE_REDUCTION:
        clean_audio = advanced_noise_suppression(clean_audio, SAMPLE_RATE)

    # VAD: filter only speech segments
    clean_audio = filter_speech_segments(clean_audio, SAMPLE_RATE)
    peak = float(np.max(np.abs(clean_audio))) if clean_audio.size > 0 else 0.0
    if peak <= 1e-8:
        return {"transcript": "", "answer": "", "audio": "", "segments": []}

    audio_for_whisper = clean_audio.astype(np.float32) / (peak + 1e-6)
    audio_for_whisper, _ = librosa.effects.trim(audio_for_whisper, top_db=25)

    min_samples = int(SAMPLE_RATE * MIN_AUDIO_SECONDS)
    if audio_for_whisper.size < min_samples:
        return {"transcript": "", "answer": "", "audio": "", "segments": []}

    if audio_for_whisper.size == 0:
        return {"transcript": "", "answer": "", "audio": "", "segments": []}

    stt_t0 = perf_counter()
    segments = transcribe_audio_segments(audio_for_whisper)
    transcript = " ".join(seg["text"] for seg in segments)
    transcript = clean_transcript(transcript)
    transcript = remove_repeated_sentences(transcript)
    stt_ms = (perf_counter() - stt_t0) * 1000
    if not transcript:
        return {"transcript": "", "answer": "", "audio": "", "segments": segments}

    if is_self_harm_request(transcript):
        answer = get_india_crisis_response()
        sessions[current_session].append({"role": "user", "content": transcript})
        sessions[current_session].append({"role": "assistant", "content": answer})
        sessions[current_session] = sessions[current_session][-MAX_CHAT_MESSAGES:]
        audio_b64 = advanced_tts(answer) if ENABLE_SERVER_TTS else ""
        return {"transcript": transcript, "answer": answer, "audio": audio_b64, "segments": segments}

    sessions[current_session].append({"role": "user", "content": transcript})
    sessions[current_session] = sessions[current_session][-MAX_CHAT_MESSAGES:]

    limit_prompt = (
        "You are a friendly story-telling assistant for kids aged 6-18. Tell engaging stories, answer questions and doubts in between, and always continue the story after answering. Use simple, clear language. If the user interrupts or asks a question, pause the story, answer, then ask 'Do you want me to continue the story?' and wait for the user's response. If the user says yes, resume the story flow. Keep the story interactive and fun."
    )
    max_tokens = 220
    if (
        "briefly" in transcript.lower()
        or "detailed" in transcript.lower()
        or "100 words" in transcript.lower()
        or "long" in transcript.lower()
        or "story" in transcript.lower()
        or "continue" in transcript.lower()
        or "question" in transcript.lower()
        or "doubt" in transcript.lower()
    ):
        limit_prompt = (
            "You are a friendly story-telling assistant for kids aged 6-18. Tell engaging stories, answer questions and doubts in between, and always continue the story after answering. Use simple, clear language. If the user interrupts or asks a question, pause the story, answer, then ask 'Do you want me to continue the story?' and wait for the user's response. If the user says yes, resume the story flow. Make the story longer and more detailed, at least 100 words. Always finish your sentences and never leave a story incomplete."
        )
        max_tokens = 1200

    rag_context = retrieve_context(transcript)
    history = sessions[current_session][-MAX_CHAT_MESSAGES:]
    system_prompt = (
        f"You are a helpful assistant. {limit_prompt} "
        "Always end with a complete final sentence and avoid cutting off mid-sentence."
    )
    if rag_context:
        system_prompt = f"{system_prompt}\n\nUse this knowledge base context when relevant:\n{rag_context}"

    llm_t0 = perf_counter()
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
    ]
    completion = OPENAI_CLIENT.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=max_tokens,
    )
    answer = completion.choices[0].message.content or ""

    finish_reason = getattr(completion.choices[0], "finish_reason", None)
    if finish_reason == "length":
        continuation = OPENAI_CLIENT.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                *messages,
                {"role": "assistant", "content": answer},
                {
                    "role": "user",
                    "content": "Continue from exactly where you stopped. Complete the unfinished sentence first, then finish briefly.",
                },
            ],
            max_tokens=120,
        )
        answer = f"{answer} {continuation.choices[0].message.content or ''}".strip()

    if answer and answer[-1] not in ".!?\"'":
        answer = f"{answer}."
    llm_ms = (perf_counter() - llm_t0) * 1000

    sessions[current_session].append({"role": "assistant", "content": answer})
    sessions[current_session] = sessions[current_session][-MAX_CHAT_MESSAGES:]
    audio_b64 = ""
    if ENABLE_SERVER_TTS:
        audio_b64 = advanced_tts(answer)

    total_ms = (perf_counter() - pipeline_t0) * 1000
    print(
        f"[latency] total={total_ms:.0f}ms stt={stt_ms:.0f}ms llm={llm_ms:.0f}ms "
        f"noise_reduction={'on' if ENABLE_NOISE_REDUCTION else 'off'} tts={'on' if ENABLE_SERVER_TTS else 'off'}"
    )

    return {"transcript": transcript, "answer": answer, "audio": audio_b64, "segments": segments}


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    sessions_data, current = get_state()
    return jsonify({"sessions": sessions_data, "current_session": current})


@app.post("/api/chat/new")
def api_chat_new():
    sessions_data, _ = get_state()
    i = 1
    while f"Chat {i}" in sessions_data:
        i += 1
    new_chat = f"Chat {i}"
    sessions_data[new_chat] = []
    session["sessions"] = sessions_data
    session["current_session"] = new_chat
    session.modified = True
    return jsonify({"ok": True, "current_session": new_chat})


@app.post("/api/chat/clear")
def api_chat_clear():
    sessions_data, current = get_state()
    sessions_data[current] = []
    session["sessions"] = sessions_data
    session.modified = True
    return jsonify({"ok": True})


@app.post("/api/chat/switch")
def api_chat_switch():
    data = request.get_json(silent=True) or {}
    target = data.get("chat")
    sessions_data, current = get_state()
    if not target or target not in sessions_data:
        return jsonify({"ok": False, "error": "Chat not found"}), 404
    session["current_session"] = target
    session.modified = True
    return jsonify({"ok": True, "current_session": target, "previous": current})


@app.post("/api/chat/delete")
def api_chat_delete():
    data = request.get_json(silent=True) or {}
    target = data.get("chat")
    sessions_data, current = get_state()

    if not target or target not in sessions_data:
        return jsonify({"ok": False, "error": "Chat not found"}), 404

    del sessions_data[target]
    if not sessions_data:
        sessions_data["Chat 1"] = []

    if current == target:
        session["current_session"] = next(iter(sessions_data.keys()))

    session["sessions"] = sessions_data
    session.modified = True
    return jsonify({"ok": True, "current_session": session["current_session"]})


@app.post("/api/voice")
def api_voice():
    if not API_KEY:
        return jsonify({"ok": False, "error": "Missing GROK_API_KEY in environment."}), 400

    if "audio" not in request.files:
        return jsonify({"ok": False, "error": "No audio file uploaded."}), 400

    try:
        raw_audio = decode_audio_upload(request.files["audio"])
        sessions_data, current = get_state()
        result = run_pipeline(raw_audio, sessions_data, current)
        session["sessions"] = sessions_data
        session.modified = True
        return jsonify(
            {
                "ok": True,
                "transcript": result["transcript"],
                "answer": result["answer"],
                "audio": result["audio"],
                "sessions": sessions_data,
                "current_session": current,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Keep startup fast/reliable by default; warmup is optional via env toggle.
    if ENABLE_ASR_WARMUP:
        try:
            if USE_FASTER_WHISPER:
                load_faster_whisper_model()
            else:
                load_whisper_model()
        except RuntimeError as exc:
            print(f"[startup] Whisper warmup skipped: {exc}")
        except Exception as exc:
            print(f"[startup] ASR warmup skipped: {exc}")
    app.run(host="0.0.0.0", port=8000, debug=FLASK_DEBUG, use_reloader=False, threaded=True)
