# StoryBuddy - Voice Story Assistant (Easy Guide)

StoryBuddy - Voice Story Assistant is an interactive web app that turns spoken input into kid-friendly storytelling conversations.
It supports both one-shot voice requests and chunked streaming voice input, keeps multi-chat session history in Flask session state, and returns AI-generated story responses with optional server-side audio.

This project is a website where you can:
1. Click a microphone button.
2. Speak.
3. See your speech turned into text.
4. Get a story answer from AI.

The website has 2 parts:
- Frontend: what you see in the browser (`HTML`, `CSS`, `JavaScript`).
- Backend: Python server that does speech-to-text and AI reply.

Core capabilities:
- Speech-to-text with Whisper (or `faster-whisper` when enabled).
- Story/Q&A generation through Groq's OpenAI-compatible chat API.
- Lightweight retrieval-augmented context using `knowledge.txt` and a FAISS vector store.
- Session-based chat management (`new`, `switch`, `clear`, `delete`).
- Safety guardrails for self-harm requests with crisis-support response.

## Architecture

High-level request flow:

1. Browser UI (`templates/index.html`, `static/app.js`) records microphone audio.
2. Frontend sends audio to Flask API:
- `POST /api/voice` for full audio upload.
- `POST /api/voice/stream` for chunked streaming uploads.
3. Backend (`main.py`) decodes and normalizes WAV audio, then applies preprocessing:
- optional noise reduction,
- silence trimming,
- short-input filtering.
4. ASR pipeline transcribes audio into text using Whisper/faster-whisper.
5. Backend optionally retrieves relevant context from `knowledge.txt` using LangChain + FAISS.
6. Prompt + conversation history are sent to Groq (`llama-3.1-8b-instant`) via OpenAI SDK.
7. Response is post-processed, saved into session chat history, and returned as JSON.
8. If enabled, backend also synthesizes TTS and returns base64 WAV audio.

Main architectural components:
- Presentation layer: HTML/CSS/JS single-page interface.
- API layer: Flask routes for state, chat lifecycle, and voice processing.
- AI pipeline layer: ASR, transcript cleanup, RAG retrieval, LLM response generation.
- State layer: per-user session storage in Flask signed cookies/session state.

## Tech Stack

Frontend:
- HTML5 templates (`Jinja2` render via Flask).
- CSS3 (`static/styles.css`).
- Vanilla JavaScript (`static/app.js`) for recording, API calls, and UI updates.

Backend:
- Python 3.x.
- Flask (`main.py`) for HTTP API and templating.
- `python-dotenv` for environment configuration.

AI/Audio/ML:
- `openai-whisper` and optional `faster-whisper` for ASR.
- `openai` SDK with Groq OpenAI-compatible endpoint.
- `librosa`, `numpy`, `scipy`, `noisereduce` for audio preprocessing.
- `pyttsx3` for optional server-side TTS generation.
- `torch` for model runtime support.

RAG and retrieval:
- LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`).
- `FAISS` vector index (`faiss-cpu`).
- Hugging Face sentence embeddings (`all-MiniLM-L6-v2` via `langchain-huggingface`).

Legacy/alternate interface:
- `app.py` contains an older Streamlit-based interface retained for reference.

## What You Need First

1. Windows computer.
2. Internet connection.
3. Python installed (we used Python 3.13).
4. `ffmpeg` installed and added to `PATH`.
5. A Groq API key.

## Step-by-Step Setup (Very Simple)

### Step 1: Open the project folder

Open PowerShell and run:

```powershell
cd "c:\Users\DELL\Downloads\AweTales_StoryTelling-main\AweTales_StoryTelling-main"
```

### Step 2: Create `.env` file

In this project folder, create a file named `.env` and add:

```env
GROK_API_KEY=your_real_api_key_here
APP_SECRET=anything_you_like
```

Optional settings (you can skip these at first):

```env
WHISPER_MODEL=tiny
ENABLE_ASR_WARMUP=0
FLASK_DEBUG=0
```

### Step 3: Install Python packages

Run this command:

```powershell
c:/python313/python.exe -m pip install -r requirements.txt
```

If you still get missing-package errors, run:

```powershell
c:/python313/python.exe -m pip install librosa noisereduce pyttsx3 openai openai-whisper python-dotenv flask torch
```

### Step 4: Start the server

Run:

```powershell
c:/python313/python.exe "c:\Users\DELL\Downloads\AweTales_StoryTelling-main\AweTales_StoryTelling-main\main.py"
```

Keep this terminal open.

### Step 5: Open website

Open this in your browser:

```text
http://127.0.0.1:8000
```

### Step 6: Use it

1. Click `Start Talking`.
2. Allow microphone permission.
3. Speak clearly for at least 1 second.
4. Wait for AI answer.

## Quick Check (Is Backend Running?)

Run this in PowerShell:

```powershell
try { (Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/state" -UseBasicParsing -TimeoutSec 5).StatusCode } catch { $_.Exception.Message }
```

If you see `200`, backend is running.

## What Every File Does

### Main files

- `main.py`
   - Main Flask backend.
   - Receives recorded audio from browser.
   - Converts speech to text with Whisper.
   - Calls Groq model for answer.
   - Returns transcript and answer to frontend.
   - Has safety reply for self-harm prompts.

- `app.py`
   - Older Streamlit version of this project.
   - Kept as reference.
   - Not needed when running the Flask web app.

- `requirements.txt`
   - List of Python libraries needed by project.

- `.env`
   - Secret and config values.
   - Stores `GROK_API_KEY`.

- `knowledge.txt`
   - Small text knowledge used by RAG retrieval.

- `knowledge_base.txt`
   - Larger notes and context text for project domain.

### Frontend folder

- `templates/index.html`
   - Page structure (buttons, chat panel, mic button).

- `static/styles.css`
   - All design styles (colors, layout, animations).

- `static/app.js`
   - Browser logic.
   - Records microphone input.
   - Stops recording after silence.
   - Sends audio to backend endpoints.
   - Shows chat messages and plays AI audio.

### Other files/folders

- `.gitignore`
   - Tells Git which files to ignore.

- `.git/`
   - Git history metadata.

- `.venv/`
   - Virtual environment folder.
   - In this workspace it may be Linux-style and not usable directly on Windows.

- `__MACOSX/`
   - Extra folder from zip extraction on macOS.
   - Not required for running app.

## Common Problems and Easy Fixes

### Problem: "Cannot reach backend server"

Fix:
1. Start `main.py` in terminal.
2. Keep terminal open.
3. Refresh browser with `Ctrl+F5`.

### Problem: microphone button clicked but no text

Fix:
1. Allow mic permission in browser.
2. Speak a little longer (at least 1 second).
3. Check backend health endpoint returns `200`.

### Problem: `KeyboardInterrupt`

Meaning:
- Server was interrupted (often by `Ctrl+C`) while loading.

Fix:
1. Run server command again.
2. Wait until you see Flask "Running on ..." lines.

### Problem: `ModuleNotFoundError`

Fix:

```powershell
c:/python313/python.exe -m pip install <missing_package_name>
```

## Safe Use Note

This assistant should not provide harmful self-harm instructions.
If a user asks such content, backend returns a safety response with India helpline details.
