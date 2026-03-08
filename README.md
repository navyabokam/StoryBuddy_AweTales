# Voice Story Assistant (Easy Guide)

This project is a website where you can:
1. Click a microphone button.
2. Speak.
3. See your speech turned into text.
4. Get a story answer from AI.

The website has 2 parts:
- Frontend: what you see in the browser (`HTML`, `CSS`, `JavaScript`).
- Backend: Python server that does speech-to-text and AI reply.

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
