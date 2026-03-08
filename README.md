# Conversational Voice AI Assistant

This project now uses a browser-based HTML/CSS/JavaScript frontend with a Python Flask backend.

The backend workflow remains the same:
1. Record voice input
2. Reduce noise
3. Transcribe with Whisper
4. Generate response with Groq (OpenAI-compatible client)
5. Convert response to speech with `pyttsx3`

## Features
- Browser mic recording with silence auto-stop
- Multi-session chat history (`Chat 1`, `Chat 2`, ...)
- Whisper ASR + Groq LLM response
- TTS audio playback in the browser

## Setup
1. Add `GROK_API_KEY` to your `.env` file.
2. Ensure `ffmpeg` is installed and available in `PATH`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python main.py
   ```
5. Open:
   ```text
   http://127.0.0.1:8000
   ```

## Project Files
- `main.py`: Flask backend and voice pipeline
- `templates/index.html`: Main page markup
- `static/styles.css`: UI styling
- `static/app.js`: Frontend behavior and audio capture
- `app.py`: Original Streamlit version (kept as reference)
