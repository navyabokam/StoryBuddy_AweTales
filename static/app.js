const state = {
  sessions: {},
  currentSession: "Chat 1",
  recorder: null,
  mediaStream: null,
  audioContext: null,
  processor: null,
  sourceNode: null,
  recordedChunks: [],
  sampleRate: 16000,
  isRecording: false,
  silenceSince: null,
  recordingStartedAt: 0,
};

const SILENCE_RMS_THRESHOLD = 0.008;
const AUTO_STOP_SILENCE_MS = 900;
const MIN_RECORDING_MS = 700;
const PROCESSOR_BUFFER_SIZE = 512; // was 1024, now lower latency

const chatListEl = document.getElementById("chatList");
const chatTitleEl = document.getElementById("chatTitle");
const chatMessagesEl = document.getElementById("chatMessages");
const micBtn = document.getElementById("micBtn");
const micStatusEl = document.getElementById("micStatus");
const newChatBtn = document.getElementById("newChatBtn");
const clearChatBtn = document.getElementById("clearChatBtn");
const stopAudioBtn = document.getElementById("stopAudioBtn");
const aiAudio = document.getElementById("aiAudio");

let activeUtterance = null;

async function api(path, method = "GET", body = null, isForm = false) {
  const options = { method, headers: {} };
  if (body && isForm) {
    options.body = body;
  } else if (body) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  async function doFetch() {
    const res = await fetch(path, options);
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      data = { error: `Request failed (${res.status})` };
    }
    if (!res.ok || data.ok === false) {
      throw new Error(data.error || "Request failed");
    }
    return data;
  }

  try {
    return await doFetch();
  } catch (err) {
    if (err instanceof TypeError) {
      // Retry once for transient backend reconnects/network hiccups.
      await new Promise((resolve) => setTimeout(resolve, 400));
      try {
        return await doFetch();
      } catch (retryErr) {
        if (retryErr instanceof TypeError) {
          throw new Error("Cannot reach backend server. Please refresh and try again.");
        }
        throw retryErr;
      }
    }
    throw err;
  }
}

function render() {
  chatTitleEl.textContent = `🎙 ${state.currentSession}`;

  chatListEl.innerHTML = "";
  Object.keys(state.sessions).forEach((chat) => {
    const row = document.createElement("div");
    row.className = "chat-item";

    const mainBtn = document.createElement("button");
    mainBtn.className = `chat-main ${chat === state.currentSession ? "active" : ""}`;
    mainBtn.textContent = chat === state.currentSession ? `💬 ${chat}` : chat;
    mainBtn.onclick = () => switchChat(chat);

    const delBtn = document.createElement("button");
    delBtn.className = "chat-del";
    delBtn.textContent = "🗑";
    delBtn.title = `Delete ${chat}`;
    delBtn.onclick = () => deleteChat(chat);

    row.appendChild(mainBtn);
    row.appendChild(delBtn);
    chatListEl.appendChild(row);
  });

  chatMessagesEl.innerHTML = "";
  const messages = state.sessions[state.currentSession] || [];
  messages.forEach((m) => {
    const msg = document.createElement("div");
    msg.className = `msg ${m.role}`;
    msg.textContent = m.content;
    chatMessagesEl.appendChild(msg);
  });
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
}

async function refreshState() {
  const data = await api("/api/state");
  state.sessions = data.sessions;
  state.currentSession = data.current_session;
  render();
}

async function newChat() {
  await api("/api/chat/new", "POST");
  await refreshState();
}

async function clearChat() {
  await api("/api/chat/clear", "POST");
  await refreshState();
}

async function switchChat(chat) {
  await api("/api/chat/switch", "POST", { chat });
  await refreshState();
}

async function deleteChat(chat) {
  await api("/api/chat/delete", "POST", { chat });
  await refreshState();
}

function mergeBuffers(chunks) {
  const totalLength = chunks.reduce((sum, arr) => sum + arr.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function downsample(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) {
    return buffer;
  }
  const ratio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = count > 0 ? accum / count : 0;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

function floatTo16BitPCM(view, offset, input) {
  for (let i = 0; i < input.length; i += 1, offset += 2) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i += 1) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function setRecordingUI(isRecording) {
  state.isRecording = isRecording;
  micBtn.classList.toggle("recording", isRecording);
  micStatusEl.textContent = isRecording ? "Recording..." : "Start Talking";
}

async function startRecording() {
  if (state.isRecording) {
    stopRecording();
    return;
  }

  // Avoid starting capture if backend is currently unreachable.
  try {
    await api("/api/state");
  } catch (err) {
    const message = err && err.message ? err.message : "Backend is unavailable.";
    alert(`${message}\n\nStart the Flask server and try again.`);
    return;
  }

  aiAudio.pause();
  aiAudio.currentTime = 0;

  try {
    state.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const inputRate = state.audioContext.sampleRate;
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
    // Smaller buffer lowers end-to-end latency for voice activity checks.
    state.processor = state.audioContext.createScriptProcessor(PROCESSOR_BUFFER_SIZE, 1, 1);
    state.recordedChunks = [];
    state.silenceSince = null;
    state.recordingStartedAt = Date.now();

    state.processor.onaudioprocess = (event) => {
      if (!state.isRecording) {
        return;
      }
      const data = event.inputBuffer.getChannelData(0);
      state.recordedChunks.push(new Float32Array(data));

      let rms = 0;
      for (let i = 0; i < data.length; i += 1) {
        rms += data[i] * data[i];
      }
      rms = Math.sqrt(rms / data.length);

      if (rms < SILENCE_RMS_THRESHOLD && Date.now() - state.recordingStartedAt > MIN_RECORDING_MS) {
        if (!state.silenceSince) {
          state.silenceSince = Date.now();
        } else if (Date.now() - state.silenceSince > AUTO_STOP_SILENCE_MS) {
          stopRecording();
        }
      } else {
        state.silenceSince = null;
      }
    };

    state.sourceNode.connect(state.processor);
    state.processor.connect(state.audioContext.destination);
    setRecordingUI(true);

    state.stopTimer = setTimeout(() => {
      if (state.isRecording) {
        stopRecording();
      }
    }, 20000);

    state.finishRecording = async () => {
      clearTimeout(state.stopTimer);

      const merged = mergeBuffers(state.recordedChunks);
      const mono16k = downsample(merged, inputRate, state.sampleRate);
      if (mono16k.length < state.sampleRate * 0.25) {
        micStatusEl.textContent = "Speak a bit longer...";
        return;
      }

      const wavBlob = encodeWAV(mono16k, state.sampleRate);
      await sendAudio(wavBlob);
    };
  } catch (err) {
    setRecordingUI(false);
    cleanupAudioNodes();
    const name = err && typeof err === "object" ? err.name : "";
    if (name === "NotAllowedError") {
      alert("Microphone permission is blocked. Allow mic access and try again.");
    } else if (name === "NotFoundError") {
      alert("No microphone device found. Connect a mic and try again.");
    } else {
      alert("Unable to start recording. Please check microphone settings.");
    }
  }
}

function cleanupAudioNodes() {
  if (state.processor) {
    state.processor.disconnect();
  }
  if (state.sourceNode) {
    state.sourceNode.disconnect();
  }
  if (state.audioContext) {
    state.audioContext.close();
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
  }

  state.processor = null;
  state.sourceNode = null;
  state.audioContext = null;
  state.mediaStream = null;
}

async function stopRecording() {
  if (!state.isRecording) {
    return;
  }
  setRecordingUI(false);
  cleanupAudioNodes();
  if (state.finishRecording) {
    await state.finishRecording();
    state.finishRecording = null;
  }
}

async function sendAudio(blob) {
  const form = new FormData();
  form.append("audio", blob, "voice.wav");

  micStatusEl.textContent = "Thinking...";
  try {
    const data = await api("/api/voice", "POST", form, true);
    state.sessions = data.sessions;
    state.currentSession = data.current_session;
    render();
    if (data.audio) {
      aiAudio.src = `data:audio/wav;base64,${data.audio}`;
      aiAudio.play().catch(() => {});
    } else {
      speakWithBrowser(data.answer);
    }
  } catch (err) {
    alert(err.message);
  } finally {
    micStatusEl.textContent = "Start Talking";
  }
}

function stopAiAudio() {
  aiAudio.pause();
  aiAudio.currentTime = 0;
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  activeUtterance = null;
}

function speakWithBrowser(text) {
  if (!("speechSynthesis" in window) || !text) {
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  activeUtterance = utterance;
  window.speechSynthesis.speak(utterance);
}

newChatBtn.addEventListener("click", newChat);
clearChatBtn.addEventListener("click", clearChat);
micBtn.addEventListener("click", startRecording);
stopAudioBtn.addEventListener("click", stopAiAudio);

refreshState().catch((e) => alert(e.message));
