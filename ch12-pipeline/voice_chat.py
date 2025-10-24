import streamlit as st
import sounddevice as sd
import numpy as np
import queue
import threading
import openai
import time
import requests
import os 
import io
import soundfile as sf

# Replace with your own OpenAI API key or other LLM/ASR/TTS endpoints
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ASR_ENDPOINT = 'https://api.openai.com/v1/audio/transcriptions'  # Whisper ASR
LLM_ENDPOINT = 'https://api.openai.com/v1/chat/completions'      # GPT-4o-mini
TTS_ENDPOINT = 'https://api.openai.com/v1/audio/speech'          # OpenAI TTS

# Queue to communicate between audio thread and Streamlit
audio_q = queue.Queue()
stop_stream = threading.Event()

# Record audio from mic in a background thread
def record_audio(duration=30, samplerate=16000, channels=1):
    audio_frames = []
    def callback(indata, frames, time, status):
        if stop_stream.is_set():
            raise sd.CallbackAbort()
        audio_frames.append(indata.copy())
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        sd.sleep(int(duration * 1000))
    audio = np.concatenate(audio_frames, axis=0)
    audio_q.put(audio)

def start_recording_thread():
    stop_stream.clear()
    threading.Thread(target=record_audio, daemon=True).start()

def stop_recording():
    stop_stream.set()

# Send audio to ASR (OpenAI Whisper)
def transcribe_audio(audio, samplerate=16000):


    buf = io.BytesIO()
    sf.write(buf, audio, samplerate, format='WAV')
    buf.seek(0)
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    files = {'file': ('audio.wav', buf, 'audio/wav')}
    data = {'model': 'whisper-1', 'response_format': 'text'}
    resp = requests.post(ASR_ENDPOINT, headers=headers, files=files, data=data)
    return resp.text.strip()

# Send user text to LLM (OpenAI GPT-4o-mini)
def stream_llm_response(user_input):
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    data = {
        'model': 'gpt-4o',
        'messages': [{'role': 'user', 'content': user_input}],
        'stream': True
    }
    response = requests.post(LLM_ENDPOINT, headers=headers, json=data, stream=True)
    for line in response.iter_lines():
        if line:
            if line.startswith(b'data: '):
                payload = line[6:]
                if payload == b'[DONE]':
                    break
                import json
                chunk = json.loads(payload)
                if 'choices' in chunk and chunk['choices']:
                    content = chunk['choices'][0]['delta'].get('content', '')
                    yield content

# Convert LLM output to speech
# For demonstration, this simply returns TTS audio bytes
# You may wish to stream TTS audio for smoother experience

def text_to_speech(text):
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json'}
    data = {
        'model': 'tts-1',
        'input': text,
        'voice': 'alloy',  # you can choose other voices
    }
    response = requests.post(TTS_ENDPOINT, headers=headers, json=data)
    audio_bytes = response.content
    return audio_bytes

# --- Streamlit UI ---
st.set_page_config(page_title="Real-Time Voice Chatbot", layout="wide")
st.title("🗣️ Real-Time Voice Chatbot Demo")

# Sidebar controls
with st.sidebar:
    st.header("Instructions")
    st.markdown("1. Click 'Record' to speak.\n2. Wait for reply, or click 'Stop' to interrupt.")
    record_btn = st.button("Record", key="record")
    stop_btn = st.button("Stop", key="stop")

# State
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Audio recording and interruption
audio = None
if record_btn:
    start_recording_thread()
    st.info("Recording... Speak now.")
    # Poll for audio
    while audio_q.empty():
        if stop_btn or stop_stream.is_set():
            stop_recording()
            break
        time.sleep(0.1)
    if not audio_q.empty():
        audio = audio_q.get()
        stop_recording()
if audio is not None:
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, 16000, format='WAV')
    st.audio(audio_buffer.getvalue(), format='audio/wav')
    st.success("Audio captured. Transcribing...")

    user_text = transcribe_audio(audio)
    st.write(f"**You:** {user_text}")
    st.session_state['history'].append({'role': 'user', 'content': user_text})
    st.info("🤖 Generating response (you can interrupt)...")
    # Streaming response
    response_text = ''
    response_placeholder = st.empty()
    try:
        for chunk in stream_llm_response(user_text):
            response_text += chunk
            response_placeholder.markdown(f"**Bot:** {response_text}")
            if stop_btn or stop_stream.is_set():
                break
        st.session_state['history'].append({'role': 'assistant', 'content': response_text})
        # Text-to-speech
        st.audio(text_to_speech(response_text), format='audio/mp3')
    except Exception as e:
        st.error(f"Error: {e}")

# Display chat history
st.divider()
st.subheader("Conversation History")
for msg in st.session_state['history']:
    role = '🧑' if msg['role'] == 'user' else '🤖'
    st.markdown(f"{role}: {msg['content']}")
