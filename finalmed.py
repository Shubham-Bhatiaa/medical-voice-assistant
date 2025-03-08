import streamlit as st
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import g4f
import wave
import pyaudio
import os
import pyttsx3
import os
import subprocess
import asyncio
import nest_asyncio
from gtts import gTTS
pip install --upgrade pip
pip install --upgrade streamlit


nest_asyncio.apply()  # This helps run Streamlit and asyncio concurrently
try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    subprocess.run(["pip", "install", "transformers"])
    from transformers import WhisperProcessor, WhisperForConditionalGeneration


# Load Whisper Model
device = "cpu"  # Force CPU usage

whisper_model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

# Load Medical AI Model
medical_model_name = "stanford-crfm/BioMedLM"
tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
medical_model = AutoModelForCausalLM.from_pretrained(medical_model_name).to(device)

# Memory for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit App UI
st.title("🧑‍⚕️ Medical Voice Assistant")
st.write("###  Powered by Whisper, g4f, and Pyttsx3")

# Function to record audio
def record_audio(filename="input.wav", duration=4, rate=16000):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    sample_rate = rate
    record_seconds = duration

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

    st.write("🎤 Listening...")
    frames = []

    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    return filename

# Function to transcribe audio
def transcribe_audio(filename="input.wav"):
    audio_input, _ = torchaudio.load(filename)
    input_features = processor(audio_input.squeeze(0), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.lower()

# Function to get AI response
def get_ai_response(prompt):
    response_generator = g4f.ChatCompletion.create(
        model=g4f.models.default,
        messages=st.session_state.messages,
        stream=True
    )

    response_text = ""
    for chunk in response_generator:
        if isinstance(chunk, str):  # ✅ Ensure only string responses are processed
            response_text += chunk  

    return response_text

# Function to get Medical AI response
def get_medical_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = medical_model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Function to speak using Pyttsx3
def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Text Input Option (First)
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    ai_response = get_ai_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    with st.chat_message("assistant"):
        st.write(ai_response)

    # Speak AI response
    speak(ai_response)

# Add Mic Button for Voice Input (At the Bottom)
if st.button("🎙️ Speak"):
    audio_file = record_audio()
    user_input = transcribe_audio(audio_file)
    st.write(f"🗣️ You said: {user_input}")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    ai_response = get_ai_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    with st.chat_message("assistant"):
        st.write(ai_response)

    # Speak AI response
    speak(ai_response)
st.markdown(
    """
    <style>
    .watermark {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 18px;
        color: rgba(200, 200, 200, 0.3);
        z-index: 1000;
    }
    </style>
    <div class="watermark">© Rena Sebastian </div>
    """,
    unsafe_allow_html=True
)
