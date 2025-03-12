import streamlit as st
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import g4f
import wave
import pyaudio
import os
import subprocess
import nest_asyncio
from gtts import gTTS

nest_asyncio.apply()

try:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
except ImportError:
    subprocess.run(["pip", "install", "transformers"])

device = "cpu"  
whisper_model_name = "openai/whisper-tiny"
processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

medical_model_name = "stanford-crfm/BioMedLM"
tokenizer = AutoTokenizer.from_pretrained(medical_model_name)
medical_model = AutoModelForCausalLM.from_pretrained(medical_model_name).to(device)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧑‍⚕️ Medical Voice Assistant")
st.write("### Powered by Whisper, g4f, and gTTS")

st.warning(
    "⚠️ **Note:** The 'Speak' option requires local execution. "
    "If you're using this app on Streamlit Cloud, text-to-speech (TTS) will not work. "
    "To enable this feature, fork the app’s [GitHub repository](your-github-repo-link) "
    "and run it locally using:\n\n"
    "```bash\n"
    "streamlit run app.py\n"
    "```",
)

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


def transcribe_audio(filename="input.wav"):
    # Use librosa or soundfile instead of torchaudio
    import soundfile as sf
    import numpy as np

    audio_input, sample_rate = sf.read(filename)
    audio_input = torch.tensor(audio_input).unsqueeze(0)

    input_features = processor(
        audio_input.squeeze(0), sampling_rate=16000, return_tensors="pt"
    ).input_features
    predicted_ids = whisper_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription.lower()


def get_ai_response(prompt):
    response_generator = g4f.ChatCompletion.create(
        model=g4f.models.default,
        messages=st.session_state.messages,
        stream=True
    )

    response_text = ""
    for chunk in response_generator:
        if isinstance(chunk, str): 
            response_text += chunk  

    return response_text


def get_medical_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = medical_model.generate(**inputs, max_length=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")

    audio_html = f"""
    <audio autoplay hidden>
        <source src="output.mp3" type="audio/mp3">
    </audio>
    """

    st.markdown(audio_html, unsafe_allow_html=True)

    os.remove("output.mp3")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    ai_response = get_ai_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    with st.chat_message("assistant"):
        st.write(ai_response)

   
    speak(ai_response)

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
