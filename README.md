🎙️ Streamlit Voice Assistant

📌 Project Overview

This is an AI-powered Voice Assistant built using Streamlit, Whisper, g4f, and Pyttsx3. The assistant transcribes speech, generates AI responses, and speaks back the response. It also includes a medical-focused AI response system using BioMedLM for medical queries.

🚀 Features

🎤 Speech-to-Text using OpenAI's Whisper model.

🤖 AI-generated responses using g4f.

🗣️ Text-to-Speech with pyttsx3.

🏥 Medical AI responses with BioMedLM.

📜 Conversation history for seamless interaction.

🛠️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/voice-assistant.git
cd voice-assistant

2️⃣ Create a Virtual Environment (Recommended)

python -m venv voice_assistant_env
source voice_assistant_env/bin/activate  # On macOS/Linux
voice_assistant_env\Scripts\activate  # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Run the Streamlit App

streamlit run final.py  # Or finalmed.py for medical responses

📜 Usage Guide

Text Input: Type your query in the chat box.

Voice Input: Click the 🎙️ button to speak.

AI Response: The assistant replies via text & voice.

Medical Mode: For medical queries, the assistant uses BioMedLM.

🔧 Technologies Used

Python

Streamlit (UI)

Whisper (Speech-to-Text)

g4f (AI responses)

Pyttsx3 (Text-to-Speech)

Transformers (BioMedLM model for medical queries)
