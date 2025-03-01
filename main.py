import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import datetime
from openai import OpenAI
import streamlit as st
from dataclasses import asdict
from models import Message
import logging
from pathlib import Path
import dotenv
import speech_recognition as sr
from kokoro import KPipeline
import soundfile as sf
import io
import base64
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv('.env')

# Initialize Kokoro pipeline with error handling
try:
    kokoro_pipeline = KPipeline(lang_code='a')
except Exception as e:
    st.error(f"Failed to initialize Kokoro: {str(e)}")
    kokoro_pipeline = None

# Initialize OpenAI client
oai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROK_API_KEY")
)

# Constants
CHAT_MODEL = "mixtral-8x7b-32768"
MODEL_TEMPERATURE = 0.5

# Available models dictionary
AVAILABLE_MODELS = {
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "llama-3.3-70b-versatile": "LLaMA 3.3 70B",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B",
    "gemma2-9b-it": "Gemma 2 9B",
    "qwen-2.5-32b": "Qwen 2.5 32B",
    "mistral-saba-24b": "Mistral Saba 24B"
}

# Default prompt
DEFAULT_PROMPT = """You are an AI medical assistant designed to help gather patient information and assess their symptoms. Your role is to:

1. Collect essential patient information:
   * Full name
   * Age
   * Contact number
   * Email address
   * Current address
   * Medical history
   * Current medications (if any)
   * Known allergies
   * Family medical history (if relevant)

2. Gather detailed information about their symptoms:
   * What symptoms are they experiencing?
   * When did the symptoms start?
   * Severity of symptoms (scale 1-10)
   * Any triggers or patterns noticed
   * Any previous similar experiences

3. Assess the urgency level:
   * Emergency (Requires immediate medical attention)
   * Urgent (Should see a doctor within 24 hours)
   * Non-urgent (Routine medical care)

Emergency symptoms to watch for:
* Chest pain or difficulty breathing
* Severe abdominal pain
* Sudden confusion or difficulty speaking
* Severe headache with neck stiffness
* Loss of consciousness
* Severe bleeding
* Suicidal thoughts

Guidelines for interaction:
* Always maintain a calm and professional tone
* If you detect emergency symptoms, immediately advise seeking emergency care
* Don't make definitive diagnoses - only suggest possible conditions
* If unsure, always err on the side of caution and recommend professional medical evaluation
* Respect medical privacy and confidentiality

Remember: You are not a replacement for professional medical care. Your role is to gather information, assess urgency, and guide users to appropriate medical care.

Begin by asking for the patient's name and their primary concern today."""

def ask_gpt_chat(prompt: str, messages: list[Message]):
    """Returns ChatGPT's response to the given prompt."""
    system_message = [{"role": "system", "content": prompt}]
    message_dicts = [asdict(message) for message in messages]
    conversation_messages = system_message + message_dicts
    response = oai_client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=conversation_messages,
        temperature=MODEL_TEMPERATURE
    )
    return response.choices[0].message.content

def record_audio():
    """Record audio from microphone and return the transcript."""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("🎤 Listening...")
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # Increased timeout to 10 seconds, phrase_time_limit to 5 seconds
            # phrase_time_limit allows for pauses between phrases
            audio = r.listen(source, timeout=5, phrase_time_limit=3)
            st.write("Processing...")
            # Using Google Speech Recognition
            text = r.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
            return None
        except sr.UnknownValueError:
            st.warning("Could not understand audio. Please try again.")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
            return None

def kokoro_text_to_speech(text: str):
    """Converts text to speech using Kokoro TTS."""
    if kokoro_pipeline is None:
        st.error("Kokoro TTS is not initialized")
        return
        
    try:
        # Generate audio using global pipeline
        generator = kokoro_pipeline(
            text,
            voice='af_heart',
            speed=1,
            split_pattern=r'\n+'
        )
        
        # Process all segments and combine audio
        for i, (gs, ps, audio) in enumerate(generator):
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            # Convert audio bytes to base64
            b64 = base64.b64encode(audio_bytes).decode()
            
            # Create HTML with autoplay
            audio_html = f"""
                <audio autoplay>
                    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                </audio>
            """
            
            # Display using components.html
            st.components.v1.html(audio_html, height=0)
            
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

def main():
    st.set_page_config(page_title="Medical Assistant", page_icon="🏥", layout="wide")
    st.title("AI Medical Assistant")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'prompt' not in st.session_state:
        st.session_state.prompt = DEFAULT_PROMPT
    if 'listening' not in st.session_state:
        st.session_state.listening = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = CHAT_MODEL

    # Sidebar for prompt editing
    with st.sidebar:
        st.header("Settings")
        selected_model = st.selectbox(
            "Select Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.session_state.messages = []
            st.success("Model changed and chat history cleared!")
        
        st.header("System Prompt")
        new_prompt = st.text_area("Edit prompt here:", st.session_state.prompt, height=400)
        if st.button("Update Prompt"):
            st.session_state.prompt = new_prompt
            st.session_state.messages = []
            st.success("Prompt updated and chat history cleared!")
        
        if st.button("Reset to Default"):
            st.session_state.prompt = DEFAULT_PROMPT
            st.session_state.messages = []
            st.success("Prompt reset to default and chat history cleared!")
        
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Main chat interface
    chat_col, control_col = st.columns([2, 1])

    with chat_col:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.role):
                st.write(message.content)

    with control_col:
        st.header("Voice Controls")
        
        # Voice input button
        if st.button("🎤 Start Speaking"):
            user_input = record_audio()
            if user_input:
                # Add user message to chat
                st.session_state.messages.append(Message(role="user", content=user_input))
                with st.chat_message("user"):
                    st.write(user_input)

                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = ask_gpt_chat(st.session_state.prompt, st.session_state.messages)
                        st.write(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append(Message(role="assistant", content=response))

        st.info("""
        Instructions:
        1. Click 'Start Speaking' to begin
        2. Speak clearly into your microphone
        3. Wait for the AI to process and respond
        4. The response will be spoken automatically
        
        Note: This is not a replacement for professional medical care.
        """)

if __name__ == "__main__":
    main()