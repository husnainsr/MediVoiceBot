import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

from datetime import datetime
from openai import OpenAI
import streamlit as st
# First Streamlit command must be set_page_config
st.set_page_config(page_title="Medical Assistant", page_icon="��", layout="wide")

# Add this near the top of the file, after st.set_page_config()
st.markdown("""
    <style>
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 0;
        background: white;
        padding: 1rem;
        z-index: 100;
    }
    .stChatContainer {
        height: calc(100vh - 200px);
        overflow-y: auto;
    }
    </style>
""", unsafe_allow_html=True)

from dataclasses import asdict
from models import Message, PatientForm
import logging
from pathlib import Path
import dotenv
import speech_recognition as sr
from kokoro import KPipeline
import time
import soundfile as sf
import io
import base64
import warnings
import asyncio
import numpy as np
import pandas as pd
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
dotenv.load_dotenv('.env')

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Kokoro pipeline with error handling and model loading
try:
    import torch
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
    
    # Set event loop policy for Windows
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    logger.debug("Initializing Kokoro pipeline...")
    
    # Create a container for loading messages
    loading_container = st.empty()
    with loading_container:
        loading_container.info("Loading TTS models... This may take a moment.")
        
        kokoro_pipeline = KPipeline(
            lang_code='a',
            repo_id='hexgrad/Kokoro-82M'
        )
        
        # Warm up the model with a test inference
        logger.debug("Warming up TTS model...")
        _ = kokoro_pipeline("Test.", voice='af_heart')
        
        logger.debug("Kokoro pipeline initialized successfully")
        loading_container.success("TTS models loaded successfully!")
        # Clear the messages after a short delay
        time.sleep(1)
        loading_container.empty()

except Exception as e:
    logger.error(f"Failed to initialize Kokoro: {str(e)}", exc_info=True)
    kokoro_pipeline = None
    st.error("Failed to initialize TTS system. Some features may not be available.")

# Add this before initializing the OpenAI client
if 'GROQ_API_KEY' not in st.session_state:
    st.session_state.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Sidebar for API key input
with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input(
        "Enter your Groq API Key",
        value=st.session_state.GROQ_API_KEY,
        type="password",
        help="Get your API key from https://console.groq.com",
        key="global_api_key"
    )
    if api_key != st.session_state.GROQ_API_KEY:
        st.session_state.GROQ_API_KEY = api_key
        st.success("API Key updated!")

# Initialize OpenAI client with the session state API key
oai_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.session_state.GROQ_API_KEY
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

# Hospital database - in a real app this would be in a database
HOSPITALS = {
    "New York": [
        {"name": "NYC General Hospital", "address": "123 Main St, New York, NY", "phone": "212-555-1000"},
        {"name": "Manhattan Medical Center", "address": "456 Park Ave, New York, NY", "phone": "212-555-2000"}
    ],
    "Boston": [
        {"name": "Boston Medical Center", "address": "789 Washington St, Boston, MA", "phone": "617-555-3000"},
        {"name": "Massachusetts General Hospital", "address": "55 Fruit St, Boston, MA", "phone": "617-555-4000"}
    ],
    "Chicago": [
        {"name": "Chicago Memorial Hospital", "address": "100 Lake Shore Dr, Chicago, IL", "phone": "312-555-5000"},
        {"name": "Northwestern Medical Center", "address": "200 Michigan Ave, Chicago, IL", "phone": "312-555-6000"}
    ],
    "Los Angeles": [
        {"name": "LA County Hospital", "address": "1200 N State St, Los Angeles, CA", "phone": "323-555-7000"},
        {"name": "Cedars-Sinai Medical Center", "address": "8700 Beverly Blvd, Los Angeles, CA", "phone": "310-555-8000"}
    ],
    "Other": [
        {"name": "Regional Medical Center", "address": "Please call for directions", "phone": "800-555-9000"}
    ]
}

def find_nearest_hospital(location):
    """Find hospitals near the patient's location."""
    location = location.strip().title()
    
    # Check if we have hospitals in this location
    for city in HOSPITALS.keys():
        if city.lower() in location.lower():
            return city, HOSPITALS[city]
    
    # If no match, return the default "Other" hospitals
    return "Other", HOSPITALS["Other"]

def refer_to_hospital(patient_form):
    """Refer the patient to a hospital based on their location and symptoms."""
    if not patient_form.location:
        return False
    
    city, hospitals = find_nearest_hospital(patient_form.location)
    
    # For simplicity, choose the first hospital in the list
    # In a real app, you would use more sophisticated matching
    hospital = hospitals[0]
    
    # Update the patient form with referral info
    patient_form.referred_hospital = hospital["name"]
    # Simulate acceptance (in a real app, the hospital would respond)
    patient_form.referral_status = "accepted"
    
    return hospital

# Default prompt
DEFAULT_PROMPT = """You are an AI medical assistant designed to help gather patient information and direct them to appropriate medical facilities. Your role is to:

1. Ask only ONE question at a time
2. Keep responses brief and focused
3. Remember previous information shared
4. After each patient response, update the form with any new information in this format:
   [FORM_UPDATE]field:value[/FORM_UPDATE]
   
   Example:
   [FORM_UPDATE]name:John Doe,age:45,primary_concern:headache[/FORM_UPDATE]

Follow this sequence of information gathering:
   - Name
   - Primary concern
   - Age
   - Location (city/state) - THIS IS CRITICAL for hospital matching
   - Specific symptoms
   - Duration of symptoms
   - Severity (1-10)
   - Previous occurrences
   - Current medications
   - Allergies

Guidelines:
* Ask only ONE question at a time
* Keep responses under 3 sentences
* If you detect emergency symptoms, immediately advise seeking emergency care
* Don't make diagnoses - only suggest possible conditions
* Use previous chat context to avoid asking repeated questions
* ALWAYS ask for the patient's location (city/state) as this is essential for hospital matching

Remember: You are not a replacement for professional medical care.

Begin by asking ONLY for the patient's name and their primary concern today."""

# Add this check before making API calls
def check_api_key():
    if not st.session_state.GROQ_API_KEY:
        st.error("Please enter your Groq API key in the sidebar to continue.")
        return False
    return True

# Modify the ask_gpt_chat function to include the API key check
def ask_gpt_chat(prompt: str, messages: list[Message]):
    """Returns ChatGPT's response to the given prompt."""
    if not check_api_key():
        return "Please enter your Groq API key to continue."
        
    # Move the form update instruction to the main system prompt instead of a separate message
    system_prompt = prompt + "\n\nAfter each patient response, update the form with any new information provided. Format: [FORM_UPDATE]field:value[/FORM_UPDATE]. IMPORTANT: Do not include any text about form updates in your visible response to the patient."
    
    response = oai_client.chat.completions.create(
        model=st.session_state.selected_model,
        messages=[
            {"role": "system", "content": system_prompt},
            *[asdict(message) for message in messages]
        ],
        temperature=MODEL_TEMPERATURE
    )
    
    content = response.choices[0].message.content
    
    # Extract form updates if present
    if "[FORM_UPDATE]" in content:
        try:
            form_updates = content.split("[FORM_UPDATE]")[1].split("[/FORM_UPDATE]")[0].split(",")
            for update in form_updates:
                if ":" in update:
                    field, value = update.strip().split(":", 1)
                    if hasattr(st.session_state.patient_form, field.lower()):
                        setattr(st.session_state.patient_form, field.lower(), value.strip())
            
            # Return only the response without the form updates and any instructions
            clean_response = content.split("[FORM_UPDATE]")[0].strip()
            # Remove any remaining instructions about updating forms
            clean_response = clean_response.replace("After the patient responds, please update the form", "")
            clean_response = clean_response.replace("with their name.", "")
            clean_response = clean_response.replace("with their primary concern.", "")
            clean_response = clean_response.replace("with their age.", "")
            clean_response = clean_response.replace("with their location.", "")
            return clean_response.strip()
        except Exception as e:
            logger.error(f"Error processing form updates: {str(e)}")
            # If there's an error processing the form updates, just return the content without them
            return content.split("[FORM_UPDATE]")[0].strip()
    
    return content

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
        logger.error("Kokoro pipeline is None - initialization failed")
        st.error("Kokoro TTS is not initialized")
        return
        
    try:
        logger.debug(f"Starting TTS for text: {text[:50]}...")
        logger.debug(f"Pipeline type: {type(kokoro_pipeline)}")
        
        generator = kokoro_pipeline(
            text,
            voice='af_heart',
            speed=1,
            split_pattern=r'\n+'
        )
        
        logger.debug("Generator created successfully")
        
        # Collect all audio segments
        all_audio = []
        for gs, ps, audio in generator:
            all_audio.append(audio)
        
        # Concatenate all audio segments
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            
            # Convert to audio file
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio, 24000, format='WAV')
            buffer.seek(0)
            audio_bytes = buffer.read()
            
            logger.debug(f"Combined audio bytes length: {len(audio_bytes)}")
            b64 = base64.b64encode(audio_bytes).decode()
            
            # Single audio element for the entire response with hidden controls
            audio_html = f"""
                <div id="audio-container" style="display: none;">
                    <audio id="audio-player" autoplay>
                        <source src="data:audio/wav;base64,{b64}" type="audio/wav">
                    </audio>
                </div>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {{
                        const audioElement = document.getElementById('audio-player');
                        console.log('Audio element:', audioElement);
                        audioElement.play().catch(e => console.error('Playback failed:', e));
                    }});
                </script>
            """
            
            logger.debug("Displaying combined audio component")
            st.components.v1.html(audio_html, height=0)
            
    except Exception as e:
        logger.error(f"TTS error: {str(e)}", exc_info=True)
        st.error(f"Error in TTS: {str(e)}")

def main():
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
    if 'patient_form' not in st.session_state:
        st.session_state.patient_form = PatientForm(timestamp=datetime.now())
    if 'show_hospital_info' not in st.session_state:
        st.session_state.show_hospital_info = False
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False

    # Sidebar for essential controls only
    with st.sidebar:
        selected_model = st.selectbox(
            "Select Model",
            options=list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x],
            index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model)
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.patient_form = PatientForm(timestamp=datetime.now())
            st.session_state.assessment_complete = False
            st.session_state.show_hospital_info = False
            st.rerun()
            
        # Show patient form data for debugging/admin purposes
        with st.expander("Patient Data (Admin View)"):
            if st.session_state.patient_form.name:
                for field, value in st.session_state.patient_form.to_dict().items():
                    st.write(f"**{field}:** {value}")
                
                if st.button("Download Patient Form"):
                    df = pd.DataFrame([st.session_state.patient_form.to_dict()])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name=f"patient_{st.session_state.patient_form.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No patient information collected yet.")

    # Main chat interface with columns
    chat_col, control_col = st.columns([2, 1])

    with chat_col:
        # Create a container for messages with fixed height and scrolling
        chat_container = st.container()
        
        # Create a container for the input at the bottom
        input_container = st.container()
        
        # Display messages in scrollable container
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message.role):
                    st.write(message.content)
            
            # Show hospital recommendation when assessment is complete
            if st.session_state.assessment_complete and st.session_state.patient_form.location:
                # Find and display hospital recommendation
                city, hospitals = find_nearest_hospital(st.session_state.patient_form.location)
                
                with st.container():
                    st.markdown("### 🏥 Hospital Recommendation")
                    st.markdown(f"Based on your symptoms and location in **{city}**, we recommend:")
                    
                    for i, hospital in enumerate(hospitals[:2]):  # Show top 2 hospitals
                        st.markdown(f"""
                        #### {hospital['name']}
                        **Address:** {hospital['address']}  
                        **Phone:** {hospital['phone']}
                        """)
                        
                        # Create a unique key for each button
                        if st.button(f"Select {hospital['name']}", key=f"select_hospital_{i}"):
                            st.session_state.patient_form.referred_hospital = hospital['name']
                            st.session_state.patient_form.referral_status = "accepted"
                            st.session_state.show_hospital_info = True
                            st.rerun()
                    
                    st.info("Please select a hospital to proceed with your care.")
        
        # Handle input
        with input_container:
            user_input = st.chat_input("Type your message here...")
            
        # Process user input
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
                    # Add assistant message to chat history before TTS
                    st.session_state.messages.append(Message(role="assistant", content=response))
                    # Generate speech after message is displayed
                    kokoro_text_to_speech(response)
            
            # Check if we have enough information to complete assessment
            form = st.session_state.patient_form
            if (form.name and form.primary_concern and form.location and 
                form.specific_symptoms and form.severity):
                st.session_state.assessment_complete = True
                st.rerun()  # Refresh to show hospital recommendations

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
                        # Add assistant message to chat history before TTS
                        st.session_state.messages.append(Message(role="assistant", content=response))
                        # Generate speech after message is displayed
                        kokoro_text_to_speech(response)
                
                # Check if we have enough information to complete assessment
                form = st.session_state.patient_form
                if (form.name and form.primary_concern and form.location and 
                    form.specific_symptoms and form.severity):
                    st.session_state.assessment_complete = True
                    st.rerun()  # Refresh to show hospital recommendations

        # Show selected hospital confirmation
        if st.session_state.show_hospital_info and st.session_state.patient_form.referred_hospital:
            st.success("Hospital selected!")
            city, hospitals = find_nearest_hospital(st.session_state.patient_form.location)
            for hospital in hospitals:
                if hospital["name"] == st.session_state.patient_form.referred_hospital:
                    st.markdown(f"""
                    ### Your Referral
                    
                    You've been referred to:
                    
                    **{hospital["name"]}**  
                    {hospital["address"]}  
                    📞 {hospital["phone"]}
                    
                    Please contact them to schedule your appointment.
                    """)
                    
                    # Add directions link
                    address_query = hospital["address"].replace(" ", "+")
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={address_query}"
                    st.markdown(f"[Get Directions]({maps_url})")
                    break

        st.info("""
        Instructions:
        1. Type in the chat box or click 'Start Speaking'
        2. For voice: Speak clearly into your microphone
        3. Answer all questions about your symptoms
        4. You'll receive hospital recommendations based on your location
        5. Select a hospital to get contact information
        
        Note: This is not a replacement for professional medical care.
        """)

if __name__ == "__main__":
    main()