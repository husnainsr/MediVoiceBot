# AI Medical Assistant

A Streamlit-based medical assistant application that uses AI to gather patient information and assess symptoms through both text and voice interactions.

## Features

- 🤖 AI-powered medical conversation system
- 🎤 Voice input capabilities
- 🔊 Text-to-speech output using Kokoro TTS
- 💬 Multiple AI model support through Groq API
- 📝 Customizable system prompts
- 🔄 Real-time conversation history

## Supported AI Models

- Mixtral 8x7B (Default)
- LLaMA 3.3 70B
- LLaMA 3.1 8B
- Gemma 2 9B
- Qwen 2.5 32B
- Mistral Saba 24B

## Prerequisites

- Python 3.8+
- Groq API key
- Microphone for voice input

## Installation

1. Clone the repository
2. Install Python 3.10:

```bash
conda create -n medical-assistant python=3.10
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
4. Install espeak-ng.1
```bash
https://github.com/espeak-ng/espeak-ng/releases/download/1.52.0/espeak-ng.msi
```

5. Create a `.env` file with your API keys:

```bash
GROQ_API_KEY="your_groq_api_key"
```


## Usage

1. Start the application:

```bash
streamlit run main.py
```


2. Use the sidebar to:
   - Select your preferred AI model
   - Customize the system prompt
   - Reset to default settings
   - Clear chat history

3. Interact with the assistant using:
   - Voice input (click "Start Speaking")
   - Text input (chat interface)

## Safety Notes

This application is designed for information gathering and preliminary assessment only. It:
- Does not replace professional medical care
- Will recommend emergency care for serious symptoms
- Maintains medical privacy and confidentiality
- Avoids making definitive diagnoses

## Technical Details

The application uses:
- Streamlit for the web interface
- Groq API for AI model access
- SpeechRecognition for voice input
- Kokoro TTS for voice output
- Session state management for conversation history

## Project Structure

- `main.py`: Core application logic
- `models.py`: Data models for message handling
- `prompts/`: Directory containing system prompts
- `.env`: Configuration and API keys
- `requirements.txt`: Project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

