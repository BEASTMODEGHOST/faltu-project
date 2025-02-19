import os
from flask import Flask, request, jsonify, render_template, send_file
from google.cloud import translate_v2 as translate
from pydub import AudioSegment
from langchain_google_genai import ChatGoogleGenerativeAI
from groq import Groq
import tempfile
import json
import logging
from google.cloud import texttospeech

app = Flask(__name__)


logging.basicConfig(level=logging.INFO)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"
translate_client = translate.Client()
GOOGLE_API_KEY = "AIzaSyCqNzDqQ6grXOKAdLIkOKjcD0AIqApNcGg"
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key="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")
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"
text_to_speech_client = texttospeech.TextToSpeechClient()

user_data = {
    "name": None,
    "language": "en",
    "literacy_level": "good"
}

def log_event(message):
    logging.info(message)

def translate_to_english(text):
    try:
        detection = translate_client.detect_language(text)
        detected_language = detection.get('language', 'und')  

        if detected_language == "und":
            raise ValueError("Language could not be detected.")

        if detected_language != "en":  
            translation = translate_client.translate(text, target_language="en")
            return translation['translatedText'], detected_language
        return text, detected_language
    except Exception as e:
        print(f"Error in translate_to_english: {str(e)}")
        raise RuntimeError(f"Translation to English failed: {str(e)}")


def translate_from_english(text, target_language):
    try:
        if target_language != "en":
            if target_language == "und":
                raise ValueError("Target language is undefined or invalid.")
            translation = translate_client.translate(text, target_language=target_language)
            return translation['translatedText']
        return text
    except Exception as e:
        print(f"Error in translate_from_english: {str(e)}")
        raise RuntimeError(f"Translation from English failed: {str(e)}")

def transcribe_audio(audio_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio = AudioSegment.from_file(audio_file)
            audio.export(temp_audio_file.name, format="wav")
            with open(temp_audio_file.name, "rb") as audio:
                transcription = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=audio)
            log_event(f"Transcription result: {transcription.text}")
            return transcription.text
    except Exception as e:
        log_event(f"Error during audio transcription: {str(e)}")
        raise RuntimeError(f"Error during audio transcription: {str(e)}")
    

def text_to_speech(text, language_code="en"):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    if language_code == "hi-IN":
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            name="hi-IN-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
    elif language_code == "ta-IN":
        voice = texttospeech.VoiceSelectionParams(
            language_code="ta-IN",
            name="ta-IN-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
    else: 
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = text_to_speech_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    with open(temp_audio_file.name, "wb") as out:
        out.write(response.audio_content)

    return temp_audio_file.name

@app.route("/")
def index():
    return render_template("indexey.html")

@app.route("/get_name", methods=["POST"])
def get_name():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    try:
        name = transcribe_audio(audio_file)
        translated_name, detected_language = translate_to_english(name)
        user_data["name"] = translated_name
        user_data["language"] = detected_language
        log_event(f"User name set to: {translated_name}, Language: {detected_language}")
        return jsonify({
            "message": f"Hello {translated_name}!",
            "detected_language": detected_language
        })
    except Exception as e:
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

@app.route("/set_literacy_level", methods=["POST"])
def set_literacy_level():
    level = request.json.get("literacy_level", "good")
    user_data["literacy_level"] = level
    log_event(f"User literacy level set to: {level}")
    return jsonify({"message": f"Literacy level set to {level}."})

@app.route("/query", methods=["POST"])
def query():
    if "audio" not in request.files:
        log_event("No audio file uploaded to /query.")
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    try:
        log_event("Received audio for /query.")

        
        query = transcribe_audio(audio_file)
        log_event(f"Transcribed query: {query}")

        
        translated_query, detected_language = translate_to_english(query)
        log_event(f"Translated query: {translated_query} (Detected Language: {detected_language})")

        prompt = (
            f"The user {user_data['name']} is interacting in {user_data['language']}.\n"
            f"The user's financial literacy level is {user_data['literacy_level']}.\n"
            f"Query: {translated_query}\n"
            f"Provide a detailed, tailored response suitable for this literacy level."
        )
        log_event(f"Prompt for LLM: {prompt}")

        llm_response = llm.invoke(prompt)
        english_response = llm_response.content.strip()
        log_event(f"LLM response: {english_response}")

        localized_response = translate_from_english(english_response, detected_language)
        log_event(f"Localized response: {localized_response}")

        return jsonify({
            "query": query,  
            "response": localized_response,  
            "language_code": detected_language 
        })
    except Exception as e:
        log_event(f"Error processing query: {str(e)}")
        return jsonify({"error": f"Error processing query: {str(e)}"}), 500


@app.route("/play_response", methods=["POST"])
def play_response():
    text = request.json.get("text")
    language_code = request.json.get("language_code", "en")

    if not text or not language_code:
        return jsonify({"error": "Missing text or language code for speech synthesis"}), 400

    google_language_codes = {
        "hi": "hi-IN",  
        "ta": "ta-IN",  
        "en": "en-US"   
    }
    language_code = google_language_codes.get(language_code, "en-US")

    try:
        audio_file = text_to_speech(text, language_code)
        log_event(f"Generated speech for: {text} in language: {language_code}")
        return send_file(audio_file, mimetype="audio/mpeg", as_attachment=False)
    except Exception as e:
        log_event(f"Error generating speech: {str(e)}")
        return jsonify({"error": f"Error generating speech: {str(e)}"}), 500
    
    
if __name__ == "__main__":
    app.run(debug=True)
