import os
import time
import logging
import base64
import threading
import tempfile
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import pandas as pd
import gradio as gr
from groq import Groq
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
app = Flask(__name__)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"
text_to_speech_client = texttospeech.TextToSpeechClient()
translate_client = translate.Client()

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'sv2_pt_30d8bc22e01048e6ac9036fbcff466d2_ffa1e3c34d'

pc = Pinecone(api_key="9f8aa1d2-88ea-41a8-a731-bde767f0bfff")
pinecone_index_name = "flipkart-products"

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

GOOGLE_API_KEY = 'AIzaSyCqNzDqQ6grXOKAdLIkOKjcD0AIqApNcGg'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(model_name="gemini-pro")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

if 'flipkart-products' not in pc.list_indexes().names():
    pc.create_index(
        name='flipkart-products',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

index = pc.Index('flipkart-products')

df = pd.read_csv("TVS Credit Data - Sheet1 (2).csv")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
df['id'] = df['id'].astype('object')
for i in range(len(df)):
    string = df.iloc[i]['Questions']
    embeddings = genai.embed_content(
        model="models/text-embedding-004",
        content=string
    )
    index.upsert(
        vectors=[
            {"id": "prod" + str(i), "values": embeddings['embedding']}
        ]
    )
    df.at[i, 'id'] = "prod" + str(i)


interrupt_flag = False
playback_thread = None
chat_history = []

BANK_PHONE_NUMBERS = {
    "TVS Bank": "+91-9999999999",
    "Bank of Baroda": "+91-8888888888",
    "Punjab National Bank": "+91-7777777777"
}

def stop_playback():
    global interrupt_flag, playback_thread
    if playback_thread and playback_thread.is_alive():
        interrupt_flag = True
        playback_thread.join()

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file)
    return transcription.text


def transcribe_audio_step(audio_file):
    if audio_file is None:
        return "No audio file received."

    
    audio_file_path = audio_file  
    user_input = transcribe_audio(audio_file_path)

    return user_input
def translate_to_english(text):
    detection = translate_client.detect_language(text)
    detected_language = detection['language']

    if detected_language == "hi":  
        logging.info("Detected Hindi, translating to English...")
        translation = translate_client.translate(text, target_language="en")
        return translation['translatedText'], detected_language
    
    
    logging.info("Detected English, no translation needed.")
    return text, detected_language


def translate_to_original_language(text, target_language):
    translation = translate_client.translate(text, target_language=target_language)
    return translation['translatedText']

def query_pinecone(user_query):
    embeddings = genai.embed_content(model="models/text-embedding-004", content=user_query)
    
    query_results = index.query(
        vector=embeddings['embedding'],
        top_k=5,
        include_values=True
    )
    
    best_match = None
    highest_score = -1
    
    
    for result in query_results['matches']:
        product_id = result['id']
        row = df[df['id'] == product_id]
        score = result['score']
        if not row.empty and score > highest_score:
            best_match = row
            highest_score = score

    
    if best_match is not None and 'Answers' in best_match.columns and not best_match['Answers'].empty:
        return best_match['Answers'].values[0]
    else:
        return "Sorry, I couldn't find the answer to that. Please contact your local branch or call at 910-888-2341 for assistance."

def generate_response(user_query, end_conversation=False):
    global chat_history

    product_answer = query_pinecone(user_query)

    context_str = ""
    for entry in chat_history:
        context_str += f"Q: {entry['user']}\nA: {entry['assistant']}\n"

    chat_history.append({"user": user_query, "assistant": product_answer})

    
    inputs = {
        "context": context_str,
        "Question": user_query,
    }

    if end_conversation:
        chat_history = []  
        return "Anything else you want to know?"  

    
    return product_answer

template = """You are an intelligent and human-like assistant that answers customer queries related to movie and TV show recommendations on Netflix.  
Your primary role is to help the customer by providing accurate, concise, and helpful suggestions based on their preferences.

### Recommendation Database Information:
You have access to a database with common movie-related questions, genres, and corresponding recommendations.  
Whenever the customer asks for a recommendation, you should first check for the most similar genre or type of movie in the database and provide suggestions accordingly.

If the customer asks something that is not explicitly covered in the database, use the context provided below to generate appropriate movie or TV show recommendations based on the available information.

### Context:
This is the conversation so far between you and the customer. Ensure that the response stays relevant to the ongoing discussion:
{context}

### Instructions:
1. **Use the context first**: Refer to the previous conversation to provide recommendations that fit the customer’s preferences.
2. **Query Matching**: If the customer query matches a question or genre in the database, offer movie or TV show suggestions from the database.
3. **Handle Out-of-Scope Questions**: If there’s no close match in the database, generate a list of movie recommendations based on genres, themes, and features from the current Netflix library.
4. **Politeness and Clarity**: Always respond politely, clearly, and concisely. Ensure the recommendations match the customer's mood or genre preferences.
5. **Summarize if asked**: If the user asks for a summary, extract the key points from the chat history and summarize their preferences and recommendations.
6. **Incorporate Dynamic Responses**: Use Netflix’s latest titles and critically acclaimed films that match the customer’s specific preferences when possible.

### Answer the following:
Question: {Question}
"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": RunnablePassthrough(), "Question": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def detect_end_of_conversation(user_query):
    end_phrases = ["Thanks"]
    s = user_query.lower()
    s1 = s.split()
    for i in s1:
        if i in end_phrases:
            return True
    return False


def text_to_speech(text, language_code="en-US"):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    
    if language_code == "hi-IN":
        voice = texttospeech.VoiceSelectionParams(
            language_code="hi-IN",
            name="hi-IN-Wavenet-A",  
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

    
    output_dir = "static/audio"
    os.makedirs(output_dir, exist_ok=True)  

    audio_filename = f"response_{int(time.time())}.mp3"
    audio_path = os.path.join(output_dir, audio_filename)

    with open(audio_path, "wb") as out:
        out.write(response.audio_content)

    return audio_path


def process_transcription_step(transcription):
    if transcription is None or transcription.strip() == "":
        return "No transcription available."

    translated_text, original_language = translate_to_english(transcription)
    
    response = generate_response(translated_text)

    final_response = translate_to_original_language(response, original_language)

    return final_response



def process_transcription_and_generate_audio(transcription):
    text_response = process_transcription_step(transcription)

    translated_text, original_language = translate_to_english(transcription)

    language_code = "hi-IN" if original_language == "hi" else "en-US"

    audio_file_path = text_to_speech(text_response, language_code=language_code)

    return text_response, audio_file_path

def stop_audio():
    stop_playback()
    return "Playback stopped"

@app.route('/')
def indexx():
    return render_template('index_loan.html', banks=BANK_PHONE_NUMBERS.keys(),chat_history=chat_history)

@app.route('/submit_audio', methods=['POST'])
def submit_audio():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    file.save(temp_audio_path)

    transcription = transcribe_audio(temp_audio_path)
    return jsonify(transcription=transcription)

@app.route('/process_query', methods=['POST'])
def process_query():
    transcription = request.form['transcription']
    bank_name = request.form['bank_name']
    translated_text, original_language = translate_to_english(transcription)
    response = generate_response(translated_text)
    final_response = translate_to_original_language(response, original_language)
    phone_number = BANK_PHONE_NUMBERS.get(bank_name, "Unknown")
    language_code = "hi-IN" if original_language == "hi" else "en-US"
    audio_path = text_to_speech(final_response, language_code=language_code)
    audio_filename = os.path.basename(audio_path)
    return jsonify(response=final_response, phone_number=phone_number, audio_url=f"/static/audio/{audio_filename}",chat_history=chat_history)

@app.route('/handle_choice', methods=['POST'])
def handle_choice():
    global chat_history
    user_choice = request.form['choice']
    
    if user_choice == "Yes":
        message = "Your history saved"
    else:
        chat_history = []
        message = "Chat history reset."

    return jsonify(message=message, chat_history=chat_history)


if __name__ == '__main__':
    app.run(debug=True)


