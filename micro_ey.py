import os
import time
import logging
import base64
import threading
import tempfile
import json
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

GOOGLE_API_KEY = "AIzaSyCqNzDqQ6grXOKAdLIkOKjcD0AIqApNcGg"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = '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'

translate_client = translate.Client()
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key="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")



pc = Pinecone(api_key="9f8aa1d2-88ea-41a8-a731-bde767f0bfff")
pinecone_index_name = "gov-schemes"

index = pc.Index('gov-schemes')

df = pd.read_csv("EY_GovPolicies - Set 1.csv")
df['id'] = df.index.astype(str) 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
df['id'] = df['id'].astype('object')
for i in range(len(df)):
    string = df.iloc[i]['How It Can Help You']
    embeddings = genai.embed_content(
        model="models/text-embedding-004",
        content=string
    )
    index.upsert(
        vectors=[
            {"id": "prodd" + str(i), "values": embeddings['embedding']}
        ]
    )
    df.at[i, 'id'] = "prodd" + str(i)

full_chat_history = []
user_data = {}


INVESTMENT_OPTIONS = {
    "low": [
        "Fixed Deposits", "Public Provident Fund (PPF)", "Government Bonds", "Recurring Deposits", "Gold ETFs"
    ],
    "medium": [
        "Debt Mutual Funds", "Hybrid Mutual Funds", "Index Funds", "Low Volatility ETFs"
    ],
    "high": [
        "Stocks", "Equity Mutual Funds", "Sectoral ETFs", "Cryptocurrency (high risk)"
    ]
}
def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio = AudioSegment.from_file(audio_file, format="webm")  
        audio.export(temp_audio_file.name, format="wav")

        with open(temp_audio_file.name, "rb") as audio:
            transcription = "Sample Transcription" 
    
    return transcription


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
    """Search Pinecone for relevant government schemes."""
    try:
        
        embedding_response = genai.embed_content(model="models/text-embedding-004", content=user_query)

        if not embedding_response or "embedding" not in embedding_response:
            raise ValueError("Failed to generate embeddings for the query.")

        embeddings = embedding_response["embedding"]

        if not isinstance(embeddings, list) or len(embeddings) == 0:
            raise ValueError("Generated embeddings are empty or invalid.")

        query_results = index.query(vector=embeddings, top_k=5, include_metadata=True)

        if not query_results["matches"]:
            return "No suitable government scheme found."

        best_match = max(query_results["matches"], key=lambda x: x["score"])
        scheme_id = best_match["id"]

        scheme_row = df[df["id"] == scheme_id]

        if scheme_row.empty:
            return "No relevant government scheme found in the dataset."

        return scheme_row.iloc[0].to_dict()

    except Exception as e:
        logging.error(f"Pinecone query error: {e}")
        return "Error retrieving scheme."

@app.route('/')
def indexx():
    return render_template('index_micro.html')


@app.route('/get_user_input', methods=['POST'])
def get_user_input():
    """ Capture user preferences and store them """
    data = request.json
    user_data['risk'] = data.get("risk_level", "medium")
    user_data['income'] = data.get("income", 10000)
    user_data['time_horizon'] = data.get("investment_period", 1)
    
    return jsonify({"message": "User preferences saved", "user_data": user_data})


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


@app.route('/get_investment_recommendation', methods=['POST'])
def get_investment_recommendation():
    """ Receive user details & voice input, send to LLM for investment suggestions """
    data = request.json
    transcription = data.get('transcription', '')
    risk_level = data.get('risk_level', 'medium')
    income = data.get('income', 10000)
    time_horizon = data.get('investment_period', 1)
    voice_input = data.get('voice_input', '')
    #transcription = request.form['transcription']
    translated_text, original_language = translate_to_english(transcription)
    prompt = f"""
    A rural Indian user wants investment suggestions based on:
    - Risk Level: {risk_level}
    - Monthly Income: {income} INR
    - Investment Period: {time_horizon} years
    - Additional input: "{translated_text}"

    Suggest the best micro-investment opportunities in India. 
    Return a JSON list with fields:
    {{"Investment Type": "Name", "Expected Returns": "low/medium/high", "Min Investment": "INR", "Liquidity": "low/medium/high", "Details": "More information on how to avail the investment"}}
    """

    llm_response = llm.invoke(prompt)
    response_content = llm_response.content.strip()

    if response_content.startswith("```json") and response_content.endswith("```"):
        response_content = response_content.replace("```json", "").replace("```", "").strip()

    try:
        investment_suggestions = json.loads(response_content)
    except json.JSONDecodeError:
        investment_suggestions = [{"error": "Invalid response from LLM"}]

    if original_language != "en":
        for item in investment_suggestions:
            for key in item:
                item[key] = translate_to_original_language(item[key], original_language)

    return jsonify({
        "investment_suggestions": investment_suggestions,
        "language": original_language,  
        "user_query": voice_input 
    })


@app.route('/government_schemes', methods=['POST'])
def government_schemes():
    """ Handles voice input, finds matching government schemes, and returns results in the correct language. """
    data = request.json
    voice_input = data.get('voice_input', '')

    translated_text, detected_language = translate_to_english(voice_input)

    scheme_response = query_pinecone(translated_text)

    if isinstance(scheme_response, str):
        final_response = translate_to_original_language(scheme_response, detected_language)
    elif isinstance(scheme_response, dict):
        final_response = "\n".join([f"{key}: {value}" for key, value in scheme_response.items()])
        final_response = translate_to_original_language(final_response, detected_language)
    else:
        final_response = "Error processing scheme response."

    return jsonify({
        "user_query": voice_input,
        "response": final_response,
        "language": detected_language
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
