import os
import json
import io
import base64
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from collections import defaultdict
from google.cloud import translate_v2 as translate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydub import AudioSegment
from groq import Groq
from rapidfuzz import process as fuzz_process

app = Flask(__name__)

GOOGLE_API_KEY = "AIzaSyCqNzDqQ6grXOKAdLIkOKjcD0AIqApNcGg"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"translation.json"
translate_client = translate.Client()
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key="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")
chat_history_expenses = []
chat_history_earnings = []
full_chat_history = []
expense_data = defaultdict(lambda: defaultdict(float))
earning_data = defaultdict(lambda: defaultdict(float))

def transcribe_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        audio = AudioSegment.from_file(audio_file)
        audio.export(temp_audio_file.name, format="wav")
        with open(temp_audio_file.name, "rb") as audio:
            transcription = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=audio)
    return transcription.text

def translate_to_english(text):
    detection = translate_client.detect_language(text)
    detected_language = detection['language']
    if detected_language != "en":
        translation = translate_client.translate(text, target_language="en")
        return translation['translatedText'], detected_language
    return text, detected_language

def process_voice_input(audio_file):
    text = transcribe_audio(audio_file)
    translated_text, detected_language = translate_to_english(text)

    full_chat_history.append({"original": text, "translated": translated_text, "language": detected_language})

    prompt = f"""
    Classify the following statement into 'expense' or 'earning'. Extract:
    - category: The broad category (e.g., Food, Loan, Crop)
    - sub_category: The specific subcategory (e.g., Wheat, Electricity, Fertilizers)
    - amount: The numeric value in INR
    - type: 'expense' if money is spent, 'earning' if money is earned
    
    Example Inputs:
    - "I spent 3000 rupees on fertilizers" → Expense (Category: Crop, Sub-category: Fertilizers, Amount: 3000)
    - "I earned 5000 rupees by selling wheat" → Earning (Category: Crop, Sub-category: Wheat, Amount: 5000)
    
    Statement: '{translated_text}'
    
    Return JSON format:
    {{
        "type": "expense" or "earning",
        "amount": 1234,
        "category": "Category Name",
        "sub_category": "Sub-category Name"
    }}
    """

    llm_response = llm.invoke(prompt)
    response_content = llm_response.content.strip()

    if response_content.startswith("```json") and response_content.endswith("```"):
        response_content = response_content.replace("```json", "").replace("```", "").strip()

    parsed_response = json.loads(response_content)
    transaction_type = parsed_response.get("type", "")
    amount = parsed_response.get("amount", 0)
    category = parsed_response.get("category", "Unknown")
    sub_category = parsed_response.get("sub_category", "Unknown")

    if transaction_type == "expense":
        chat_history_expenses.append(translated_text)
        expense_data[category][sub_category] += amount
    elif transaction_type == "earning":
        chat_history_earnings.append(translated_text)
        earning_data[category][sub_category] += amount

    return f"Processed: {transaction_type} of {amount} INR for {sub_category} ({category})"



def generate_pie_chart(data, title):
    total_per_category = {cat: sum(sub.values()) for cat, sub in data.items()}
    plt.figure(figsize=(6, 4))
    plt.pie(total_per_category.values(), labels=total_per_category.keys(), autopct="%1.1f%%")
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_bar_chart(data, title):
    sub_categories = []
    amounts = []
    for category, sub_data in data.items():
        for sub, amount in sub_data.items():
            sub_categories.append(f"{category} - {sub}")
            amounts.append(amount)

    plt.figure(figsize=(8, 5))
    plt.bar(sub_categories, amounts)
    plt.xlabel("Sub-category")
    plt.ylabel("Amount (INR)")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def search_chat_history(audio_file, full_chat_history):
    try:
        # Step 1: Convert Audio to Text (Speech-to-Text)
        #query_text = transcribe_audio(audio_file)

        # Step 2: Translate Query if Needed
        #translated_query, detected_language = translate_to_english(query_text)
        #print(f"Original Query: {query_text} | Translated Query: {translated_query} | Language: {detected_language}")

        # Step 3: Prepare LLM Prompt for Searching Chat History
        prompt = (
            f"You are an AI assistant. Search the following chat history and return the most relevant entries for the given query.\n"
            f"Chat History:\n{json.dumps(full_chat_history, indent=2)}\n\n"
            f"Query: '{audio_file}'\n\n"
            f"Find and return the most relevant results in JSON format as an array of matched entries."
        )

        llm_response = llm.invoke(prompt)
        response_content = llm_response.content.strip()

        
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content.replace("```json", "").replace("```", "").strip()

        try:
            matched_entries = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(f"Response Content: {response_content}")
            return []

        matched_entries = [str(entry) for entry in matched_entries]

        return matched_entries

    except Exception as e:
        print(f"Error in search_chat_history: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('indexeybud.html',
                           expense_pie=generate_pie_chart(expense_data, "Expense Categories"),
                           expense_bar=generate_bar_chart(expense_data, "Expense Sub-categories"),
                           earning_pie=generate_pie_chart(earning_data, "Earning Categories"),
                           earning_bar=generate_bar_chart(earning_data, "Earning Sub-categories"),
                           total_expenditure=sum(sum(sub.values()) for sub in expense_data.values()),
                           total_earnings=sum(sum(sub.values()) for sub in earning_data.values()),
                           chat_history_expenses=chat_history_expenses,
                           chat_history_earnings=chat_history_earnings)

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    response_message = process_voice_input(audio_file)

    return jsonify({
        "message": response_message,
        "expense_pie": generate_pie_chart(expense_data, "Expense Categories"),
        "expense_bar": generate_bar_chart(expense_data, "Expense Sub-categories"),
        "earning_pie": generate_pie_chart(earning_data, "Earning Categories"),
        "earning_bar": generate_bar_chart(earning_data, "Earning Sub-categories"),
        "total_expenditure": sum(sum(sub.values()) for sub in expense_data.values()),
        "total_earnings": sum(sum(sub.values()) for sub in earning_data.values()),
        "chat_history_expenses": chat_history_expenses,
        "chat_history_earnings": chat_history_earnings
    })



@app.route('/search_chat', methods=['POST'])
def search_chat():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400
    # I am translating here only
    audio_file = request.files['audio']
    query_text = transcribe_audio(audio_file)
    translated_query, _ = translate_to_english(query_text)
    
    results = search_chat_history(translated_query,full_chat_history)
    
    return jsonify({"results": results})
@app.route('/get_budget_advice', methods=['POST'])
def get_budget_advice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files['audio']
    user_text = transcribe_audio(audio_file)
    translated_text, detected_language = translate_to_english(user_text)

    prompt = f"""
    You are a financial advisor providing budget management recommendations. 
    Use the following expense and earning history to analyze the user's financial pattern:

    Expense History:
    {json.dumps(chat_history_expenses, indent=2)}

    Earning History:
    {json.dumps(chat_history_earnings, indent=2)}

    The user has asked for budget advice: '{translated_text}'

    Based on their expense and earning history, provide recommendations on how they can improve their financial management.
    """

    llm_response = llm.invoke(prompt)
    response_content = llm_response.content.strip()

    return jsonify({"query": translated_text, "advice": response_content})
if __name__ == '__main__':
    app.run(debug=True)
