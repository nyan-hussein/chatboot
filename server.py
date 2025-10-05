from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import requests
from glob import glob

# ==============================================================================
# Configuration
# ==============================================================================

GEMINI_API_KEY = "AIzaSyDXLZN9A_0p97RvEyfp4JvD5JiOE4Sdo9A"
DATA_DIRECTORY = "./chuncked_json"
COMBINED_DATA_FILE = "combined_json_cache.txt"
GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# ==============================================================================
# Flask App Setup
# ==============================================================================

app = Flask(__name__)
CORS(app)

# Global variable to store the combined data and chat history
combined_data = None
chat_history = []

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_data():
    """Load combined JSON data from cache or create it."""
    global combined_data
    
    if os.path.exists(COMBINED_DATA_FILE):
        with open(COMBINED_DATA_FILE, 'r', encoding='utf-8') as f:
            combined_data = f.read()
        print("Data loaded from cache.")
    else:
        print("Creating data cache...")
        json_files = glob(os.path.join(DATA_DIRECTORY, "*.json"))
        combined_contents = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    combined_contents.append(json.dumps(content))
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
        
        if combined_contents:
            combined_data = "[" + ",\n".join(combined_contents) + "]"
            with open(COMBINED_DATA_FILE, 'w', encoding='utf-8') as f:
                f.write(combined_data)
            print("Data cache created.")
        else:
            print("No data found!")


def initialize_history():
    """Initialize chat history with data injection."""
    global chat_history
    
    chat_history = [
        {
            "role": "user",
            "parts": [
                {"text": "You can ONLY answer using the following dataset. Never use external knowledge."},
                {"text": f"--- DATA ---\n{combined_data}\n--- END ---"}
            ]
        }
    ]
    
    # Get initial confirmation from Gemini
    system_prompt = (
        "You are a data analyst. You can ONLY answer questions using the provided JSON dataset. "
        "If a question cannot be answered from the dataset, say so. "
        "Never use external knowledge or web search."
    )
    
    payload = {
        "contents": chat_history,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.1, "topP": 0.8, "topK": 20}
    }
    
    response = requests.post(
        f"{API_URL}?key={GEMINI_API_KEY}",
        headers={'Content-Type': 'application/json'},
        json=payload
    )
    
    if response.ok:
        result = response.json()
        text = result['candidates'][0]['content']['parts'][0]['text']
        chat_history.append({"role": "model", "parts": [{"text": text}]})


# ==============================================================================
# Single Route
# ==============================================================================

@app.route('/chat', methods=['POST'])
def chat():
    """Send a message and get a response."""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    user_message = data['message']
    
    # Add user message to history
    chat_history.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })
    
    # Send to Gemini
    system_prompt = (
        "You are a data analyst. You can ONLY answer questions using the provided JSON dataset. "
        "If a question cannot be answered from the dataset, say so. "
        "Never use external knowledge or web search."
    )
    
    payload = {
        "contents": chat_history,
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.1, "topP": 0.8, "topK": 20}
    }
    
    try:
        response = requests.post(
            f"{API_URL}?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result['candidates'][0]['content']['parts'][0]['text']
        
        # Add response to history
        chat_history.append({
            "role": "model",
            "parts": [{"text": response_text}]
        })
        
        return jsonify({"response": response_text}), 200
        
    except Exception as e:
        chat_history.pop()  # Remove failed message
        return jsonify({"error": str(e)}), 500


# ==============================================================================
# Startup
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  GEMINI CHAT SERVER")
    print("="*50)
    
    # Load data on startup
    load_data()
    
    if combined_data:
        initialize_history()
        print("Ready! POST to http://localhost:8080/chat")
        print('Example: {"message": "What data do you have?"}')
        print("="*50 + "\n")
        app.run(host='0.0.0.0', port=8080, debug=True)
    else:
        print("ERROR: No data loaded. Check your JSON files.")
        print("="*50)