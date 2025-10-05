import os
import json
import time
import requests
from glob import glob
from typing import Optional, List, Dict, Any

# ==============================================================================
# 1. Configuration Area (Edit these variables)
# ==============================================================================

# **MANDATORY**: Paste your Gemini API Key here.
GEMINI_API_KEY = "AIzaSyDXLZN9A_0p97RvEyfp4JvD5JiOE4Sdo9A"

# **MANDATORY**: Specify the directory path where your 500 JSON files are located.
DATA_DIRECTORY = "./chuncked_json"

# INTERNAL: File where the combined JSON data is temporarily stored.
# This prevents reading 500 files every time you run the script.
COMBINED_DATA_FILE = "combined_json_cache.txt" 

# ==============================================================================
# 2. Core Functions
# ==============================================================================

GEMINI_MODEL = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

def combine_json_files(directory_path: str) -> Optional[str]:
    """
    Reads all JSON files, validates them, and returns a single large JSON array string.
    This function is run only if the cache file (COMBINED_DATA_FILE) is missing.
    """
    print(f"Searching for JSON files in: {directory_path}...")
    json_files = glob(os.path.join(directory_path, "*.json"))
    
    if not json_files:
        print(f"Error: No .json files found in the directory '{directory_path}'.")
        return None

    print(f"Found {len(json_files)} files. Reading and combining data...")
    combined_contents = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                content_string = json.dumps(content)
                combined_contents.append(content_string)
        except Exception as e:
            print(f"Warning: Skipping file {file_path} due to error: {e}")

    if not combined_contents:
        print("Error: All found files were invalid or empty.")
        return None
        
    print(f"Successfully combined {len(combined_contents)} valid file(s) into a single JSON array.")
    return "[" + ",\n".join(combined_contents) + "]"


def send_chat_message(api_key: str, history: List[Dict[str, Any]], max_retries: int = 5) -> Optional[str]:
    """
    Sends the entire chat history (which includes the data injection) to the Gemini API.
    """
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print("Error: Please set your GEMINI_API_KEY in the configuration area.")
        return None

    # This system instruction defines the model's role as a data analyst
    system_prompt = (
        "You are a specialized large-scale data analyst AI. All preceding user turns contain "
        "a massive JSON dataset that you must use as your core reference. Respond to subsequent "
        "user queries based ONLY on the data provided, providing professional, concise, "
        "and accurate analysis formatted clearly using Markdown."
    )

    payload = {
        "contents": history, # The full history array is sent here
        "tools": [{"google_search": {} }], # Optional: grounding for external context
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    headers = {
        'Content-Type': 'application/json'
    }

    print("\nSending query (and history) to Gemini API...")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_URL}?key={api_key}", 
                headers=headers, 
                data=json.dumps(payload)
            )
            response.raise_for_status() 
            
            result = response.json()
            
            # Extract the text from the response
            text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No response text found.')
            return text

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                delay = 2 ** attempt
                print(f"Rate limit hit (429). Retrying in {delay} seconds...")
                time.sleep(delay)
                continue
            elif response.status_code == 400:
                print(f"API Error (400 - Bad Request). Check payload size (too large?). Details: {response.text}")
                return None
            else:
                print(f"HTTP Error: {e}\nResponse: {response.text}")
                return None
        
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
            
    return None


def interactive_chat_loop(data_path: str):
    """
    Loads the data cache and runs the stateful interactive chat session.
    """
    print("\n" + "="*70)
    print(f"L O A D I N G   D A T A   F R O M   C A C H E: {data_path}")
    print("="*70)

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            combined_json_string = f.read()
    except FileNotFoundError:
        print(f"Critical Error: Data cache file '{data_path}' not found. Run the script once to generate it.")
        return

    print("Data loaded successfully. Starting stateful chat session.")

    # Initialize chat history. This first turn is the DATA INJECTION ("feeding").
    chat_history: List[Dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "The following content is the full, combined dataset you must use for "
                        "all subsequent analysis. Confirm receipt of this data, and state you "
                        "are ready for the first question. Do not summarize the data yet."
                    )
                },
                # The massive data is added here as part of the initial user turn
                {
                    "text": f"--- DATA START ---\n{combined_json_string}\n--- DATA END ---"
                }
            ]
        }
    ]

    # Send the data injection message (Turn 1)
    initial_response_text = send_chat_message(GEMINI_API_KEY, chat_history)
    
    if initial_response_text is None:
        print("Failed to initialize chat session with data. Exiting.")
        return
        
    # The model's confirmation response is added to the history
    chat_history.append({
        "role": "model",
        "parts": [{"text": initial_response_text}]
    })
    
    print("\n" + "="*50)
    print("         G E M I N I   R E S P O N S E")
    print("="*50)
    print(initial_response_text) # Show the model's confirmation
    print("="*50)

    # Start the interactive loop
    while True:
        print("-" * 70)
        user_prompt = input("Enter analysis question (or type 'exit' or 'quit'):\n> ")
        
        if user_prompt.lower() in ['exit', 'quit']:
            print("Exiting stateful analysis mode. Goodbye!")
            break
        
        if not user_prompt.strip():
            print("Please enter a non-empty query.")
            continue

        # Add the new user prompt to the history
        chat_history.append({
            "role": "user",
            "parts": [{"text": user_prompt}]
        })

        # Send the entire updated history array (data + all turns)
        model_response_text = send_chat_message(GEMINI_API_KEY, chat_history)

        if model_response_text is None:
            # If the request failed, remove the user's last turn so they can try again
            chat_history.pop() 
            continue
        
        # Add the new model response to the history
        chat_history.append({
            "role": "model",
            "parts": [{"text": model_response_text}]
        })
        
        print("\n" + "="*50)
        print("         G E M I N I   A N A L Y S I S")
        print("="*50)
        print(model_response_text)
        print("="*50)


# ==============================================================================
# 3. Execution Block
# ==============================================================================

if __name__ == "__main__":
    
    # Setup directory check (retains the logic you selected in the Canvas)
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"Directory not found: '{DATA_DIRECTORY}'. Creating it now. Please populate it with your JSON files.")
        os.makedirs(DATA_DIRECTORY)

    # Check for the cached combined data file
    if not os.path.exists(COMBINED_DATA_FILE):
        print("\n--- FIRST RUN: Combining data and creating cache file ---")
        combined_json_string = combine_json_files(DATA_DIRECTORY)
        
        if combined_json_string:
            # Save the result for future runs
            try:
                with open(COMBINED_DATA_FILE, 'w', encoding='utf-8') as f:
                    f.write(combined_json_string)
                print(f"\nData successfully saved to cache file: {COMBINED_DATA_FILE}")
                print("Starting interactive query loop...")
                interactive_chat_loop(COMBINED_DATA_FILE)
            except Exception as e:
                print(f"Error saving data to cache file: {e}")
        else:
            print("\nProcess aborted due to data combination failure.")

    else:
        # Cache file already exists, jump directly to interactive mode
        print(f"\n--- DATA ALREADY FED (Cache file {COMBINED_DATA_FILE} found) ---")
        interactive_chat_loop(COMBINED_DATA_FILE)
