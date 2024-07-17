import os
from dotenv import load_dotenv
import json
from flask import Flask, request ,jsonify
from langchain_community.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

app = Flask(__name__)

# Load variables from the .env file
load_dotenv()

# Access variables
api_key = os.environ.get("API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)

persist_directory = 'docs/chroma/'

# Function to be called when the server starts
def startup_function():
    print("Server is starting...")
    json_file = "SQLite.json_modified.json"
    num_messages_to_store = 1000
    texts = log_messages_to_list(json_file, num_messages_to_store)
    global vector_database
    vector_database = Chroma.from_texts(texts, embeddings)

# Route handler for /ask
@app.route('/ask', methods=['POST'])
def ask_handler():
    if request.method == 'POST':
        question = request.form.get('question')
        print (question)
        if question:
            # Similarity search with k = 5
            docs = vector_database.similarity_search(question, k=5)
            all_job_postings = ""
            for i in range(5):
                job_posting = f"Job {i+1}:\n"
                job_posting += f"  Text: {docs[i]}\n"
                job_posting += "-------------------\n"
                all_job_postings += job_posting

            # Generate response using generative model
            response = model.generate_content("""Input:

A list containing the text descriptions of 5 job postings.

{""" +  all_job_postings +"""}

Output:

a string of 3 arrays with no name and as elemts of another array. just the aquare brackets[] and the values intide them.

the first array should contain The title of the job (if available, otherwise "unknown").

the second array should contain The name of the company offering the job (if available, otherwise "unknown").

the third array should contain The location of the company (city, state, country, etc., if available, otherwise "unknown", if it is "Anywhere/remote" also mark it as "unknown" ).""")


           
            print(response.text)
            return jsonify(response.text)
        else:
            return 'No question provided'
    else:
        return 'Invalid request method'

def log_messages_to_list(filename, n):
    """
    Reads messages from a JSON file and stores the first n messages in a list named 'splits'.

    Args:
        filename: Path to the JSON file.
        n: Number of messages to store.
    """

    with open(filename, "r") as f:
        data = json.load(f)

    # Ensure data is a list of messages
    if not isinstance(data, list):
        raise ValueError("JSON data is not a list of messages.")

    splits = []  # Initialize an empty list to store messages

    for message in data[:n]:
        if len(splits) >= n:  # Limit to n messages
            break

        try:
            message_content = message["message"]
            splits.append(message_content)  # Add message content to the list
        except KeyError:
            print(f"Message does not contain a 'message' key.")

    return splits  # Return the list of messages

if __name__ == '__main__':
    # Call the startup function when the server starts
    startup_function()

    # Run the Flask app
    app.run(debug=True)